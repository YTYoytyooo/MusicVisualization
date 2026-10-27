"""Frozen B00 inference with explicit units and no usable confidence claim."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from tools.train_next.core import atomic_json, sha256
from tools.train_next.features import FeatureStore, feature_spec, local_clap_snapshot
from tools.train_next_v2.common import ROOT, AXES, read_json, verify_hashes
from tools.train_next_v2.calibration.job_v2 import load_models


def contained(path):
    value = (ROOT / path).resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError('Research inputs and outputs must stay inside Vm')
    return value


def validate_features(features):
    x = np.asarray(features)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 531 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty raw float32 CLAP512 + acoustic19 features')
    return x


def standardize(features, mean, std):
    x = validate_features(features)
    if (mean.shape != (531,) or std.shape != (531,) or mean.dtype != np.float32
            or std.dtype != np.float32 or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError('Invalid frozen training scaler; refitting is forbidden')
    result = (x - mean) / std
    if not np.isfinite(result).all():
        raise ValueError('Non-finite standardized input')
    return result


def validate_clock(times, frames, duration_seconds):
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError('Invalid native decoded duration')
    expected = np.arange(math.floor(duration_seconds / .5) + 1, dtype=np.float64) * .5
    t = np.asarray(times)
    if t.shape != (frames,) or not np.array_equal(t, expected):
        raise ValueError('Feature clock must match the unshifted frozen half-second grid')


def combine_members(member_predictions):
    members = np.asarray(member_predictions, dtype=np.float64)
    if (members.ndim != 3 or members.shape[0] != 3 or members.shape[2] != 2
            or members.shape[1] == 0 or not np.isfinite(members).all()
            or np.any(np.abs(members) > 1)):
        raise ValueError('Invalid three-member raw V/A output')
    return dict(raw_va=members.mean(axis=0), seed_disagreement_std=members.std(axis=0, ddof=1),
        disagreement_is_calibrated_confidence=False,
        calibration=dict(status='unavailable_failed_audit', intervals=None,
                         reason='The frozen grouped calibration audit failed eligibility; agreement is not accuracy.'),
        display=dict(status='not_applied', values=None),
        tension=dict(status='unvalidated_no_supervision', values=None))


def acoustic_properties(features):
    x = validate_features(features).astype(np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        rms = np.maximum(np.exp(x[:, 512]) - 1e-7, 0.)
        centroid = np.expm1(x[:, 514])
    if not np.isfinite(rms).all() or not np.isfinite(centroid).all() or np.any(centroid < 0):
        raise ValueError('Invalid acoustic magnitude feature')
    floor = 1e-8
    return dict(rms_amplitude=rms, rms_dbfs=20 * np.log10(np.maximum(rms, floor)),
        spectral_centroid_hz=centroid, dbfs_floor_amplitude=floor,
        meaning=dict(energy='RMS relative amplitude/dBFS, not psychological arousal or LUFS.',
                     brightness='Spectral centroid in Hz, not a normalized visual brightness control.'),
        method='Invert the unnormalized training log-RMS and log1p-centroid features. Mono 22050 Hz, 2048-sample frames, 512-sample hop; log features interpolated on the frozen 0.5 s clock. These are feature-derived local acoustic measures, not song-level loudness.')


class ResearchPredictor:
    def __init__(self, manifest_path, device='cpu'):
        self.manifest_path = contained(manifest_path)
        self.manifest = read_json(self.manifest_path)
        self.manifest_sha256 = sha256(self.manifest_path)
        m = self.manifest
        if (m['id'] != 'research-reference-v1' or m['promoted_model'] is not False
                or m['label_order'] != list(AXES) or m['pseudo_training_enabled'] is not False):
            raise ValueError('Unsupported research manifest contract')
        verify_hashes(m['pinned_hashes'])
        b = read_json(contained(m['B_protocol']))
        if b['feature_spec'] != m['feature_spec'] or b['feature_key'] != m['feature_key']:
            raise ValueError('Frozen B feature contract mismatch')
        current = feature_spec(local_clap_snapshot())
        if current != m['feature_spec']:
            raise ValueError('Encoder, extractor or dependency version changed')
        expected_ids = [f'b00-reference-s{seed}' for seed in (20261003, 20261004, 20261005)]
        if [member['run_id'] for member in m['members']] != expected_ids:
            raise ValueError('Reference members changed')
        for member in m['members']:
            config = read_json(contained(member['config']))
            if (config['feature_key'] != m['feature_key'] or config['protocol_sha256'] != m['B_protocol_sha256']
                    or m['pinned_hashes'][member['checkpoint']] != member['checkpoint_sha256']):
                raise ValueError('Checkpoint feature/protocol binding changed')
        self.device = torch.device(device)
        # This frozen loader checks checkpoint provenance, axis order, strict
        # state loading and equality of all three training-only scalers.
        self.models, self.scaler = load_models(m, self.device)

    @torch.inference_mode()
    def predict_features(self, times, features, feature_key, duration_seconds):
        if feature_key != self.manifest['feature_key']:
            raise ValueError('Features belong to a different extractor')
        x = standardize(features, *self.scaler)
        validate_clock(times, len(x), duration_seconds)
        tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(x)], dtype=torch.int64)
        members = np.stack([model(tensor, lengths).cpu().numpy()[0] for model in self.models])
        return dict(schema_version=1, manifest_sha256=self.manifest_sha256,
            label_order=list(AXES), times_seconds=np.asarray(times, np.float64).copy(),
            duration_seconds=float(duration_seconds), prediction_scope='offline whole-file bidirectional inference',
            time_origin='Native decoded sample zero of this local file; no reaction-delay correction or time stretching.',
            feature_key=feature_key, member_raw_va=members, **combine_members(members),
            acoustic=acoustic_properties(features), promoted_model=False)


class FrozenFeatureStore(FeatureStore):
    """Use the exact training extractor in a separate receipted cache."""
    def __init__(self, manifest, device, directory, progress=None):
        super().__init__(torch.device(device), batch_size=8, progress=progress)
        if self.spec != manifest['feature_spec'] or self.key != manifest['feature_key']:
            raise ValueError('Feature extractor contract changed')
        self.directory = contained(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, record, root=ROOT):
        if Path(root).resolve() != ROOT:
            raise ValueError('Use the Vm workspace root')
        contained(record['audio_path'])
        cache = self.directory / (record['audio_sha256'] + '.npz')
        receipt = cache.with_suffix('.receipt.json')
        if cache.exists() != receipt.exists():
            raise ValueError('Unreceipted feature cache: preserve and inspect it before recovery')
        if cache.exists():
            prior = read_json(receipt)
            if (prior['audio_sha256'] != record['audio_sha256'] or prior['feature_key'] != self.key
                    or prior['cache_sha256'] != sha256(cache)):
                raise ValueError('Feature cache receipt mismatch')
        times, features, reused = super().get(record, root)
        validate_clock(times, len(features), record['duration_seconds'])
        if not reused:
            atomic_json(receipt, dict(audio_sha256=record['audio_sha256'], feature_key=self.key,
                cache_sha256=sha256(cache), frames=len(times), extractor_sha256=self.spec['extractor_sha256']))
        return times, features, reused


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
