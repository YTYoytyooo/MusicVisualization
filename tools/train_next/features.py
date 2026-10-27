"""Versioned, resumable frozen CLAP + acoustic feature extraction."""
from __future__ import annotations

import json
import hashlib
from importlib.metadata import version
from pathlib import Path
import time
import uuid

import numpy as np
import soundfile as sf
import torch

from .core import ROOT, sha256


def local_clap_snapshot():
    repository = ROOT / 'versions/v2.0/.runtime/huggingface/hub/models--laion--clap-htsat-fused'
    revision = (repository / 'refs/main').read_text().strip()
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Invalid local CLAP revision')
    snapshot = repository / 'snapshots' / revision
    for filename in ('config.json', 'model.safetensors', 'preprocessor_config.json'):
        if not (snapshot / filename).is_file():
            raise FileNotFoundError(f'Local CLAP file missing: {filename}')
    return snapshot


def centered_window(audio, sample_rate, timestamp, seconds=2):
    size = round(seconds * sample_rate)
    start = round((timestamp - seconds / 2) * sample_rate)
    result = np.zeros(size, dtype=np.float32)
    source_start, source_end = max(0, start), min(len(audio), start + size)
    if source_end > source_start:
        result[source_start - start:source_end - start] = audio[source_start:source_end]
    return result


def acoustic_features(audio, sample_rate, times):
    import librosa
    y = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050, res_type='soxr_hq')
    magnitude = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    power = magnitude ** 2
    columns = [
        np.log(librosa.feature.rms(y=y, frame_length=2048, hop_length=512) + 1e-7),
        librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512),
        np.log1p(librosa.feature.spectral_centroid(S=magnitude, sr=22050)),
        np.log1p(librosa.feature.spectral_bandwidth(S=magnitude, sr=22050)),
        np.log1p(librosa.feature.spectral_rolloff(S=magnitude, sr=22050)),
        librosa.feature.spectral_flatness(S=magnitude),
        librosa.onset.onset_strength(S=librosa.power_to_db(power), sr=22050, hop_length=512)[None],
        librosa.feature.chroma_stft(S=power, sr=22050, hop_length=512, tuning=0),
    ]
    matrix = np.concatenate(columns, axis=0)
    clock = librosa.frames_to_time(np.arange(matrix.shape[1]), sr=22050, hop_length=512)
    return np.stack([np.interp(times, clock, row) for row in matrix], axis=1).astype(np.float32)


def feature_spec(snapshot):
    return dict(schema_version=1, extractor_sha256=sha256(__file__), clap_revision=snapshot.name,
                clap_weights_sha256=sha256(snapshot / 'model.safetensors'),
                clap_config_sha256=sha256(snapshot / 'config.json'),
                clap_processor_sha256=sha256(snapshot / 'preprocessor_config.json'),
                libraries={name: version(name) for name in ('torch', 'transformers', 'librosa', 'numpy', 'soundfile', 'scipy', 'soxr', 'numba')},
                hop_seconds=.5, clap_window_seconds=2, window_anchor='center',
                edge_padding='zeros', audio_channels='mean', clap_sample_rate=48000,
                clap_padding='repeatpad', clap_truncation='fusion', is_longer=False,
                acoustic_schema='log_rms,zcr,log_centroid,log_bandwidth,log_rolloff,flatness,onset,chroma12',
                dimensions=531)


class FeatureStore:
    def __init__(self, device, batch_size=8, progress=None):
        self.snapshot = local_clap_snapshot()
        self.spec = feature_spec(self.snapshot)
        self.key = hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()[:20]
        self.directory = ROOT / 'data/training/next_va/features' / self.key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device, self.batch_size, self.progress = device, batch_size, progress
        self.model = self.processor = None

    def _load(self):
        if self.model is None:
            from transformers import ClapFeatureExtractor, ClapModel
            self.processor = ClapFeatureExtractor.from_pretrained(self.snapshot, local_files_only=True)
            self.model = ClapModel.from_pretrained(self.snapshot, local_files_only=True).to(self.device).eval()

    def get(self, record, root):
        path = Path(root) / record['audio_path']
        if sha256(path) != record['audio_sha256']:
            raise ValueError(f'Source audio changed: {record["song_id"]}')
        destination = self.directory / (record['audio_sha256'] + '.npz')
        times = np.arange(int(np.floor(record['duration_seconds'] / .5)) + 1, dtype=np.float64) * .5
        if destination.is_file():
            with np.load(destination, allow_pickle=False) as cached:
                if str(cached['feature_key'].item()) != self.key:
                    raise ValueError('Feature cache configuration mismatch')
                np.testing.assert_array_equal(cached['times'], times)
                values = cached['features'].copy()
            self._validate(values, len(times))
            return times, values, True
        self._load()
        audio, sample_rate = sf.read(path, dtype='float32', always_2d=True)
        if audio.shape != (record['decoded_frames'], record['channels']) or sample_rate != record['sample_rate']:
            raise ValueError(f'Audio decode no longer matches prepared data: {record["song_id"]}')
        mono = audio.mean(axis=1)
        import librosa
        audio48 = librosa.resample(mono, orig_sr=sample_rate, target_sr=48000, res_type='soxr_hq')
        embeddings = []
        started = time.monotonic()
        for start in range(0, len(times), self.batch_size):
            selected = times[start:start + self.batch_size]
            windows = [centered_window(audio48, 48000, t) for t in selected]
            inputs = self.processor(windows, sampling_rate=48000, return_tensors='pt',
                                    padding='repeatpad', truncation='fusion')
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            # Every source window is exactly 2 s. Avoid the feature extractor's
            # random choice of a nominally "long" sample in an all-short batch.
            is_longer = torch.zeros((len(selected), 1), dtype=torch.bool, device=self.device)
            with torch.inference_mode():
                output = self.model.audio_model(input_features=inputs['input_features'], is_longer=is_longer)
                embedding = self.model.audio_projection(output.pooler_output)
            embeddings.append(embedding.cpu().float().numpy())
            if self.progress:
                self.progress(dict(song_id=record['song_id'], frames_done=start + len(selected),
                                   frames_total=len(times), song_seconds=round(time.monotonic() - started, 1)))
        values = np.concatenate([np.concatenate(embeddings), acoustic_features(mono, sample_rate, times)], axis=1).astype(np.float32)
        self._validate(values, len(times))
        temporary = destination.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, times=times, features=values, feature_key=self.key)
        temporary.replace(destination)
        return times, values, False

    @staticmethod
    def _validate(values, length):
        if values.shape != (length, 531) or not np.isfinite(values).all():
            raise ValueError('Invalid feature matrix')

    def unload(self):
        self.model = self.processor = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
