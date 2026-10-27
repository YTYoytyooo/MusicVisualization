"""Verify screened clip classes and unchanged continuous-label training inputs."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.datasets.deam_dataset import contained_path
from tools.train_next.core import atomic_json, fit_scaler, sha256
from tools.train_next.runtime import is_running
from tools.train_next_v2.common import BASE, ROOT, GPU_LEASE, code_hashes, read_json, verify_hashes, now
from tools.train_next_v2.data import load_internal

PROTOCOL = BASE / 'protocols/d-merge-quadrant-v2.json'
CONTRACT = BASE / 'protocols/d-merge-training-inputs-v1.json'
FEATURE_JOB = BASE / 'feature_jobs/d-merge-clap-features-v2'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def array_digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def training_code_hashes():
    result = code_hashes()
    result.update({relative(p): sha256(p) for p in Path(__file__).parent.glob('*.py')})
    fixture = ROOT/'tools/train_next_v2/tests/test_experiments.py'
    result[relative(fixture)] = sha256(fixture)
    return result


def validate_candidates(rows, protocol, main_audio_hashes):
    if len(rows) != protocol['auxiliary_songs'] or len({r['song_id'] for r in rows}) != len(rows):
        raise ValueError('Auxiliary membership/count changed or duplicate song IDs')
    hashes = [r['audio']['audio_sha256'] for r in rows]
    if len(set(hashes)) != len(hashes) or set(hashes) & set(main_audio_hashes):
        raise ValueError('Auxiliary duplicate/overlapping audio')
    if dict(Counter(r['quadrant'] for r in rows)) != protocol['candidate_quadrants']:
        raise ValueError('Auxiliary quadrant membership changed')
    for row in rows:
        if (row['official_role'] != 'train' or row['quadrant'] not in ('Q1', 'Q2', 'Q3', 'Q4')
                or row['label_scope'] != 'whole_released_clip_static' or row['time_varying_labels']
                or row['numeric_regression_enabled'] or not row['audio_gate_passed']):
            raise ValueError('Invalid auxiliary role or label semantics: ' + row['song_id'])
        for field in ('exclusion_reasons', 'linked_heldout_ids', 'exact_main_identity_ids', 'possible_title_overlap_ids'):
            if row[field]:
                raise ValueError('Unresolved auxiliary identity exclusion: ' + field)
        for field in ('strict_decode', 'native_stderr_guard', 'bulk_stream_parity'):
            if not row[field]['passed'] or row[field]['audio_sha256'] != row['audio']['audio_sha256']:
                raise ValueError('Decoder gate is missing or stale: ' + field)
        if row['audio']['native_audio']['local_time_origin_seconds'] != 0:
            raise ValueError('Released clip must retain its own local time origin')


def read_auxiliary_feature(entry):
    path = contained_path(ROOT, entry['path'])
    verify_hashes({entry['path']: entry['sha256'], entry['receipt']: entry['receipt_sha256'],
                   entry['audio_path']: entry['audio_sha256']})
    receipt = read_json(ROOT / entry['receipt'])
    for key in ('song_id', 'audio_sha256', 'feature_key', 'path', 'sha256', 'frames', 'dimensions'):
        if receipt[key] != entry[key]:
            raise ValueError('Feature receipt identity differs: ' + key)
    times = np.arange(int(np.floor(entry['duration_seconds']/.5))+1, dtype=np.float64) * .5
    with np.load(path, allow_pickle=False) as cached:
        if set(cached.files) != {'times', 'features', 'feature_key'}:
            raise ValueError('Auxiliary cache contains unexpected targets or fields')
        np.testing.assert_array_equal(cached['times'], times)
        if str(cached['feature_key'].item()) != entry['feature_key']:
            raise ValueError('Auxiliary feature version differs')
        values = cached['features'].copy()
    if (values.dtype != np.float32 or values.shape != (len(times), entry['dimensions'])
            or entry['frames'] != len(times) or not np.isfinite(values).all()):
        raise ValueError('Auxiliary feature dimensions, clock or finite values invalid')
    if entry['class_index'] not in range(4) or entry['role'] != 'auxiliary_fit':
        raise ValueError('Invalid whole-clip class/role')
    return dict(song_id=entry['song_id'], group_id=entry['group_id'], audio_sha256=entry['audio_sha256'],
                x=values, times=times, class_index=entry['class_index'])


def main_membership(songs):
    return [dict(song_id=s['song_id'], role=role, source_dataset=s['source_dataset'],
        group_id=s['group_id'], audio_sha256=s['audio_sha256'],
        target_digest=array_digest(s['times'], s['y'], s['mask']), native_seconds=s['native_seconds'])
        for role in ('fit', 'development') for s in songs[role]]


def prepare_contract(device, progress=None):
    if CONTRACT.exists():
        raise FileExistsError('Preserve the existing frozen D input contract')
    if is_running(GPU_LEASE) or is_running('d-merge-clap-features-v2'):
        raise RuntimeError('Feature producer is still active; do not freeze partial inputs')
    protocol = read_json(PROTOCOL)
    bpath = ROOT / protocol['source_B_protocol']
    source = ROOT / protocol['auxiliary_candidates']
    verify_hashes({relative(bpath): protocol['source_B_protocol_sha256'],
                   relative(source): protocol['auxiliary_candidates_sha256']})
    b = read_json(bpath)
    report = read_json(FEATURE_JOB / 'report.json')
    launch = read_json(FEATURE_JOB / 'launch.json')
    verify_hashes(launch['protected_hashes'])
    if (report['status'] != 'completed_features_verified' or report['songs'] != protocol['auxiliary_songs']
            or report['launch_sha256'] != sha256(FEATURE_JOB/'launch.json')
            or report['source_sha256'] != sha256(source) or report['feature_key'] != b['feature_key']
            or launch['feature_spec'] != b['feature_spec'] or report['labels_used']):
        raise ValueError('Feature extraction has not passed the frozen completion contract')
    manifest = Path(b['source_manifest'])
    verify_hashes({relative(manifest): b['manifest_sha256']})
    # Identity metadata only: no heldout target files are read here.
    main_hashes = {json.loads(line)['audio_sha256'] for line in manifest.read_text(encoding='utf-8').splitlines() if line}
    rows = read_json(source)
    validate_candidates(rows, protocol, main_hashes)
    features = {r['song_id']: r for r in report['feature_manifest']}
    if len(features) != len(rows) or set(features) != {r['song_id'] for r in rows}:
        raise ValueError('Feature manifest IDs differ from the approved pool')
    admitted = []
    for row in sorted(rows, key=lambda r: r['song_id']):
        feature = features[row['song_id']]
        audio = row['audio']
        receipt = (ROOT / feature['path']).with_suffix('.json')
        if feature['audio_sha256'] != audio['audio_sha256'] or feature['dimensions'] != 531:
            raise ValueError('Feature/audio identity or baseline width differs')
        entry = dict(song_id=row['song_id'], group_id=row['group_id'], role='auxiliary_fit',
            class_index=int(row['quadrant'][1])-1, audio_path=audio['audio_path'],
            audio_sha256=audio['audio_sha256'], duration_seconds=audio['native_audio']['duration_seconds'],
            local_time_origin_seconds=0, source_song_offset_seconds=None,
            path=feature['path'], sha256=feature['sha256'], feature_key=feature['feature_key'],
            frames=feature['frames'], dimensions=feature['dimensions'],
            receipt=relative(receipt), receipt_sha256=sha256(receipt))
        read_auxiliary_feature(entry)
        admitted.append(entry)
    songs = load_internal(b, device, progress)
    if len(songs['fit']) != 3070 or len(songs['development']) != 383:
        raise ValueError('Main split must remain 3070 fit / 383 internal development')
    mean, std = fit_scaler(songs['fit'])
    history_path = ROOT/'data/training/next_va/completion-20261003.json'
    history = read_json(history_path)
    files = [PROTOCOL, bpath, source, FEATURE_JOB/'launch.json', FEATURE_JOB/'report.json', history_path,
        BASE/'analyses/d-merge-audio-audit-v1/report.json',
        BASE/'analyses/d-merge-native-decoder-v1/report.json', BASE/'analyses/d-merge-read-parity-v1/report.json']
    protected = {relative(p): sha256(p) for p in files}
    protected['models/emotion_model.pth'] = b['legacy_model_sha256']
    protected['models/next_va/mixed-20261003-recovery/best.pth'] = b['baseline_checkpoint_sha256']
    protected[relative(history['runs']['teacher']['checkpoint'])] = history['runs']['teacher']['checkpoint_sha256']
    verify_hashes(protected)
    contract = dict(schema_version=1, id='d-merge-training-inputs-v1', created_at=now(),
        protocol=relative(PROTOCOL), protocol_sha256=sha256(PROTOCOL),
        source_B_protocol=relative(bpath), source_B_protocol_sha256=sha256(bpath),
        protected_hashes=protected, builder_code_hashes=training_code_hashes(),
        main_songs=main_membership(songs), auxiliary=admitted, feature_dimensions=531,
        scaler_digest=array_digest(mean, std), scaler_fit_songs=3070,
        supervision=dict(main='continuous_VA_independent_masks', auxiliary='one_quadrant_class_per_whole_clip',
                         numeric_MERGE_VA=False, pseudo_labels=False),
        training_admitted=True, calibration_validation_test_targets_read=0,
        protocol_activation='Implements the immutable pending specification after data/code preflight; original protocol is preserved.',
        limitations=protocol['limitations']+protocol.get('extra_limitations', []))
    atomic_json(CONTRACT, contract)
    return contract


def load_inputs(contract, device, progress=None):
    verify_hashes(contract['protected_hashes'])
    verify_hashes(contract['builder_code_hashes'])
    if not contract['training_admitted'] or contract['supervision']['numeric_MERGE_VA'] or contract['supervision']['pseudo_labels']:
        raise ValueError('Wrong D supervision contract')
    songs = load_internal(read_json(ROOT/contract['source_B_protocol']), device, progress)
    if main_membership(songs) != contract['main_songs']:
        raise ValueError('Main song roles/labels/native clocks changed')
    mean, std = fit_scaler(songs['fit'])
    if array_digest(mean, std) != contract['scaler_digest']:
        raise ValueError('Fit-only scaler changed')
    auxiliary = [read_auxiliary_feature(entry) for entry in contract['auxiliary']]
    return songs, auxiliary
