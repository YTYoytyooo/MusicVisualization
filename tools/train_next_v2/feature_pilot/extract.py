"""Registered, resumable C encoder extraction on the fixed 90-song pilot."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from tools.train_next.core import ROOT, atomic_json, sha256
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import BASE, GPU_LEASE, PLAN, assert_gpu_available, now, process_identity, read_json, record_event, verify_hashes
from .cache import read_shard, write_shard
from .encoders import FrozenEncoder
from .preflight import PACKAGES, PROTOCOL, PROTOCOL_SHA, relative


JOB_ID = 'c-pilot-encoder-features-v1'
JOB_DIR = BASE / 'feature_jobs' / JOB_ID
CACHE_DIR = BASE / 'feature_caches' / JOB_ID


def register():
    assert_gpu_available()
    if is_running(GPU_LEASE) or is_running(JOB_ID):
        raise RuntimeError('GPU extraction/training is already active')
    if sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Frozen C protocol changed')
    preflight_dir = BASE / 'preflights/c-encoder-preflight-v2'
    verified = read_json(preflight_dir / 'verification.json')
    if verified['status'] != 'completed_verified' or verified['report_sha256'] != sha256(preflight_dir / 'report.json'):
        raise ValueError('C runtime preflight is not independently verified')
    preflight = read_json(preflight_dir / 'launch.json')
    verify_hashes(preflight['protected_hashes'])
    verify_hashes(preflight['code_hashes'])
    verify_hashes(preflight['runtime_source_hashes'])
    protocol = read_json(PROTOCOL)
    songs = read_json(ROOT / protocol['pilot_songs'])['songs']
    if len(songs) != 90 or sum(row['role'] == 'fit' for row in songs) != 60 or sum(row['role'] == 'development' for row in songs) != 30:
        raise ValueError('Pilot membership changed')
    protected = dict(preflight['protected_hashes'])
    protected[relative(preflight_dir / 'report.json')] = sha256(preflight_dir / 'report.json')
    protected[relative(preflight_dir / 'verification.json')] = sha256(preflight_dir / 'verification.json')
    for row in songs:
        protected[row['audio_path']] = row['audio_sha256']
        cache = ROOT / preflight['existing_feature_cache'] / (row['audio_sha256'] + '.npz')
        protected[relative(cache)] = row['original_feature_sha256']
    verify_hashes(protected)
    code = {relative(path): sha256(path) for path in Path(__file__).parent.glob('*.py')}
    for name in ('test_pilot_timing.py', 'test_pilot_cache.py'):
        path = ROOT / 'tools/train_next_v2/tests' / name
        code[relative(path)] = sha256(path)
    variants = [item for item in protocol['experiments'] if item['encoder'] in ('music-clap', 'mert95m')]
    feature_specs = {}
    for variant in variants:
        spec = dict(variant=variant, protocol_sha256=PROTOCOL_SHA,
            encoder=next(e for e in protocol['encoders'] if e['name'] == variant['encoder']),
            code_hashes=code, runtime_source_hashes=preflight['runtime_source_hashes'],
            libraries=preflight['libraries'], precision=preflight['precision'],
            grid_seconds=.5, batch_size=4, batching=preflight['batching'], acoustic19='unchanged verified original cache')
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
        feature_specs[variant['id']] = dict(key=key, specification=spec)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    if (JOB_DIR / 'launch.json').exists():
        raise FileExistsError('Extraction already registered; inspect status and use --resume only if appropriate')
    argv = [sys.executable, '-u', '-m', 'tools.train_next_v2.feature_pilot.extract', '--launch', str(JOB_DIR / 'launch.json')]
    launch = dict(schema_version=1, job_id=JOB_ID, registered_at=now(), cwd=str(ROOT), python=sys.executable,
        original_argv=argv, resume_argv=argv + ['--resume'], environment=preflight['environment'],
        protected_hashes=protected, code_hashes=code, runtime_source_hashes=preflight['runtime_source_hashes'],
        libraries=preflight['libraries'], precision=preflight['precision'], songs=songs,
        variants=variants, feature_specs=feature_specs, encoders=protocol['encoders'],
        preflight_verification=relative(preflight_dir / 'verification.json'),
        existing_feature_cache=preflight['existing_feature_cache'], cache_dir=str(CACHE_DIR),
        expected_shards=len(songs)*len(variants), os_lock=str(lock_path(JOB_ID)), gpu_lock=str(lock_path(GPU_LEASE)),
        process_path=str(JOB_DIR/'process.json'), status_path=str(JOB_DIR/'status.json'), report_path=str(JOB_DIR/'report.json'),
        log_pattern=str(JOB_DIR/'attempt-*/console.log'), checkpoint_path=str(JOB_DIR/'shards.json'),
        status_argv=[sys.executable,'-m','tools.train_next_v2.feature_pilot.status','--job-id',JOB_ID],
        recovery='After confirming this OS lease is inactive, diagnose failure then use the exact registered resume_argv. Verify code/data/runtime and each completed cache hash. Recompute orphan shards exactly; never silently overwrite mismatched caches.',
        labels_read=False, validation_test_calibration_songs_used=0,
        remaining_variants='c00 reuses original caches; c05 rhythm/harmony preparation is separate and remains pending.')
    atomic_json(JOB_DIR / 'launch.json', launch)
    plan = read_json(PLAN)
    stages = plan['stages']
    stages['C'].update(phase='pilot_encoder_extraction_registered', feature_extraction=dict(
        job_id=JOB_ID, status='registered', launch=relative(JOB_DIR/'launch.json'),
        status_path=relative(JOB_DIR/'status.json'), report_path=relative(JOB_DIR/'report.json'),
        status_argv=launch['status_argv'], expected_shards=launch['expected_shards'], completed_shards=0))
    record_event(dict(event='C_90_song_encoder_extraction_registered', job_id=JOB_ID,
        songs=90, variants=len(variants), expected_shards=launch['expected_shards'], labels_read=False),
        updates=dict(stages=stages, next_action='Check C feature extraction via its own status command and real OS leases; do not start a competing GPU job. Prepare c05 rhythm/harmony separately; verify all feature clocks and V/A targets/masks before pilot fitting.'))
    print(JOB_DIR / 'launch.json')


def run(path, resume=False):
    import librosa
    import soundfile as sf
    import torch

    if path.resolve() != (JOB_DIR / 'launch.json').resolve():
        raise ValueError('Unknown C feature extraction launch')
    launch = read_json(path)
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != Path(launch['python']).resolve():
        raise ValueError('CWD or interpreter changed')
    for key, value in launch['environment'].items():
        if os.environ.get(key) != value:
            raise ValueError('Environment changed: ' + key)
    for field in ('code_hashes', 'protected_hashes', 'runtime_source_hashes'):
        verify_hashes(launch[field])
    if {name: version(name) for name in PACKAGES} != launch['libraries']:
        raise ValueError('Runtime versions changed')
    assert_gpu_available()
    with run_lease(GPU_LEASE), run_lease(JOB_ID):
        attempts = sorted(JOB_DIR.glob('attempt-*'))
        if attempts and not resume:
            raise RuntimeError('Existing extraction requires --resume after diagnosis')
        if (JOB_DIR / 'report.json').exists() and read_json(JOB_DIR / 'report.json')['status'] == 'completed':
            raise RuntimeError('Extraction already completed; do not restart')
        attempt = JOB_DIR / f'attempt-{len(attempts)+1:03d}'
        attempt.mkdir()
        identity = process_identity()
        atomic_json(JOB_DIR/'process.json', dict(**identity, argv=sys.argv, started_at=now(), log=str(attempt/'console.log')))
        with (attempt/'console.log').open('x', encoding='utf-8', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            report = dict(job_id=JOB_ID, attempt=len(attempts)+1, started_at=now(), status='running',
                launch_sha256=sha256(path), expected_shards=launch['expected_shards'], completed_shards=0,
                reused_shards=0, labels_read=False, validation_test_calibration_songs_used=0)
            index = {}
            started = time.monotonic()
            def progress(**data):
                state = dict(job_id=JOB_ID, state='running', updated_at=now(), process_identity=identity,
                    completed_shards=len(index), expected_shards=launch['expected_shards'], **data)
                atomic_json(JOB_DIR/'status.json', state)
                print(json.dumps(state), flush=True)
            progress(phase='loading_encoders')
            encoder = None
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA is unavailable')
                torch.set_num_threads(8)
                torch.manual_seed(20261003)
                np.random.seed(20261003)
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.allow_tf32 = launch['precision']['cudnn_allow_tf32']
                torch.backends.cuda.matmul.allow_tf32 = launch['precision']['matmul_allow_tf32']
                report['gpu'] = torch.cuda.get_device_name()
                for specification in launch['encoders']:
                    variants = [v for v in launch['variants'] if v['encoder'] == specification['name']]
                    encoder = None
                    for row in launch['songs']:
                        times = np.arange(int(np.floor(row['duration_seconds']/.5))+1) * .5
                        waveform = None
                        for variant in variants:
                            key = launch['feature_specs'][variant['id']]['key']
                            output = Path(launch['cache_dir']) / variant['id'] / (row['audio_sha256']+'.npz')
                            metadata = read_shard(output, times, key, row['audio_sha256'], variant['feature_dimensions'])
                            cache_key = variant['id'] + '/' + row['song_id']
                            if metadata is not None:
                                report['reused_shards'] += 1
                            else:
                                if encoder is None:
                                    progress(phase='load_encoder', encoder=specification['name'])
                                    encoder = FrozenEncoder(specification)
                                if waveform is None:
                                    decode_started = time.monotonic()
                                    audio, rate = sf.read(ROOT/row['audio_path'], dtype='float32', always_2d=True)
                                    if audio.shape != (row['decoded_frames'], row['channels']) or rate != row['sample_rate']:
                                        raise ValueError('Audio decode changed: ' + row['song_id'])
                                    waveform = librosa.resample(audio.mean(1), orig_sr=rate, target_sr=encoder.rate, res_type='soxr_hq')
                                    if len(waveform) != math.ceil(len(audio)*encoder.rate/rate):
                                        raise ValueError('Resampled audio length changed')
                                    decode_seconds = time.monotonic()-decode_started
                                source_cache = ROOT / launch['existing_feature_cache'] / (row['audio_sha256']+'.npz')
                                with np.load(source_cache, allow_pickle=False) as original:
                                    np.testing.assert_array_equal(original['times'], times)
                                    acoustic = original['features'][:, -19:].copy()
                                progress(phase='extracting', encoder=encoder.name, variant=variant['id'], song_id=row['song_id'], frames=len(times))
                                torch.cuda.reset_peak_memory_stats()
                                torch.cuda.synchronize()
                                extraction_started = time.monotonic()
                                batch_size, oom_fallback = 4, False
                                try:
                                    values, timing, batches = encoder.encode(waveform, times, variant['context_seconds'], batch_size)
                                except torch.cuda.OutOfMemoryError:
                                    torch.cuda.empty_cache()
                                    batch_size, oom_fallback = 1, True
                                    values, timing, batches = encoder.encode(waveform, times, variant['context_seconds'], batch_size)
                                torch.cuda.synchronize()
                                elapsed = time.monotonic()-extraction_started
                                features = np.concatenate([values, acoustic], axis=1)
                                if features.shape != (len(times), variant['feature_dimensions']):
                                    raise ValueError('Unexpected C feature width')
                                metadata = write_shard(output, times, features, key, row['audio_sha256'], dict(
                                    completed_at=now(), song_id=row['song_id'], role=row['role'], source_dataset=row['source_dataset'],
                                    variant=variant['id'], source_sample_rate=rate, resampled_rate=encoder.rate,
                                    resampled_samples=len(waveform), decode_resample_seconds=decode_seconds,
                                    extraction_seconds=elapsed, windows_per_second=len(times)/elapsed,
                                    maximum_batch_size=max(batches), requested_batch_size=4, oom_fallback=oom_fallback,
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), timing=timing,
                                    output=relative(output), source_feature_sha256=row['original_feature_sha256']))
                            index[cache_key] = dict(output=relative(output), output_sha256=metadata['output_sha256'],
                                sidecar_sha256=sha256(output.with_suffix('.json')), frames=len(times), dimensions=variant['feature_dimensions'])
                            atomic_json(JOB_DIR/'shards.json', dict(job_id=JOB_ID, feature_keys={k:v['key'] for k,v in launch['feature_specs'].items()}, shards=index))
                            progress(phase='shard_complete', variant=variant['id'], song_id=row['song_id'])
                    if encoder is not None:
                        encoder.unload()
                        encoder = None
                if len(index) != launch['expected_shards']:
                    raise ValueError('Incomplete feature cache set')
                for field in ('code_hashes', 'protected_hashes', 'runtime_source_hashes'):
                    verify_hashes(launch[field])
                report.update(status='completed', completed_shards=len(index), protected_hashes_verified=True,
                    shard_index_sha256=sha256(JOB_DIR/'shards.json'), feature_keys={k:v['key'] for k,v in launch['feature_specs'].items()})
            except Exception as error:
                traceback.print_exc()
                report.update(status='failed', completed_shards=len(index), error_type=type(error).__name__, error=str(error))
            finally:
                if encoder is not None:
                    encoder.unload()
            report.update(completed_at=now(), elapsed_seconds=time.monotonic()-started)
            atomic_json(attempt/'report.json', report)
            atomic_json(JOB_DIR/'report.json', report)
            atomic_json(JOB_DIR/'status.json', dict(job_id=JOB_ID, state=report['status'], updated_at=now(), process_identity=identity,
                completed_shards=len(index), expected_shards=launch['expected_shards'], report=str(JOB_DIR/'report.json')))
            print(json.dumps(report), flush=True)
    plan = read_json(PLAN)
    stages = plan['stages']
    stages['C']['feature_extraction'].update(status=report['status'], completed_shards=len(index), process_identity=identity)
    stages['C']['phase'] = 'pilot_encoder_extraction_complete_pending_verification' if report['status']=='completed' else 'pilot_encoder_extraction_needs_diagnosis'
    record_event(dict(event='C_pilot_encoder_extraction_finished', job_id=JOB_ID, status=report['status'], completed_shards=len(index), report=relative(JOB_DIR/'report.json')),
        updates=dict(stages=stages))
    return 0 if report['status']=='completed' else 1


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--register', action='store_true')
    group.add_argument('--launch', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.register:
        register()
        return 0
    return run(args.launch, args.resume)


if __name__ == '__main__':
    raise SystemExit(main())
