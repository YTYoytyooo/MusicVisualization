"""Register and run the finite, offline three-song C encoder preflight."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
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
from tools.train_next_v2.common import (
    BASE, GPU_LEASE, PLAN, assert_gpu_available, now, process_identity,
    read_json, record_event, verify_hashes,
)
from .encoders import FrozenEncoder, RESOURCES


PROTOCOL = BASE / 'protocols/c-feature-pilot-v1.json'
PROTOCOL_SHA = '5cdcf511f550b1716da3d71925472a39ccb21c20d6606a45a849bb6290848a91'
PACKAGES = ('torch', 'transformers', 'librosa', 'numpy', 'soundfile', 'scipy', 'soxr', 'numba')


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def register(job_id):
    import re
    if not re.fullmatch(r'c-encoder-preflight-[a-zA-Z0-9_-]+', job_id):
        raise ValueError('Invalid C preflight job ID')
    assert_gpu_available()
    if is_running(GPU_LEASE) or is_running(job_id):
        raise RuntimeError('A GPU job is already active')
    if sha256(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Frozen C protocol changed')
    protocol = read_json(PROTOCOL)
    plan = read_json(PLAN)
    frozen_B = read_json(BASE / 'runs/b00-reference-s20261003/launch.json')['code_hashes']
    verify_hashes(frozen_B)
    protected = {**frozen_B, relative(PROTOCOL): PROTOCOL_SHA,
        protocol['pilot_songs']: protocol['pilot_songs_sha256'],
        protocol['resource_manifest']: protocol['resource_manifest_sha256'],
        plan['baseline']['legacy_model']: plan['baseline']['legacy_model_sha256'],
        plan['baseline']['checkpoint']: plan['baseline']['checkpoint_sha256']}
    retrieval = read_json(RESOURCES / 'pinned-model-files-retrieval.json')
    for row in retrieval['files']:
        protected[relative(RESOURCES / row['destination'])] = row['sha256']
    for specification in protocol['encoders']:
        weight = RESOURCES / 'encoder-weights' / specification['name'] / specification['revision'] / 'pytorch_model.bin'
        protected[relative(weight)] = specification['weight']['sha256']
    songs = read_json(ROOT / protocol['pilot_songs'])['songs']
    selected = [next(row for row in songs if row['role'] == 'fit' and row['source_dataset'] == dataset)
                for dataset in ('deam', 'memo2496', 'pmemo')]
    for row in selected:
        # Only audio and pre-existing feature files are read; no label content.
        protected[row['audio_path']] = row['audio_sha256']
        cache = ROOT / plan['paths']['existing_feature_cache'] / (row['audio_sha256'] + '.npz')
        protected[relative(cache)] = row['original_feature_sha256']
        count = int(math.floor(row['duration_seconds'] / .5)) + 1
        row['preflight_indices'] = sorted({0, 1, count//2-1, count//2, count//2+1, count//2+2, count-1})
    verify_hashes(protected)
    code = {relative(path): sha256(path) for path in Path(__file__).parent.glob('*.py')}
    code['tools/train_next_v2/tests/test_pilot_timing.py'] = sha256(ROOT / 'tools/train_next_v2/tests/test_pilot_timing.py')
    transformer_directory = Path(sys.executable).parent.parent / 'Lib/site-packages/transformers'
    runtime_sources = {relative(transformer_directory / path): sha256(transformer_directory / path)
        for path in ('models/clap/modeling_clap.py', 'models/clap/configuration_clap.py',
            'models/clap/feature_extraction_clap.py', 'models/hubert/modeling_hubert.py',
            'models/wav2vec2/feature_extraction_wav2vec2.py', 'configuration_utils.py')}
    environment = dict(plan['stages']['C']['sample_preparation']['environment'])
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='0')
    destination = BASE / 'preflights' / job_id
    destination.mkdir(parents=True, exist_ok=True)
    launch_path = destination / 'launch.json'
    if launch_path.exists():
        raise FileExistsError('Preflight is immutable; inspect previous outcome before a new run')
    launch = dict(schema_version=1, job_id=job_id, stage='C', registered_at=now(),
        cwd=str(ROOT), python=sys.executable, original_argv=[sys.executable, '-u', '-m',
            'tools.train_next_v2.feature_pilot.preflight', '--launch', str(launch_path)],
        environment=environment, code_hashes=code, protected_hashes=protected,
        runtime_source_hashes=runtime_sources,
        precision=dict(cudnn_allow_tf32=False, matmul_allow_tf32=False,
            reason='Recorded two-mode C diagnostic shows TF32 causes the MERT batch-parity failure; IEEE FP32 passes the original tolerance.'),
        libraries={name: version(name) for name in PACKAGES}, protocol=str(PROTOCOL),
        songs=selected, encoders=protocol['encoders'], contexts=[2, 10], batch_size=4,
        repeat_tolerance=dict(atol=1e-6, rtol=1e-6), batch_parity_tolerance=dict(atol=1e-4, rtol=1e-4),
        batching='Exact waveform length buckets avoid padding-dependent temporal GroupNorm. Maximum batch 4, smaller bucket tails, fallback 1 only on CUDA OOM.',
        os_lock=str(lock_path(job_id)), gpu_lock=str(lock_path(GPU_LEASE)),
        log=str(destination / 'console.log'), status_path=str(destination / 'status.json'),
        process_path=str(destination / 'process.json'), report_path=str(destination / 'report.json'),
        existing_feature_cache=plan['paths']['existing_feature_cache'],
        recovery='Finite preflight has no optimizer or resume checkpoint. Preserve logs/results; diagnose once and register a new ID if code/environment changes. Never repeat an unchanged failed invocation.',
        labels_read=False, heldout_songs_used=0, predictor_weights_loaded=False)
    atomic_json(launch_path, launch)
    stages = plan['stages']
    stages['C'].update(phase='runtime_preflight_registered', runtime_preflight=dict(
        job_id=job_id, status='registered', launch=relative(launch_path),
        report=relative(destination / 'report.json'), songs=[row['song_id'] for row in selected]))
    record_event(dict(event='C_runtime_preflight_registered', job_id=job_id,
        launch=relative(launch_path), source_songs=[row['song_id'] for row in selected],
        labels_read=False), updates=dict(stages=stages))
    return launch_path


def run(launch_path):
    import librosa
    import soundfile as sf
    import torch

    launch_path = launch_path.resolve()
    if not launch_path.is_relative_to(BASE / 'preflights'):
        raise ValueError('Launch path must be inside the C preflight directory')
    launch = read_json(launch_path)
    destination = launch_path.parent
    if (destination / 'report.json').exists():
        raise FileExistsError('Completed/failed preflight report already exists; no blind reruns')
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != Path(launch['python']).resolve():
        raise ValueError('CWD or Python identity mismatch')
    for name, expected in launch['environment'].items():
        if os.environ.get(name) != expected:
            raise ValueError('Launch environment mismatch: ' + name)
    verify_hashes(launch['code_hashes'])
    verify_hashes(launch['protected_hashes'])
    verify_hashes(launch['runtime_source_hashes'])
    if {name: version(name) for name in PACKAGES} != launch['libraries']:
        raise ValueError('Runtime package versions changed')
    assert_gpu_available()
    report = dict(job_id=launch['job_id'], started_at=now(), status='running',
        launch_sha256=sha256(launch_path), libraries=launch['libraries'], cases=[], encoder_loads=[],
        labels_read=False, heldout_songs_used=0, full_pilot_extracted=False)
    with run_lease(GPU_LEASE), run_lease(launch['job_id']), (destination / 'console.log').open('x', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            identity = process_identity()
            atomic_json(destination / 'process.json', dict(**identity, started_at=report['started_at'], argv=sys.argv))
            def status(**progress):
                atomic_json(destination / 'status.json', dict(job_id=launch['job_id'], updated_at=now(),
                    process_identity=identity, **progress))
                print(json.dumps(progress, ensure_ascii=False), flush=True)
            status(state='running', phase='decode_and_runtime_checks')
            started = time.monotonic()
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError('Registered C preflight requires CUDA')
                torch.set_num_threads(8)
                torch.manual_seed(20261003)
                np.random.seed(20261003)
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.allow_tf32 = launch['precision']['cudnn_allow_tf32']
                torch.backends.cuda.matmul.allow_tf32 = launch['precision']['matmul_allow_tf32']
                report['precision'] = launch['precision']
                report['runtime_source_hashes'] = launch['runtime_source_hashes']
                report['gpu'] = torch.cuda.get_device_name()
                report['cuda_version'] = torch.version.cuda
                report['cudnn_version'] = torch.backends.cudnn.version()
                report['performance_context'] = 'Observed local timings, including warmup; not an isolated GPU performance benchmark.'
                decoded = []
                for row in launch['songs']:
                    audio, rate = sf.read(ROOT / row['audio_path'], dtype='float32', always_2d=True)
                    if audio.shape != (row['decoded_frames'], row['channels']) or rate != row['sample_rate']:
                        raise ValueError('Prepared audio decode mismatch: ' + row['song_id'])
                    mono = audio.mean(axis=1)
                    cache = ROOT / launch['existing_feature_cache'] / (row['audio_sha256'] + '.npz')
                    with np.load(cache, allow_pickle=False) as cached:
                        expected_times = np.arange(int(np.floor(row['duration_seconds'] / .5)) + 1) * .5
                        np.testing.assert_array_equal(cached['times'], expected_times)
                        if cached['features'].shape != (len(expected_times), 531):
                            raise ValueError('Existing feature shape changed')
                        indices = row['preflight_indices']
                        acoustic = cached['features'][indices, -19:].copy()
                    decoded.append((row, mono, rate, expected_times[indices], acoustic))
                for specification in launch['encoders']:
                    encoder = None
                    load_started = time.monotonic()
                    status(state='running', phase='load_encoder', encoder=specification['name'])
                    try:
                        encoder = FrozenEncoder(specification)
                        report['encoder_loads'].append(dict(encoder=encoder.name, status='passed',
                            strict_weights=True, weight_tensors=encoder.weight_tensor_count,
                            frozen_parameters=encoder.parameter_count, load_seconds=time.monotonic()-load_started))
                        for row, mono, rate, times, acoustic in decoded:
                            resample_started = time.monotonic()
                            waveform = librosa.resample(mono, orig_sr=rate, target_sr=encoder.rate, res_type='soxr_hq')
                            if len(waveform) != math.ceil(len(mono) * encoder.rate / rate):
                                raise ValueError('Resampling changed the physical sample clock')
                            resample_seconds = time.monotonic() - resample_started
                            for context in launch['contexts']:
                                case = dict(encoder=encoder.name, song_id=row['song_id'], context_seconds=context,
                                    timestamps=times.tolist(), source_rate=rate, resampled_rate=encoder.rate,
                                    source_frames=len(mono), resampled_frames=len(waveform), resample_seconds=resample_seconds)
                                status(state='running', phase='encode', encoder=encoder.name,
                                    song_id=row['song_id'], context_seconds=context, cases_completed=len(report['cases']))
                                torch.cuda.reset_peak_memory_stats()
                                torch.cuda.synchronize()
                                inference_started = time.monotonic()
                                batch_size = launch['batch_size']
                                try:
                                    values, metadata, batches = encoder.encode(waveform, times, context, batch_size)
                                except torch.cuda.OutOfMemoryError:
                                    torch.cuda.empty_cache()
                                    batch_size = 1
                                    case['oom_fallback'] = True
                                    values, metadata, batches = encoder.encode(waveform, times, context, batch_size)
                                torch.cuda.synchronize()
                                case['extraction_seconds'] = time.monotonic() - inference_started
                                repeated, _, _ = encoder.encode(waveform, times, context, batch_size)
                                np.testing.assert_allclose(values, repeated, **launch['repeat_tolerance'])
                                case['repeat_max_absolute_difference'] = float(np.max(np.abs(values - repeated)))
                                # An interior window is also run alone to catch batching/padding dependence.
                                parity_index = len(times) // 2
                                single, _, _ = encoder.encode(waveform, times[parity_index:parity_index+1], context, 1)
                                np.testing.assert_allclose(values[parity_index], single[0], **launch['batch_parity_tolerance'])
                                case['single_batch_max_absolute_difference'] = float(np.max(np.abs(values[parity_index] - single[0])))
                                combined = np.concatenate([values, acoustic], axis=1)
                                if combined.shape != (len(times), encoder.dimensions + 19) or not np.isfinite(combined).all():
                                    raise ValueError('Invalid C feature vector')
                                output = destination / (encoder.name + '-' + row['song_id'].replace(':', '-') + '-' + str(context) + 's.npz')
                                with output.open('xb') as stream:
                                    np.savez_compressed(stream, times=times, features=combined, encoder_features=values)
                                case.update(status='passed', dimensions=combined.shape[1], windows=len(times),
                                    batches=batches, requested_batch_size=launch['batch_size'], actual_max_batch_size=max(batches),
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                    source_frame_metadata=metadata, output=relative(output), output_sha256=sha256(output))
                                report['cases'].append(case)
                                atomic_json(destination / 'partial-report.json', report)
                    except Exception as error:
                        traceback.print_exc()
                        report.setdefault('errors', []).append(dict(encoder=specification['name'],
                            type=type(error).__name__, message=str(error)))
                    finally:
                        if encoder is not None:
                            encoder.unload()
                expected_cases = len(launch['encoders']) * len(launch['songs']) * len(launch['contexts'])
                report['status'] = 'completed' if not report.get('errors') and len(report['cases']) == expected_cases else 'failed'
                report['expected_cases'] = expected_cases
                report['actual_cases'] = len(report['cases'])
                verify_hashes(launch['protected_hashes'])
                verify_hashes(launch['runtime_source_hashes'])
                report['protected_hashes_verified_after_run'] = True
            except Exception as error:
                traceback.print_exc()
                report['status'] = 'failed'
                report.setdefault('errors', []).append(dict(type=type(error).__name__, message=str(error)))
            report.update(completed_at=now(), elapsed_seconds=time.monotonic()-started)
            atomic_json(destination / 'report.json', report)
            status(state=report['status'], report=str(destination / 'report.json'), cases_completed=len(report['cases']))
    plan = read_json(PLAN)
    stages = plan['stages']
    stages['C']['phase'] = 'runtime_preflight_passed' if report['status'] == 'completed' else 'runtime_preflight_needs_diagnosis'
    stages['C']['runtime_preflight'].update(status=report['status'], process_identity=identity,
        cases_completed=len(report['cases']), completed_at=report['completed_at'])
    next_action = ('Independently verify the C preflight report and outputs; register the fixed 90-song feature extraction with strict provenance, then verify targets/masks before any fitting.'
        if report['status'] == 'completed' else 'Diagnose the recorded C runtime incompatibility; preserve this failed preflight and use a new job/environment only after resolving its specific cause.')
    record_event(dict(event='C_runtime_preflight_finished', job_id=launch['job_id'], status=report['status'],
        cases_completed=len(report['cases']), report=relative(destination / 'report.json')),
        updates=dict(stages=stages, next_action=next_action))
    return 0 if report['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--register', metavar='JOB_ID')
    group.add_argument('--launch', type=Path)
    args = parser.parse_args()
    if args.register:
        print(register(args.register))
        return 0
    return run(args.launch)


if __name__ == '__main__':
    raise SystemExit(main())
