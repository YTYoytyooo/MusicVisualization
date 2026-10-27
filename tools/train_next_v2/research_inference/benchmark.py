"""Registered whole-audio research benchmark; no emotion accuracy evaluation."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
import unittest

import numpy as np
import soundfile as sf
import torch

from tools.train_next.core import atomic_json, collate_songs, sha256
from tools.train_next.runtime import run_lease, is_running, lock_path
from tools.train_next_v2.common import (BASE, ROOT, PLAN, GPU_LEASE, now, read_json,
    record_event, verify_hashes, process_identity, assert_gpu_available)
from tools.train_next_v2.calibration.job_v2 import PROTECTED
from tools.train_next_v2.run import Tee
from .adapter import ResearchPredictor, FrozenFeatureStore, contained, jsonable


JOB = 'f-research-reference-benchmark-v1'
DIRECTORY = BASE / 'inference_jobs' / JOB
MANIFEST = ROOT / 'models/next_va_v2/research-reference-v1/manifest.json'
PROTOCOL = BASE / 'protocols' / (JOB + '.json')
TEST_DIRECTORY = BASE / 'analyses/f-research-inference-tests-v1'
TEST_REPORT = TEST_DIRECTORY / 'report.json'
B_PROTOCOL = BASE / 'protocols/b-v1.json'
CAL_PROTOCOL = BASE / 'protocols/e-b00-grouped-calibration-v2.json'
CAL_DIRECTORY = BASE / 'calibration_jobs/e-b00-grouped-calibration-v2'
NATIVE_DIRECTORY = BASE / 'annotation_pilot/e-target-native-audit-v1'
ANNOTATION_INVENTORY = BASE / 'annotation_pilot/e-target-annotation-inventory-v1/inventory.json'
CACHE_DIRECTORY = BASE / 'inference_features/research-reference-v1'


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def code_hashes():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [Path(__file__).parent / 'README.md',
        ROOT / 'tools/train_next_v2/tests/test_research_inference.py',
        ROOT / 'tools/train_next_v2/calibration/job_v2.py']
    return {rel(p): sha256(p) for p in files}


def state(stage, **extra):
    value = dict(job_id=JOB, stage=stage, updated_at=now(), **extra)
    atomic_json(DIRECTORY / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def tests():
    if TEST_REPORT.exists():
        raise ValueError('Preserve the registered test report')
    TEST_DIRECTORY.mkdir(parents=True, exist_ok=True)
    atomic_json(TEST_DIRECTORY / 'launch.json', dict(
        original_argv=[sys.executable, '-u', '-m', __package__ + '.benchmark', '--tests'],
        cwd=str(ROOT), python=sys.executable, code=code_hashes(), registered_at=now(),
        initial_red_check='Before adapter.py existed, unittest failed with ModuleNotFoundError for the missing adapter.'))
    atomic_json(TEST_DIRECTORY / 'process.json', dict(process_identity=process_identity(), started_at=now()))
    with (TEST_DIRECTORY / 'console.log').open('w', encoding='utf-8') as stream:
        suite = unittest.defaultTestLoader.loadTestsFromName('tools.train_next_v2.tests.test_research_inference')
        result = unittest.TextTestRunner(stream=Tee(sys.stdout, stream), verbosity=2).run(suite)
    report = dict(status='passed' if result.wasSuccessful() else 'failed',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), code=code_hashes(), completed_at=now())
    atomic_json(TEST_REPORT, report)
    record_event(dict(event='F_research_inference_tests_finished', **report))
    if not result.wasSuccessful():
        raise ValueError('Inference contract tests failed')


def prepare():
    if MANIFEST.exists() or PROTOCOL.exists() or (DIRECTORY / 'launch.json').exists():
        raise ValueError('Preserve the frozen manifest, protocol and launch')
    check = read_json(TEST_REPORT)
    if check['status'] != 'passed' or check['tests'] < 10 or check['skipped']:
        raise ValueError('Fresh inference tests required')
    verify_hashes(check['code'])
    verify_hashes(PROTECTED)
    cal, verified = read_json(CAL_PROTOCOL), read_json(CAL_DIRECTORY / 'verification.json')
    cal_report = read_json(CAL_DIRECTORY / 'report.json')
    if (verified['status'] != 'passed' or verified['report_sha256'] != sha256(CAL_DIRECTORY / 'report.json')
            or cal_report['decision'] != 'pseudo_labels_not_eligible'
            or verified['protocol_sha256'] != sha256(CAL_PROTOCOL)):
        raise ValueError('Calibration evidence changed; this contract intentionally exposes no confidence interval')
    verify_hashes(cal['protected_hashes'])
    b = read_json(B_PROTOCOL)
    if sha256(B_PROTOCOL) != '5ee479f2740f9fd2ddff0f11ed9bd381e70477ecaccb9493d686875f8bf09e6b':
        raise ValueError('Frozen B protocol changed')
    pinned = {**PROTECTED, **code_hashes()}
    for member in cal['members']:
        config = read_json(ROOT / member['config'])
        if is_running(member['run_id']):
            raise ValueError('Reference training is still active')
        verify_hashes(config['code'])
        pinned.update(config['code'])
        pinned[member['checkpoint']] = member['checkpoint_sha256']
        pinned[member['config']] = sha256(ROOT / member['config'])
    for path in [B_PROTOCOL, CAL_PROTOCOL, CAL_DIRECTORY / 'report.json', CAL_DIRECTORY / 'verification.json',
                 TEST_REPORT, ROOT / 'versions/v2.0/studio/paths.py', ROOT / 'versions/v2.0/emotion_model.py']:
        pinned[rel(path)] = sha256(path)
    manifest = dict(id='research-reference-v1', schema_version=1, created_at=now(),
        promoted_model=False, pseudo_training_enabled=False, label_order=['valence', 'arousal'],
        members=cal['members'], B_protocol=rel(B_PROTOCOL), B_protocol_sha256=sha256(B_PROTOCOL),
        feature_spec=b['feature_spec'], feature_key=b['feature_key'], pinned_hashes=pinned,
        deployment_status='research-only reference; no existing application model selection changed',
        calibration=dict(status='unavailable_failed_audit', report=rel(CAL_DIRECTORY / 'report.json')),
        output_contract='Raw arithmetic mean of three seeds, sample seed std without confidence claim, physical acoustic measures, no calibration or display smoothing, tension unavailable.')
    audit_report = read_json(NATIVE_DIRECTORY / 'report.json')
    if audit_report['status'] != 'completed_native_identity_audit':
        raise ValueError('Target native identity audit is incomplete')
    candidates = read_json(NATIVE_DIRECTORY / 'candidate-audit.json')
    # The existing inventory contains identity/count metadata, not raw labels.
    inventory = read_json(ANNOTATION_INVENTORY)
    annotated_hashes = {row['audio_sha256'] for row in inventory['candidates'] if row['annotations']}
    eligible = sorted([r for r in candidates if r['annotation_candidate'] and r['native_audio_passed']
                       and r['audio_sha256'] not in annotated_hashes],
                      key=lambda r: (r['native_audio']['duration_seconds'], r['audio_sha256']))
    if len(eligible) < 3:
        raise ValueError('Need three native-audited unannotated benchmark songs')
    rows = []
    for rank in [0, len(eligible) // 2, len(eligible) - 1]:
        row = eligible[rank]
        n = row['native_audio']
        rows.append(dict(song_id=row['song_id'], audio_sha256=row['audio_sha256'],
            audio_path=row['audio_path'], duration_seconds=n['duration_seconds'], decoded_frames=n['frames'],
            channels=n['channels'], sample_rate=n['sample_rate'], native_audio=n, duration_rank=rank))
    # Audio audit is benchmark provenance, separate from the reusable model.
    protocol_hashes = {rel(NATIVE_DIRECTORY / name): sha256(NATIVE_DIRECTORY / name)
                       for name in ('report.json', 'candidate-audit.json')}
    protocol_hashes[rel(ANNOTATION_INVENTORY)] = sha256(ANNOTATION_INVENTORY)
    protocol_hashes.update({row['audio_path']: row['audio_sha256'] for row in rows})
    verify_hashes(protocol_hashes)
    protocol = dict(id=JOB, schema_version=1, registered_at=now(), songs=rows,
        selection='Sort native-audited target songs without local annotation identity by duration then SHA; use shortest, floor(n/2), longest. Metadata-only selection; no accuracy-based selection.',
        eligible_songs=len(eligible), annotated_hashes_excluded=sorted(x for x in annotated_hashes if x),
        annotation_exclusion_snapshot=rel(ANNOTATION_INVENTORY),
        device='cuda', threads=8, feature_batch_size=8, repeats=3, seed=20261004,
        feature_cache=rel(CACHE_DIRECTORY), input_hashes=protocol_hashes,
        original_validation_labels_used=0, final_test_labels_used=0, accuracy_evaluated=False,
        scope='Full-file audio decode/features, three-seed inference and warmed feature-cache load. First encoder load and overall startup reported separately. No emotional accuracy, user preference or listening claim.',
        recovery='Original argv plus --resume; complete song receipts must match output/cache/manifest/protocol hashes. Inspect unreceipted artifacts rather than accepting them. No live run may share the run ID or GPU lease.')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(MANIFEST, manifest)
    protocol.update(manifest=rel(MANIFEST), manifest_sha256=sha256(MANIFEST))
    atomic_json(PROTOCOL, protocol)
    python = str(Path(read_json(PLAN)['python']).resolve())
    environment = dict(read_json(CAL_DIRECTORY / 'launch.json')['environment'])
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    argv = [python, '-u', '-m', __package__ + '.benchmark', '--launch', str(DIRECTORY / 'launch.json')]
    launch = dict(job_id=JOB, registered_at=now(), original_argv=argv, resume_argv=argv + ['--resume'],
        verification_argv=argv + ['--verify'], cwd=str(ROOT), python=python, environment=environment,
        protocol=rel(PROTOCOL), protocol_sha256=sha256(PROTOCOL), manifest=rel(MANIFEST),
        manifest_sha256=sha256(MANIFEST), pinned_hashes={**pinned, **protocol_hashes},
        os_lock=str(lock_path(JOB)), gpu_lock=str(lock_path(GPU_LEASE)), log=rel(DIRECTORY / 'console.log'),
        checkpoint_directory=rel(DIRECTORY / 'receipts'), recovery=protocol['recovery'],
        status_module=__package__ + '.benchmark --status')
    atomic_json(DIRECTORY / 'launch.json', launch)
    stages = read_json(PLAN)['stages']
    stages['F'].update(status='in_progress', phase='research_adapter_benchmark_registered',
        research_manifest=rel(MANIFEST), benchmark_job=JOB, final_test_evaluation_started=False,
        promoted_model=False, human_listening_verified=False)
    record_event(dict(event='F_research_benchmark_registered', job_id=JOB, songs=[r['song_id'] for r in rows]),
        run=dict(run_id=JOB, status='registered', stage='F', launch=rel(DIRECTORY / 'launch.json')),
        updates=dict(stages=stages, next_action='Run the registered research-reference audio benchmark and independently reproduce its predictions. E audio/human-review gaps remain open; no test evaluation or pseudo-label training.'))
    print(json.dumps(dict(job_id=JOB, songs=len(rows), total_audio_seconds=sum(r['duration_seconds'] for r in rows),
                          launch=rel(DIRECTORY / 'launch.json')), ensure_ascii=False))


def receipt_paths(row):
    stem = row['audio_sha256']
    return DIRECTORY / 'predictions' / (stem + '.json'), DIRECTORY / 'receipts' / (stem + '.json')


def read_receipt(row, protocol):
    output_path, receipt_path = receipt_paths(row)
    receipt = read_json(receipt_path)
    if (receipt['protocol_sha256'] != sha256(PROTOCOL)
            or receipt['manifest_sha256'] != protocol['manifest_sha256']
            or receipt['audio_sha256'] != row['audio_sha256']):
        raise ValueError('Completed-song receipt provenance changed')
    verify_hashes(receipt['files'])
    return receipt, read_json(output_path)


def native_check(row):
    audio, rate = sf.read(contained(row['audio_path']), dtype='float32', always_2d=True)
    if (rate != row['sample_rate'] or audio.shape != (row['decoded_frames'], row['channels'])
            or not np.isfinite(audio).all()
            or hashlib.sha256(audio.tobytes(order='C')).hexdigest() != row['native_audio']['float32_sample_sha256']):
        raise ValueError('Native decoded PCM no longer matches the audited recording')


def run(launch, protocol, resume):
    if (DIRECTORY / 'report.json').exists():
        raise ValueError('Completed benchmark exists; use independent verification')
    if (DIRECTORY / 'process.json').exists() and not resume:
        raise ValueError('Existing benchmark requires --resume')
    assert_gpu_available()
    with run_lease(GPU_LEASE):
        if not torch.cuda.is_available():
            raise ValueError('Independent GPU environment unavailable')
        torch.set_num_threads(protocol['threads'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        random.seed(protocol['seed']); np.random.seed(protocol['seed']); torch.manual_seed(protocol['seed'])
        atomic_json(DIRECTORY / 'process.json', dict(started_at=now(), process_identity=process_identity(), resumed=resume))
        record_event(dict(event='F_research_benchmark_started', job_id=JOB, resumed=resume),
            run=dict(run_id=JOB, status='running', stage='F'))
        start = time.perf_counter()
        predictor = ResearchPredictor(MANIFEST, 'cuda')
        def progress(value):
            if value['frames_done'] % 160 == 0 or value['frames_done'] == value['frames_total']:
                state('extracting_features', **value)
        store = FrozenFeatureStore(predictor.manifest, 'cuda', protocol['feature_cache'], progress)
        startup = time.perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        state('loading_frozen_encoder')
        load_start = time.perf_counter(); store._load(); torch.cuda.synchronize()
        encoder_load = time.perf_counter() - load_start
        rows = []
        for index, row in enumerate(protocol['songs'], 1):
            output_path, receipt_path = receipt_paths(row)
            if receipt_path.exists():
                if not resume:
                    raise ValueError('Existing completed-song receipt requires --resume')
                receipt, _ = read_receipt(row, protocol)
                rows.append(receipt['measurements']); continue
            if output_path.exists():
                raise ValueError('Unreceipted prediction output; preserve and inspect it')
            state('checking_native_audio', song=index, total=len(protocol['songs']), song_id=row['song_id'])
            native_start = time.perf_counter(); native_check(row)
            native_seconds = time.perf_counter() - native_start
            feature_start = time.perf_counter()
            times, values, reused = store.get(row)
            torch.cuda.synchronize(); feature_seconds = time.perf_counter() - feature_start
            timings = []; output = None
            for _ in range(protocol['repeats']):
                torch.cuda.synchronize(); call_start = time.perf_counter()
                current = predictor.predict_features(times, values, store.key, row['duration_seconds'])
                torch.cuda.synchronize(); timings.append(time.perf_counter() - call_start)
                if output is not None:
                    np.testing.assert_array_equal(current['member_raw_va'], output['member_raw_va'])
                output = current
            warm_start = time.perf_counter(); warm_t, warm_x, cached = store.get(row)
            warm_seconds = time.perf_counter() - warm_start
            if not cached:
                raise ValueError('Warmed feature read did not reuse the receipted cache')
            np.testing.assert_array_equal(times, warm_t); np.testing.assert_array_equal(values, warm_x)
            measured = dict(song_id=row['song_id'], audio_path=row['audio_path'], duration_seconds=row['duration_seconds'],
                feature_frames=len(times), initial_feature_cache_reused=reused, native_pcm_check_seconds=native_seconds,
                feature_seconds=feature_seconds, inference_seconds=timings,
                inference_median_seconds=statistics.median(timings), warm_cache_read_seconds=warm_seconds,
                decode_features_and_inference_seconds=feature_seconds + timings[0],
                peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                peak_memory_scope='Invocation cumulative peak including frozen CLAP and all three prediction heads',
                labels_used=0, accuracy_evaluated=False)
            output.update(source=dict(song_id=row['song_id'], audio_path=row['audio_path'], audio_sha256=row['audio_sha256']),
                          native_pcm_hash_verified=True)
            atomic_json(output_path, jsonable(output))
            cache_path = store.directory / (row['audio_sha256'] + '.npz')
            files = {rel(p): sha256(p) for p in (output_path, cache_path, cache_path.with_suffix('.receipt.json'))}
            atomic_json(receipt_path, dict(song_id=row['song_id'], audio_sha256=row['audio_sha256'],
                protocol_sha256=sha256(PROTOCOL), manifest_sha256=sha256(MANIFEST), files=files, measurements=measured))
            rows.append(measured)
            state('song_completed', done=index, total=len(protocol['songs']), **measured)
        store.unload()
        verify_hashes(launch['pinned_hashes'])
        report = dict(job_id=JOB, status='completed_pending_independent_verification', completed_at=now(),
            protocol_sha256=sha256(PROTOCOL), manifest_sha256=sha256(MANIFEST), measurements=rows,
            device=torch.cuda.get_device_name(), startup_seconds=startup, encoder_load_seconds=encoder_load,
            elapsed_this_invocation_seconds=time.perf_counter() - start,
            total_audio_seconds=sum(r['duration_seconds'] for r in rows), feature_frames=sum(r['feature_frames'] for r in rows),
            final_test_labels_used=0, original_validation_labels_used=0, training_labels_used=0,
            accuracy_evaluated=False, protected_models_unchanged=True, application_defaults_changed=False,
            limits=['Three songs measure engineering behavior, not emotional accuracy or general runtime distribution.',
                    'Frozen B00 reference ensemble has not been promoted and its calibration failed eligibility.',
                    'Full-file bidirectional inference is offline; half-second output cadence is not streaming latency.',
                    'Feature timing excludes separately reported encoder load and startup; metadata uses native local-file time.'])
        atomic_json(DIRECTORY / 'report.json', report)
        state('completed_pending_independent_verification', songs=len(rows), feature_frames=report['feature_frames'])
        record_event(dict(event='F_research_benchmark_finished', report=rel(DIRECTORY / 'report.json')),
            run=dict(run_id=JOB, status='pending_verification', stage='F', report=rel(DIRECTORY / 'report.json')))


@torch.inference_mode()
def verify(protocol):
    if (DIRECTORY / 'verification.json').exists():
        raise ValueError('Independent verification already exists')
    atomic_json(DIRECTORY / 'verification-process.json', dict(started_at=now(), process_identity=process_identity()))
    torch.set_num_threads(2)
    report = read_json(DIRECTORY / 'report.json')
    if report['protocol_sha256'] != sha256(PROTOCOL) or report['manifest_sha256'] != sha256(MANIFEST):
        raise ValueError('Benchmark report provenance mismatch')
    predictor = ResearchPredictor(MANIFEST, 'cpu')
    difference = 0.
    for row in protocol['songs']:
        receipt, saved = read_receipt(row, protocol)
        cache = contained(protocol['feature_cache']) / (row['audio_sha256'] + '.npz')
        with np.load(cache, allow_pickle=False) as data:
            times, features, key = data['times'].copy(), data['features'].copy(), str(data['feature_key'].item())
        fresh = predictor.predict_features(times, features, key, row['duration_seconds'])
        # Independent training collator checks normalization, axis ordering and
        # whole-song sequence lengths against all three freshly loaded models.
        song = dict(x=features, y=np.zeros((len(times), 2), np.float32), mask=np.zeros((len(times), 2), bool))
        tensor, _, _, lengths = collate_songs([song], *predictor.scaler)
        direct = np.stack([model(tensor, lengths).numpy()[0] for model in predictor.models])
        np.testing.assert_array_equal(fresh['member_raw_va'], direct)
        np.testing.assert_allclose(fresh['member_raw_va'], saved['member_raw_va'], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(fresh['raw_va'], saved['raw_va'], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(fresh['seed_disagreement_std'], saved['seed_disagreement_std'], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(fresh['times_seconds'], saved['times_seconds'])
        if jsonable(fresh['acoustic']) != saved['acoustic']:
            raise ValueError('Acoustic outputs did not reproduce')
        for name in ('calibration', 'display', 'tension', 'label_order', 'feature_key'):
            if fresh[name] != saved[name]:
                raise ValueError('Public output contract mismatch: ' + name)
        difference = max(difference, float(np.max(np.abs(direct - np.asarray(saved['member_raw_va'])))))
    verify_hashes(PROTECTED)
    evidence = dict(status='passed', verified_at=now(), job_id=JOB,
        report_sha256=sha256(DIRECTORY / 'report.json'), protocol_sha256=sha256(PROTOCOL),
        manifest_sha256=sha256(MANIFEST), fresh_cpu_model_loads=3,
        all_song_receipts_verified=len(protocol['songs']), maximum_cpu_gpu_prediction_difference=difference,
        independent_training_collator_equal=True, native_half_second_times_equal=True,
        acoustic_units_reproduced=True, failed_calibration_remains_unavailable=True,
        original_validation_and_test_labels_used=0, protected_models_unchanged=True)
    atomic_json(DIRECTORY / 'verification.json', evidence)
    stages = read_json(PLAN)['stages']
    stages['F'].update(phase='research_adapter_benchmark_verified', research_manifest=rel(MANIFEST),
        research_benchmark_report=rel(DIRECTORY / 'report.json'), research_benchmark_verification=rel(DIRECTORY / 'verification.json'),
        final_test_evaluation_started=False, promoted_model=False, human_listening_verified=False)
    record_event(dict(event='F_research_adapter_independently_verified', **evidence),
        run=dict(run_id=JOB, status='completed_verified', stage='F', report=rel(DIRECTORY / 'report.json'), verification=rel(DIRECTORY / 'verification.json')),
        updates=dict(stages=stages, next_action='Prepare a fixed same-song V2/mixed/B01 comparison protocol and compatible independent loaders without opening final-test labels. Fabio annotations still need exact original audio and reviewer completion/confidence review; do not train on them or claim a direct V2 improvement.'))
    state('completed_verified', maximum_cpu_gpu_prediction_difference=difference)


def process_alive(identity):
    if not identity or process_identity(identity['pid']) != identity:
        return False
    if os.name != 'nt':
        return None
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, identity['pid'])
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def status():
    held = is_running(JOB)
    process_path = DIRECTORY / ('verification-process.json' if (DIRECTORY / 'verification-process.json').exists() else 'process.json')
    process = read_json(process_path) if process_path.exists() else None
    verified = False
    if (DIRECTORY / 'verification.json').exists() and (DIRECTORY / 'report.json').exists():
        v = read_json(DIRECTORY / 'verification.json')
        verified = (v['status'] == 'passed' and v['report_sha256'] == sha256(DIRECTORY / 'report.json')
                    and v['protocol_sha256'] == sha256(PROTOCOL) and v['manifest_sha256'] == sha256(MANIFEST))
    tail = []
    log = DIRECTORY / 'console.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 8192))
            tail = stream.read().decode('utf-8', errors='replace').splitlines()[-3:]
    print(json.dumps(dict(job_id=JOB, checked_at=now(), os_lease_active=held,
        gpu_lease_active=is_running(GPU_LEASE), process_alive=process_alive(process['process_identity']) if process else None,
        outcome='active' if held else 'completed_verified' if verified else 'inactive_requires_inspection',
        progress=read_json(DIRECTORY / 'status.json') if (DIRECTORY / 'status.json').exists() else None,
        feature_files=len(list(CACHE_DIRECTORY.glob('*.npz'))), completed_song_receipts=len(list((DIRECTORY / 'receipts').glob('*.json'))),
        bounded_log_tail=tail), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true'); modes.add_argument('--tests', action='store_true')
    modes.add_argument('--status', action='store_true'); modes.add_argument('--launch', type=Path)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.status:
        status(); return
    with run_lease(JOB):
        if args.tests:
            tests(); return
        if args.prepare:
            prepare(); return
        if args.launch.resolve() != DIRECTORY / 'launch.json':
            raise ValueError('Use the registered launch path')
        launch = read_json(args.launch)
        if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != Path(launch['python']).resolve():
            raise ValueError('Wrong workspace or independent Python')
        for key, value in launch['environment'].items():
            if os.environ.get(key) != value:
                raise ValueError('Frozen environment changed: ' + key)
        verify_hashes(launch['pinned_hashes'])
        verify_hashes({launch['protocol']: launch['protocol_sha256'], launch['manifest']: launch['manifest_sha256']})
        protocol = read_json(PROTOCOL)
        with (DIRECTORY / 'console.log').open('a', encoding='utf-8', buffering=1) as stream:
            with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                try:
                    if args.verify:
                        verify(protocol)
                    else:
                        run(launch, protocol, args.resume)
                except BaseException as error:
                    traceback.print_exc()
                    atomic_json(DIRECTORY / 'failure.json', dict(failed_at=now(), error=f'{type(error).__name__}: {error}',
                        mode='verification' if args.verify else 'benchmark'))
                    record_event(dict(event='F_research_benchmark_failed', error=str(error)),
                        run=dict(run_id=JOB, status='failed', stage='F'))
                    raise


if __name__ == '__main__':
    main()
