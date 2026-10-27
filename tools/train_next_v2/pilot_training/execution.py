"""Frozen, sequential C pilot runs using the unchanged B training engine."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from tools.train_next.core import atomic_json, fit_scaler, sha256, write_status
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import (BASE, ROOT, PLAN, GPU_LEASE, assert_gpu_available,
    code_hashes, now, paths, process_identity, read_json, record_event, verify_hashes)
from tools.train_next_v2.evaluation import evaluate, assert_metrics_match, score
from tools.train_next_v2.model import SequenceModel
from tools.train_next_v2.run import Tee, train_loop
from .data import CONTRACT, load_variant

QUEUE_ID = 'c-pilot-training-v1'
QUEUE_DIR = BASE / 'queues' / QUEUE_ID
MATRIX = QUEUE_DIR / 'matrix.json'
PROTOCOL = BASE / 'protocols/c-feature-pilot-v1.json'
MODULE = 'tools.train_next_v2.pilot_training.execution'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def runtime_signature():
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__)


def training_environment():
    source = BASE / 'feature_jobs/c-pilot-encoder-features-v2/launch.json'
    result = dict(read_json(source)['environment'])
    result.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return result


def update_stage(event, **updates):
    # record_event serializes read/modify/write with all other workers.
    with run_lease('next-va-v2-c-ledger'):
        stages = read_json(PLAN)['stages']
        stages['C'].update(updates)
        record_event(event, updates=dict(current_stage='C', stages=stages))


def immutable_json(path, data):
    if path.exists():
        if read_json(path) != data:
            raise ValueError('Existing immutable record differs: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, data)


def register_matrix():
    if MATRIX.exists() or is_running(GPU_LEASE):
        raise ValueError('Matrix already registered or GPU workload still active')
    verification = BASE / 'analyses/c-pilot-input-verification-v1/report.json'
    audit = read_json(verification)
    contract, protocol = read_json(CONTRACT), read_json(PROTOCOL)
    if audit['status'] != 'completed_verified' or audit['contract_sha256'] != sha256(CONTRACT):
        raise ValueError('All feature/target verification must pass before registration')
    if contract['protocol_sha256'] != sha256(PROTOCOL):
        raise ValueError('C feature protocol changed')
    verify_hashes(contract['protected_hashes'])
    verify_hashes(contract['builder_code_hashes'])
    code = code_hashes()
    code.update({relative(p): sha256(p) for p in Path(__file__).parent.glob('*.py')})
    env = training_environment()
    commands, protected = [], dict(code)
    for seed in protocol['seeds']:
        for variant in protocol['experiments']:
            run_id = variant['id'] + '-s' + str(seed)
            run_dir, model_dir = paths(run_id)
            if run_dir.exists() or model_dir.exists():
                raise ValueError('Refuse to overwrite an existing C run: ' + run_id)
            config = dict(schema_version=1, run_id=run_id, stage='C', experiment=variant['id'],
                settings={**protocol['settings'], 'seed': seed}, feature_dimensions=variant['feature_dimensions'],
                protocol=relative(PROTOCOL), protocol_sha256=sha256(PROTOCOL),
                contract=relative(CONTRACT), contract_sha256=sha256(CONTRACT),
                code=code, runtime=runtime_signature(), environment=env,
                arithmetic=dict(deterministic_algorithms=True, cudnn_benchmark=False,
                                cudnn_deterministic=True, cudnn_tf32=False, matmul_tf32=False),
                initialization='random; no B/teacher/mixed checkpoint loading',
                normalization='Only the 60 fit songs; scaler is persisted and checked on resume',
                fit_songs=60, development_songs=30, calibration_songs_used=0,
                original_validation_songs_used=0, test_songs_used=0)
            run_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            atomic_json(run_dir / 'config.json', config)
            argv = [str(Path(sys.executable).resolve()), '-u', '-m', MODULE, '--run-id', run_id, '--resume']
            verify_argv = [str(Path(sys.executable).resolve()), '-u', '-m', MODULE, '--verify', run_id]
            launch = dict(run_id=run_id, stage='C', original_argv=argv, resume_argv=argv,
                verify_argv=verify_argv, cwd=str(ROOT), python=str(Path(sys.executable).resolve()),
                environment=env, config=relative(run_dir / 'config.json'),
                config_sha256=sha256(run_dir / 'config.json'), code_hashes=code,
                data_hashes=dict(contract=sha256(CONTRACT), protocol=sha256(PROTOCOL)),
                os_lock=str(lock_path(run_id)), gpu_lock=str(lock_path(GPU_LEASE)),
                log_paths=[str(run_dir / 'console.log')],
                checkpoint_paths=dict(best=str(model_dir / 'best.pth'), last=str(model_dir / 'last.pth')),
                step_budget=protocol['settings']['max_steps'],
                recovery='Stop on failure; diagnose with OS lease and PID identity before exact --resume. Full last.pth restores model, optimizer, scaler, RNG and step. Never change configuration or hash checks.')
            atomic_json(run_dir / 'launch.json', launch)
            for name in ('config.json', 'launch.json'):
                protected[relative(run_dir / name)] = sha256(run_dir / name)
            commands.append(dict(run_id=run_id, launch=relative(run_dir / 'launch.json')))
            record_event(dict(event='C_pilot_run_registered', run_id=run_id),
                         run=dict(run_id=run_id, stage='C', status='registered', launch=relative(run_dir / 'launch.json'),
                                  original_argv=argv, cwd=str(ROOT), python=launch['python'],
                                  environment=env, config_sha256=launch['config_sha256'],
                                  log_paths=launch['log_paths'], os_lock=launch['os_lock'],
                                  checkpoint_paths=launch['checkpoint_paths'], step_budget=launch['step_budget']))
    for path in (CONTRACT, PROTOCOL, verification):
        protected[relative(path)] = sha256(path)
    matrix = dict(schema_version=1, job_id=QUEUE_ID, registered_at=now(), runs=commands,
                  protected_hashes=protected, ordering='seed first, then c00 through c05',
                  fixed_runs=18, steps_per_run=2000, automatic_retry=False,
                  independent_reload_required_before_next_run=True)
    immutable_json(MATRIX, matrix)
    argv = [str(Path(sys.executable).resolve()), '-u', '-m', MODULE, '--queue']
    launch = dict(job_id=QUEUE_ID, original_argv=argv, resume_argv=argv + ['--resume'],
        cwd=str(ROOT), python=str(Path(sys.executable).resolve()), environment=env,
        matrix=relative(MATRIX), matrix_sha256=sha256(MATRIX), code_hashes=code,
        os_lock=str(lock_path(QUEUE_ID)), log=str(QUEUE_DIR / 'console.log'),
        recovery='No automatic retry. Inspect failed run and actual OS locks before --queue --resume. Completed verified runs are skipped; incomplete runs use unchanged launch argv with --resume.')
    immutable_json(QUEUE_DIR / 'launch.json', launch)
    update_stage(dict(event='C_pilot_matrix_registered', runs=len(commands)),
                 phase='pilot_training_registered', training_queue=dict(job_id=QUEUE_ID, status='registered',
                 launch=relative(QUEUE_DIR / 'launch.json'), matrix=relative(MATRIX), runs=18,
                 status_argv=[launch['python'], '-m', 'tools.train_next_v2.pilot_training.status']))
    print(json.dumps(dict(registered_runs=len(commands), matrix=str(MATRIX), launch=str(QUEUE_DIR / 'launch.json'))))


def validate_run(run_id):
    run_dir, model_dir = paths(run_id)
    launch, config = read_json(run_dir / 'launch.json'), read_json(run_dir / 'config.json')
    if launch['run_id'] != run_id or config['run_id'] != run_id or config['stage'] != 'C':
        raise ValueError('Wrong C run identity')
    if sha256(run_dir / 'config.json') != launch['config_sha256']:
        raise ValueError('C configuration changed')
    verify_hashes(config['code'])
    verify_hashes({config['protocol']: config['protocol_sha256'], config['contract']: config['contract_sha256']})
    if runtime_signature() != config['runtime']:
        raise ValueError('Frozen runtime changed')
    for name, expected in config['environment'].items():
        if os.environ.get(name) != expected:
            raise ValueError('Registered environment mismatch: ' + name)
    protocol = read_json(ROOT / config['protocol'])
    variant = next(row for row in protocol['experiments'] if row['id'] == config['experiment'])
    seed = config['settings']['seed']
    if seed not in protocol['seeds'] or config['settings'] != {**protocol['settings'], 'seed': seed}:
        raise ValueError('Run settings differ from the frozen C experiment')
    if config['feature_dimensions'] != variant['feature_dimensions']:
        raise ValueError('Feature dimensions differ from frozen experiment')
    contract = read_json(ROOT / config['contract'])
    verify_hashes(contract['builder_code_hashes'])
    return run_dir, model_dir, config, contract


def configure_torch(config):
    assert_gpu_available()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(config['settings']['threads'])
    torch.manual_seed(config['settings']['seed'])
    random.seed(config['settings']['seed'])
    np.random.seed(config['settings']['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    return torch.device('cuda')


def load_inputs(config, contract):
    # Sidecars carry encoder clocks and provenance, even though fitting uses only npz.
    for song in contract['songs']:
        entry = song['features'][config['experiment']]
        if entry.get('sidecar'):
            verify_hashes({entry['sidecar']: entry['sidecar_sha256']})
    songs = load_variant(contract, config['experiment'])
    if len(songs['fit']) != 60 or len(songs['development']) != 30:
        raise ValueError('C pilot membership changed')
    if any(song['x'].shape[1] != config['feature_dimensions'] for rows in songs.values() for song in rows):
        raise ValueError('C input width changed')
    return songs


def run_one(run_id, resume):
    run_dir, model_dir, config, contract = validate_run(run_id)
    if (run_dir / 'report.json').exists():
        raise ValueError('Training already completed; verify instead of retraining')
    if not resume:
        raise ValueError('Registered runs require --resume, including their first execution')
    identity = process_identity()
    attempt = len(list(run_dir.glob('attempt-*'))) + 1
    attempt_dir = run_dir / f'attempt-{attempt:03d}'
    attempt_dir.mkdir()
    process = dict(run_id=run_id, started_at=now(), process_identity=identity, attempt=attempt,
                   log=str(run_dir / 'console.log'), config_sha256=sha256(run_dir / 'config.json'))
    atomic_json(run_dir / 'process.json', process)
    atomic_json(attempt_dir / 'process.json', process)
    record_event(dict(event='C_pilot_training_started', run_id=run_id, attempt=attempt),
                 run=dict(run_id=run_id, stage='C', status='running', process_identity=identity))

    def progress(row):
        item = dict(**row, run_id=run_id, process_identity=identity)
        write_status(run_dir / 'status.json', item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    try:
        started = time.monotonic()
        device = configure_torch(config)
        progress(dict(stage='verifying_and_loading_C_inputs'))
        songs = load_inputs(config, contract)
        torch.cuda.reset_peak_memory_stats()
        result = train_loop(songs, config['settings'], config, run_dir, model_dir, device, True, progress)
        validate_run(run_id)
        verify_hashes(contract['protected_hashes'])
        best = torch.load(model_dir / 'best.pth', weights_only=True, map_location='cpu')
        result.update(run_id=run_id, stage='C', experiment=config['experiment'], seed=config['settings']['seed'],
            completed_at=now(), elapsed_seconds=round(time.monotonic()-started, 3),
            legacy_model_unchanged=True, baseline_model_unchanged=True,
            checkpoint=str(model_dir / 'best.pth'), checkpoint_sha256=sha256(model_dir / 'best.pth'),
            last_checkpoint_sha256=sha256(model_dir / 'last.pth'),
            config_sha256=sha256(run_dir / 'config.json'), contract_sha256=config['contract_sha256'],
            protocol_sha256=config['protocol_sha256'],
            parameter_count=sum(value.numel() for value in best['state_dict'].values()),
            gpu_name=torch.cuda.get_device_name(), runtime=config['runtime'],
            independent_reload_pending=True, limitations=read_json(PROTOCOL)['limitations'])
        atomic_json(run_dir / 'report.json', result)
        atomic_json(attempt_dir / 'result.json', dict(status='completed', completed_at=now()))
        progress(dict(stage='completed_pending_independent_reload', step=result['completed_steps'],
                      best_step=result['best_step'], best_development_score=result['best_development_score']))
        record_event(dict(event='C_pilot_training_completed', run_id=run_id, score=result['best_development_score']),
                     run=dict(run_id=run_id, status='completed_pending_verification', result=relative(run_dir / 'report.json')))
    except BaseException as exc:
        failure = dict(status='failed', error=f'{type(exc).__name__}: {exc}', failed_at=now())
        atomic_json(attempt_dir / 'failure.json', failure)
        progress(dict(stage='failed', error=failure['error']))
        record_event(dict(event='C_pilot_training_failed', run_id=run_id, **failure),
                     run=dict(run_id=run_id, status='failed', error=failure['error']))
        raise


def verification_matches(run_id):
    run_dir, model_dir = paths(run_id)
    if not all((run_dir / name).is_file() for name in ('report.json', 'verification.json')):
        return False
    report, checked = read_json(run_dir / 'report.json'), read_json(run_dir / 'verification.json')
    if checked.get('status') != 'completed_verified' or report.get('status') != 'completed':
        return False
    if not all(checked.get(name) is True for name in ('separate_process', 'strict_state_load',
            'full_resume_state_verified', 'fit_only_scaler_verified', 'legacy_model_unchanged', 'baseline_model_unchanged')):
        return False
    if checked.get('calibration_validation_test_songs_used') != 0:
        return False
    if not all((model_dir / (name + '.pth')).is_file() for name in ('best', 'last')):
        return False
    return (checked['report_sha256'] == sha256(run_dir / 'report.json')
            and checked['config_sha256'] == sha256(run_dir / 'config.json')
            and checked['best_sha256'] == report['checkpoint_sha256'] == sha256(model_dir / 'best.pth')
            and checked['last_sha256'] == report['last_checkpoint_sha256'] == sha256(model_dir / 'last.pth')
            and report['completed_steps'] == 2000 and checked['fit_songs'] == 60 and checked['development_songs'] == 30)


def verify_one(run_id):
    if is_running(run_id):
        raise RuntimeError('Cannot verify a run while its training lease is active')
    run_dir, model_dir, config, contract = validate_run(run_id)
    if (run_dir / 'verification.json').exists():
        if not verification_matches(run_id):
            raise ValueError('Existing completed verification no longer matches; preserve evidence')
        print(json.dumps(dict(run_id=run_id, status='already_verified')))
        return
    identity = process_identity()
    atomic_json(run_dir / 'verification-process.json', dict(started_at=now(), process_identity=identity))
    report = read_json(run_dir / 'report.json')
    if report['status'] != 'completed' or report['completed_steps'] != config['settings']['max_steps']:
        raise ValueError('Training has not completed its fixed step budget')
    if report['config_sha256'] != sha256(run_dir / 'config.json'):
        raise ValueError('Training report config mismatch')
    for name, field in [('best', 'checkpoint_sha256'), ('last', 'last_checkpoint_sha256')]:
        if sha256(model_dir / (name + '.pth')) != report[field]:
            raise ValueError('Checkpoint changed: ' + name)
    device = configure_torch(config)
    songs = load_inputs(config, contract)
    best = torch.load(model_dir / 'best.pth', map_location='cpu', weights_only=True)
    last = torch.load(model_dir / 'last.pth', map_location='cpu', weights_only=True)
    for saved in (best, last):
        if saved['provenance'] != config or saved['settings'] != config['settings']:
            raise ValueError('Checkpoint provenance/settings mismatch')
    required = {'optimizer', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng', 'best_state_dict'}
    if not required.issubset(last) or last['step'] != 2000 or last['best_step'] != best['step']:
        raise ValueError('Full resume checkpoint is incomplete')
    for name, value in best['state_dict'].items():
        torch.testing.assert_close(value, last['best_state_dict'][name], rtol=0, atol=0)
    mean, std = fit_scaler(songs['fit'])
    for saved in (best, last):
        np.testing.assert_array_equal(saved['mean'].numpy(), mean)
        np.testing.assert_array_equal(saved['std'].numpy(), std)
    model = SequenceModel(config['feature_dimensions'], config['settings']).to(device)
    model.load_state_dict(best['state_dict'], strict=True)
    torch.cuda.synchronize()
    started = time.monotonic()
    metrics = evaluate(model, songs['development'], mean, std, device, config['settings']['batch_size'])
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    difference = max(assert_metrics_match(best['development'], metrics),
                     assert_metrics_match(report['best_development'], metrics))
    np.testing.assert_allclose(score(metrics), report['best_development_score'], rtol=1e-5, atol=1e-6)
    validate_run(run_id)
    verify_hashes(contract['protected_hashes'])
    checked = dict(run_id=run_id, status='completed_verified', verified_at=now(), process_identity=identity,
        separate_process=identity != read_json(run_dir / 'process.json')['process_identity'],
        report_sha256=sha256(run_dir / 'report.json'), config_sha256=sha256(run_dir / 'config.json'),
        best_sha256=sha256(model_dir / 'best.pth'), last_sha256=sha256(model_dir / 'last.pth'),
        strict_state_load=True, full_resume_state_verified=True, fit_only_scaler_verified=True,
        maximum_metric_difference=difference, development_score=score(metrics),
        fit_songs=60, development_songs=30, calibration_validation_test_songs_used=0,
        development_metrics=metrics, cached_feature_inference_seconds=seconds,
        cached_feature_inference_frames=sum(len(song['x']) for song in songs['development']),
        legacy_model_unchanged=True, baseline_model_unchanged=True)
    if not checked['separate_process']:
        raise ValueError('Independent reload requires a fresh process')
    atomic_json(run_dir / 'verification.json', checked)
    record_event(dict(event='C_pilot_independent_reload_verified', run_id=run_id, score=score(metrics)),
                 run=dict(run_id=run_id, status='completed_verified', verification=relative(run_dir / 'verification.json')))
    print(json.dumps(dict(run_id=run_id, status='completed_verified', score=score(metrics), maximum_metric_difference=difference)))


def queue(resume):
    launch = read_json(QUEUE_DIR / 'launch.json')
    if sha256(MATRIX) != launch['matrix_sha256']:
        raise ValueError('Registered matrix changed')
    matrix = read_json(MATRIX)
    verify_hashes(matrix['protected_hashes'])
    for name, value in launch['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('Queue environment differs: ' + name)
    if (QUEUE_DIR / 'report.json').exists():
        raise ValueError('Queue already finished; inspect its report')
    if (QUEUE_DIR / 'process.json').exists() and not resume:
        raise ValueError('Previously started queue requires diagnosis and explicit --resume')
    identity = process_identity()
    attempt = len(list(QUEUE_DIR.glob('attempt-*'))) + 1
    attempt_dir = QUEUE_DIR / f'attempt-{attempt:03d}'
    attempt_dir.mkdir()
    process = dict(started_at=now(), process_identity=identity, attempt=attempt)
    atomic_json(QUEUE_DIR / 'process.json', process)
    atomic_json(attempt_dir / 'process.json', process)
    update_stage(dict(event='C_pilot_queue_started', job_id=QUEUE_ID, attempt=attempt),
                 phase='pilot_training_running', training_queue={**launch, 'status': 'running', 'process_identity': identity})
    done = []
    active = None
    try:
        for row in matrix['runs']:
            active = row['run_id']
            if is_running(active) or is_running('verify-' + active):
                raise RuntimeError('A same-number C worker is already active: ' + active)
            if verification_matches(active):
                done.append(active)
                continue
            if is_running(GPU_LEASE):
                raise RuntimeError('GPU lease is held by another workload; no competing launch')
            run_dir, _ = paths(active)
            child = read_json(ROOT / row['launch'])
            for mode, argv in [('training', child['original_argv']), ('verification', child['verify_argv'])]:
                if mode == 'training' and (run_dir / 'report.json').exists():
                    continue
                if mode == 'training' and (run_dir / 'process.json').exists() and not resume:
                    raise ValueError('Interrupted run needs diagnosis before queue --resume: ' + active)
                item = dict(stage=mode, active_run=active, completed_runs=len(done), total_runs=len(matrix['runs']),
                            process_identity=identity)
                write_status(QUEUE_DIR / 'status.json', item)
                print(json.dumps(item), flush=True)
                subprocess.run(argv, cwd=child['cwd'], env={**os.environ, **child['environment']}, check=True)
            if not verification_matches(active):
                raise ValueError('Fresh-process completion verification missing: ' + active)
            done.append(active)
        report = dict(job_id=QUEUE_ID, status='completed_awaiting_paired_analysis', completed_at=now(),
                      completed_runs=done, count=len(done), matrix_sha256=sha256(MATRIX),
                      original_validation_test_calibration_songs_used=0)
        atomic_json(QUEUE_DIR / 'report.json', report)
        write_status(QUEUE_DIR / 'status.json', dict(stage=report['status'], completed_runs=len(done), total_runs=18,
                                                   process_identity=identity))
        update_stage(dict(event='C_pilot_queue_completed', completed_runs=len(done)),
                     phase='pilot_training_complete_pending_selection',
                     training_queue={**launch, 'status': report['status'], 'report': relative(QUEUE_DIR / 'report.json')})
        record_event(dict(event='C_pilot_paired_analysis_ready'), updates=dict(next_action=
            '汇总六组各三个种子的逐库逐维表现、常量曲线对照、动态误差和成本；按冻结门槛做配对歌曲组bootstrap，最多选择一个候选扩大内部验证，继续封存原验证和测试集。'))
    except BaseException as exc:
        failure = dict(job_id=QUEUE_ID, status='failed', failed_at=now(), active_run=active,
                       completed_runs=done, error=f'{type(exc).__name__}: {exc}')
        atomic_json(attempt_dir / 'failure.json', failure)
        write_status(QUEUE_DIR / 'status.json', dict(stage='failed', active_run=active, completed_runs=len(done),
                                                   total_runs=18, error=failure['error'], process_identity=identity))
        update_stage(dict(event='C_pilot_queue_failed', active_run=active, error=failure['error']),
                     phase='pilot_training_requires_diagnosis',
                     training_queue={**launch, 'status': 'failed', 'failure': relative(attempt_dir / 'failure.json')})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register-matrix', action='store_true')
    mode.add_argument('--run-id')
    mode.add_argument('--verify')
    mode.add_argument('--queue', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.register_matrix:
        with run_lease(QUEUE_ID):
            register_matrix()
        return
    run_id = args.run_id or args.verify
    directory = paths(run_id)[0] if run_id else QUEUE_DIR
    logfile = directory / ('verification.log' if args.verify else 'console.log')
    lease_id = ('verify-' + run_id) if args.verify else (run_id or QUEUE_ID)
    with run_lease(lease_id), logfile.open('a', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                if args.queue:
                    queue(args.resume)
                else:
                    with run_lease(GPU_LEASE):
                        verify_one(run_id) if args.verify else run_one(run_id, args.resume)
            except BaseException:
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
