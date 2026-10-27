"""Frozen six-run D queue; no automatic retries, independent reload between runs."""
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
    now, paths, process_identity, read_json, record_event, verify_hashes)
from tools.train_next_v2.evaluation import evaluate, assert_metrics_match, score
from tools.train_next_v2.run import Tee
from .data import CONTRACT, PROTOCOL, FEATURE_JOB, relative, training_code_hashes, prepare_contract, load_inputs
from .engine import train_loop, check_resume, tensor_digest, ARCHITECTURE
from .model import AuxiliaryModel
from .sampling import PairedSampler

QUEUE_ID = 'd-merge-paired-training-v1'
QUEUE_DIR = BASE/'queues'/QUEUE_ID
MATRIX = QUEUE_DIR/'matrix.json'
MODULE = 'tools.train_next_v2.auxiliary_training.execution'
CHECKS = BASE/'analyses/d-auxiliary-code-tests-v1/report.json'


def runtime_signature():
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__)


def environment():
    env = dict(read_json(FEATURE_JOB/'launch.json')['environment'])
    env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env


def update_stage(event, **updates):
    with run_lease('next-va-v2-d-ledger'):
        stages = read_json(PLAN)['stages']
        stages['D'].update(updates)
        record_event(event, updates=dict(current_stage='D', stages=stages))


def checks_passed():
    checks = read_json(CHECKS)
    if (checks['status'] != 'passed' or checks['code_hashes'] != training_code_hashes()
            or not checks.get('cuda_resume_verified') or checks.get('skipped') != 0):
        raise ValueError('Fresh focused code verification must pass before D admission')


def settings_for(protocol, variant, seed):
    if seed not in protocol['optimization']['seeds']:
        raise ValueError('Seed outside frozen D protocol')
    return dict(protocol['model']['configuration'], seed=seed,
        quadrant_loss_weight=variant['quadrant_loss_weight'], aux_batch_size=protocol['optimization']['aux_batch_clips'])


def prepare():
    checks_passed()
    contract = prepare_contract(torch.device('cpu'), progress=lambda row: print(json.dumps(row), flush=True))
    update_stage(dict(event='D_auxiliary_inputs_verified', contract=relative(CONTRACT)),
        phase='MERGE_auxiliary_training_ready', training_inputs=relative(CONTRACT),
        admitted_auxiliary_clips=len(contract['auxiliary']), numeric_MERGE_VA_regression_enabled=False)
    print(json.dumps(dict(contract=relative(CONTRACT), auxiliary_clips=len(contract['auxiliary']),
                          main_fit=3070, internal_development=383)))


def register_matrix():
    if MATRIX.exists() or is_running(GPU_LEASE):
        raise ValueError('Matrix already registered or GPU still occupied')
    checks_passed()
    contract, protocol = read_json(CONTRACT), read_json(PROTOCOL)
    verify_hashes(contract['protected_hashes']); verify_hashes(contract['builder_code_hashes'])
    if contract['protocol_sha256'] != sha256(PROTOCOL) or not contract['training_admitted']:
        raise ValueError('Input admission does not match D protocol')
    env, code = environment(), training_code_hashes()
    commands, protected = [], dict(code)
    run_specs = [(variant, seed) for seed in protocol['optimization']['seeds'] for variant in protocol['variants']]
    # Preflight all destinations before any registration; preserve partially registered runs on error.
    for variant, seed in run_specs:
        rid = variant['id']+'-v1-s'+str(seed)
        if any(p.exists() for p in paths(rid)):
            raise FileExistsError('Existing D run must be preserved: '+rid)
    for variant, seed in run_specs:
        rid = variant['id']+'-v1-s'+str(seed)
        run_dir, model_dir = paths(rid)
        config = dict(schema_version=1, run_id=rid, stage='D', experiment=variant['id'],
            settings=settings_for(protocol, variant, seed), feature_dimensions=531,
            protocol=relative(PROTOCOL), protocol_sha256=sha256(PROTOCOL),
            contract=relative(CONTRACT), contract_sha256=sha256(CONTRACT), code=code,
            runtime=runtime_signature(), environment=env,
            arithmetic=dict(deterministic=True, fp32=True, amp=False, tf32=False, cudnn_benchmark=False),
            initialization='Fresh paired random backbone and auxiliary head; no trained checkpoint initialization',
            fit_songs=3070, development_songs=383, auxiliary_clips=1001,
            calibration_songs_used=0, original_validation_songs_used=0, test_songs_used=0,
            normalization='Main internal fit only, identical in each paired run')
        run_dir.mkdir(parents=True); model_dir.mkdir(parents=True)
        atomic_json(run_dir/'config.json', config)
        argv = [str(Path(sys.executable).resolve()), '-u', '-m', MODULE, '--run-id', rid, '--resume']
        launch = dict(run_id=rid, stage='D', original_argv=argv, resume_argv=argv,
            verify_argv=[argv[0], '-u', '-m', MODULE, '--verify', rid], cwd=str(ROOT), python=argv[0], environment=env,
            config=relative(run_dir/'config.json'), config_sha256=sha256(run_dir/'config.json'), code_hashes=code,
            data_hashes=dict(contract=sha256(CONTRACT), protocol=sha256(PROTOCOL)),
            os_lock=str(lock_path(rid)), gpu_lock=str(lock_path(GPU_LEASE)), log_paths=[str(run_dir/'console.log')],
            checkpoint_paths=dict(best=str(model_dir/'best.pth'), last=str(model_dir/'last.pth')),
            step_budget=20000, main_exposures=80000, auxiliary_forward_exposures=80000,
            recovery='Diagnose exit and verify same-ID/GPU leases first; exact argv --resume restores last.pth, optimizer, both samplers, scaler and all RNG. No automatic retry or relaxed hashes.')
        atomic_json(run_dir/'launch.json', launch)
        for name in ('config.json', 'launch.json'):
            protected[relative(run_dir/name)] = sha256(run_dir/name)
        commands.append(dict(run_id=rid, launch=relative(run_dir/'launch.json')))
        record_event(dict(event='D_auxiliary_run_registered', run_id=rid),
            run=dict(run_id=rid, stage='D', status='registered', launch=relative(run_dir/'launch.json'),
                     **{k: launch[k] for k in ('original_argv', 'cwd', 'python', 'environment', 'config_sha256',
                         'log_paths', 'os_lock', 'checkpoint_paths', 'step_budget')}))
    for p in (CONTRACT, PROTOCOL, CHECKS):
        protected[relative(p)] = sha256(p)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    matrix = dict(schema_version=1, job_id=QUEUE_ID, registered_at=now(), runs=commands, protected_hashes=protected,
        fixed_runs=6, steps_per_run=20000, ordering='seed then control and auxiliary', automatic_retry=False,
        independent_reload_required_before_next_run=True)
    atomic_json(MATRIX, matrix)
    argv = [str(Path(sys.executable).resolve()), '-u', '-m', MODULE, '--queue']
    launch = dict(job_id=QUEUE_ID, original_argv=argv, resume_argv=argv+['--resume'], cwd=str(ROOT),
        python=argv[0], environment=env, matrix=relative(MATRIX), matrix_sha256=sha256(MATRIX), code_hashes=code,
        os_lock=str(lock_path(QUEUE_ID)), log=str(QUEUE_DIR/'console.log'),
        recovery='Stop on failure; inspect actual leases and failure logs before --queue --resume. Skip only verified completed runs; preserve and validate original configuration.')
    atomic_json(QUEUE_DIR/'launch.json', launch)
    update_stage(dict(event='D_auxiliary_matrix_registered', runs=6), phase='MERGE_paired_training_registered',
        training_queue=dict(job_id=QUEUE_ID, status='registered', launch=relative(QUEUE_DIR/'launch.json'),
        matrix=relative(MATRIX), runs=6, status_argv=[argv[0], '-m', 'tools.train_next_v2.auxiliary_training.status']))
    print(json.dumps(dict(registered_runs=6, launch=relative(QUEUE_DIR/'launch.json'))))


def validate_run(rid):
    run_dir, model_dir = paths(rid)
    launch, config = read_json(run_dir/'launch.json'), read_json(run_dir/'config.json')
    if launch['run_id'] != rid or config['run_id'] != rid or config['stage'] != 'D':
        raise ValueError('Wrong D run identity')
    if launch['config_sha256'] != sha256(run_dir/'config.json'):
        raise ValueError('Frozen configuration changed')
    verify_hashes(config['code'])
    verify_hashes({config['protocol']: config['protocol_sha256'], config['contract']: config['contract_sha256']})
    if config['runtime'] != runtime_signature():
        raise ValueError('Frozen runtime changed')
    for name, value in config['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('Frozen environment differs: '+name)
    protocol = read_json(ROOT/config['protocol'])
    variant = next(v for v in protocol['variants'] if v['id'] == config['experiment'])
    if config['settings'] != settings_for(protocol, variant, config['settings']['seed']):
        raise ValueError('Settings differ from frozen paired protocol')
    contract = read_json(ROOT/config['contract'])
    verify_hashes(contract['builder_code_hashes'])
    return run_dir, model_dir, config, contract


def configure_torch(settings):
    assert_gpu_available()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(settings['threads'])
    torch.manual_seed(settings['seed']); np.random.seed(settings['seed']); random.seed(settings['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    return torch.device('cuda')


def run_one(rid, resume):
    run_dir, model_dir, config, contract = validate_run(rid)
    if (run_dir/'report.json').exists():
        raise FileExistsError('Completed training must only be verified')
    if not resume:
        raise ValueError('Registered runs require --resume including their first execution')
    identity = process_identity()
    attempt = len(list(run_dir.glob('attempt-*')))+1
    attempt_dir = run_dir/f'attempt-{attempt:03d}'; attempt_dir.mkdir()
    process = dict(run_id=rid, started_at=now(), process_identity=identity, attempt=attempt,
                   config_sha256=sha256(run_dir/'config.json'))
    atomic_json(run_dir/'process.json', process); atomic_json(attempt_dir/'process.json', process)
    record_event(dict(event='D_auxiliary_run_started', run_id=rid, attempt=attempt),
                 run=dict(run_id=rid, stage='D', status='running', process_identity=identity))
    def progress(row):
        row = dict(row, run_id=rid, process_identity=identity)
        write_status(run_dir/'status.json', row); print(json.dumps(row), flush=True)
    try:
        started = time.monotonic()
        device = configure_torch(config['settings'])
        progress(dict(stage='verifying_main_and_auxiliary_inputs'))
        songs, auxiliary = load_inputs(contract, device, progress)
        torch.cuda.reset_peak_memory_stats()
        result = train_loop(songs, auxiliary, config['settings'], config, run_dir, model_dir, device, True, progress)
        validate_run(rid); verify_hashes(contract['protected_hashes'])
        result.update(run_id=rid, stage='D', experiment=config['experiment'], seed=config['settings']['seed'],
            completed_at=now(), elapsed_seconds=round(time.monotonic()-started, 3),
            legacy_model_unchanged=True, baseline_model_unchanged=True, independent_reload_pending=True,
            checkpoint=str(model_dir/'best.pth'), checkpoint_sha256=sha256(model_dir/'best.pth'),
            last_checkpoint_sha256=sha256(model_dir/'last.pth'), config_sha256=sha256(run_dir/'config.json'),
            contract_sha256=config['contract_sha256'], protocol_sha256=config['protocol_sha256'],
            runtime=config['runtime'], gpu_name=torch.cuda.get_device_name(), limitations=contract['limitations'])
        atomic_json(run_dir/'report.json', result)
        atomic_json(attempt_dir/'result.json', dict(status='completed', completed_at=now()))
        progress(dict(stage='completed_pending_independent_reload', step=result['completed_steps'],
            best_step=result['best_step'], best_development_score=result['best_development_score']))
        record_event(dict(event='D_auxiliary_run_completed', run_id=rid, score=result['best_development_score']),
            run=dict(run_id=rid, status='completed_pending_verification', result=relative(run_dir/'report.json')))
    except BaseException as exc:
        failure = dict(status='failed', failed_at=now(), error=f'{type(exc).__name__}: {exc}')
        atomic_json(attempt_dir/'failure.json', failure)
        progress(dict(stage='failed', error=failure['error']))
        record_event(dict(event='D_auxiliary_run_failed', run_id=rid, **failure),
                     run=dict(run_id=rid, status='failed', error=failure['error']))
        raise


def verification_matches(rid):
    run_dir, model_dir = paths(rid)
    if not all((run_dir/name).is_file() for name in ('report.json', 'verification.json')):
        return False
    report, checked = read_json(run_dir/'report.json'), read_json(run_dir/'verification.json')
    if report.get('status') != 'completed' or checked.get('status') != 'completed_verified':
        return False
    if not all(checked.get(k) is True for k in ('separate_process', 'strict_state_load', 'full_resume_state_verified',
            'fit_only_scaler_verified', 'legacy_model_unchanged', 'baseline_model_unchanged')):
        return False
    if checked.get('calibration_validation_test_songs_used') != 0 or report.get('completed_steps') != 20000:
        return False
    if not all((model_dir/(name+'.pth')).is_file() for name in ('best', 'last')):
        return False
    return (checked['report_sha256'] == sha256(run_dir/'report.json')
        and checked['config_sha256'] == sha256(run_dir/'config.json')
        and checked['best_sha256'] == report['checkpoint_sha256'] == sha256(model_dir/'best.pth')
        and checked['last_sha256'] == report['last_checkpoint_sha256'] == sha256(model_dir/'last.pth')
        and checked['fit_songs'] == 3070 and checked['development_songs'] == 383 and checked['auxiliary_clips'] == 1001
        and checked['samplers']['main_exposures'] == 80000 and checked['samplers']['auxiliary_forward_exposures'] == 80000)


def verify_one(rid):
    if is_running(rid):
        raise RuntimeError('Cannot verify an active training worker')
    run_dir, model_dir, config, contract = validate_run(rid)
    if (run_dir/'verification.json').exists():
        if not verification_matches(rid):
            raise ValueError('Previous verification changed; preserve evidence')
        return
    identity = process_identity()
    atomic_json(run_dir/'verification-process.json', dict(started_at=now(), process_identity=identity))
    report = read_json(run_dir/'report.json')
    if (report['status'] != 'completed' or report['completed_steps'] != 20000
            or report['config_sha256'] != sha256(run_dir/'config.json')):
        raise ValueError('Training budget or report/config check failed')
    for name, field in [('best', 'checkpoint_sha256'), ('last', 'last_checkpoint_sha256')]:
        if sha256(model_dir/(name+'.pth')) != report[field]:
            raise ValueError('Checkpoint bytes changed: '+name)
    device = configure_torch(config['settings'])
    songs, auxiliary = load_inputs(contract, device)
    best = torch.load(model_dir/'best.pth', map_location='cpu', weights_only=True)
    last = torch.load(model_dir/'last.pth', map_location='cpu', weights_only=True)
    mean, std = fit_scaler(songs['fit'])
    sampler = PairedSampler(songs['fit'], auxiliary, config['settings'])
    check_resume(last, config['settings'], config, mean, std, sampler)
    if (last['step'] != 20000 or best['step'] != last['best_step'] or best['architecture'] != ARCHITECTURE
            or best['provenance'] != config or best['settings'] != config['settings']):
        raise ValueError('Best/full checkpoint consistency failure')
    if last['samplers'] != report['samplers'] or last['exposure_chain'] != report['exposure_chain']:
        raise ValueError('Exposure accounting mismatch')
    if best['initial_state_sha256'] != last['initial_state_sha256'] or last['initial_state_sha256'] != report['initial_state_sha256']:
        raise ValueError('Paired random initialization receipt mismatch')
    for name, value in best['state_dict'].items():
        torch.testing.assert_close(value, last['best_state_dict'][name], rtol=0, atol=0)
    for saved in (best, last):
        np.testing.assert_array_equal(saved['mean'].numpy(), mean)
        np.testing.assert_array_equal(saved['std'].numpy(), std)
    # Actually restore optimizer and all RNG containers, not just check key existence.
    restored = AuxiliaryModel(531, config['settings']).to(device)
    if tensor_digest(restored.state_dict()) != last['initial_state_sha256']:
        raise ValueError('Fresh paired random initialization is not reproducible')
    restored.load_state_dict(last['state_dict'], strict=True)
    optimizer = torch.optim.AdamW(restored.parameters(), lr=config['settings']['lr'], weight_decay=config['settings']['weight_decay'])
    optimizer.load_state_dict(last['optimizer'])
    for parameter in restored.parameters():
        state = optimizer.state[parameter]
        if int(state['step']) != 20000:
            raise ValueError('Incomplete optimizer step history')
        for key in ('exp_avg', 'exp_avg_sq'):
            if state[key].shape != parameter.shape or not torch.isfinite(state[key]).all():
                raise ValueError('Malformed optimizer state')
    torch.set_rng_state(last['torch_rng']); torch.cuda.set_rng_state_all(last['cuda_rng'])
    random.setstate(last['python_rng'])
    ns = last['numpy_rng']; np.random.set_state((ns[0], np.array(ns[1], np.uint32), ns[2], ns[3], ns[4]))
    restored.load_state_dict(best['state_dict'], strict=True)
    torch.cuda.synchronize(); started = time.monotonic()
    metrics = evaluate(restored, songs['development'], mean, std, device, config['settings']['batch_size'])
    torch.cuda.synchronize(); seconds = time.monotonic()-started
    difference = max(assert_metrics_match(best['development'], metrics), assert_metrics_match(report['best_development'], metrics))
    np.testing.assert_allclose(score(metrics), report['best_development_score'], rtol=1e-5, atol=1e-6)
    validate_run(rid); verify_hashes(contract['protected_hashes'])
    checked = dict(run_id=rid, status='completed_verified', verified_at=now(), process_identity=identity,
        separate_process=identity != read_json(run_dir/'process.json')['process_identity'],
        report_sha256=sha256(run_dir/'report.json'), config_sha256=sha256(run_dir/'config.json'),
        best_sha256=sha256(model_dir/'best.pth'), last_sha256=sha256(model_dir/'last.pth'),
        strict_state_load=True, full_resume_state_verified=True, fit_only_scaler_verified=True,
        maximum_metric_difference=difference, fit_songs=3070, development_songs=383, auxiliary_clips=1001,
        calibration_validation_test_songs_used=0, development_score=score(metrics), development_metrics=metrics,
        cached_feature_inference_seconds=seconds, cached_feature_inference_frames=sum(len(s['x']) for s in songs['development']),
        samplers=last['samplers'], exposure_chain=last['exposure_chain'], initial_state_sha256=last['initial_state_sha256'],
        legacy_model_unchanged=True, baseline_model_unchanged=True)
    if not checked['separate_process']:
        raise ValueError('A separate interpreter is required for final checkpoint verification')
    atomic_json(run_dir/'verification.json', checked)
    record_event(dict(event='D_auxiliary_independent_reload_verified', run_id=rid, score=score(metrics)),
        run=dict(run_id=rid, status='completed_verified', verification=relative(run_dir/'verification.json')))
    print(json.dumps(dict(run_id=rid, status='completed_verified', score=score(metrics))))


def queue(resume):
    launch, matrix = read_json(QUEUE_DIR/'launch.json'), read_json(MATRIX)
    if launch['matrix_sha256'] != sha256(MATRIX):
        raise ValueError('Frozen D matrix changed')
    verify_hashes(matrix['protected_hashes'])
    for name, value in launch['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('Queue environment differs: '+name)
    if (QUEUE_DIR/'report.json').exists():
        raise FileExistsError('Completed queue must not be restarted')
    if (QUEUE_DIR/'process.json').exists() and not resume:
        raise ValueError('Started queue needs diagnosis before --resume')
    identity = process_identity()
    attempt = len(list(QUEUE_DIR.glob('attempt-*')))+1
    attempt_dir = QUEUE_DIR/f'attempt-{attempt:03d}'; attempt_dir.mkdir()
    process = dict(started_at=now(), process_identity=identity, attempt=attempt)
    atomic_json(QUEUE_DIR/'process.json', process); atomic_json(attempt_dir/'process.json', process)
    update_stage(dict(event='D_auxiliary_queue_started', attempt=attempt), phase='MERGE_paired_training_running',
        training_queue=dict(job_id=QUEUE_ID, launch=relative(QUEUE_DIR/'launch.json'), matrix=relative(MATRIX),
                            status='running', process_identity=identity, runs=6))
    done, active = [], None
    try:
        for row in matrix['runs']:
            active = row['run_id']
            if is_running(active) or is_running('verify-'+active):
                raise RuntimeError('A same-ID worker already holds its lease: '+active)
            if verification_matches(active):
                done.append(active); continue
            if is_running(GPU_LEASE):
                raise RuntimeError('GPU is leased; refuse a competing worker')
            child = read_json(ROOT/row['launch']); run_dir, _ = paths(active)
            for mode, argv in [('training', child['original_argv']), ('verification', child['verify_argv'])]:
                if mode == 'training' and (run_dir/'report.json').exists():
                    continue
                if mode == 'training' and (run_dir/'process.json').exists() and not resume:
                    raise ValueError('Interrupted worker requires diagnosis before queue --resume')
                status = dict(stage=mode, active_run=active, completed_runs=len(done), total_runs=6, process_identity=identity)
                write_status(QUEUE_DIR/'status.json', status); print(json.dumps(status), flush=True)
                subprocess.run(argv, cwd=child['cwd'], env={**os.environ, **child['environment']}, check=True)
            if not verification_matches(active):
                raise ValueError('Fresh-process verification is incomplete: '+active)
            done.append(active)
        report = dict(job_id=QUEUE_ID, status='completed_awaiting_paired_analysis', completed_at=now(),
            completed_runs=done, count=6, matrix_sha256=sha256(MATRIX), calibration_validation_test_songs_used=0)
        atomic_json(QUEUE_DIR/'report.json', report)
        write_status(QUEUE_DIR/'status.json', dict(stage=report['status'], completed_runs=6, total_runs=6, process_identity=identity))
        update_stage(dict(event='D_auxiliary_queue_completed', completed_runs=6), phase='MERGE_paired_analysis_pending',
            training_queue=dict(job_id=QUEUE_ID, status=report['status'], launch=relative(QUEUE_DIR/'launch.json'),
                                report=relative(QUEUE_DIR/'report.json'), runs=6))
        record_event(dict(event='D_auxiliary_paired_analysis_ready'), updates=dict(next_action=
            'Verify paired initialization/exposure parity, compare six runs on383 internal-development songs with2000 paired song-group bootstrap replicates, and apply frozen D gates. Do not use sealed validation/test scores to tune.'))
    except BaseException as exc:
        failure = dict(job_id=QUEUE_ID, status='failed', failed_at=now(), active_run=active,
            completed_runs=done, error=f'{type(exc).__name__}: {exc}')
        atomic_json(attempt_dir/'failure.json', failure)
        write_status(QUEUE_DIR/'status.json', dict(stage='failed', active_run=active, completed_runs=len(done),
            total_runs=6, error=failure['error'], process_identity=identity))
        update_stage(dict(event='D_auxiliary_queue_failed', active_run=active, error=failure['error']),
            phase='MERGE_training_requires_diagnosis', training_queue=dict(job_id=QUEUE_ID, status='failed',
                launch=relative(QUEUE_DIR/'launch.json'), failure=relative(attempt_dir/'failure.json')))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--register-matrix', action='store_true')
    mode.add_argument('--run-id'); mode.add_argument('--verify'); mode.add_argument('--queue', action='store_true')
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    if args.prepare or args.register_matrix:
        with run_lease(QUEUE_ID):
            prepare() if args.prepare else register_matrix()
        return
    rid = args.run_id or args.verify
    directory = paths(rid)[0] if rid else QUEUE_DIR
    lease = ('verify-'+rid) if args.verify else (rid or QUEUE_ID)
    with run_lease(lease), (directory/('verification.log' if args.verify else 'console.log')).open('a', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                if args.queue:
                    queue(args.resume)
                else:
                    with run_lease(GPU_LEASE):
                        verify_one(rid) if args.verify else run_one(rid, args.resume)
            except BaseException:
                traceback.print_exc(); raise


if __name__ == '__main__':
    main()
