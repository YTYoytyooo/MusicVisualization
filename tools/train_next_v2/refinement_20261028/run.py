"""Register, run, or inspect one bounded cache-only refinement experiment."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from tools.train_next.core import atomic_json, sha256, write_status
from tools.train_next.runtime import run_lease, lock_path, is_running
from tools.train_next_v2.common import (ROOT, BASE, PLAN, GPU_LEASE, now, read_json,
    process_identity, code_hashes as reference_hashes, verify_hashes, assert_gpu_available)
from tools.train_next_v2.data import load_internal
from tools.train_next_v2.run import Tee
from .engine import train_loop

WINDOW = ROOT / 'data/v3.0/optimization/20261028'
PROTOCOL = WINDOW / 'protocol.json'
PACKAGE = Path(__file__).parent
ENV_NAMES = ('PYTHONUTF8', 'HF_HUB_OFFLINE', 'HF_HUB_DISABLE_PROGRESS_BARS', 'TEMP', 'TMP',
             'CUBLAS_WORKSPACE_CONFIG', 'CUDA_CACHE_PATH', 'HF_HOME', 'TORCH_HOME', 'NUMBA_CACHE_DIR')


class WindowEnded(RuntimeError):
    pass


def expired(utc_text, current=None):
    return (current or datetime.now(timezone.utc)) >= datetime.fromisoformat(utc_text)


def locations(arm, seed):
    if arm not in ('control', 'mean_loss', 'tone_residual') or seed not in (20261003, 20261004, 20261005):
        raise ValueError('Unknown frozen arm or seed')
    identifier = f'refinement-20261028-{arm}-s{seed}'
    return identifier, WINDOW / 'runs' / identifier, ROOT / 'models/next_va_v2' / identifier


def sources():
    result = reference_hashes()
    for name in ('__init__.py', 'model.py', 'engine.py', 'run.py'):
        path = PACKAGE / name
        result[path.relative_to(ROOT).as_posix()] = sha256(path)
    return result


def configuration(arm, seed):
    protocol = read_json(PROTOCOL)
    frozen = ROOT / protocol['source_protocol']
    if sha256(frozen) != protocol['source_protocol_sha256']:
        raise ValueError('Historical split/protocol changed')
    source = read_json(frozen)
    spec = next(r for r in protocol['arms'] if r['id'] == arm)
    run_id, _, _ = locations(arm, seed)
    return protocol, source, dict(schema_version=1, run_id=run_id, arm=arm,
        settings={**protocol['defaults'], **spec['overrides'], 'seed': seed},
        protocol_sha256=sha256(PROTOCOL), source_protocol_sha256=sha256(frozen),
        code_hashes=sources(), python=str(Path(sys.executable).resolve()), device='cuda',
        environment={name: os.environ.get(name) for name in ENV_NAMES},
        runtime={'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda': torch.version.cuda},
        worker_stop_utc=protocol['worker_stop_utc'], initialization='random; no prior trained weights')


def ledger(event, row):
    # Same mutex as historical orchestration; retain unrelated state verbatim.
    with run_lease('next-va-v2-plan'):
        local = read_json(WINDOW / 'plan.json')
        old = next((j for j in local['jobs'] if j['run_id'] == row['run_id']), {})
        merged = {**old, **row}
        local['jobs'] = [j for j in local['jobs'] if j['run_id'] != row['run_id']] + [merged]
        local['updated_at'] = now()
        atomic_json(WINDOW / 'plan.json', local)
        global_plan = read_json(PLAN)
        section = global_plan['v3_optimization_20261028']
        section['jobs'] = local['jobs']
        section['updated_at'] = local['updated_at']
        global_plan['active_jobs'] = [j for j in global_plan['active_jobs'] if j['run_id'] != row['run_id']]
        if row['status'] in ('registered', 'running'):
            global_plan['active_jobs'].append(merged)
        global_plan['updated_at'] = now()
        atomic_json(PLAN, global_plan)
        with (BASE / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(at=now(), event=event, run_id=row['run_id'], status=row['status']), ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def validate_budget(protocol, arm, seed):
    if expired(protocol['latest_new_training_utc']):
        raise WindowEnded('No new training after09:30 Beijing')
    if seed != protocol['seeds'][0]:
        selection = WINDOW / 'selection.json'
        if not selection.is_file():
            raise ValueError('Repeat seeds require a completed, frozen screening selection')
        picked = read_json(selection)
        if (picked.get('protocol_sha256') != sha256(PROTOCOL) or not picked.get('eligible')
                or arm not in ('control', picked.get('selected_arm'))):
            raise ValueError('Not authorized by screening selection')
    registered = list((WINDOW / 'runs').glob('*/config.json'))
    if len(registered) >= protocol['budget']['max_runs']:
        raise ValueError('Seven-run fixed budget exhausted')


def register(args):
    protocol, source, config = configuration(args.arm, args.seed)
    run_id, run_dir, model_dir = locations(args.arm, args.seed)
    with run_lease('refinement-20261028-registration'), run_lease(run_id):
        validate_budget(protocol, args.arm, args.seed)
        if run_dir.exists() or model_dir.exists():
            raise ValueError('Existing run must be inspected/resumed; registration never overwrites it')
        expected_python = ROOT / 'data/training/next_va/env/Scripts/python.exe'
        if Path(sys.executable).resolve() != expected_python.resolve():
            raise ValueError('Use the registered independent GPU environment')
        run_dir.mkdir(parents=True)
        model_dir.mkdir(parents=True)
        atomic_json(run_dir / 'config.json', config)
        argv = [config['python'], '-u', '-m', 'tools.train_next_v2.refinement_20261028.run',
                '--arm', args.arm, '--seed', str(args.seed), '--resume']
        row = dict(run_id=run_id, status='registered', original_argv=argv, cwd=str(ROOT),
            environment=config['environment'], python=config['python'], config_hash=sha256(run_dir / 'config.json'),
            code_hashes=config['code_hashes'], protocol_sha256=config['protocol_sha256'],
            source_protocol_sha256=config['source_protocol_sha256'], manifest_sha256=source['manifest_sha256'],
            feature_key=source['feature_key'], expected_cached_songs=3453,
            expected_fit_songs=3070, expected_development_songs=383,
            label_and_feature_hashes='Frozen source protocol contains complete per-song label and feature SHA256 maps',
            os_lock=str(lock_path(run_id)), gpu_lock=str(lock_path(GPU_LEASE)),
            log=str(run_dir / 'console.log'), status_path=str(run_dir / 'status.json'),
            checkpoints={name: str(model_dir / (name + '.pth')) for name in ('best', 'last')},
            process_identity=None, registered_at=now(), worker_stop_utc=protocol['worker_stop_utc'],
            recovery='Confirm lock released AND previous creation identity absent. Original argv includes --resume; exact config/data/code/environment and full last.pth required. Never restart a completed run or extend deadline without new authorization.')
        atomic_json(run_dir / 'launch.json', row)
        ledger('refinement_registered', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)


def actual_status(arm, seed):
    run_id, run_dir, model_dir = locations(arm, seed)
    row = read_json(run_dir / 'process.json') if (run_dir / 'process.json').is_file() else {}
    identity = row.get('process_identity')
    alive = bool(identity and identity == process_identity(identity['pid']))
    held = is_running(run_id)
    last = read_json(run_dir / 'status.json') if (run_dir / 'status.json').exists() else None
    report = read_json(run_dir / 'report.json') if (run_dir / 'report.json').exists() else None
    return dict(run_id=run_id, observed_at=now(), process_identity=identity, same_process_alive=alive,
        run_lock_held=held, gpu_lock_held=is_running(GPU_LEASE), active=bool(alive and held),
        indeterminate=bool(alive != held), last_progress=last,
        report={k: report.get(k) for k in ('status', 'completed_steps', 'best_step', 'best_development_score', 'checkpoint_reload_verified')} if report else None,
        checkpoints={p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in model_dir.glob('*.pth')},
        log_tail=bounded_tail(run_dir / 'console.log'))


def bounded_tail(path):
    if not path.exists():
        return []
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 16384))
        return stream.read(16384).decode('utf-8', errors='replace').splitlines()[-5:]


def execute(args):
    protocol, source, config = configuration(args.arm, args.seed)
    run_id, run_dir, model_dir = locations(args.arm, args.seed)
    if not args.resume or not (run_dir / 'launch.json').is_file():
        raise ValueError('Register first, then use the original --resume command')
    if read_json(run_dir / 'config.json') != config:
        raise ValueError('Registered configuration/code/environment/runtime changed; do not weaken checks')
    if (run_dir / 'report.json').exists():
        raise ValueError('Run already completed; never restart')
    if expired(config['worker_stop_utc']):
        raise WindowEnded('Worker deadline reached; no training/inference after09:50')
    if not (model_dir / 'last.pth').exists() and expired(protocol['latest_new_training_utc']):
        raise WindowEnded('Too late to start a new training run')
    old_process = read_json(run_dir / 'process.json') if (run_dir / 'process.json').exists() else {}
    identity = old_process.get('process_identity')
    if identity and identity == process_identity(identity['pid']):
        raise RuntimeError('Previous worker creation identity still active')
    assert_gpu_available()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Deterministic cuBLAS configuration missing')
    verify_hashes(config['code_hashes'])
    protected = {'models/emotion_model.pth': source['legacy_model_sha256'],
                 'models/next_va/mixed-20261003-recovery/best.pth': source['baseline_checkpoint_sha256']}
    verify_hashes(protected)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(config['settings']['threads'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    row = {**read_json(run_dir / 'launch.json'), 'status': 'running',
           'started_at': now(), 'process_identity': process_identity(), 'gpu_name': torch.cuda.get_device_name(0)}
    atomic_json(run_dir / 'process.json', row)
    ledger('refinement_started', row)

    def progress(data):
        write_status(run_dir / 'status.json', dict(**data, process_identity=row['process_identity']))
        print(json.dumps(data, ensure_ascii=False), flush=True)

    def loading_progress(data):
        progress(data)
        if expired(config['worker_stop_utc']):
            raise WindowEnded('Deadline reached during cache verification/loading')

    started = time.monotonic()
    try:
        loading_progress(dict(stage='verifying_cache', expected=3453))
        device = torch.device('cuda')
        songs = load_internal(source, device, loading_progress)
        if len(songs['fit']) != 3070 or len(songs['development']) != 383:
            raise ValueError('Frozen data count mismatch')
        torch.cuda.reset_peak_memory_stats()
        result = train_loop(songs, config['settings'], config, run_dir, model_dir, device,
                            resume=True, progress=progress, should_stop=lambda: expired(config['worker_stop_utc']))
        verify_hashes(config['code_hashes'])
        verify_hashes(protected)
        result.update(run_id=run_id, arm=args.arm, seed=args.seed, recorded_at=now(),
            elapsed_seconds=time.monotonic()-started, protocol_sha256=config['protocol_sha256'],
            protected_hashes=protected, checkpoint_hashes={p.name: sha256(p) for p in model_dir.glob('*.pth')})
        name = 'report.json' if result['status'] == 'completed' else 'pause.json'
        atomic_json(run_dir / name, result)
        progress(dict(stage=result['status'], step=result['completed_steps'],
                      best_development_score=result.get('best_development_score'), report=str(run_dir / name)))
        ledger('refinement_' + result['status'], dict(run_id=run_id, status=result['status'],
            result=str(run_dir / name), completed_steps=result['completed_steps'],
            best_development_score=result.get('best_development_score')))
    except BaseException as exc:
        status = 'paused_deadline' if isinstance(exc, WindowEnded) else 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        progress(dict(stage=status, error=f'{type(exc).__name__}: {exc}'))
        ledger('refinement_' + status, dict(run_id=run_id, status=status, error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=('control', 'mean_loss', 'tone_residual'))
    parser.add_argument('--seed', required=True, type=int, choices=(20261003, 20261004, 20261005))
    parser.add_argument('--register-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(actual_status(args.arm, args.seed), ensure_ascii=False, indent=2))
    elif args.register_only:
        register(args)
    else:
        run_id, run_dir, _ = locations(args.arm, args.seed)
        if not run_dir.is_dir():
            raise ValueError('Register launch before executing')
        with run_lease(GPU_LEASE), run_lease(run_id):
            with (run_dir / 'console.log').open('a', encoding='utf-8', buffering=1) as log:
                with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                    execute(args)


if __name__ == '__main__':
    main()
