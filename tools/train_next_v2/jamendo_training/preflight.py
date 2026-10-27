"""Record and execute CPU/GPU interruption and data-boundary checks once."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys
import time
import unittest

import torch

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, is_running, lock_path
from tools.train_next_v2.common import BASE, ROOT, GPU_LEASE, now, process_identity, read_json, record_event, verify_hashes
from tools.train_next_v2.run import Tee
from .data import PROTOCOL, relative, training_code_hashes
from .execution import CHECKS, environment

JOB = 'd-jamendo-training-code-tests-v1'
FOLDER = CHECKS.parent


def register():
    if FOLDER.exists() or is_running(GPU_LEASE):
        raise ValueError('Preflight already registered or GPU is occupied')
    env = environment()
    argv = [str(Path(sys.executable).resolve()), '-u', '-m', 'tools.train_next_v2.jamendo_training.preflight', '--run']
    launch = dict(job_id=JOB, stage='D', registered_at=now(), original_argv=argv, cwd=str(ROOT),
        python=argv[0], environment=env, code_hashes=training_code_hashes(),
        data_hashes={relative(PROTOCOL):sha256(PROTOCOL)},
        os_lock=str(lock_path(JOB)), gpu_lock=str(lock_path(GPU_LEASE)),
        log=relative(FOLDER/'console.log'), report=relative(CHECKS),
        scope='Synthetic CPU/CUDA boundary and resume tests plus read-only admitted Jamendo cache checks. No heldout targets or production models modified.',
        recovery='No automatic retry; preserve failure evidence and diagnose before any revised preflight version.')
    FOLDER.mkdir(parents=True)
    atomic_json(FOLDER/'launch.json', launch)
    record_event(dict(event='D_Jamendo_auxiliary_code_preflight_registered', job_id=JOB, launch=relative(FOLDER/'launch.json')))
    print(json.dumps(dict(launch=relative(FOLDER/'launch.json'))))


def run():
    launch = read_json(FOLDER/'launch.json')
    verify_hashes(launch['code_hashes']); verify_hashes(launch['data_hashes'])
    for name, value in launch['environment'].items():
        if __import__('os').environ.get(name) != value:
            raise ValueError('Preflight environment differs: '+name)
    if (FOLDER/'process.json').exists() or CHECKS.exists():
        raise ValueError('Preflight already started; preserve its evidence')
    identity = process_identity(); started = time.monotonic()
    atomic_json(FOLDER/'process.json', dict(started_at=now(), process_identity=identity))
    if not torch.cuda.is_available():
        raise RuntimeError('GPU resume check must execute before admission')
    with (FOLDER/'console.log').open('w', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            from . import tests
            suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
            result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    verify_hashes(launch['code_hashes']); verify_hashes(launch['data_hashes'])
    report = dict(job_id=JOB, status='passed' if result.wasSuccessful() and not result.skipped else 'failed',
        completed_at=now(), process_identity=identity, tests_run=result.testsRun, skipped=len(result.skipped),
        failures=[str(t) for t, _ in result.failures], errors=[str(t) for t, _ in result.errors],
        cuda_resume_verified=result.wasSuccessful() and not result.skipped,
        elapsed_seconds=time.monotonic()-started, code_hashes=launch['code_hashes'], launch_sha256=sha256(FOLDER/'launch.json'),
        covered=['CPU/CUDA exact optimizer/RNG/sampler resume', 'VA backbone equivalence and padding',
                 'whole-track observed-tag distribution and artist-group sampling', 'zero-weight control and paired exposure/RNG',
                 'fit-only normalization', 'heldout/duplicate and feature/clock rejection', 'real OS lock status'])
    atomic_json(CHECKS, report)
    record_event(dict(event='D_Jamendo_auxiliary_code_preflight_finished', status=report['status'], report=relative(CHECKS)))
    print(json.dumps({k:v for k,v in report.items() if k not in ('code_hashes','covered')}))
    if report['status'] != 'passed':
        raise RuntimeError('Auxiliary code preflight failed; no training admission')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); m = p.add_mutually_exclusive_group(required=True)
    m.add_argument('--register', action='store_true'); m.add_argument('--run', action='store_true'); args = p.parse_args()
    with run_lease(JOB):
        if args.register:
            register()
        else:
            with run_lease(GPU_LEASE):
                run()
