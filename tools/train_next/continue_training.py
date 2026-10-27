"""Finite foreground queue: wait for the DEAM teacher, then train the audited mix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .core import ROOT, atomic_json, sha256, output_paths, write_status
from .runtime import run_lease, is_running


def main(args):
    output_paths(args.teacher)
    output_paths(args.run_id)
    if not args.prepared.resolve().is_relative_to(ROOT / 'data/datasets'):
        raise ValueError('Prepared data must remain inside Vm/data/datasets')
    teacher = ROOT / 'data/training/next_va/runs' / args.teacher
    queue = ROOT / 'data/training/next_va/queues' / args.run_id
    queue.mkdir(parents=True, exist_ok=args.resume)
    command = [sys.executable, '-u', '-m', 'tools.train_next.mixed', '--run-id', args.run_id,
               '--initialize', str(ROOT / 'models/next_va' / args.teacher / 'best.pth'),
               '--prepared', str(args.prepared), '--device', 'cuda']
    manifest_hash = sha256(args.prepared / 'manifest.jsonl')
    atomic_json(queue / 'command.json', dict(argv=command, cwd=str(ROOT), waiting_for=str(teacher),
                data_manifest_sha256=manifest_hash))
    try:
        start = time.monotonic()
        while True:
            state_path = teacher / 'status.json'
            if not state_path.exists():
                raise ValueError('Teacher run is missing; refusing to wait indefinitely')
            state = json.loads(state_path.read_text())
            running = is_running(args.teacher)
            report_path = teacher / 'report.json'
            if report_path.exists() and not running:
                report = json.loads(report_path.read_text())
                if report['status'] != 'completed' or not report['legacy_model_unchanged']:
                    raise ValueError('Teacher did not pass completion checks')
                if not (ROOT / 'models/next_va' / args.teacher / 'best.pth').is_file():
                    raise ValueError('Teacher completion report has no checkpoint')
                break
            if not running:
                raise RuntimeError(f'Teacher stopped: no active run lock; last recorded status: {state}')
            if time.monotonic() - start > 12 * 3600:
                raise TimeoutError('Teacher did not finish within the queue time limit')
            write_status(queue / 'status.json', dict(stage='waiting_for_teacher', teacher_stage=state['stage']))
            time.sleep(15)
        if sha256(args.prepared / 'manifest.jsonl') != manifest_hash:
            raise ValueError('Prepared manifest changed while waiting')
        if args.resume:
            mixed_run, mixed_models = output_paths(args.run_id)
            if (mixed_run / 'config.json').is_file():
                command.append('--resume')
            elif mixed_run.exists() or mixed_models.exists():
                raise ValueError('Incomplete mixed run directories without configuration; inspect before restarting')
        atomic_json(queue / 'command.json', dict(argv=command, cwd=str(ROOT), waiting_for=str(teacher),
                    data_manifest_sha256=manifest_hash))
        write_status(queue / 'status.json', dict(stage='mixed_training_running'))
        with (queue / 'mixed.console.log').open('a', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report_path = ROOT / 'data/training/next_va/runs' / args.run_id / 'report.json'
        result = json.loads(report_path.read_text())
        if result.get('status') != 'completed' or not result.get('legacy_model_unchanged'):
            raise RuntimeError('Mixed process returned without a verified completion report')
        write_status(queue / 'status.json', dict(stage='completed', report=str(report_path)))
    except BaseException as exc:
        write_status(queue / 'status.json', dict(stage='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', default='teacher-20261002')
    p.add_argument('--run-id', default='mixed-20261003')
    p.add_argument('--prepared', type=Path, default=ROOT / 'data/datasets/combined/prepared/v1')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    with run_lease('queue-' + args.run_id):
        main(args)
