"""Bounded C training status from OS leases, process identity and checkpoint evidence."""
from __future__ import annotations

import argparse
import json

from tools.train_next.runtime import is_running
from tools.train_next_v2.common import GPU_LEASE, now, paths, process_identity, read_json
from tools.train_next_v2.status import recent_progress
from .execution import MATRIX, QUEUE_DIR, QUEUE_ID, verification_matches


def optional(path):
    return read_json(path) if path.is_file() else {}


def identity_matches(value):
    identity = value.get('process_identity')
    return bool(identity and process_identity(identity['pid']) == identity)


def snapshot(run_id):
    folder, models = paths(run_id)
    config = read_json(folder / 'config.json')
    if config['stage'] != 'C':
        raise ValueError('Status is restricted to registered C experiments')
    process = optional(folder / 'process.json')
    status = optional(folder / 'status.json')
    report = optional(folder / 'report.json')
    active, verifying = is_running(run_id), is_running('verify-' + run_id)
    verified = verification_matches(run_id) if not (active or verifying) else False
    outcome = ('running' if active and identity_matches(process) else
               'lease_held_identity_unconfirmed' if active else 'independent_reload_running' if verifying else
               'completed_verified' if verified else 'completed_pending_verification' if report else
               'stopped_without_completion' if process else 'registered_not_started')
    checkpoints = {name: dict(bytes=path.stat().st_size, modified=path.stat().st_mtime)
                   for name in ('best', 'last') if (path := models / (name + '.pth')).is_file()}
    return dict(run_id=run_id, outcome=outcome, os_lease_held=active,
                verification_lease_held=verifying, process_identity=process.get('process_identity'),
                process_identity_matches=identity_matches(process), status=status,
                recent_progress=recent_progress(folder / 'console.log'), checkpoints=checkpoints,
                completed_steps=report.get('completed_steps'), best_step=report.get('best_step'),
                best_development_score=report.get('best_development_score'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    matrix = read_json(MATRIX)
    ids = [args.run_id] if args.run_id else [row['run_id'] for row in matrix['runs']]
    runs = [snapshot(run) for run in ids]
    queue_process = optional(QUEUE_DIR / 'process.json')
    print(json.dumps(dict(checked_at=now(), stage='C', gpu_lease_active=is_running(GPU_LEASE),
        queue=dict(job_id=QUEUE_ID, os_lease_held=is_running(QUEUE_ID),
                   process_identity_matches=identity_matches(queue_process),
                   process_identity=queue_process.get('process_identity'),
                   status=optional(QUEUE_DIR / 'status.json')),
        completed_verified=sum(row['outcome']=='completed_verified' for row in runs),
        registered_runs=len(matrix['runs']), runs=runs), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
