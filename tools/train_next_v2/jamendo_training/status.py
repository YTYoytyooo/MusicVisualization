"""Bounded D queue/worker status backed by OS leases and checkpoint evidence."""
import argparse
import json

from tools.train_next.runtime import is_running
from tools.train_next_v2.common import GPU_LEASE, now, paths, process_identity, read_json
from tools.train_next_v2.status import recent_progress
from .execution import MATRIX, QUEUE_ID, QUEUE_DIR, verification_matches


def optional(path):
    return read_json(path) if path.is_file() else {}


def identity_matches(process):
    identity = process.get('process_identity')
    return bool(identity and process_identity(identity['pid']) == identity)


def snapshot(rid):
    folder, models = paths(rid)
    if read_json(folder/'config.json')['stage'] != 'D':
        raise ValueError('D status requires a registered D experiment')
    process, status, report = [optional(folder/name) for name in ('process.json', 'status.json', 'report.json')]
    active, verifying = is_running(rid), is_running('verify-'+rid)
    verification_process = optional(folder/'verification-process.json')
    verified = verification_matches(rid) if not (active or verifying) else False
    outcome = ('running' if active and identity_matches(process) else
        'lease_held_identity_unconfirmed' if active else
        'independent_reload_running' if verifying and identity_matches(verification_process) else
        'verification_lease_identity_unconfirmed' if verifying else 'completed_verified' if verified else
        'completed_pending_verification' if report else 'stopped_without_completion' if process else 'registered_not_started')
    checkpoints = {name: dict(bytes=p.stat().st_size, modified=p.stat().st_mtime)
        for name in ('best', 'last') if (p := models/(name+'.pth')).is_file()}
    return dict(run_id=rid, outcome=outcome, os_lease_held=active, verification_lease_held=verifying,
        process_identity=process.get('process_identity'), process_identity_matches=identity_matches(process),
        status=status, recent_progress=recent_progress(folder/'console.log'), checkpoints=checkpoints,
        completed_steps=report.get('completed_steps'), best_step=report.get('best_step'),
        best_development_score=report.get('best_development_score'))


def all_status(run_id=None):
    matrix = read_json(MATRIX)
    ids = [run_id] if run_id else [r['run_id'] for r in matrix['runs']]
    runs = [snapshot(rid) for rid in ids]
    process = optional(QUEUE_DIR/'process.json')
    return dict(checked_at=now(), stage='D', gpu_lease_active=is_running(GPU_LEASE),
        queue=dict(job_id=QUEUE_ID, os_lease_held=is_running(QUEUE_ID), process_identity=process.get('process_identity'),
            process_identity_matches=identity_matches(process), status=optional(QUEUE_DIR/'status.json')),
        completed_verified=sum(r['outcome']=='completed_verified' for r in runs), registered_runs=len(matrix['runs']), runs=runs)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-id'); args = p.parse_args()
    print(json.dumps(all_status(args.run_id), ensure_ascii=False, indent=2))
