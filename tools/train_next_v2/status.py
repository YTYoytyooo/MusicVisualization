"""Bounded new-version status, backed by OS leases and PID creation identity."""
from __future__ import annotations

import argparse
import json

from tools.train_next.runtime import is_running
from tools.train_next.core import sha256
from .common import PLAN, GPU_LEASE, paths, read_json, now, process_identity


def recent_progress(path):
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        lines = stream.read(65536).decode('utf-8', errors='replace').splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict) and any(key in row for key in ('step', 'done', 'stage')):
            return row
    return None


def snapshot(run_id):
    run_dir, model_dir = paths(run_id)
    running = is_running(run_id)
    process = read_json(run_dir / 'process.json') if (run_dir / 'process.json').exists() else None
    identity = process.get('process_identity') if process else None
    current_identity = process_identity(identity['pid']) if identity else None
    report = read_json(run_dir / 'report.json') if (run_dir / 'report.json').exists() else None
    complete = bool(report and report['status'] == 'completed' and report.get('checkpoint_reload_verified')
                    and report.get('legacy_model_unchanged') and report.get('baseline_model_unchanged')
                    and (model_dir / 'best.pth').is_file()
                    and sha256(model_dir / 'best.pth') == report['checkpoint_sha256'])
    checkpoints = {name: dict(bytes=path.stat().st_size, modified=path.stat().st_mtime)
                   for name in ('best', 'last') if (path := model_dir / (name + '.pth')).is_file()}
    return dict(run_id=run_id, running=running, process_identity=identity,
        process_identity_matches=identity is not None and identity == current_identity,
        outcome='running' if running else 'completed_verified' if complete else
                'stopped_without_completion' if process else 'registered' if run_dir.exists() else 'not_started',
        status=read_json(run_dir / 'status.json') if (run_dir / 'status.json').exists() else None,
        recent_progress=recent_progress(run_dir / 'console.log'), checkpoints=checkpoints,
        best_development_score=report.get('best_development_score') if report else None,
        completed_steps=report.get('completed_steps') if report else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    plan = read_json(PLAN)
    runs = [args.run_id] if args.run_id else [row['run_id'] for row in plan['experiments']]
    print(json.dumps(dict(checked_at=now(), current_stage=plan['current_stage'],
        gpu_lease_active=is_running(GPU_LEASE), runs=[snapshot(run) for run in runs]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
