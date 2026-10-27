"""Real OS lease/process status for stage C extraction jobs."""
import argparse
import json
from pathlib import Path

from tools.train_next.runtime import is_running
from tools.train_next_v2.common import BASE, now, process_identity, read_json


def snapshot(job_id):
    if '/' in job_id or '\\' in job_id or not job_id.startswith('c-'):
        raise ValueError('Invalid C job ID')
    directory = BASE / 'feature_jobs' / job_id
    launch = read_json(directory / 'launch.json')
    status = read_json(directory / 'status.json') if (directory / 'status.json').exists() else {}
    identity = status.get('process_identity')
    locked = is_running(job_id)
    matches = bool(identity and process_identity(identity['pid']) == identity)
    report = read_json(directory / 'report.json') if (directory / 'report.json').exists() else None
    outcome = ('running' if locked and matches else 'lease_held_identity_unconfirmed' if locked else
        'completed_pending_verification' if report and report['status']=='completed' else
        'stopped_after_failure' if report and report['status']=='failed' else
        'stopped_without_completion' if status else 'registered_not_started')
    actual = list(Path(launch['cache_dir']).glob('*/*.npz'))
    process = read_json(directory/'process.json') if (directory/'process.json').exists() else {}
    tail = []
    if process.get('log'):
        with Path(process['log']).open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-16000))
            tail = stream.read().decode('utf-8', errors='replace').splitlines()[-6:]
    return dict(checked_at=now(), job_id=job_id, outcome=outcome, os_lease_held=locked,
        process_identity_matches=matches, process_identity=identity, status=status,
        expected_shards=launch['expected_shards'], actual_npz_count=len(actual),
        actual_npz_bytes=sum(p.stat().st_size for p in actual), recent_log_lines=tail,
        report=report, recovery=launch['recovery'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', required=True)
    print(json.dumps(snapshot(parser.parse_args().job_id), ensure_ascii=False, indent=2))
