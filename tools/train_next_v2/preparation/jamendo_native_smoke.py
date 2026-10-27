"""Exercise strict native decoding on three pinned, already downloaded MP3 tracks."""
import json
from pathlib import Path

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import BASE, ROOT, now, read_json, verify_hashes, record_event, process_identity
from tools.train_next_v2.jamendo_audit.native import inspect
from tools.train_next_v2.auxiliary_data.merge_audio_audit import fingerprint

JOB = 'd-jamendo-native-smoke-v1'
DIRECTORY = BASE/'analyses'/JOB


def main():
    with run_lease(JOB):
        launch = read_json(DIRECTORY/'launch.json')
        verify_hashes(launch['protected_hashes'])
        if (DIRECTORY/'report.json').exists():
            raise FileExistsError('Preserve completed MP3 preflight')
        atomic_json(DIRECTORY/'process.json', dict(started_at=now(), process_identity=process_identity()))
        checks = []
        for row in launch['tracks']:
            try:
                native = inspect(ROOT/row['audio_path'], row['metadata_duration_seconds'], DIRECTORY/'stderr'/row['track_id'])
                fp, _ = fingerprint(ROOT/row['audio_path'], row['audio_sha256'], ROOT/launch['fingerprint_cache'])
                check = dict(track_id=row['track_id'], passed=True, native_audio=native, fingerprint=fp)
            except Exception as exc:
                check = dict(track_id=row['track_id'], passed=False, error=f'{type(exc).__name__}: {exc}')
            checks.append(check)
        verify_hashes(launch['protected_hashes'])
        report = dict(status='passed' if all(r['passed'] for r in checks) else 'failed_audio_gates',
            completed_at=now(), checks=checks, count=len(checks), passed=sum(r['passed'] for r in checks),
            gpu_used=False, training_admitted=False, interpretation='Execution preflight on three already downloaded training candidates; not a full-pool audit or admission.')
        atomic_json(DIRECTORY/'report.json', report)
        record_event(dict(event='D_Jamendo_native_MP3_preflight_completed', passed=report['passed'], total=len(checks),
            report=str((DIRECTORY/'report.json').relative_to(ROOT))))
        print(json.dumps(report))


if __name__ == '__main__':
    main()
