"""One bounded probe of the exact mirror URL that previously lost its TLS connection."""
import json
from pathlib import Path
import sys

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, run_lease
from tools.train_next_v2.common import BASE, ROOT, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.jamendo_audio.ranges import ArchiveRanges, Budget, parse_header

JOB = 'd-jamendo-recovery-probe-v1'
DIRECTORY = BASE/'resources/20261004/jobs'/JOB
DOWNLOAD = BASE/'resources/20261004/jobs/d-jamendo-selected-audio-v1'
DEST = ROOT/'data/datasets/mtg_jamendo/raw/recovery-probe-v1'


def main():
    with run_lease(JOB):
        launch = read_json(DIRECTORY/'launch.json')
        verify_hashes(launch['protected_hashes'])
        verify_hashes(read_json(DOWNLOAD/'launch.json')['protected_hashes'])
        if is_running('d-jamendo-selected-audio-v1'):
            raise RuntimeError('Acquisition is active; no recovery probe required')
        if DEST.exists() or (DIRECTORY/'report.json').exists():
            raise FileExistsError('Preserve one-shot probe evidence')
        if Path(sys.executable).resolve() != Path(launch['python']):
            raise ValueError('Wrong Python')
        DEST.mkdir(parents=True)
        atomic_json(DIRECTORY/'process.json', dict(started_at=now(), process_identity=process_identity()))
        budget = Budget(DIRECTORY/'network-budget.json', 512, 1)
        reader = ArchiveRanges(launch['url'], DEST, budget)
        try:
            data = reader.header(0)
            header = parse_header(data, '12', 0, reader.source['bytes'])
            report = dict(status='completed_exact_source_probe', completed_at=now(),
                request_count=1, bytes=512, source=reader.source, header=header,
                tls_verification_disabled=False, previous_failure=read_json(DOWNLOAD/'failure.json'),
                recovery_ready=True, interpretation='Exact same official URL now serves a verified512-byte range. Original failure was a TLS transport interruption before any archive header was saved; code, inputs and existing hashes remain unchanged. One original-command resume is justified; no automatic retry loop.')
        except Exception as exc:
            report = dict(status='failed', completed_at=now(), error=f'{type(exc).__name__}: {exc}', recovery_ready=False)
        finally:
            reader.close()
        atomic_json(DIRECTORY/'report.json', report)
        record_event(dict(event='D_Jamendo_recovery_probe_finished', recovery_ready=report['recovery_ready'],
                          report=str((DIRECTORY/'report.json').relative_to(ROOT))))
        print(json.dumps(report))


if __name__ == '__main__':
    main()
