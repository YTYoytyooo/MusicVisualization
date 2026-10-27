"""Probe exact-range access to the two official mirrors without fetching a tar."""
import argparse
import json
from pathlib import Path
import re
import sys
import tarfile
from urllib.request import Request, urlopen

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, lock_path
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes, process_identity

JOB = 'd-jamendo-audio-probe-v1'
DIRECTORY = BASE / 'resources/20261004/jobs' / JOB
DEST = ROOT / 'data/datasets/mtg_jamendo/raw/audio-probe-v1'
SELECTED = ROOT / 'data/datasets/mtg_jamendo/prepared/candidates-v1/selected.json'
SOURCE = ROOT / 'data/datasets/mtg_jamendo/raw/source-details-v1/jamendo'
MIRRORS = (
    'https://cdn.freesound.org/mtg-jamendo',
    'https://essentia.upf.edu/documentation/datasets/mtg-jamendo',
)


def register():
    if DIRECTORY.exists():
        raise FileExistsError('Preserve registered probe')
    python = str(Path(read_json(PLAN)['python']).resolve())
    environment = read_json(BASE/'analyses/d-jamendo-candidates-v1/launch.json')['environment']
    files = [Path(__file__), SELECTED, SOURCE/'scripts/download/download.py',
             SOURCE/'data/download/autotagging_moodtheme_audio-low_sha256_tars.txt',
             ROOT/'tools/train_next/core.py', ROOT/'tools/train_next/runtime.py',
             ROOT/'tools/train_next_v2/common.py']
    launch = dict(job_id=JOB, registered_at=now(), python=python, cwd=str(ROOT),
        original_argv=[python, '-u', '-m', 'tools.train_next_v2.jamendo_audio.probe', '--run'],
        environment=environment, os_lock=str(lock_path(JOB)),
        protected_hashes={p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
        urls=[f'{m}/autotagging_moodtheme/audio-low/autotagging_moodtheme_audio-low-45.tar' for m in MIRRORS],
        range_start=0, range_end=511, max_response_bytes=512, max_requests=2,
        log=str(DIRECTORY/'events.jsonl'), output=str(DEST.relative_to(ROOT)),
        recovery='No automatic retry. Preserve mirror evidence and version a new probe if sources change.',
        scope='Read at most 512 bytes per official mirror. Reject HTTP200 before reading body. No full archive or audio acquisition; no training admission.')
    DIRECTORY.mkdir(parents=True)
    atomic_json(DIRECTORY/'launch.json', launch)
    record_event(dict(event='D_Jamendo_audio_probe_registered', launch=str((DIRECTORY/'launch.json').relative_to(ROOT))))
    print(json.dumps(dict(launch=str(DIRECTORY/'launch.json'))))


def run():
    with run_lease(JOB):
        launch = read_json(DIRECTORY/'launch.json')
        verify_hashes(launch['protected_hashes'])
        if Path(sys.executable).resolve() != Path(launch['python']):
            raise ValueError('Wrong Python')
        if DEST.exists() or (DIRECTORY/'report.json').exists():
            raise FileExistsError('Preserve previous probe evidence')
        DEST.mkdir(parents=True)
        atomic_json(DIRECTORY/'process.json', dict(started_at=now(), process_identity=process_identity()))
        rows = []
        for i, url in enumerate(launch['urls']):
            row = dict(url=url, requested_range='bytes=0-511', observed_at=now(), viable=False)
            try:
                request = Request(url, headers={'Range': 'bytes=0-511', 'Accept-Encoding': 'identity', 'User-Agent': 'MusicVisualization-research/1.0'})
                with urlopen(request, timeout=15) as response:
                    row.update(http_status=response.status, final_url=response.url, headers=dict(response.headers.items()))
                    match = re.fullmatch(r'bytes 0-511/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status != 206 or not match or response.headers.get('Content-Encoding', 'identity') != 'identity':
                        raise ValueError('Exact unencoded Range not honored; no body read')
                    total = int(match.group(1))
                    if not 1024 <= total <= 4*1024**3:
                        raise ValueError('Unexpected tar size')
                    data = response.read(513)
                    if len(data) != 512:
                        raise ValueError('Range body length mismatch')
                    path = DEST/f'mirror-{i}-header.bin'
                    with path.open('xb') as stream:
                        stream.write(data)
                    header = tarfile.TarInfo.frombuf(data, 'utf-8', 'strict')
                    row.update(viable=True, tar_bytes=total, etag=response.headers.get('ETag'),
                        last_modified=response.headers.get('Last-Modified'), transferred_bytes=len(data),
                        header_file=str(path.relative_to(ROOT)), header_sha256=sha256(path),
                        first_header=dict(name=header.name, size=header.size, type=header.type.decode('ascii')))
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
            with (DIRECTORY/'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
        verify_hashes(launch['protected_hashes'])
        report = dict(status='completed_probe_only', job_id=JOB, completed_at=now(), mirrors=rows,
            viable_mirrors=sum(r['viable'] for r in rows), audio_downloaded=False, training_admitted=False)
        atomic_json(DIRECTORY/'report.json', report)
        with run_lease('next-va-v2-d-ledger'):
            stages = read_json(PLAN)['stages']
            stages['D']['jamendo_audio_probe'] = str((DIRECTORY/'report.json').relative_to(ROOT))
            record_event(dict(event='D_Jamendo_audio_probe_completed', viable_mirrors=report['viable_mirrors']), updates=dict(stages=stages))
        print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.register == args.run:
        parser.error('Choose exactly one action')
    register() if args.register else run()
