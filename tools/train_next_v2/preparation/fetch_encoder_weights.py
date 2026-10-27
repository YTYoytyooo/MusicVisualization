"""Retrieve two pinned public encoder weights, with bounded resumable reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import traceback
import urllib.request

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import BASE, ROOT, now, process_identity, read_json, verify_hashes


def checked_destination(root, entry):
    expected = f"https://huggingface.co/{entry['repository']}/resolve/{entry['revision']}/pytorch_model.bin"
    if entry['repository'] not in ('laion/larger_clap_music', 'm-a-p/MERT-v1-95M') or entry['url'] != expected:
        raise ValueError('Only the pinned public encoder URLs are supported')
    if not re.fullmatch(r'[a-f0-9]{40}', entry['revision']) or not re.fullmatch(r'[a-f0-9]{64}', entry['sha256']):
        raise ValueError('Pinned revision/hash missing')
    if not 1 <= entry['bytes'] <= 800_000_000:
        raise ValueError('Unexpected weight size')
    path = (root / entry['destination']).resolve()
    if not path.is_relative_to(root.resolve()) or path.name != 'pytorch_model.bin':
        raise ValueError('Destination escapes encoder weight directory')
    return path


def check_response(status, headers, offset, expected):
    if offset:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', headers.get('Content-Range', ''))
        if status != 206 or not match or tuple(map(int, match.groups())) != (offset, expected - 1, expected):
            raise ValueError('Server did not honor the exact resume range; partial file preserved')
    elif status != 200:
        raise ValueError('Unexpected initial download status')
    length = headers.get('Content-Length')
    if length is not None and int(length) != expected - offset:
        raise ValueError('Response length differs from the pinned weight')
    if 'text/html' in headers.get('Content-Type', ''):
        raise ValueError('Received an HTML page instead of weights')


def download(root, entry, publish, opener=urllib.request.urlopen):
    path = checked_destination(root, entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    if path.exists():
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('Existing weight differs from pinned metadata; preserved')
        return dict(name=entry['name'], status='existing_verified', path=str(path), bytes=entry['bytes'], sha256=entry['sha256'])
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry['bytes']:
        raise ValueError('Existing partial is larger than the pinned weight; preserved')
    if offset < entry['bytes']:
        headers = {'User-Agent': 'MusicVisualization-C-pilot/1.0', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        started, last_update = time.monotonic(), 0.
        request = urllib.request.Request(entry['url'], headers=headers)
        with opener(request, timeout=30) as response:
            if not response.geturl().startswith('https://'):
                raise ValueError('HTTPS redirect required')
            check_response(response.status, response.headers, offset, entry['bytes'])
            with partial.open('ab' if partial.exists() else 'xb') as stream:
                received = offset
                while True:
                    if time.monotonic() - started > 1200:
                        raise TimeoutError('Per-file 20-minute transfer budget exhausted; partial preserved')
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    if received + len(block) > entry['bytes']:
                        raise ValueError('Response exceeds pinned size; partial preserved')
                    stream.write(block)
                    received += len(block)
                    if time.monotonic() - last_update >= 10:
                        stream.flush()
                        publish(dict(stage='downloading', name=entry['name'], bytes=received, expected_bytes=entry['bytes']))
                        last_update = time.monotonic()
    if partial.stat().st_size != entry['bytes']:
        raise ValueError('Incomplete transfer; partial preserved for explicit recovery')
    publish(dict(stage='verifying_sha256', name=entry['name'], bytes=entry['bytes'], expected_bytes=entry['bytes']))
    digest = sha256(partial)
    if digest != entry['sha256']:
        raise ValueError('SHA256 mismatch; partial preserved and not admitted')
    partial.replace(path)
    return dict(name=entry['name'], status='downloaded_verified', path=str(path), bytes=entry['bytes'], sha256=digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', required=True, type=Path)
    args = parser.parse_args()
    launch_path = args.launch.resolve()
    if not launch_path.is_relative_to(BASE / 'resources'):
        raise ValueError('Launch record must be in the optimization resources directory')
    launch = read_json(launch_path)
    verify_hashes(launch['code_hashes'])
    verify_hashes(launch['data_hashes'])
    spec = read_json(ROOT / launch['download_plan'])
    root = (ROOT / spec['output_root']).resolve()
    if not root.is_relative_to(BASE / 'resources'):
        raise ValueError('All weights must stay inside Vm optimization resources')
    root.mkdir(parents=True, exist_ok=True)
    job_dir = launch_path.parent
    with run_lease(launch['job_id']):
        identity = process_identity()
        atomic_json(job_dir / 'process.json', dict(process_identity=identity, original_argv=launch['original_argv'], started_at=now()))
        with (job_dir / 'console.log').open('a', encoding='utf-8') as log:
            def publish(value):
                value = dict(**value, process_identity=identity, updated_at=now())
                atomic_json(job_dir / 'status.json', value)
                line = json.dumps(value)
                log.write(line + '\n'); log.flush()
                print(line, flush=True)

            publish(dict(stage='starting', files=len(spec['files'])))
            results = []
            for entry in spec['files']:
                try:
                    result = download(root, entry, publish)
                    results.append(result)
                    publish(dict(stage='file_verified', **result))
                except Exception as exc:
                    result = dict(name=entry['name'], status='failed', error=f'{type(exc).__name__}: {exc}')
                    results.append(result)
                    publish(dict(stage='file_failed', **result))
                    log.write(traceback.format_exc() + '\n'); log.flush()
            successful = all(r['status'] in ('existing_verified', 'downloaded_verified') for r in results)
            report = dict(status='completed_verified' if successful else 'failed', completed_at=now(), process_identity=identity,
                          download_plan_sha256=sha256(ROOT / launch['download_plan']), code_hashes=launch['code_hashes'],
                          files=results, weights_loaded=False, remote_code_executed=False, gpu_used=False)
            atomic_json(job_dir / 'report.json', report)
            publish(dict(stage=report['status'], report=str(job_dir / 'report.json'), verified_files=sum(r['status'] != 'failed' for r in results)))
    raise SystemExit(0 if successful else 1)


if __name__ == '__main__':
    main()
