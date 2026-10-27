"""Fetch only pinned public metadata; never download audio or execute source code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.request

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, process_identity, verify_hashes

JOB = 'd-jamendo-metadata-v1'
DIRECTORY = BASE / 'resources/20261004/jobs' / JOB
DESTINATION = ROOT / 'data/datasets/mtg_jamendo/raw/metadata-v1'
REPOSITORY = 'MTG/mtg-jamendo-dataset'
API = 'https://api.github.com/repos/' + REPOSITORY
MAX_FILE = 20 * 1024 * 1024
MAX_TOTAL = 64 * 1024 * 1024
MODULE = 'tools.train_next_v2.jamendo_data.metadata'


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def update_ledger(status, **fields):
    with run_lease('next-va-v2-d-ledger'):
        plan = read_json(PLAN)
        stages = plan['stages']
        entry = dict(stages['D'].get('jamendo_metadata', {}))
        entry.update(job_id=JOB, status=status, launch=relative(DIRECTORY / 'launch.json'), **fields)
        stages['D']['jamendo_metadata'] = entry
        record_event(dict(event='D_Jamendo_metadata_' + status, **fields), updates=dict(stages=stages))


def fetch_bytes(url, limit):
    request = urllib.request.Request(url, headers={
        'User-Agent': 'MusicVisualization-research-metadata/1.0',
        'Accept': 'application/vnd.github+json' if 'api.github.com' in url else '*/*',
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.headers.get('Content-Length') and int(response.headers['Content-Length']) > limit:
            raise ValueError('Metadata response exceeds its byte limit')
        result = response.read(limit + 1)
    if len(result) > limit:
        raise ValueError('Metadata response exceeds its byte limit')
    return result


def write_once(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Preserved metadata differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('wb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def blob_sha(content):
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()


def select_files(tree):
    if tree.get('truncated') is not False:
        raise ValueError('A complete Git tree is required')
    required = {'README.md', 'LICENSE', 'audio_licenses.txt', 'data/raw.meta.tsv',
                'data/autotagging_moodtheme.tsv', 'data/tag_map.json'}
    selected = []
    for entry in tree['tree']:
        path = entry['path']
        wanted = path in required or (
            path.startswith('data/splits/split-0/') and 'moodtheme' in path and path.endswith('.tsv'))
        if not wanted:
            continue
        parts = PurePosixPath(path)
        if parts.is_absolute() or '..' in parts.parts or '\\' in path or ':' in path:
            raise ValueError('Unsafe metadata path')
        if entry['type'] != 'blob' or entry['mode'] != '100644':
            raise ValueError('Only regular metadata blobs are allowed: ' + path)
        if not re.fullmatch(r'[0-9a-f]{40}', entry['sha']) or not 0 < entry['size'] <= MAX_FILE:
            raise ValueError('Invalid Git blob or byte bound: ' + path)
        selected.append(dict(path=path, size=entry['size'], git_blob_sha1=entry['sha']))
    names = {r['path'] for r in selected}
    if not required <= names or len(names) != len(selected):
        raise ValueError('Missing or duplicate required metadata')
    if sum(r['size'] for r in selected) > MAX_TOTAL:
        raise ValueError('Metadata selection exceeds the 64 MiB total budget')
    return sorted(selected, key=lambda r: r['path'])


def validate_blob(data, spec):
    if len(data) != spec['size'] or blob_sha(data) != spec['git_blob_sha1']:
        raise ValueError('Git blob content mismatch: ' + spec['path'])


def register():
    if (DIRECTORY / 'launch.json').exists() or DESTINATION.exists():
        raise FileExistsError('Existing metadata job must be preserved')
    python = str(Path(read_json(PLAN)['python']).resolve())
    code = {relative(p): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    for path in ('tools/train_next_v2/common.py', 'tools/train_next/core.py', 'tools/train_next/runtime.py'):
        code[path] = sha256(ROOT / path)
    env = dict(PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TEMP=str(BASE / 'tmp'), TMP=str(BASE / 'tmp'), HF_HUB_OFFLINE='1',
               HF_HOME=str(BASE / 'cache/huggingface'), TORCH_HOME=str(BASE / 'cache/torch'),
               NUMBA_CACHE_DIR=str(BASE / 'cache/numba'), CUDA_CACHE_PATH=str(BASE / 'cache/cuda'))
    launch = dict(job_id=JOB, stage='D', created_at=now(), python=python, cwd=str(ROOT),
        original_argv=[python, '-u', '-m', MODULE, '--run'],
        recovery_argv=[python, '-u', '-m', MODULE, '--run', '--resume'], environment=env,
        code_sha256=code, os_lock=str(lock_path(JOB)), log=str(DIRECTORY / 'events.jsonl'),
        report=str(DIRECTORY / 'report.json'), destination=str(DESTINATION),
        sources=[API + '/commits/master', API + '/git/trees/<pinned-commit>?recursive=1',
                 'https://raw.githubusercontent.com/' + REPOSITORY + '/<pinned-commit>/<allowlisted-path>'],
        maximum_file_bytes=MAX_FILE, maximum_total_file_bytes=MAX_TOTAL,
        recovery='Inspect the failure first. Resume only when the OS lease is free; preserve and revalidate pinned API receipts and Git blob checksums. No automatic retries.',
        gpu_used=False, audio_downloads=False, remote_code_execution=False,
        training_admitted=False, labels='Uploader whole-track mood/theme tags; no continuous VA labels.')
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    atomic_json(DIRECTORY / 'launch.json', launch)
    update_ledger('registered')
    print(json.dumps(dict(job_id=JOB, status='registered', launch=relative(DIRECTORY / 'launch.json'))))


def api_receipt(name, url):
    path = DESTINATION / (name + '.json')
    receipt_path = DESTINATION / (name + '.receipt.json')
    if path.exists() or receipt_path.exists():
        if not path.exists() or not receipt_path.exists():
            raise ValueError('Incomplete API receipt; inspect before recovery: ' + name)
        receipt = read_json(receipt_path)
        if receipt['url'] != url or receipt['sha256'] != sha256(path):
            raise ValueError('Pinned API receipt changed: ' + name)
        return read_json(path)
    data = fetch_bytes(url, 8 * 1024 * 1024)
    result = json.loads(data)
    write_once(path, data)
    atomic_json(receipt_path, dict(url=url, retrieved_at=now(), bytes=len(data), sha256=sha256(path)))
    return result


def run(resume):
    launch = read_json(DIRECTORY / 'launch.json')
    verify_hashes(launch['code_sha256'])
    if Path(sys.executable).resolve() != Path(launch['python']) or Path.cwd().resolve() != ROOT:
        raise ValueError('Metadata runtime/cwd differs from registration')
    for name, value in launch['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('Metadata environment changed: ' + name)
    if (DIRECTORY / 'report.json').exists():
        raise FileExistsError('Completed metadata job is immutable')
    if (DIRECTORY / 'process.json').exists() and not resume:
        raise ValueError('Existing attempt requires --resume after failure inspection')
    identity = process_identity()
    atomic_json(DIRECTORY / 'process.json', dict(started_at=now(), process_identity=identity, resumed=resume))
    update_ledger('running', process_identity=identity)
    try:
        commit = api_receipt('commit', API + '/commits/master')
        revision = commit['sha']
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('GitHub did not return a full commit SHA')
        tree = api_receipt('tree', API + '/git/trees/' + revision + '?recursive=1')
        if tree['sha'] != commit['commit']['tree']['sha']:
            raise ValueError('Pinned Git tree does not match the commit')
        selected = select_files(tree)
        atomic_json(DIRECTORY / 'selection.json', dict(revision=revision, files=selected))
        files = []
        for spec in selected:
            path = DESTINATION / 'source' / spec['path']
            url = 'https://raw.githubusercontent.com/' + REPOSITORY + '/' + revision + '/' + spec['path']
            data = path.read_bytes() if path.exists() else fetch_bytes(url, spec['size'])
            validate_blob(data, spec)
            write_once(path, data)
            row = dict(**spec, file=relative(path), url=url, sha256=sha256(path))
            files.append(row)
            status = dict(job_id=JOB, stage='metadata_download', revision=revision, completed_files=len(files),
                          total_files=len(selected), bytes_verified=sum(r['size'] for r in files),
                          process_identity=identity, updated_at=now())
            atomic_json(DIRECTORY / 'status.json', status)
            with (DIRECTORY / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(time=now(), **row)) + '\n')
            print(json.dumps(dict(file=spec['path'], completed=len(files), total=len(selected))), flush=True)
        report = dict(status='completed_verified', job_id=JOB, completed_at=now(), repository=REPOSITORY,
            revision=revision, files=files, bytes_verified=sum(r['size'] for r in files),
            code_sha256=launch['code_sha256'], gpu_used=False, audio_downloaded=False,
            remote_code_executed=False, training_admitted=False,
            pending=['License and schema audit', 'Selected subset identity exclusions and audio screening',
                     'Frozen classification-only auxiliary protocol; never expand static tags into continuous VA'])
        atomic_json(DIRECTORY / 'report.json', report)
        update_ledger('completed_verified', report=relative(DIRECTORY / 'report.json'), revision=revision,
                      bytes_verified=report['bytes_verified'])
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        atomic_json(DIRECTORY / 'failure.json', dict(failed_at=now(), error=error))
        update_ledger('failed', error=error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with run_lease(JOB):
        register() if args.register else run(args.resume)


if __name__ == '__main__':
    main()
