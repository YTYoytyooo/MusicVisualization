"""Use the commit's explicit tree object ID, preserving failed v1 evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

from tools.train_next.runtime import is_running
from .metadata import (BASE, ROOT, PLAN, API, REPOSITORY, MAX_TOTAL, MAX_FILE,
    atomic_json, read_json, sha256, verify_hashes, lock_path, run_lease, now, record_event,
    process_identity, relative, fetch_bytes, write_once, select_files, validate_blob)

JOB = 'd-jamendo-metadata-v2'
DIRECTORY = BASE / 'resources/20261004/jobs' / JOB
DESTINATION = ROOT / 'data/datasets/mtg_jamendo/raw/metadata-v2'
PREVIOUS = BASE / 'resources/20261004/jobs/d-jamendo-metadata-v1'
COMMIT = ROOT / 'data/datasets/mtg_jamendo/raw/metadata-v1/commit.json'


def tree_request(commit):
    revision, tree_id = commit['sha'], commit['commit']['tree']['sha']
    if any(not re.fullmatch(r'[0-9a-f]{40}', item) for item in (revision, tree_id)):
        raise ValueError('A full commit and tree object ID are required')
    return revision, tree_id, API + '/git/trees/' + tree_id + '?recursive=1'


def update(status, **fields):
    with run_lease('next-va-v2-d-ledger'):
        stages = read_json(PLAN)['stages']
        stages['D']['jamendo_metadata'] = dict(job_id=JOB, status=status,
            launch=relative(DIRECTORY / 'launch.json'), previous_failed_job=relative(PREVIOUS), **fields)
        record_event(dict(event='D_Jamendo_metadata_v2_' + status, **fields), updates=dict(stages=stages))


def register():
    if is_running('d-jamendo-metadata-v1') or (DIRECTORY / 'launch.json').exists() or DESTINATION.exists():
        raise ValueError('Previous job active or v2 already registered')
    prior = read_json(PREVIOUS / 'launch.json')
    receipt = read_json(COMMIT.with_name('commit.receipt.json'))
    if receipt['sha256'] != sha256(COMMIT):
        raise ValueError('Preserved commit receipt changed')
    revision, tree_id, url = tree_request(read_json(COMMIT))
    code = {relative(p): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    protected = dict(code)
    for path in (COMMIT, COMMIT.with_name('commit.receipt.json'), PREVIOUS / 'launch.json', PREVIOUS / 'failure.json'):
        protected[relative(path)] = sha256(path)
    for path in ('tools/train_next_v2/common.py', 'tools/train_next/core.py', 'tools/train_next/runtime.py'):
        protected[path] = sha256(ROOT / path)
    argv = [prior['python'], '-u', '-m', 'tools.train_next_v2.jamendo_data.metadata_v2', '--run']
    launch = dict(job_id=JOB, stage='D', registered_at=now(), python=prior['python'], cwd=str(ROOT),
        original_argv=argv, recovery_argv=argv + ['--resume'], environment=prior['environment'],
        protected_hashes=protected, revision=revision, tree_id=tree_id, tree_url=url,
        destination=str(DESTINATION), os_lock=str(lock_path(JOB)), log=str(DIRECTORY / 'events.jsonl'),
        report=str(DIRECTORY / 'report.json'), maximum_file_bytes=MAX_FILE, maximum_total_file_bytes=MAX_TOTAL,
        recovery='Inspect error and released OS lock; preserve config and use --resume, revalidating every cached blob.',
        correction='The GitHub recursive-tree endpoint echoed the supplied commit ID in its sha field. v2 requests the explicit tree object SHA pinned in the already-received commit; retains strict SHA equality and all blob checks. v1 files/config/failure are preserved.',
        audio_downloaded=False, gpu_used=False, remote_code_executed=False, training_admitted=False)
    DIRECTORY.mkdir(parents=True, exist_ok=False)
    atomic_json(DIRECTORY / 'launch.json', launch)
    update('registered', revision=revision)
    print(json.dumps(dict(status='registered', job_id=JOB)))


def run(resume):
    launch = read_json(DIRECTORY / 'launch.json')
    verify_hashes(launch['protected_hashes'])
    if Path(sys.executable).resolve() != Path(launch['python']) or Path.cwd().resolve() != ROOT:
        raise ValueError('Runtime/cwd changed')
    for name, value in launch['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('Environment changed: ' + name)
    if (DIRECTORY / 'report.json').exists() or ((DIRECTORY / 'process.json').exists() and not resume):
        raise ValueError('Completed evidence is immutable; an interrupted attempt requires --resume')
    identity = process_identity()
    atomic_json(DIRECTORY / 'process.json', dict(started_at=now(), process_identity=identity, resumed=resume))
    update('running', process_identity=identity, revision=launch['revision'])
    try:
        tree_path = DESTINATION / 'tree.json'
        receipt_path = DESTINATION / 'tree.receipt.json'
        if tree_path.exists() or receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt['sha256'] != sha256(tree_path) or receipt['url'] != launch['tree_url']:
                raise ValueError('Pinned tree receipt changed')
            tree = read_json(tree_path)
        else:
            payload = fetch_bytes(launch['tree_url'], 8 * 1024 * 1024)
            tree = json.loads(payload)
            if tree['sha'] != launch['tree_id']:
                raise ValueError('Explicit tree object ID mismatch')
            write_once(tree_path, payload)
            atomic_json(receipt_path, dict(url=launch['tree_url'], sha256=sha256(tree_path), retrieved_at=now()))
        if tree['sha'] != launch['tree_id']:
            raise ValueError('Preserved tree identity mismatch')
        selected = select_files(tree)
        expected_splits = {'data/splits/split-0/autotagging_moodtheme-' + s + '.tsv'
                           for s in ('train', 'validation', 'test')}
        if not expected_splits <= {row['path'] for row in selected}:
            raise ValueError('Official split metadata is missing')
        selection = dict(revision=launch['revision'], tree_id=launch['tree_id'], files=selected)
        write_once(DIRECTORY / 'selection.json', (json.dumps(selection, indent=2) + '\n').encode())
        files = []
        for spec in selected:
            path = DESTINATION / 'source' / spec['path']
            url = 'https://raw.githubusercontent.com/' + REPOSITORY + '/' + launch['revision'] + '/' + spec['path']
            payload = path.read_bytes() if path.exists() else fetch_bytes(url, spec['size'])
            validate_blob(payload, spec)
            write_once(path, payload)
            row = dict(**spec, file=relative(path), url=url, sha256=sha256(path))
            files.append(row)
            status = dict(job_id=JOB, stage='metadata_download', completed_files=len(files), total_files=len(selected),
                verified_bytes=sum(item['size'] for item in files), process_identity=identity, updated_at=now())
            atomic_json(DIRECTORY / 'status.json', status)
            with (DIRECTORY / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(time=now(), **row)) + '\n')
            print(json.dumps(dict(file=spec['path'], completed=len(files), total=len(selected))), flush=True)
        verify_hashes(launch['protected_hashes'])
        report = dict(status='completed_verified', completed_at=now(), job_id=JOB, repository=REPOSITORY,
            revision=launch['revision'], tree_id=launch['tree_id'], tree_sha256=sha256(tree_path),
            files=files, bytes_verified=sum(row['size'] for row in files), protected_hashes=launch['protected_hashes'],
            audio_downloaded=False, gpu_used=False, remote_code_executed=False, training_admitted=False,
            pending=['License/schema audit', 'Selected subset identity exclusions and audio screening',
                     'Frozen whole-track category auxiliary protocol; never expand tags into continuous VA'])
        atomic_json(DIRECTORY / 'report.json', report)
        update('completed_verified', report=relative(DIRECTORY / 'report.json'), revision=launch['revision'],
               bytes_verified=report['bytes_verified'])
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        atomic_json(DIRECTORY / 'failure.json', dict(failed_at=now(), error=error))
        update('failed', error=error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true'); mode.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    with run_lease(JOB):
        register() if args.register else run(args.resume)


if __name__ == '__main__': main()
