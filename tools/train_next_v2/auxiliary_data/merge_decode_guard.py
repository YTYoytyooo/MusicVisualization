"""Associate strict MP3 decoder errors with individual candidate songs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, process_identity, read_json, record_event, verify_hashes

JOB = 'd-merge-strict-decode-v1'
FOLDER = BASE / 'analyses' / JOB
FFMPEG = ROOT / 'versions/v2.0/.runtime/bin/ffmpeg.exe'
SOURCE = ROOT / 'data/datasets/merge/prepared/audio-audit-v1/eligible-quadrant-candidates.json'
OUTPUT = ROOT / 'data/datasets/merge/prepared/strict-audio-v1'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def command(audio):
    return [str(FFMPEG),'-nostdin','-hide_banner','-v','error','-xerror','-err_detect','explode',
            '-threads','1','-i',str(audio),'-map','0:a:0','-f','null','-']


def inspect(row):
    path = ROOT / row['audio']['audio_path']
    if sha256(path) != row['audio']['audio_sha256']:
        raise ValueError('Source audio changed: '+row['song_id'])
    result = subprocess.run(command(path),capture_output=True,text=True,timeout=45,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    passed = result.returncode == 0 and not result.stderr.strip()
    return dict(song_id=row['song_id'],passed=passed,returncode=result.returncode,
                error=result.stderr[-4000:],audio_sha256=row['audio']['audio_sha256'])


def register():
    if FOLDER.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve prior decoder guard')
    prior = read_json(BASE/'analyses/d-merge-audio-audit-v1/launch.json')
    hashes = {relative(p):sha256(p) for p in [Path(__file__),FFMPEG,SOURCE,
        BASE/'analyses/d-merge-audio-audit-v1/report.json',ROOT/'tools/train_next_v2/common.py',
        ROOT/'tools/train_next/core.py',ROOT/'tools/train_next/runtime.py']}
    argv = [str(Path(sys.executable).resolve()),'-u','-m','tools.train_next_v2.auxiliary_data.merge_decode_guard','--run']
    launch = dict(job_id=JOB,registered_at=now(),original_argv=argv,cwd=str(ROOT),python=argv[0],
        environment=prior['environment'],protected_hashes=hashes,source=relative(SOURCE),
        os_lock=str(lock_path(JOB)),log=relative(FOLDER/'console.log'),report=relative(FOLDER/'report.json'),
        status_path=relative(FOLDER/'status.json'),gpu_used=False,worker_count=4,
        per_file_argv=command('<verified candidate audio path>'),
        recovery='No automatic retry; preserve failed-song evidence. No model labels or scores used.')
    FOLDER.mkdir(parents=True)
    atomic_json(FOLDER/'launch.json',launch)
    stages=read_json(PLAN)['stages']
    stages['D'].update(phase='MERGE_strict_decode_quality_gate',strict_decode=dict(job_id=JOB,launch=relative(FOLDER/'launch.json'),status='registered'))
    record_event(dict(event='D_MERGE_strict_decoder_registered',launch=relative(FOLDER/'launch.json')),updates=dict(stages=stages))


def run():
    launch=read_json(FOLDER/'launch.json');verify_hashes(launch['protected_hashes'])
    if OUTPUT.exists():
        raise FileExistsError('Decoder guard output already exists')
    identity=process_identity()
    atomic_json(FOLDER/'process.json',dict(process_identity=identity,started_at=now()))
    rows=read_json(SOURCE);checks=[]
    with (FOLDER/'console.log').open('a',encoding='utf-8') as log:
        def publish(stage):
            value=dict(stage=stage,completed=len(checks),total=len(rows),failed=sum(not x['passed'] for x in checks),
                       process_identity=identity,updated_at=now())
            atomic_json(FOLDER/'status.json',value)
            log.write(str(value)+'\n');log.flush();print(value,flush=True)
        publish('strict_mp3_decode')
        with ThreadPoolExecutor(max_workers=launch['worker_count']) as pool:
            for result in pool.map(inspect,rows):
                checks.append(result)
                if len(checks)%100==0 or len(checks)==len(rows):publish('strict_mp3_decode')
        okay={row['song_id']:row for row in checks if row['passed']}
        accepted=[dict(**row,strict_decode=okay[row['song_id']]) for row in rows if row['song_id'] in okay]
        OUTPUT.mkdir(parents=True)
        atomic_json(OUTPUT/'checks.json',checks)
        atomic_json(OUTPUT/'eligible-quadrant-candidates.json',accepted)
        report=dict(job_id=JOB,status='completed_strict_audio_gate',completed_at=now(),process_identity=identity,
            candidates_checked=len(rows),passed=len(accepted),quarantined=len(rows)-len(accepted),
            quadrant_counts=dict(Counter(x['quadrant'] for x in accepted)),
            rejected=[row for row in checks if not row['passed']],
            source_hashes=launch['protected_hashes'],outputs={relative(p):sha256(p) for p in OUTPUT.glob('*.json')},
            numeric_regression_enabled=False,fitting_enabled=False,original_heldout_targets_used=False,
            next_gate='Freeze paired auxiliary/control protocol and version eligible-clip features before fitting.')
        atomic_json(FOLDER/'report.json',report)
        stages=read_json(PLAN)['stages']
        stages['D'].update(phase='MERGE_strict_audio_ready_protocol_and_features_pending')
        stages['D']['strict_decode'].update(status=report['status'],report=relative(FOLDER/'report.json'),passed=len(accepted))
        record_event(dict(event='D_MERGE_strict_audio_gate_completed',passed=len(accepted),quarantined=report['quarantined']),updates=dict(stages=stages))
        publish(report['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    with run_lease(JOB):register() if args.register else run()
