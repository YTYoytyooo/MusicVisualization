"""Exclude clips whose bulk decode differs in length from the audited native stream."""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

from tools.train_next.core import atomic_json,sha256
from tools.train_next.runtime import lock_path,run_lease
from tools.train_next_v2.common import BASE,ROOT,PLAN,now,process_identity,read_json,record_event,verify_hashes

JOB='d-merge-read-parity-v1';FOLDER=BASE/'analyses'/JOB
SOURCE=ROOT/'data/datasets/merge/prepared/native-audio-v1/eligible-quadrant-candidates.json'
OUTPUT=ROOT/'data/datasets/merge/prepared/read-consistent-v1'


def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def register():
    if FOLDER.exists() or OUTPUT.exists():raise FileExistsError('Preserve prior parity checks')
    prior=read_json(BASE/'analyses/d-merge-native-decoder-v1/launch.json')
    files=[Path(__file__),SOURCE,BASE/'analyses/d-merge-native-decoder-v1/report.json',
           ROOT/'tools/train_next_v2/common.py',ROOT/'tools/train_next/core.py',ROOT/'tools/train_next/runtime.py']
    argv=[str(Path(sys.executable).resolve()),'-u','-m','tools.train_next_v2.auxiliary_data.merge_read_parity','--run']
    launch=dict(job_id=JOB,registered_at=now(),original_argv=argv,cwd=str(ROOT),python=argv[0],
        environment=prior['environment'],protected_hashes={rel(p):sha256(p) for p in files},
        os_lock=str(lock_path(JOB)),log=rel(FOLDER/'console.log'),report=rel(FOLDER/'report.json'),
        recovery='No automatic retries. Bulk/stream metadata mismatches are excluded, never patched or rescaled.',
        gpu_used=False,labels_used=False)
    FOLDER.mkdir(parents=True);atomic_json(FOLDER/'launch.json',launch)
    stages=read_json(PLAN)['stages']
    stages['D']['feature_job'].update(status='failed_data_decode_inconsistency',completed_songs=11,
        cause='A143-72 block decode frames601859 vs bulk decode599040; strict feature guard correctly rejected input.',
        recovery='Do not repeat unchanged v1; screen all clips for parity and use a new frozen candidate/protocol/job version.')
    stages['D'].update(phase='MERGE_bulk_stream_decode_consistency_gate',read_parity=dict(job_id=JOB,launch=rel(FOLDER/'launch.json'),status='registered'))
    record_event(dict(event='D_MERGE_read_parity_registered',launch=rel(FOLDER/'launch.json')),updates=dict(stages=stages))


def run():
    launch=read_json(FOLDER/'launch.json');verify_hashes(launch['protected_hashes'])
    if OUTPUT.exists():raise FileExistsError('Preserve read parity output')
    identity=process_identity();atomic_json(FOLDER/'process.json',dict(started_at=now(),process_identity=identity))
    rows=read_json(SOURCE);OUTPUT.mkdir(parents=True);logs=OUTPUT/'stderr';logs.mkdir()
    accepted=[];checks=[]
    with (FOLDER/'console.log').open('a',encoding='utf-8') as log:
        for row in rows:
            audio=row['audio'];native=audio['native_audio'];path=ROOT/audio['audio_path']
            if sha256(path)!=audio['audio_sha256']:raise ValueError('Audio changed: '+row['song_id'])
            stderr=logs/(row['release_id']+'.log');fd=os.dup(2);error=None;shape=None;rate=None;finite=False
            try:
                with stderr.open('wb') as stream:
                    sys.stderr.flush();os.dup2(stream.fileno(),2)
                    try:
                        values,rate=sf.read(path,dtype='float32',always_2d=True)
                        shape=list(values.shape);finite=bool(np.isfinite(values).all())
                    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                    finally:sys.stderr.flush()
            finally:os.dup2(fd,2);os.close(fd)
            warning=stderr.read_text(encoding='utf-8',errors='replace')
            expected=[native['frames'],native['channels']]
            passed=error is None and not warning.strip() and finite and shape==expected and rate==native['sample_rate']
            result=dict(song_id=row['song_id'],passed=passed,bulk_shape=shape,expected_shape=expected,
                sample_rate=rate,error=error,native_stderr=warning[-4000:],audio_sha256=audio['audio_sha256'])
            checks.append(result)
            if passed:accepted.append(dict(**row,bulk_stream_parity=result))
            if len(checks)%100==0 or len(checks)==len(rows):
                status=dict(stage='bulk_stream_parity',completed=len(checks),total=len(rows),
                    failed=len(checks)-len(accepted),process_identity=identity,updated_at=now())
                atomic_json(FOLDER/'status.json',status);log.write(str(status)+'\n');log.flush();print(status,flush=True)
        atomic_json(OUTPUT/'checks.json',checks);atomic_json(OUTPUT/'eligible-quadrant-candidates.json',accepted)
        regression=next(x for x in checks if x['song_id']=='merge:A143-72')
        if regression['passed']:raise ValueError('Known bulk/stream inconsistency was not detected')
        report=dict(job_id=JOB,status='completed_exact_read_parity_gate',completed_at=now(),
            checked=len(rows),passed=len(accepted),quarantined=len(rows)-len(accepted),regression_case=regression,
            rejected=[x for x in checks if not x['passed']],source_hashes=launch['protected_hashes'],
            outputs={rel(p):sha256(p) for p in OUTPUT.glob('*.json')},original_heldout_targets_used=False,
            existing_feature_checks_relaxed=False,timeline_stretched=False,training_started=False)
        atomic_json(FOLDER/'report.json',report)
        stages=read_json(PLAN)['stages']
        stages['D'].update(phase='MERGE_read_consistent_pool_ready')
        stages['D']['read_parity'].update(status=report['status'],report=rel(FOLDER/'report.json'),passed=len(accepted))
        record_event(dict(event='D_MERGE_read_parity_completed',passed=len(accepted),quarantined=report['quarantined']),updates=dict(stages=stages))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true');mode.add_argument('--run',action='store_true');args=p.parse_args()
    with run_lease(JOB):register() if args.register else run()
