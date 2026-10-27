"""Capture libsndfile's native stderr per eligible clip, without threaded stderr mixing."""
import argparse
import os
from pathlib import Path
import sys

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, process_identity, read_json, record_event, verify_hashes
from .merge_audio_audit import decode

JOB='d-merge-native-decoder-v1'
FOLDER=BASE/'analyses'/JOB
SOURCE=ROOT/'data/datasets/merge/prepared/strict-audio-v1/eligible-quadrant-candidates.json'
OUTPUT=ROOT/'data/datasets/merge/prepared/native-audio-v1'


def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def register():
    if FOLDER.exists() or OUTPUT.exists():raise FileExistsError('Preserve prior native decoder check')
    prior=read_json(BASE/'analyses/d-merge-strict-decode-v1/launch.json')
    files=[Path(__file__),SOURCE,BASE/'analyses/d-merge-strict-decode-v1/report.json',
           ROOT/'tools/train_next_v2/auxiliary_data/merge_audio_audit.py',ROOT/'tools/train_next_v2/common.py',
           ROOT/'tools/train_next/core.py',ROOT/'tools/train_next/runtime.py']
    argv=[str(Path(sys.executable).resolve()),'-u','-m','tools.train_next_v2.auxiliary_data.merge_native_guard','--run']
    launch=dict(job_id=JOB,registered_at=now(),original_argv=argv,cwd=str(ROOT),python=argv[0],
        environment=prior['environment'],protected_hashes={rel(p):sha256(p) for p in files},
        os_lock=str(lock_path(JOB)),log=rel(FOLDER/'console.log'),report=rel(FOLDER/'report.json'),
        recovery='Finite sequential audit. Native stderr is isolated by song. Preserve all per-song error logs; no automatic retries.',
        gpu_used=False,labels_used=False)
    FOLDER.mkdir(parents=True);atomic_json(FOLDER/'launch.json',launch)
    stages=read_json(PLAN)['stages']
    stages['D'].update(phase='MERGE_native_decoder_warning_gate',native_decode=dict(job_id=JOB,launch=rel(FOLDER/'launch.json'),status='registered'))
    record_event(dict(event='D_MERGE_native_decoder_registered',launch=rel(FOLDER/'launch.json')),updates=dict(stages=stages))


def run():
    launch=read_json(FOLDER/'launch.json');verify_hashes(launch['protected_hashes'])
    if OUTPUT.exists():raise FileExistsError('Preserve existing native decoder output')
    identity=process_identity();atomic_json(FOLDER/'process.json',dict(started_at=now(),process_identity=identity))
    OUTPUT.mkdir(parents=True);logs=OUTPUT/'stderr';logs.mkdir()
    rows=read_json(SOURCE);checks=[];accepted=[]
    with (FOLDER/'console.log').open('a',encoding='utf-8') as log:
        for row in rows:
            path=ROOT/row['audio']['audio_path']
            if sha256(path)!=row['audio']['audio_sha256']:raise ValueError('Audio changed: '+row['song_id'])
            logpath=logs/(row['release_id']+'.log')
            error=None
            original_fd=os.dup(2)
            try:
                with logpath.open('wb') as native:
                    sys.stderr.flush();os.dup2(native.fileno(),2)
                    try:details=decode(path)
                    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                    finally:sys.stderr.flush()
            finally:
                os.dup2(original_fd,2);os.close(original_fd)
            warning=logpath.read_text(encoding='utf-8',errors='replace')
            passed=error is None and not warning.strip()
            if passed and details!=row['audio']['native_audio']:
                raise ValueError('Native decode result changed: '+row['song_id'])
            result=dict(song_id=row['song_id'],passed=passed,error=error,native_stderr=warning[-4000:],
                        stderr_path=rel(logpath),stderr_sha256=sha256(logpath),audio_sha256=row['audio']['audio_sha256'])
            checks.append(result)
            if passed:accepted.append(dict(**row,native_stderr_guard=result))
            if len(checks)%100==0 or len(checks)==len(rows):
                status=dict(stage='native_decoder_stderr',completed=len(checks),total=len(rows),
                            failed=len(checks)-len(accepted),process_identity=identity,updated_at=now())
                atomic_json(FOLDER/'status.json',status);log.write(str(status)+'\n');log.flush();print(status,flush=True)
        atomic_json(OUTPUT/'checks.json',checks);atomic_json(OUTPUT/'eligible-quadrant-candidates.json',accepted)
        report=dict(job_id=JOB,status='completed_both_decoder_gates',completed_at=now(),
            checked=len(rows),passed=len(accepted),quarantined=len(rows)-len(accepted),
            rejected=[x for x in checks if not x['passed']],source_hashes=launch['protected_hashes'],
            outputs={rel(p):sha256(p) for p in OUTPUT.glob('*.json')},original_heldout_targets_used=False,
            fitting_enabled=False,numeric_regression_enabled=False)
        atomic_json(FOLDER/'report.json',report)
        stages=read_json(PLAN)['stages']
        stages['D'].update(phase='MERGE_audio_ready_protocol_and_features_pending')
        stages['D']['native_decode'].update(status=report['status'],report=rel(FOLDER/'report.json'),passed=len(accepted))
        record_event(dict(event='D_MERGE_both_decoder_gates_completed',passed=len(accepted),quarantined=report['quarantined']),updates=dict(stages=stages))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true');mode.add_argument('--run',action='store_true');args=p.parse_args()
    with run_lease(JOB):register() if args.register else run()
