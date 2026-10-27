"""Finite CPU-only, resumable, hash-bound technical intake of nine recordings."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import traceback
import unittest

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, run_lease, lock_path
from tools.train_next_v2.common import ROOT, BASE, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.research_inference.benchmark import process_alive
from .contract import VERSION, AXES, verify_native, adapt_raw_takes

JOB = 'e-fabio-certified-intake-v1'
DIRECTORY = BASE / 'annotation_pilot' / JOB
DIAG = BASE / 'annotation_pilot/e-fabio-clock-diagnosis-v1'


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def status():
    process = read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity = process.get('process_identity')
    return dict(job_id=JOB, checked_at=now(), process_identity=identity,
                process_alive=bool(identity and process_alive(identity)), os_lease_active=is_running(JOB),
                receipts=len(list((DIRECTORY/'records').glob('*.json'))),
                progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None)


def state(stage, **values):
    value=dict(job_id=JOB,stage=stage,updated_at=now(),**values)
    atomic_json(DIRECTORY/'status.json',value)
    line=json.dumps(value,ensure_ascii=False)
    with (DIRECTORY/'console.log').open('a',encoding='utf-8') as out: out.write(line+'\n')
    print(line,flush=True)


def prepare():
    if DIRECTORY.exists(): raise ValueError('Existing registered configuration must not be replaced')
    previous=read_json(DIAG/'launch.json')
    evidence=read_json(DIAG/'report.json')['evidence']
    pins=dict(previous['protected_hashes'])
    for item in evidence: pins[item['path']]=item['sha256']
    paths=list(Path(__file__).parent.glob('*.py'))+[DIAG/'launch.json',DIAG/'report.json',DIAG/'verification.json',
        ROOT/'models/emotion_model.pth', ROOT/'models/next_va/teacher-20261003-recovery/best.pth',
        ROOT/'models/next_va/mixed-20261003-recovery/best.pth',
        ROOT/'data/datasets/external/fabio/metadata/collection-index-20261005-v1.json']
    pins.update({relative(p):sha256(p) for p in paths})
    verify_hashes(pins)
    argv=[previous['python'],'-u','-m','tools.train_next_v2.certified_intake.job','--launch',str(DIRECTORY/'launch.json')]
    config=dict(job_id=JOB,contract_version=VERSION,registered_at=now(),original_argv=argv,
        resume_argv=argv+['--resume'],verification_argv=[previous['python'],'-u','-m','tools.train_next_v2.certified_intake.job','--verify'],
        cwd=str(ROOT),python=previous['python'],environment=previous['environment'],protected_hashes=pins,
        certificates=evidence,annotation_snapshot=previous['annotation_snapshot'],max_rate_gap_seconds=.5,
        rate_gap_rationale='Only adjacent valid rows within the same raw take and at most two nominal 0.25-second capture intervals; no gap filling or rate loss is applied.',
        log=relative(DIRECTORY/'console.log'),runtime_lock=str(lock_path(JOB)),
        recovery='After PID creation identity is dead and OS lease is free, preserve configuration and valid receipts; use resume_argv. Failed receipts must be diagnosed, never overwritten.',
        scope='Nine exact source recordings only; CPU EOF verification and lossless raw take export. No fitting, feature extraction, final test access or psychological quality certification.',
        acceptance=['All nine certified source/PCM hashes and EOF counts match','Raw song/take/time/value metadata retained exactly',
                    'Independent axis masks, no invalid/gap/take-crossing rate edges','Regression checks pass','Old header gate failures retained; training/formal evaluation admission false'])
    DIRECTORY.mkdir(parents=True)
    atomic_json(DIRECTORY/'launch.json',config)
    record_event(dict(event='E_certified_intake_registered',job_id=JOB,launch=relative(DIRECTORY/'launch.json')),
                 run=dict(run_id=JOB,stage='E',status='registered',launch=relative(DIRECTORY/'launch.json'),
                          launch_sha256=sha256(DIRECTORY/'launch.json'),original_argv=argv,cwd=str(ROOT),
                          runtime_lock=str(lock_path(JOB)),log=relative(DIRECTORY/'console.log')))
    print(json.dumps(dict(launch=relative(DIRECTORY/'launch.json'),pinned_files=len(pins))))


def verify_outputs(config):
    verify_hashes(config['protected_hashes'])
    source={s['source']['song_id']:s for s in read_json(ROOT/config['annotation_snapshot'])['songs']}
    count=0;points=0;axis_counts={a:0 for a in AXES};outputs=[]
    for ref in config['certificates']:
        certificate=read_json(ROOT/ref['path']);sid=certificate['annotation_song_id']
        path=DIRECTORY/'records'/(sid+'.json');row=read_json(path)
        if row['config_sha256']!=sha256(DIRECTORY/'launch.json') or row['certificate']!=ref: raise ValueError('Receipt binding changed')
        if row['adapted']['raw_song']!=source[sid]: raise ValueError('Raw labels or provenance changed')
        native=row['native']
        if (native['frames'],native['sample_rate'],native['channels'],native['float32_sha256'])!=(certificate['actual_frames'],certificate['actual_sample_rate'],certificate['actual_channels'],certificate['bulk_float32_sha256']): raise ValueError('Saved native receipt mismatch')
        if row['adapted']!=adapt_raw_takes(source[sid],certificate,config['max_rate_gap_seconds']): raise ValueError('Saved masks changed')
        for take in row['adapted']['takes']:
            points+=len(take['raw']['points'])
            for axis in AXES: axis_counts[axis]+=sum(take['valid_masks'][axis])
        count+=1;outputs.append(dict(path=relative(path),sha256=sha256(path)))
    if count!=9 or len(source)!=9: raise ValueError('Unexpected corpus size')
    verification=dict(status='passed',verified_at=now(),songs=count,raw_points=points,valid_points=axis_counts,
        outputs=outputs,training_admitted=False,formal_evaluation_admitted=False,
        browser_priming_verified=False,psychological_quality_verified=False,old_header_gate_failures_preserved=True)
    atomic_json(DIRECTORY/'verification.json',verification)
    return verification


def run(config):
    output=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromName('tools.train_next_v2.certified_intake.test_contract')
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    atomic_json(DIRECTORY/'regression-checks.json',dict(checked_at=now(),tests_run=result.testsRun,
                passed=result.wasSuccessful(),output=output.getvalue()))
    if not result.wasSuccessful(): raise ValueError('Regression checks failed: '+output.getvalue())
    songs={s['source']['song_id']:s for s in read_json(ROOT/config['annotation_snapshot'])['songs']}
    (DIRECTORY/'records').mkdir(exist_ok=True)
    config_hash=sha256(DIRECTORY/'launch.json')
    for index,ref in enumerate(config['certificates']):
        c=read_json(ROOT/ref['path']);path=DIRECTORY/'records'/(c['annotation_song_id']+'.json')
        if path.exists():
            row=read_json(path)
            if row['config_sha256']!=config_hash or row['certificate']!=ref: raise ValueError('Resume binding changed')
            # Revalidate real source PCM even when resuming an existing receipt.
            native=verify_native(ROOT/c['audio_path'],c)
            adapted=adapt_raw_takes(songs[c['annotation_song_id']],c,config['max_rate_gap_seconds'])
            if row['native']!=native or row['adapted']!=adapted: raise ValueError('Resume receipt content changed')
        else:
            native=verify_native(ROOT/c['audio_path'],c)
            adapted=adapt_raw_takes(songs[c['annotation_song_id']],c,config['max_rate_gap_seconds'])
            atomic_json(path,dict(config_sha256=config_hash,certificate=ref,native=native,adapted=adapted))
        state('checking_native_and_raw_takes',done=index+1,total=9)
    verification=verify_outputs(config)
    report=dict(job_id=JOB,status='completed_technical_intake_pending_human_review',completed_at=now(),
        contract_version=VERSION,songs=verification['songs'],raw_points=verification['raw_points'],
        valid_points=verification['valid_points'],regression_checks=result.testsRun,
        outputs=verification['outputs'],training_admitted=False,formal_evaluation_admitted=False,
        human_completion_flags_modified=False,old_header_gate_failures_preserved=True,
        browser_priming_verified=False,psychological_quality_verified=False,final_test_targets_read=0)
    atomic_json(DIRECTORY/'report.json',report);state(report['status'],songs=9,raw_points=report['raw_points'])
    record_event(dict(event='E_certified_intake_completed',**report),run=dict(run_id=JOB,stage='E',status='completed_verified',report=relative(DIRECTORY/'report.json')))


def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--launch',type=Path)
    g.add_argument('--verify',action='store_true');g.add_argument('--status',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.prepare: prepare();return
    if args.status: print(json.dumps(status()));return
    config=read_json(DIRECTORY/'launch.json');verify_hashes(config['protected_hashes'])
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(config['python']).resolve(): raise ValueError('Wrong runtime/cwd')
    if any(os.environ.get(k)!=v for k,v in config['environment'].items()): raise ValueError('Frozen environment changed')
    previous=status()
    if previous['process_alive'] or previous['os_lease_active']: raise ValueError('Existing job active')
    if args.launch and args.launch.resolve()!=(DIRECTORY/'launch.json').resolve(): raise ValueError('Wrong launch')
    if args.launch and (DIRECTORY/'report.json').exists(): raise ValueError('Completed job cannot rerun')
    if args.launch and (DIRECTORY/'process.json').exists() and not args.resume: raise ValueError('Inspect failed attempt and use --resume')
    with run_lease(JOB):
        atomic_json(DIRECTORY/'process.json',dict(process_identity=process_identity(),argv=sys.argv,started_at=now()))
        if args.verify:
            print(json.dumps(verify_outputs(config)));return
        record_event(dict(event='E_certified_intake_started',job_id=JOB,process_identity=process_identity()),run=dict(run_id=JOB,stage='E',status='running'))
        try: run(config)
        except BaseException:
            state('failed',error=traceback.format_exc())
            record_event(dict(event='E_certified_intake_failed',job_id=JOB),run=dict(run_id=JOB,stage='E',status='failed'))
            raise


if __name__=='__main__': main()
