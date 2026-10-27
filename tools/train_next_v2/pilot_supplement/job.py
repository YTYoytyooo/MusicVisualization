"""Reuse immutable native gates and jointly screen 60 previous + 20 new recordings."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import traceback

import soundfile as sf

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, run_lease, lock_path
from tools.train_next_v2.common import ROOT, BASE, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.research_inference.benchmark import process_alive
from tools.train_next_v2.pilot_native.job import decode_eof, identities
from tools.train_next_v2.auxiliary_data.merge_audio_audit import fingerprint
from tools.train_next_v2.jamendo_audit_v2.run import reference_values

JOB='e-fabio-pilot-supplement-20-v1'
DIRECTORY=BASE/'annotation_pilot'/JOB
PRIOR=BASE/'annotation_pilot/e-fabio-pilot-native-60-v1'
CANDIDATES=BASE/'annotation_pilot/e-fabio-pilot-candidates-60-v1'


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def selection():
    manifest=read_json(CANDIDATES/'manifest.json')
    excluded={r['selection']['audio_sha256'] for r in manifest['candidates']}
    chosen=sorted((r for r in manifest['all_exact_byte_groups'] if r['audio_sha256'] not in excluded),
                  key=lambda r:(r['selection_rank'],r['audio_sha256']))[:20]
    if len(chosen)!=20 or len(excluded)!=60 or any(r['annotation_song_ids'] for r in chosen):
        raise ValueError('Unexpected supplement population')
    return chosen


def status():
    process=read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity=process.get('process_identity');log=DIRECTORY/'console.log'
    return dict(job_id=JOB,checked_at=now(),process_identity=identity,
        process_alive=bool(identity and process_alive(identity)),os_lease_active=is_running(JOB),
        receipts=len(list((DIRECTORY/'records').glob('*.json'))),
        progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None,
        bounded_log_tail=log.read_text(encoding='utf-8').splitlines()[-2:] if log.exists() else [])


def state(stage,**values):
    data=dict(job_id=JOB,stage=stage,updated_at=now(),**values)
    atomic_json(DIRECTORY/'status.json',data);line=json.dumps(data,ensure_ascii=False)
    with (DIRECTORY/'console.log').open('a',encoding='utf-8') as out: out.write(line+'\n')
    print(line,flush=True)


def prepare():
    if DIRECTORY.exists(): raise ValueError('Preserve existing registered supplement')
    old=read_json(PRIOR/'launch.json');report=read_json(PRIOR/'report.json')
    pins=dict(old['protected_hashes']);pins.update(report['outputs'])
    paths=list(Path(__file__).parent.glob('*.py'))+[PRIOR/'launch.json',PRIOR/'report.json',PRIOR/'verification.json',
        PRIOR/'independent-check.json',PRIOR/'quarantine-summary.json',CANDIDATES/'manifest.json']
    pins.update({rel(p):sha256(p) for p in paths})
    for r in [read_json(p) for p in (PRIOR/'records').glob('*.json')]:
        pins.update(r['diagnostic_files'])
        if r.get('fingerprint'): pins[r['fingerprint']['path']]=r['fingerprint']['sha256']
    verify_hashes(pins)
    argv=[old['python'],'-u','-m','tools.train_next_v2.pilot_supplement.job','--launch',str(DIRECTORY/'launch.json')]
    config=dict(job_id=JOB,registered_at=now(),original_argv=argv,resume_argv=argv+['--resume'],
        verification_argv=[old['python'],'-u','-m','tools.train_next_v2.pilot_supplement.job','--verify'],
        python=old['python'],cwd=str(ROOT),environment=old['environment'],protected_hashes=pins,
        selected=selection(),selection_rule='Next 20 seeded SHA256 ranks from the existing byte-group manifest, excluding every prior 60 hash. No labels, predictions or evaluation outcomes used to rank.',
        cumulative_audit_cap=80,prior_receipts=[rel(p) for p in sorted((PRIOR/'records').glob('*.json'))],
        fingerprint_cache=old['fingerprint_cache'],worker_count=1,gpu_used=False,
        native_contract=old['native_contract'],dedup_contract='Jointly rescreen all 80 against the same 8887 references; include old quarantined recordings to propagate connected native failures. No role allocation or artist identity certification.',
        log=rel(DIRECTORY/'console.log'),runtime_lock=str(lock_path(JOB)),
        recovery='Check PID creation identity and OS lease are inactive, diagnose failure, preserve config and receipts and use resume_argv. Never delete failed receipts or weaken gates.',
        scope='20 new native/fingerprint checks and combined review manifest. No labels, model features, fitting, final-test evaluation or invented human results.')
    DIRECTORY.mkdir(parents=True);atomic_json(DIRECTORY/'launch.json',config)
    record_event(dict(event='E_target_supplement_registered',job_id=JOB,songs=20,cumulative_cap=80),run=dict(run_id=JOB,stage='E',status='registered',launch=rel(DIRECTORY/'launch.json'),launch_sha256=sha256(DIRECTORY/'launch.json'),original_argv=argv,cwd=str(ROOT),runtime_lock=str(lock_path(JOB)),log=rel(DIRECTORY/'console.log')))
    print(json.dumps(dict(job_id=JOB,pinned_inputs=len(pins))))


def run(config):
    config_hash=sha256(DIRECTORY/'launch.json');new=[]
    (DIRECTORY/'records').mkdir(exist_ok=True);cache=ROOT/config['fingerprint_cache']
    for choice in config['selected']:
        digest=choice['audio_sha256'];path=ROOT/choice['audio_path'];saved=DIRECTORY/'records'/(digest+'.json')
        if saved.exists():
            row=read_json(saved)
            if row['config_sha256']!=config_hash or row['audio_sha256']!=digest: raise ValueError('Changed resume receipt')
            verify_hashes(row['diagnostic_files'])
            if row.get('fingerprint'): reference_values(row)
        else:
            row=dict(song_id='fabio-pilot:'+digest,audio_sha256=digest,audio_path=choice['audio_path'],
                config_sha256=config_hash,passed=False,metadata={},annotation_song_ids=[],training_admitted=False)
            logs=DIRECTORY/'native_logs'/digest
            try:
                with sf.SoundFile(path) as audio: row['metadata']={k:getattr(audio,k) for k in ('artist','title','album')}
                row['native_audio']=decode_eof(path,digest,logs);row['passed']=True
            except Exception as exc: row['native_error']=f'{type(exc).__name__}: {exc}'
            try: row['fingerprint'],_=fingerprint(path,digest,cache)
            except Exception as exc: row['fingerprint_error']=f'{type(exc).__name__}: {exc}'
            row['diagnostic_files']={rel(p):sha256(p) for p in logs.glob('*') if p.is_file()}
            atomic_json(saved,row)
        new.append(row);state('native_and_fingerprint',done=len(new),total=20,passed=sum(r['passed'] for r in new))
    prior=[read_json(ROOT/p) for p in config['prior_receipts']]
    checks=prior+new
    if len(checks)!=80 or len({r['audio_sha256'] for r in checks})!=80: raise ValueError('Cumulative pool membership mismatch')
    state('joint_identity_screen',candidates=80,references=8887)
    screened,links,reference_count=identities(checks)
    candidates=[]
    by_id={r['song_id']:r for r in checks}
    for r in screened:
        if r['annotation_candidate']:
            source=by_id[r['song_id']]
            candidates.append(dict(**r,raw_annotation_song_ids=source['annotation_song_ids'],
                actual_reviewers_confirmed=[],planned_reviewers_minimum=2,planned_reviewers_target=3,
                human_completion_verified=False,artist_group_frozen=False,dataset_role='unassigned'))
    atomic_json(DIRECTORY/'screened-80.json',screened);atomic_json(DIRECTORY/'recording-links.json',links)
    atomic_json(DIRECTORY/'review-pool.json',dict(job_id=JOB,candidates=candidates,
        raw_labels_unchanged=True,missing_human_results_not_filled=True,training_admitted=False,
        formal_evaluation_admitted=False,artist_groups_frozen=False))
    verify_hashes(config['protected_hashes'])
    outputs={rel(p):sha256(p) for p in list((DIRECTORY/'records').glob('*.json'))+[DIRECTORY/'screened-80.json',DIRECTORY/'recording-links.json',DIRECTORY/'review-pool.json']}
    report=dict(job_id=JOB,status='completed_supplement_pending_verification',completed_at=now(),new_songs=20,
        new_native_passed=sum(r['passed'] for r in new),cumulative_songs=80,cumulative_native_passed=sum(r['passed'] for r in checks),
        review_candidates=len(candidates),raw_annotated_candidates=sum(bool(r['raw_annotation_song_ids']) for r in candidates),
        reference_identities=8887,reference_fingerprints=reference_count,identity_links=len(links),
        dispositions=dict(Counter(r['disposition'] for r in screened)),
        new_native_failures=[dict(song_id=r['song_id'],audio_path=r['audio_path'],error=r.get('native_error')) for r in new if not r['passed']],
        outputs=outputs,training_admitted=False,formal_evaluation_admitted=False,final_test_targets_evaluated=0,
        conclusion='Technical review pool only. Artist identities/aliases, full-song completeness, actual completion and independent human reviews remain unresolved. Do not expand the 80-song audit automatically.')
    atomic_json(DIRECTORY/'report.json',report);state(report['status'],review_candidates=len(candidates))
    record_event(dict(event='E_target_supplement_completed',job_id=JOB,new_native_passed=report['new_native_passed'],review_candidates=len(candidates)),run=dict(run_id=JOB,stage='E',status='completed_pending_verification',report=rel(DIRECTORY/'report.json')))


def verify(config):
    r=read_json(DIRECTORY/'report.json');verify_hashes(r['outputs'])
    rows=[read_json(p) for p in (DIRECTORY/'records').glob('*.json')]
    if len(rows)!=20 or {x['audio_sha256'] for x in rows}!={x['audio_sha256'] for x in config['selected']}: raise ValueError('Incomplete supplement')
    if config['selected']!=selection(): raise ValueError('Selection drift')
    for row in rows:
        if row['config_sha256']!=sha256(DIRECTORY/'launch.json'): raise ValueError('Receipt config mismatch')
        verify_hashes(row['diagnostic_files'])
        if row.get('fingerprint'): reference_values(row)
        if row['passed']:
            n=row['native_audio']
            if n['frames']!=n['stream_frames'] or n['frames']!=n['ffmpeg_frames'] or n['float32_sample_sha256']!=n['stream_float32_sha256']: raise ValueError('Native evidence mismatch')
            if n['duration_seconds']!=n['frames']/n['sample_rate'] or n['timeline_scaled']: raise ValueError('Native clock altered')
    screened=read_json(DIRECTORY/'screened-80.json');pool=read_json(DIRECTORY/'review-pool.json')['candidates']
    if len(screened)!=80 or len(pool)!=r['review_candidates'] or {x['song_id'] for x in pool}!={x['song_id'] for x in screened if x['annotation_candidate']}: raise ValueError('Review pool accounting mismatch')
    if any(x['training_admitted'] or x['formal_evaluation_admitted'] or x['final_split_assigned'] for x in screened): raise ValueError('Unjustified role/admission')
    if any(x['actual_reviewers_confirmed'] or x['human_completion_verified'] or x['artist_group_frozen'] or x['dataset_role']!='unassigned' for x in pool): raise ValueError('Invented human/split completion')
    result=dict(status='passed',verified_at=now(),new_songs=20,cumulative_songs=80,review_candidates=len(pool),
        report_sha256=sha256(DIRECTORY/'report.json'),training_admitted=False,formal_evaluation_admitted=False)
    atomic_json(DIRECTORY/'verification.json',result);state('completed_verified',review_candidates=len(pool))
    record_event(dict(event='E_target_supplement_verified',job_id=JOB,**result),run=dict(run_id=JOB,stage='E',status='completed_verified',verification=rel(DIRECTORY/'verification.json')))
    print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--launch',type=Path);g.add_argument('--verify',action='store_true');g.add_argument('--status',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.prepare: prepare();return
    if args.status: print(json.dumps(status()));return
    config=read_json(DIRECTORY/'launch.json');verify_hashes(config['protected_hashes'])
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(config['python']).resolve(): raise ValueError('Wrong cwd/runtime')
    if any(os.environ.get(k)!=v for k,v in config['environment'].items()): raise ValueError('Environment drift')
    previous=status()
    if previous['process_alive'] or previous['os_lease_active']: raise ValueError('Same job still running')
    if args.launch and args.launch.resolve()!=(DIRECTORY/'launch.json').resolve(): raise ValueError('Wrong launch path')
    if args.launch and (DIRECTORY/'report.json').exists(): raise ValueError('Completed run: verification only')
    if args.launch and (DIRECTORY/'process.json').exists() and not args.resume: raise ValueError('Inspect failure and use --resume')
    with run_lease(JOB):
        atomic_json(DIRECTORY/'process.json',dict(process_identity=process_identity(),argv=sys.argv,started_at=now()))
        record_event(dict(event='E_target_supplement_started',job_id=JOB,verification=args.verify,process_identity=process_identity()),run=dict(run_id=JOB,stage='E',status='running'))
        try:
            if args.verify: verify(config)
            else: run(config)
        except BaseException:
            state('failed',error=traceback.format_exc())
            record_event(dict(event='E_target_supplement_failed',job_id=JOB),run=dict(run_id=JOB,stage='E',status='failed'))
            raise


if __name__=='__main__': main()
