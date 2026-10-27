"""Select a reproducible 60-recording review pool from the user's local audio."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import soundfile as sf

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, is_running, lock_path
from tools.train_next_v2.common import ROOT, BASE, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.research_inference.benchmark import process_alive

JOB = 'e-fabio-pilot-candidates-60-v1'
DIRECTORY = BASE/'annotation_pilot'/JOB
INDEX = ROOT/'data/datasets/external/fabio/metadata/collection-index-20261005-v1.json'
SEED = 'fabio-target-pilot-20261005-v1'


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def select(index, count, seed):
    """Exact-byte groups first; keep annotated songs, then balance collections."""
    groups=defaultdict(list)
    for row in index['all_audio_files']:
        if row['kind']=='audio': groups[row['sha256']].append(row)
    if not 0<count<=len(groups): raise ValueError('Invalid bounded candidate count')
    candidates=[]
    for digest,aliases in sorted(groups.items()):
        aliases=sorted(aliases,key=lambda x:(not bool(x['annotation_song_id']),x['path']))
        canonical=aliases[0]
        candidates.append(dict(audio_sha256=digest,audio_path=canonical['path'],collection=canonical['collection'],
            exact_file_aliases=[x['path'] for x in aliases],
            annotation_song_ids=sorted({x['annotation_song_id'] for x in aliases if x['annotation_song_id']}),
            selection_rank=hashlib.sha256((seed+':'+digest).encode()).hexdigest()))
    chosen=[c for c in candidates if c['annotation_song_ids']]
    if len(chosen)>count: raise ValueError('Budget cannot retain existing annotated recordings')
    taken={c['audio_sha256'] for c in chosen}
    collection_counts=Counter(c['collection'] for c in chosen)
    queues=defaultdict(list)
    for c in sorted(candidates,key=lambda x:(x['selection_rank'],x['audio_sha256'])):
        if c['audio_sha256'] not in taken: queues[c['collection']].append(c)
    while len(chosen)<count:
        available=[c for c in queues if queues[c]]
        collection=min(available,key=lambda c:(collection_counts[c],c))
        chosen.append(queues[collection].pop(0));collection_counts[collection]+=1
    return sorted(chosen,key=lambda c:(c['collection'],c['selection_rank'])),candidates


def status():
    p=read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity=p.get('process_identity')
    log=DIRECTORY/'console.log'
    return dict(job_id=JOB,checked_at=now(),process_identity=identity,
        process_alive=bool(identity and process_alive(identity)),os_lease_active=is_running(JOB),
        receipts=len(list((DIRECTORY/'records').glob('*.json'))),
        progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None,
        bounded_log_tail=log.read_text(encoding='utf-8').splitlines()[-2:] if log.exists() else [])


def state(stage,**extra):
    row=dict(job_id=JOB,stage=stage,updated_at=now(),**extra)
    atomic_json(DIRECTORY/'status.json',row)
    line=json.dumps(row,ensure_ascii=False)
    with (DIRECTORY/'console.log').open('a',encoding='utf-8') as stream: stream.write(line+'\n')
    print(line,flush=True)


def prepare():
    if DIRECTORY.exists(): raise ValueError('Do not replace a registered job')
    previous=read_json(BASE/'annotation_pilot/e-fabio-certified-intake-v1/launch.json')
    pins=dict(previous['protected_hashes'])
    index=read_json(INDEX)
    pins.update({r['path']:r['sha256'] for r in index['all_audio_files']})
    pins.update({rel(p):sha256(p) for p in Path(__file__).parent.glob('*.py')})
    for name in ('report.json','verification.json','independent-check.json'):
        p=BASE/'annotation_pilot/e-fabio-certified-intake-v1'/name;pins[rel(p)]=sha256(p)
    verify_hashes(pins)
    argv=[previous['python'],'-u','-m','tools.train_next_v2.pilot_candidates.job','--launch',str(DIRECTORY/'launch.json')]
    config=dict(job_id=JOB,registered_at=now(),original_argv=argv,resume_argv=argv+['--resume'],
        verification_argv=[previous['python'],'-u','-m','tools.train_next_v2.pilot_candidates.job','--verify'],
        cwd=str(ROOT),python=previous['python'],environment=previous['environment'],protected_hashes=pins,
        source_index=rel(INDEX),count=60,seed=SEED,
        selection_rule='Group exact-byte duplicates; retain all existing annotated recordings; use seeded SHA256 rank within each collection, choosing the least represented non-exhausted collection until 60. Album collections are sampling strata, not artist or recording identities.',
        log=rel(DIRECTORY/'console.log'),runtime_lock=str(lock_path(JOB)),
        recovery='After original PID identity is dead and lease free, preserve launch and completed receipts, diagnose failure, use resume_argv. Never overwrite changed receipts.',
        scope='Candidate manifest and embedded header metadata only; no PCM/feature extraction, learned score selection, label creation, split assignment, human verification or model fitting.',
        planned_reviewers_per_song_minimum=2,planned_reviewers_per_song_target=3)
    DIRECTORY.mkdir(parents=True);atomic_json(DIRECTORY/'launch.json',config)
    record_event(dict(event='E_target_pilot_candidates_registered',job_id=JOB,count=60),run=dict(run_id=JOB,stage='E',status='registered',launch=rel(DIRECTORY/'launch.json'),launch_sha256=sha256(DIRECTORY/'launch.json'),original_argv=argv,cwd=str(ROOT),runtime_lock=str(lock_path(JOB)),log=rel(DIRECTORY/'console.log')))
    print(json.dumps(dict(job_id=JOB,pinned_files=len(pins))))


def verify(config):
    verify_hashes(config['protected_hashes'])
    selected,all_groups=select(read_json(ROOT/config['source_index']),config['count'],config['seed'])
    report=read_json(DIRECTORY/'manifest.json')
    assert len(report['candidates'])==60
    assert len({r['selection']['audio_sha256'] for r in report['candidates']})==60
    assert [r['selection'] for r in report['candidates']]==selected
    assert len({s for r in selected for s in r['annotation_song_ids']})==9
    reversed_index=read_json(ROOT/config['source_index']);reversed_index['all_audio_files'].reverse()
    assert select(reversed_index,60,config['seed'])[0]==selected
    for row in report['candidates']:
        assert row['config_sha256']==sha256(DIRECTORY/'launch.json')
        assert not row['training_admitted'] and not row['formal_evaluation_admitted']
        assert row['assigned_split'] is None and row['human_quality_verified'] is False
        assert row['actual_reviewers_confirmed']==[]
        receipt=DIRECTORY/'records'/(row['selection']['audio_sha256']+'.json')
        assert read_json(receipt)==row
    result=dict(status='passed',verified_at=now(),selected_exact_byte_groups=60,
        inventory_audio_files=400,inventory_exact_byte_groups=len(all_groups),
        collection_counts=dict(Counter(r['collection'] for r in selected)),
        retained_existing_annotation_songs=9,unannotated_selected_recordings=51,
        order_invariance_verified=True,manifest_sha256=sha256(DIRECTORY/'manifest.json'),
        native_audio_verified=False,acoustic_identity_verified=False,artist_groups_frozen=False,
        training_admitted=False,formal_evaluation_admitted=False)
    atomic_json(DIRECTORY/'verification.json',result)
    return result


def run(config):
    index=read_json(ROOT/config['source_index'])
    selected,groups=select(index,config['count'],config['seed'])
    config_hash=sha256(DIRECTORY/'launch.json');rows=[]
    (DIRECTORY/'records').mkdir(exist_ok=True)
    for choice in selected:
        path=DIRECTORY/'records'/(choice['audio_sha256']+'.json')
        if path.exists():
            row=read_json(path)
            if row['config_sha256']!=config_hash or row['selection']!=choice: raise ValueError('Resume receipt changed')
        else:
            error=None;metadata={}
            try:
                with sf.SoundFile(ROOT/choice['audio_path']) as audio:
                    metadata=dict(artist=audio.artist,title=audio.title,album=audio.album,
                        sample_rate=audio.samplerate,channels=audio.channels,header_frames=audio.frames,
                        header_duration_seconds=audio.frames/audio.samplerate)
            except Exception as exc: error=str(exc)
            row=dict(config_sha256=config_hash,selection=choice,embedded_metadata=metadata,header_read_error=error,
                header_duration_is_estimate=True,metadata_is_unverified_hint=True,
                original_song_completeness_verified=False,artist_identity_verified=False,
                acoustic_identity_verified=False,native_eof_verified_in_this_job=False,
                existing_raw_intake_report='data/training/next_va/optimization/annotation_pilot/e-fabio-certified-intake-v1/report.json' if choice['annotation_song_ids'] else None,
                raw_annotation_completion_not_overridden=True,actual_reviewers_confirmed=[],
                planned_reviewers_minimum=2,planned_reviewers_target=3,human_quality_verified=False,
                assigned_split=None,training_admitted=False,formal_evaluation_admitted=False)
            atomic_json(path,row)
        rows.append(row)
        if len(rows)%15==0: state('reading_candidate_headers',done=len(rows),total=60)
    manifest=dict(job_id=JOB,created_at=now(),source_index=config['source_index'],selection_rule=config['selection_rule'],seed=config['seed'],
        candidates=rows,all_exact_byte_groups=groups,source_files_deleted=False,
        training_admitted=False,formal_evaluation_admitted=False,
        limitations=['Exact-byte groups do not establish acoustic recording identity or artist groups.',
            'Embedded metadata/header durations are hints; EOF/native and reference acoustic checks must follow.',
            'These recordings are review candidates, not a ready labeled dataset. No reviewer results have been invented.',
            'Candidate selection uses no validation/test labels or model predictions.'])
    atomic_json(DIRECTORY/'manifest.json',manifest)
    verification=verify(config)
    report=dict(job_id=JOB,status='completed_candidate_manifest_pending_identity_and_humans',completed_at=now(),**{k:v for k,v in verification.items() if k not in ('status','verified_at')},
                header_read_errors=sum(r['header_read_error'] is not None for r in rows),
                manifest=rel(DIRECTORY/'manifest.json'),final_test_targets_read=0)
    atomic_json(DIRECTORY/'report.json',report);state(report['status'],selected=60)
    record_event(dict(event='E_target_pilot_candidates_completed',**report),run=dict(run_id=JOB,stage='E',status='completed_verified',report=rel(DIRECTORY/'report.json')))


def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--launch',type=Path)
    g.add_argument('--verify',action='store_true');g.add_argument('--status',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.prepare: prepare();return
    if args.status: print(json.dumps(status()));return
    config=read_json(DIRECTORY/'launch.json');verify_hashes(config['protected_hashes'])
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(config['python']).resolve(): raise ValueError('Wrong runtime/cwd')
    if any(os.environ.get(k)!=v for k,v in config['environment'].items()): raise ValueError('Frozen environment changed')
    previous=status()
    if previous['process_alive'] or previous['os_lease_active']: raise ValueError('Job still active')
    if args.launch and args.launch.resolve()!=(DIRECTORY/'launch.json').resolve(): raise ValueError('Wrong launch file')
    if args.launch and (DIRECTORY/'report.json').exists(): raise ValueError('Completed job must not rerun')
    if args.launch and (DIRECTORY/'process.json').exists() and not args.resume: raise ValueError('Inspect failure and use --resume')
    with run_lease(JOB):
        atomic_json(DIRECTORY/'process.json',dict(process_identity=process_identity(),argv=sys.argv,started_at=now()))
        if args.verify: print(json.dumps(verify(config)));return
        record_event(dict(event='E_target_pilot_candidates_started',job_id=JOB,process_identity=process_identity()),run=dict(run_id=JOB,stage='E',status='running'))
        try: run(config)
        except BaseException:
            state('failed',error=traceback.format_exc())
            record_event(dict(event='E_target_pilot_candidates_failed',job_id=JOB),run=dict(run_id=JOB,stage='E',status='failed'))
            raise


if __name__=='__main__': main()
