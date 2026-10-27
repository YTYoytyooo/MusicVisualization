"""Native-clock and recording/artist identity audit; no training or new labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
import importlib.metadata
import json
from pathlib import Path
import re
import os
import sys
import time
import traceback

import soundfile as sf

from tools.datasets.index_supplemental import FPCALC, MatchIndex, normalized
from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, is_running, lock_path
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.run import Tee
from tools.train_next_v2.auxiliary_data.merge_audio_audit import fingerprint
from tools.train_next_v2.auxiliary_data.merge_candidates import connected_groups
from tools.train_next_v2.jamendo_audit_v2.native import inspect, FFMPEG
from tools.train_next_v2.jamendo_audit_v2.run import reference_values

JOB='e-target-native-audit-v1'
DIRECTORY=BASE/'annotation_pilot'/JOB
INVENTORY=BASE/'annotation_pilot/e-target-annotation-inventory-v1/inventory.json'
MAIN=ROOT/'data/datasets/combined/prepared/v1/manifest.jsonl'
MERGE=ROOT/'data/datasets/merge/prepared/audio-audit-v1'
JAMENDO=ROOT/'data/datasets/mtg_jamendo/prepared/native-audit-v2/checks.json'
TESTS=BASE/'analyses/e-target-audit-tests-v1/report.json'


def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def artist_tokens(value):
    # Conservative metadata links, not a claim that aliases are resolved.
    return sorted({normalized(v) for v in re.split(r'[,;/&\u3001]|\bfeat\.?\s+|\bfeaturing\s+',value or '',flags=re.I) if normalized(v)})


def local_native(path,duration,logdir):
    result=inspect(path,duration,logdir)
    # Do not inherit Jamendo-specific release/full-track claims from its decoder.
    result.update(source_song_offset_seconds=None,
        timing_scope='Native timeline starts at this local file sample zero. Offset into an external original recording and full-song completeness are unknown.',
        original_song_completeness_verified=False)
    with sf.SoundFile(path) as stream:
        tags={name:getattr(stream,name) for name in ('artist','title','album')}
    return result,tags


def screen(checks,references,links,main_rows):
    ids={r['song_id'] for r in checks}
    if len(ids)!=len(checks) or ids & {r['song_id'] for r in references}:
        raise ValueError('Candidate/reference identities overlap or are not unique')
    graph={r['song_id']:dict(identity_keys=[('audio_sha',r['audio_sha256']),('fp_link',r['song_id'])]) for r in references}
    for row in checks:
        keys=[('audio_sha',row['audio_sha256']),('fp_link',row['song_id'])]
        if row['passed']:
            n=row['native_audio'];keys.append(('pcm',n['sample_rate'],n['channels'],n['float32_sample_sha256']))
        graph[row['song_id']]=dict(identity_keys=keys)
    for link in links:
        if link['song_id'] not in ids or link['matched_song_id'] not in graph:
            raise ValueError('Fingerprint link outside pinned inventories')
        graph[link['song_id']]['identity_keys'].append(('fp_link',link['matched_song_id']))
    groups=connected_groups(graph)
    by_id={r['song_id']:r for r in checks}
    main_artists=defaultdict(list)
    for r in main_rows:
        for token in artist_tokens(r.get('artist','')):
            main_artists[token].append(r['song_id'])
    output=[];artist_graph={}
    for row in sorted(checks,key=lambda r:r['song_id']):
        sid=row['song_id'];members=groups[sid]
        protected=[s for s in members if s not in ids]
        native_failed=[s for s in members if s in by_id and not by_id[s]['passed']]
        tokens=artist_tokens(row.get('metadata',{}).get('artist',''))
        artist_matches=sorted({s for t in tokens for s in main_artists[t]})
        reason=('native_audio_gate_failed' if not row['passed'] else
                'recording_link_to_existing_dataset' if protected else
                'linked_native_failure' if native_failed else
                'duplicate_target_recording' if sid!=min(members) else
                'known_main_artist_overlap_pending_review' if artist_matches else None)
        group_keys=[('recording',s) for s in members if s in ids]
        group_keys += [('artist',t) for t in tokens] if tokens else [('unknown_artist','all_unresolved_targets')]
        artist_graph[sid]=dict(identity_keys=group_keys)
        output.append(dict(song_id=sid,audio_sha256=row['audio_sha256'],audio_path=row['audio_path'],
            native_audio_passed=row['passed'],recording_group=members,protected_recording_ids=protected,
            metadata=row.get('metadata',{}),normalized_artist_tokens=tokens,known_main_artist_ids=artist_matches,
            disposition=reason or 'native_checked_annotation_candidate',annotation_candidate=reason is None,
            artist_identity_status='embedded_metadata_only_aliases_not_resolved' if tokens else 'missing_artist_metadata_manual_review_required',
            native_audio=row.get('native_audio'),training_admitted=False,final_split_assigned=False))
    artist_groups=connected_groups(artist_graph)
    for row in output:
        row['provisional_artist_recording_group']=artist_groups[row['song_id']]
    return output


def code_files():
    return [Path(__file__),Path(__file__).with_name('__init__.py'),
        ROOT/'tools/train_next_v2/tests/test_target_pilot_audit.py',
        ROOT/'tools/train_next_v2/jamendo_audit_v2/native.py',ROOT/'tools/train_next_v2/jamendo_audit_v2/run.py',
        ROOT/'tools/train_next_v2/auxiliary_data/merge_audio_audit.py',ROOT/'tools/train_next_v2/auxiliary_data/merge_candidates.py',
        ROOT/'tools/datasets/index_supplemental.py',ROOT/'tools/train_next/core.py',ROOT/'tools/train_next/runtime.py',
        ROOT/'tools/train_next_v2/common.py',ROOT/'tools/train_next_v2/run.py']


def register():
    if (DIRECTORY/'launch.json').exists():raise ValueError('Preserve registered audit')
    tests=read_json(TESTS)
    if tests['status']!='passed' or tests['failures'] or tests['errors'] or tests['skipped']:
        raise ValueError('Pass targeted exclusion tests before registration')
    verify_hashes(tests['code'])
    inventory=read_json(INVENTORY)
    previous=read_json(BASE/'analyses/d-jamendo-native-audit-v2/launch.json')
    jam_report=read_json(BASE/'analyses/d-jamendo-native-audit-v2/report.json')
    merge_report=read_json(BASE/'analyses/d-merge-audio-audit-v1/report.json')
    verify_hashes({rel(JAMENDO):jam_report['outputs'][rel(JAMENDO)]})
    for path in (MERGE/'main-fingerprint-receipts.json',MERGE/'audio-inventory.json'):
        verify_hashes({rel(path):merge_report['outputs'][rel(path)]})
    files=code_files()+[INVENTORY,MAIN,TESTS,MERGE/'main-fingerprint-receipts.json',MERGE/'audio-inventory.json',JAMENDO,FPCALC,FFMPEG]
    files += [Path(sf.__file__)]+list((Path(sf.__file__).parent/'_soundfile_data').glob('*.dll'))
    protected={rel(p):sha256(p) for p in files}
    protected[rel(MAIN)]=read_json(BASE/'protocols/b-v1.json')['manifest_sha256']
    protected['models/emotion_model.pth']='f0c304aec8120ba09e21b2540cc5ae0de23a1536c449673edd0388b896e31667'
    for c in inventory['candidates']:
        for f in c['files']:protected[f['path']]=f['sha256']
        for a in c['annotations']:protected[a['path']]=a['sha256']
    verify_hashes(protected)
    environment=previous['environment'];environment['CUDA_VISIBLE_DEVICES']='-1'
    argv=[read_json(PLAN)['python'],'-u','-m','tools.train_next_v2.annotation_pilot.audit','--run']
    launch=dict(job_id=JOB,registered_at=now(),original_argv=argv,resume_argv=argv+['--resume'],cwd=str(ROOT),python=argv[0],
        environment=environment,protected_hashes=protected,os_lock=str(lock_path(JOB)),log=rel(DIRECTORY/'console.log'),
        runtime_versions={k:importlib.metadata.version(k) for k in ('numpy','soundfile')},
        fingerprint_cache=rel(ROOT/'data/datasets/.fingerprints'/sha256(FPCALC)[:16]),
        expected_unique_file_hashes=36,expected_files=39,worker_count=1,gpu_used=False,targets_used=False,
        gates=['Source bytes SHA unchanged; duplicates not counted twice.',
            'Strict ffmpeg diagnostics; full native soundfile decode; exact bulk/sequential sample hashes, clocks and frame counts.',
            'Pinned Chromaprint offset screen against main 5042 songs, MERGE 3554 clips and selected Jamendo 255 tracks.',
            'Transitive recording links and decoded-sample duplicates propagate exclusions; distinct artists are not treated as distinct recordings.',
            'Embedded artist metadata links form provisional split groups; unresolved artists share a conservative single group.'],
        limitations=['Five MERGE clips lack valid acoustic fingerprints; their exact hashes remain checked.',
            'No proof against all covers, heavy edits, aliases or short overlaps. Exact tags are hints, not verified artist identity.',
            'Full original-song completeness/offset cannot be inferred from file duration. Annotation follows the local file clock.',
            'No pseudo-label training admission; statistical calibration already failed. Human annotation and target-pool completion remain necessary.'],
        recovery='Inspect bounded log and inactive OS lease. Resume identical configuration with --resume, validating all saved receipts and diagnostics. Never retry a quarantined file under relaxed gates.')
    DIRECTORY.mkdir(parents=True,exist_ok=True);atomic_json(DIRECTORY/'launch.json',launch)
    stages=read_json(PLAN)['stages'];stages['E']['target_native_audit']=dict(job_id=JOB,status='registered',launch=rel(DIRECTORY/'launch.json'))
    record_event(dict(event='E_target_native_audit_registered',job_id=JOB),
        run=dict(run_id=JOB,status='registered',stage='E',launch=rel(DIRECTORY/'launch.json')),updates=dict(stages=stages))
    print(json.dumps(dict(job_id=JOB,status='registered')))


def run(resume):
    launch=read_json(DIRECTORY/'launch.json');verify_hashes(launch['protected_hashes'])
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(launch['python']).resolve():raise ValueError('Wrong runtime or cwd')
    for name,value in launch['environment'].items():
        if os.environ.get(name)!=value:raise ValueError('Frozen environment changed: '+name)
    for name,value in launch['runtime_versions'].items():
        if importlib.metadata.version(name)!=value:raise ValueError('Native dependency changed: '+name)
    if (DIRECTORY/'report.json').exists():raise ValueError('Completed audit cannot be restarted')
    if (DIRECTORY/'process.json').exists() and not resume:raise ValueError('Existing attempt needs diagnosed --resume')
    identity=process_identity();atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=identity,resumed=resume))
    record_event(dict(event='E_target_native_audit_started'),run=dict(run_id=JOB,status='running',stage='E'))
    pin=sha256(DIRECTORY/'launch.json');inventory=read_json(INVENTORY);cache=ROOT/launch['fingerprint_cache'];cache.mkdir(exist_ok=True)
    (DIRECTORY/'records').mkdir(exist_ok=True)
    def progress(**extra):
        state=dict(updated_at=now(),process_identity=identity,**extra);atomic_json(DIRECTORY/'status.json',state);print(json.dumps(state),flush=True)
    start=time.monotonic();checks=[]
    for item in inventory['candidates']:
        file=item['files'][0];sid=item['candidate_id'];path=ROOT/file['path'];saved=DIRECTORY/'records'/(sid+'.json')
        if saved.exists():
            check=read_json(saved)
            if check['config_sha256']!=pin or check['audio_sha256']!=item['audio_sha256']:raise ValueError('Receipt identity changed')
            verify_hashes(check['diagnostic_files'])
            if check.get('fingerprint'):reference_values(check)
        else:
            check=dict(song_id=sid,audio_path=file['path'],audio_sha256=item['audio_sha256'],config_sha256=pin,passed=False)
            logs=DIRECTORY/'native_logs'/sid
            try:
                native,tags=local_native(path,file['header_duration_seconds'],logs)
                fp,_=fingerprint(path,item['audio_sha256'],cache)
                check.update(passed=True,native_audio=native,metadata=tags,fingerprint=fp)
            except Exception as exc:
                check['error']=f'{type(exc).__name__}: {exc}'
            check['diagnostic_files']={rel(p):sha256(p) for p in logs.glob('*.log')}
            atomic_json(saved,check)
        checks.append(check)
        if len(checks)%6==0 or len(checks)==len(inventory['candidates']):
            progress(stage='native_audio_and_fingerprint',done=len(checks),total=len(inventory['candidates']),passed=sum(r['passed'] for r in checks))
    main_rows=[json.loads(line) for line in MAIN.read_text(encoding='utf-8').splitlines()]
    main=read_json(MERGE/'main-fingerprint-receipts.json')
    if {r['song_id']:r['audio_sha256'] for r in main}!={r['song_id']:r['audio_sha256'] for r in main_rows}:
        raise ValueError('Main identity reference membership changed')
    references=main+read_json(MERGE/'audio-inventory.json')+read_json(JAMENDO)
    if len(references)!=8851:raise ValueError('Reference identity count changed')
    index=MatchIndex();index_ids=[]
    for n,row in enumerate(references,1):
        if row.get('fingerprint'):
            index.add(reference_values(row));index_ids.append(row['song_id'])
        if n%2000==0 or n==len(references):progress(stage='reference_fingerprint_index',done=n,total=len(references),fingerprints=len(index_ids))
    reference_fingerprints=len(index_ids);links=[]
    for n,row in enumerate(checks,1):
        if row['passed']:
            values=reference_values(row)
            for other,details in index.matches(values).items():
                links.append(dict(song_id=row['song_id'],matched_song_id=index_ids[other],**details))
            index.add(values);index_ids.append(row['song_id'])
        if n%6==0 or n==len(checks):progress(stage='recording_overlap_screen',done=n,total=len(checks),links=len(links))
    pool=screen(checks,references,links,main_rows)
    admitted=[r for r in pool if r['annotation_candidate']]
    artist_groups={tuple(r['provisional_artist_recording_group']) for r in admitted}
    atomic_json(DIRECTORY/'recording-links.json',links);atomic_json(DIRECTORY/'candidate-audit.json',pool)
    verify_hashes(launch['protected_hashes'])
    report=dict(job_id=JOB,status='completed_native_identity_audit',completed_at=now(),seconds=time.monotonic()-start,
        files=39,unique_file_hashes=len(checks),native_passed=sum(r['passed'] for r in checks),
        failures=[dict(song_id=r['song_id'],audio_path=r['audio_path'],error=r['error']) for r in checks if not r['passed']],
        annotation_candidates=len(admitted),provisional_artist_recording_groups=len(artist_groups),
        candidate_artist_metadata_missing=sum(not r['normalized_artist_tokens'] for r in admitted),
        minimum_additional_songs=max(0,50-len(admitted)),reference_identities=len(references),reference_fingerprints=reference_fingerprints,
        acoustic_links=len(links),dispositions=dict(Counter(r['disposition'] for r in pool)),
        training_admitted=False,pseudo_labels_generated=False,benchmark_targets_read=False,
        protected_models_and_current_annotations_unchanged=True,limitations=launch['limitations'],
        outputs={rel(p):sha256(p) for p in (DIRECTORY/'recording-links.json',DIRECTORY/'candidate-audit.json')},
        next_action='Resolve quarantined/duplicate/unknown-artist identities, obtain additional target songs and real human annotation. Independently prepare reference inference adapter and benchmark without final-test labels.')
    atomic_json(DIRECTORY/'report.json',report)
    stages=read_json(PLAN)['stages'];stages['E']['target_native_audit'].update(status=report['status'],report=rel(DIRECTORY/'report.json'))
    stages['E'].update(phase='target_native_audit_complete_human_annotation_pending',audited_annotation_candidates=len(admitted),
        audited_minimum_additional_songs=report['minimum_additional_songs'])
    record_event(dict(event='E_target_native_audit_completed',candidates=len(admitted),additional_songs=report['minimum_additional_songs']),
        run=dict(run_id=JOB,status='completed',stage='E',report=rel(DIRECTORY/'report.json')),
        updates=dict(stages=stages,next_action=report['next_action']))
    progress(stage=report['status'],annotation_candidates=len(admitted),minimum_additional_songs=report['minimum_additional_songs'])


def status():
    record=read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity=record.get('process_identity');lease=is_running(JOB)
    report=read_json(DIRECTORY/'report.json') if (DIRECTORY/'report.json').exists() else {}
    state=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else {}
    tail=[];log=DIRECTORY/'console.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-12000));tail=stream.read().decode('utf-8','replace').splitlines()[-3:]
    print(json.dumps(dict(job_id=JOB,checked_at=now(),os_lease_active=lease,
        process_identity_matches=bool(identity and process_identity(identity['pid'])==identity),
        native_receipts=len(list((DIRECTORY/'records').glob('*.json'))),progress=state,
        outcome='active' if lease else report.get('status','inactive_requires_inspection'),bounded_log_tail=tail)))


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--status',action='store_true')
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.status:status();return
    with run_lease(JOB):
        if args.register:register();return
        with (DIRECTORY/'console.log').open('a',encoding='utf-8',buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                try:run(args.resume)
                except BaseException as exc:
                    traceback.print_exc();error=f'{type(exc).__name__}: {exc}'
                    atomic_json(DIRECTORY/'failure.json',dict(failed_at=now(),error=error))
                    record_event(dict(event='E_target_native_audit_failed',error=error),run=dict(run_id=JOB,status='failed',stage='E'));raise


if __name__=='__main__':main()
