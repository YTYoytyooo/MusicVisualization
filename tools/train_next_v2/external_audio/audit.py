"""Bind nine real annotations to exact audio and audit native clocks, without fitting."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import soundfile as sf

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, is_running, lock_path
from tools.train_next_v2.common import ROOT, BASE, PLAN, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.research_inference.benchmark import process_alive
from tools.train_next_v2.run import Tee
from tools.train_next_v2.annotation_pilot.audit import local_native, artist_tokens
from tools.train_next_v2.auxiliary_data.merge_audio_audit import fingerprint
from tools.train_next_v2.jamendo_audit_v2.native import FFMPEG
from tools.train_next_v2.jamendo_audit_v2.run import reference_values
from tools.datasets.index_supplemental import FPCALC, MatchIndex

JOB = 'e-fabio-nine-native-audit-v1'
DIRECTORY = BASE/'annotation_pilot'/JOB
ORGANIZED = BASE/'annotation_pilot/e-fabio-audio-organize-20261005-v1'
INTAKE = BASE/'annotation_pilot/e-fabio-intake-f26ea031b964-v1'
LOCAL = BASE/'annotation_pilot/e-target-native-audit-v1'
MERGE = ROOT/'data/datasets/merge/prepared/audio-audit-v1'
JAMENDO = ROOT/'data/datasets/mtg_jamendo/prepared/native-audit-v2/checks.json'
MAIN = ROOT/'data/datasets/combined/prepared/v1/manifest.jsonl'
AXES = ('valence', 'arousal')


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def state(stage, **values):
    value=dict(job_id=JOB, stage=stage, updated_at=now(), **values)
    atomic_json(DIRECTORY/'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def references():
    rows=read_json(MERGE/'main-fingerprint-receipts.json')+read_json(MERGE/'audio-inventory.json')+read_json(JAMENDO)
    rows += [read_json(p) for p in sorted((LOCAL/'records').glob('*.json'))]
    if len(rows)!=8887 or len({r['song_id'] for r in rows})!=8887:
        raise ValueError('Frozen reference identity membership changed')
    return rows


def prepare():
    if (DIRECTORY/'launch.json').exists(): raise ValueError('Preserve existing audit configuration')
    movement=read_json(ORGANIZED/'report.json')
    if movement['status']!='completed_files_verified' or movement['source_file_count']!=422:
        raise ValueError('Finish verified folder organization first')
    source=INTAKE/'sources/continuous-fabio-json.json'; document=read_json(source)
    organization=read_json(ORGANIZED/'launch.json'); binding={r['song_id']:r for r in organization['annotation_matches']}
    songs=[]
    for annotation in document['songs']:
        src=annotation['source']; row=binding[src['song_id']]
        if len(row['matching_paths'])!=1 or src['audio_sha256']!=row['audio_sha256']:
            raise ValueError('Exact unique annotation/audio binding required')
        path=ROOT/row['matching_paths'][0]; info=sf.info(path)
        if sha256(path)!=src['audio_sha256']: raise ValueError('Audio identity changed')
        songs.append(dict(song_id='fabio:'+src['song_id'], annotation_song_id=src['song_id'], audio_path=rel(path),
            audio_sha256=src['audio_sha256'], browser_duration_seconds=annotation['duration'],
            header_duration_seconds=info.frames/info.samplerate, header_frames=info.frames,
            sample_rate=info.samplerate, channels=info.channels))
    paths=[Path(__file__),Path(__file__).with_name('__init__.py'),ORGANIZED/'launch.json',ORGANIZED/'report.json',
        INTAKE/'intake-manifest.json',INTAKE/'verification.json',source,MAIN,JAMENDO,
        MERGE/'main-fingerprint-receipts.json',MERGE/'audio-inventory.json',FPCALC,FFMPEG,
        Path(sf.__file__)]+list((Path(sf.__file__).parent/'_soundfile_data').glob('*.dll'))
    paths += [ROOT/name for name in ('tools/train_next_v2/annotation_pilot/audit.py','tools/train_next_v2/auxiliary_data/merge_audio_audit.py',
        'tools/train_next_v2/jamendo_audit_v2/native.py','tools/train_next_v2/jamendo_audit_v2/run.py',
        'tools/datasets/index_supplemental.py','tools/train_next/core.py','tools/train_next/runtime.py',
        'tools/train_next_v2/common.py','tools/train_next_v2/run.py','tools/train_next_v2/research_inference/benchmark.py')]
    paths += list((LOCAL/'records').glob('*.json'))
    pins={rel(p):sha256(p) for p in paths}
    for row in songs:pins[row['audio_path']]=row['audio_sha256']
    for row in references():
        if row.get('fingerprint'):pins[row['fingerprint']['path']]=row['fingerprint']['sha256']
    previous=read_json(BASE/'comparison_jobs/f-validation-v2-mixed-b01-cpu-v1/launch.json')
    pins.update({k:v for k,v in previous['pinned_hashes'].items() if k.startswith('models/')})
    verify_hashes(pins)
    environment={**previous['environment'],'CUDA_VISIBLE_DEVICES':'-1'}
    argv=[read_json(PLAN)['python'],'-u','-m',__package__+'.audit','--launch',str(DIRECTORY/'launch.json')]
    launch=dict(job_id=JOB,registered_at=now(),original_argv=argv,resume_argv=argv+['--resume'],
        verification_argv=argv+['--verify'],cwd=str(ROOT),python=argv[0],environment=environment,
        protected_hashes=pins,songs=songs,annotation_snapshot=rel(source),worker_count=1,gpu_used=False,
        native_policy='Existing strict FFmpeg + bulk/stream float32 parity and header frame gates, unchanged. Browser/header differences are evidence, never a time-warp instruction.',
        time_policy='Exact source file identity; retain independent V/A raw timestamps and takes. Check bounds against native samples only after native gate passes. No duration scaling, inferred delay or smoothing.',
        dedup_policy='Frozen offset-aware Chromaprint, exact SHA256 and available PCM identities against 8887 main/auxiliary/prior target recordings plus the nine candidates; metadata artists are provisional.',
        no_admission='Human completion/quality/independent review and frozen group allocation remain required. This job performs no model selection, fitting, pseudo-labeling or final-test target evaluation.',
        os_lock=str(lock_path(JOB)),log=rel(DIRECTORY/'console.log'),
        fingerprint_cache=rel(ROOT/'data/datasets/.fingerprints'/sha256(FPCALC)[:16]),
        library_versions={k:importlib.metadata.version(k) for k in ('numpy','soundfile')},
        recovery='Confirm recorded PID is dead and OS lease is free, diagnose failure, then exact argv plus --resume; validate committed per-song receipts and diagnostics. Do not relax failed native or timing checks.')
    DIRECTORY.mkdir(parents=True,exist_ok=True);atomic_json(DIRECTORY/'launch.json',launch)
    stages=read_json(PLAN)['stages']
    stages['E'].update(phase='exact_audio_received_native_clock_check_pending',external_audio_audit=dict(job_id=JOB,status='registered',launch=rel(DIRECTORY/'launch.json')))
    record_event(dict(event='E_fabio_native_audit_registered',job_id=JOB,songs=9,reference_identities=8887),
        run=dict(run_id=JOB,stage='E',status='registered',launch=rel(DIRECTORY/'launch.json')),updates=dict(stages=stages))
    print(json.dumps(dict(job_id=JOB,registered=True,pinned_inputs=len(pins))))


def annotation_bounds(annotation, native):
    output={}
    for axis in AXES:
        takes=[]
        for take in annotation['takes']:
            if axis not in take['axes']:continue
            points=[p for p in take['points'] if p.get(axis) is not None and p.get('status')=='annotated']
            t=np.asarray([p['time'] for p in points],dtype=np.float64)
            if not np.isfinite(t).all() or (len(t)>1 and np.any(np.diff(t)<=0)):raise ValueError('Invalid native annotation clock')
            in_bounds=None if native is None else bool(np.all(t>=0) and np.all(t<=native['duration_seconds']+1/native['sample_rate']))
            takes.append(dict(take_id=take['id'],points=len(points),start=float(t[0]) if len(t) else None,
                end=float(t[-1]) if len(t) else None,within_decoded_audio=in_bounds))
        output[axis]=dict(takes=takes,declared_completed=annotation['completed'][axis])
    return output


def run(launch,resume):
    config_hash=sha256(DIRECTORY/'launch.json');start=time.monotonic()
    annotations={s['source']['song_id']:s for s in read_json(ROOT/launch['annotation_snapshot'])['songs']}
    cache=ROOT/launch['fingerprint_cache'];cache.mkdir(parents=True,exist_ok=True)
    (DIRECTORY/'records').mkdir(exist_ok=True);checks=[]
    for item in launch['songs']:
        saved=DIRECTORY/'records'/(item['annotation_song_id']+'.json')
        if saved.exists():
            row=read_json(saved)
            if row['config_sha256']!=config_hash or row['audio_sha256']!=item['audio_sha256']:raise ValueError('Receipt mismatch')
            verify_hashes(row['diagnostic_files'])
            if row.get('fingerprint'):reference_values(row)
        else:
            row=dict(**item,config_sha256=config_hash,native_passed=False)
            path=ROOT/item['audio_path'];logs=DIRECTORY/'native_logs'/item['annotation_song_id']
            try:
                native,tags=local_native(path,item['header_duration_seconds'],logs)
                row.update(native_passed=True,native_audio=native,metadata=tags,
                    decoded_minus_browser_seconds=native['duration_seconds']-item['browser_duration_seconds'])
            except Exception as exc:row['native_error']=f'{type(exc).__name__}: {exc}'
            # An identity screen can still be useful for quarantined audio; it never overrides the failed native gate.
            try:row['fingerprint'],_=fingerprint(path,item['audio_sha256'],cache)
            except Exception as exc:row['fingerprint_error']=f'{type(exc).__name__}: {exc}'
            row['annotation_bounds']=annotation_bounds(annotations[item['annotation_song_id']],row.get('native_audio'))
            row['diagnostic_files']={rel(p):sha256(p) for p in logs.glob('*.log')}
            atomic_json(saved,row)
        checks.append(row);state('native_clock_and_fingerprint',done=len(checks),total=9,native_passed=sum(r['native_passed'] for r in checks))
    refs=references(); index=MatchIndex();indexed=[]
    for n,row in enumerate(refs,1):
        if row.get('fingerprint'):index.add(reference_values(row));indexed.append(row['song_id'])
        if n%2000==0:state('reference_index',done=n,total=len(refs))
    reference_fp_count=len(indexed);links=[]
    for row in checks:
        if row.get('fingerprint'):
            values=reference_values(row)
            links += [dict(song_id=row['song_id'],matched_song_id=indexed[i],**detail) for i,detail in index.matches(values).items()]
            index.add(values);indexed.append(row['song_id'])
        state('overlap_screen',song_id=row['song_id'],links=len(links))
    main_rows=[json.loads(line) for line in MAIN.read_text('utf-8').splitlines()]
    pool=[]
    for row in checks:
        native=row.get('native_audio');tokens=artist_tokens(row.get('metadata',{}).get('artist',''))
        same_bytes=[r['song_id'] for r in refs if r['audio_sha256']==row['audio_sha256']]
        same_pcm=[r['song_id'] for r in refs if native and (r.get('native_audio') or {}).get('float32_sample_sha256')==native['float32_sample_sha256']]
        linked=[r for r in links if row['song_id'] in (r['song_id'],r['matched_song_id'])]
        pool.append(dict(song_id=row['song_id'],annotation_song_id=row['annotation_song_id'],audio_path=row['audio_path'],
            audio_sha256=row['audio_sha256'],exact_annotation_audio_identity=True,native_passed=row['native_passed'],
            native_error=row.get('native_error'),browser_duration_seconds=row['browser_duration_seconds'],header_duration_seconds=row['header_duration_seconds'],
            native_audio=native,annotation_bounds=row['annotation_bounds'],exact_reference_matches=same_bytes,
            pcm_reference_matches=same_pcm,acoustic_links=linked,metadata=row.get('metadata'),artist_tokens=tokens,
            known_main_artist_ids=sorted({r['song_id'] for r in main_rows if set(artist_tokens(r.get('artist','')))&set(tokens)}),
            artist_review_required=True,human_quality_review_performed=False,browser_decoder_alignment_verified=False,
            dataset_role='unassigned',training_admitted=False,formal_evaluation_admitted=False))
    atomic_json(DIRECTORY/'recording-links.json',links);atomic_json(DIRECTORY/'audio-label-binding.json',pool)
    verify_hashes(launch['protected_hashes'])
    report=dict(job_id=JOB,status='completed_identity_screen_pending_human_and_clock_review',completed_at=now(),seconds=time.monotonic()-start,
        exact_audio_bindings=len(checks),native_passed=sum(r['native_passed'] for r in checks),
        failures=[dict(song_id=r['song_id'],error=r.get('native_error')) for r in checks if not r['native_passed']],
        reference_identities=len(refs),reference_fingerprints=reference_fp_count,acoustic_links=len(links),
        outputs={rel(p):sha256(p) for p in (DIRECTORY/'recording-links.json',DIRECTORY/'audio-label-binding.json')},
        source_files_unchanged=True,protected_models_unchanged=True,training_admitted=False,formal_evaluation_admitted=False,
        final_test_targets_read=0,pseudo_labels_generated=False,
        limitations=['Source bytes match all nine exports; raw labels and original provenance are unchanged.',
            'Native decoder gates remain strict; failures require diagnosis before use.',
            'Browser/header duration discrepancy is not evidence for linear timeline scaling or a global delay correction.',
            'Screening cannot exclude every cover, short overlap or artist alias; five MERGE reference fingerprints are unavailable.',
            'Completion flags remain false; independent human review and fixed role allocation are pending.'])
    atomic_json(DIRECTORY/'report.json',report);state(report['status'],native_passed=report['native_passed'])
    record_event(dict(event='E_fabio_native_audit_completed',job_id=JOB,exact_audio_bindings=9,native_passed=report['native_passed'],acoustic_links=len(links)),
        run=dict(run_id=JOB,stage='E',status='completed_pending_verification',report=rel(DIRECTORY/'report.json')))


def verify(launch):
    report=read_json(DIRECTORY/'report.json');verify_hashes(report['outputs'])
    rows=[read_json(p) for p in sorted((DIRECTORY/'records').glob('*.json'))]
    if len(rows)!=9 or {r['annotation_song_id'] for r in rows}!={r['annotation_song_id'] for r in launch['songs']}:raise ValueError('Receipt coverage changed')
    for row in rows:
        verify_hashes(row['diagnostic_files'])
        if row['config_sha256']!=sha256(DIRECTORY/'launch.json'):raise ValueError('Receipt configuration changed')
        if row.get('fingerprint'):reference_values(row)
    if sum(r['native_passed'] for r in rows)!=report['native_passed']:raise ValueError('Outcome count mismatch')
    binding=read_json(DIRECTORY/'audio-label-binding.json')
    if any(r['training_admitted'] or r['formal_evaluation_admitted'] or r['dataset_role']!='unassigned' for r in binding):raise ValueError('Unexpected admission')
    result=dict(job_id=JOB,status='passed',verified_at=now(),report_sha256=sha256(DIRECTORY/'report.json'),
        launch_sha256=sha256(DIRECTORY/'launch.json'),nine_exact_identity_receipts=True,raw_sources_unchanged=True,
        native_failures_preserved=True,training_admitted=False,formal_evaluation_admitted=False,final_test_targets_read=0)
    atomic_json(DIRECTORY/'verification.json',result);state('completed_verified',native_passed=report['native_passed'])
    stages=read_json(PLAN)['stages']
    stages['E'].update(phase='exact_audio_bound_pending_clock_and_human_review',external_audio_audit=dict(job_id=JOB,status='completed_verified',
        launch=rel(DIRECTORY/'launch.json'),report=rel(DIRECTORY/'report.json'),verification=rel(DIRECTORY/'verification.json'),binding=rel(DIRECTORY/'audio-label-binding.json')))
    record_event(dict(event='E_fabio_native_audit_verified',**result),run=dict(run_id=JOB,stage='E',status='completed_verified'),updates=dict(stages=stages))


def status():
    process=read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity=process.get('process_identity')
    return dict(job_id=JOB,checked_at=now(),os_lease_active=is_running(JOB),process_identity=identity,
        process_alive=process_alive(identity) if identity else False,
        progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None,
        receipts=len(list((DIRECTORY/'records').glob('*.json'))))


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--status',action='store_true');group.add_argument('--launch',type=Path)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.status:print(json.dumps(status()));return
    if args.prepare:prepare();return
    if args.launch.resolve()!=(DIRECTORY/'launch.json').resolve():raise ValueError('Wrong launch file')
    launch=read_json(args.launch);verify_hashes(launch['protected_hashes'])
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(launch['python']).resolve():raise ValueError('Wrong cwd or runtime')
    if any(os.environ.get(k)!=v for k,v in launch['environment'].items()):raise ValueError('Frozen environment changed')
    if any(importlib.metadata.version(k)!=v for k,v in launch['library_versions'].items()):raise ValueError('Frozen library version changed')
    if (DIRECTORY/'verification.json').exists():raise ValueError('Completed audit must not rerun')
    if not args.verify and (DIRECTORY/'report.json').exists():raise ValueError('Audit finished; use verification')
    if not args.verify and (DIRECTORY/'process.json').exists() and not args.resume:raise ValueError('Diagnose prior attempt and use --resume')
    if status()['process_alive']:raise ValueError('Same job process still alive')
    with run_lease(JOB):
        atomic_json(DIRECTORY/'process.json',dict(process_identity=process_identity(),started_at=now(),argv=sys.argv,verification=args.verify))
        record_event(dict(event='E_fabio_native_audit_started',job_id=JOB,verification=args.verify,process_identity=process_identity()),run=dict(run_id=JOB,stage='E',status='running'))
        with (DIRECTORY/'console.log').open('a',encoding='utf-8',buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                try:
                    if args.verify:verify(launch)
                    else:run(launch,args.resume)
                except BaseException:
                    state('failed',error=traceback.format_exc())
                    record_event(dict(event='E_fabio_native_audit_failed',job_id=JOB),run=dict(run_id=JOB,stage='E',status='failed'))
                    raise


if __name__=='__main__':main()
