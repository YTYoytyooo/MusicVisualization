"""CPU-only audit of the frozen 60-song candidate pool; no model admission."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import soundfile as sf

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, run_lease, lock_path
from tools.train_next_v2.common import ROOT, BASE, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.research_inference.benchmark import process_alive
from tools.train_next_v2.external_audio.audit import references, MAIN
from tools.train_next_v2.annotation_pilot.audit import screen
from tools.train_next_v2.auxiliary_data.merge_audio_audit import fingerprint
from tools.train_next_v2.jamendo_audit_v2.run import reference_values
from tools.train_next_v2.jamendo_audit_v2.native import FFMPEG, read_unseeked_block
from tools.train_next_v2.certified_intake.contract import verify_native
from tools.datasets.index_supplemental import FPCALC, MatchIndex

JOB='e-fabio-pilot-native-60-v1'
DIRECTORY=BASE/'annotation_pilot'/JOB
CANDIDATES=BASE/'annotation_pilot/e-fabio-pilot-candidates-60-v1'
DIAG=BASE/'annotation_pilot/e-fabio-clock-diagnosis-v1'


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def status():
    p=read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity=p.get('process_identity');log=DIRECTORY/'console.log'
    return dict(job_id=JOB,checked_at=now(),process_identity=identity,
        process_alive=bool(identity and process_alive(identity)),os_lease_active=is_running(JOB),
        receipts=len(list((DIRECTORY/'records').glob('*.json'))),
        progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None,
        bounded_log_tail=log.read_text(encoding='utf-8').splitlines()[-2:] if log.exists() else [])


def state(stage,**extra):
    row=dict(job_id=JOB,stage=stage,updated_at=now(),**extra)
    atomic_json(DIRECTORY/'status.json',row);line=json.dumps(row,ensure_ascii=False)
    with (DIRECTORY/'console.log').open('a',encoding='utf-8') as out: out.write(line+'\n')
    print(line,flush=True)


def prepare():
    if DIRECTORY.exists(): raise ValueError('Preserve registered configuration')
    previous=read_json(CANDIDATES/'launch.json')
    pins=dict(previous['protected_hashes'])
    oldaudit=read_json(BASE/'annotation_pilot/e-fabio-nine-native-audit-v1/launch.json')
    pins.update(oldaudit['protected_hashes'])
    for p in list(Path(__file__).parent.glob('*.py'))+[CANDIDATES/'manifest.json',CANDIDATES/'report.json',CANDIDATES/'verification.json',
            ROOT/'tools/train_next_v2/auxiliary_data/merge_candidates.py',ROOT/'tools/train_next_v2/certified_intake/contract.py']:
        pins[rel(p)]=sha256(p)
    certs=read_json(DIAG/'report.json')['evidence']
    pins.update({r['path']:r['sha256'] for r in certs})
    verify_hashes(pins)
    argv=[previous['python'],'-u','-m','tools.train_next_v2.pilot_native.job','--launch',str(DIRECTORY/'launch.json')]
    config=dict(job_id=JOB,registered_at=now(),original_argv=argv,resume_argv=argv+['--resume'],
        verification_argv=[previous['python'],'-u','-m','tools.train_next_v2.pilot_native.job','--verify'],
        cwd=str(ROOT),python=previous['python'],environment=previous['environment'],protected_hashes=pins,
        candidates=rel(CANDIDATES/'manifest.json'),certificates=certs,
        fingerprint_cache=oldaudit['fingerprint_cache'],reference_identities=8887,
        native_contract='Reuse nine SHA-bound certificates with fresh native PCM verification. For other 51, require finite bulk/EOF-stream float32 equality, strict independent FFmpeg EOF count equality, non-silent PCM and unchanged source hash. Header counts recorded separately, never used to scale time. Decode disagreements stay quarantined.',
        dedup_contract='Frozen offset-aware Chromaprint screen, exact bytes and native PCM identity with sample rate/channels, against existing references and within the pool. Missing fingerprints fail candidate screening; no exhaustive cover/alias exclusion is claimed.',
        worker_count=1,gpu_used=False,log=rel(DIRECTORY/'console.log'),runtime_lock=str(lock_path(JOB)),
        recovery='After PID creation identity is dead and OS lease free, inspect failure, preserve launch and per-song receipts, run resume_argv. Source/diagnostic hashes must match; do not override failed native gates.',
        scope='Native timing and recording identity only. Do not read final-test target labels, train, extract learned features, generate labels, or assign fitting/evaluation roles.')
    DIRECTORY.mkdir(parents=True);atomic_json(DIRECTORY/'launch.json',config)
    record_event(dict(event='E_pilot_native_registered',job_id=JOB,songs=60),run=dict(run_id=JOB,stage='E',status='registered',launch=rel(DIRECTORY/'launch.json'),launch_sha256=sha256(DIRECTORY/'launch.json'),original_argv=argv,cwd=str(ROOT),runtime_lock=str(lock_path(JOB)),log=rel(DIRECTORY/'console.log')))
    print(json.dumps(dict(job_id=JOB,pinned_inputs=len(pins))))


def decode_eof(path,expected_hash,logdir):
    if sha256(path)!=expected_hash: raise ValueError('Source bytes changed')
    samples,rate=sf.read(path,dtype='float32',always_2d=True)
    frames,channels=samples.shape
    if not 8000<=rate<=192000 or not 1<=channels<=8 or frames==0 or not np.isfinite(samples).all(): raise ValueError('Invalid native samples/geometry')
    digest=hashlib.sha256(samples.tobytes(order='C')).hexdigest()
    rms=float(np.sqrt(np.mean(np.square(samples,dtype=np.float64))))
    if rms<=1e-7: raise ValueError('Silent recording')
    del samples
    h=hashlib.sha256();count=0
    with sf.SoundFile(path) as stream:
        header_frames=stream.frames
        while True:
            block=read_unseeked_block(stream,65536)
            if not len(block): break
            if not np.isfinite(block).all(): raise ValueError('Non-finite streaming PCM')
            h.update(block.tobytes(order='C'));count+=len(block)
    if count!=frames or h.hexdigest()!=digest: raise ValueError('Bulk and native EOF stream disagree')
    argv=[str(FFMPEG),'-nostdin','-hide_banner','-v','error','-xerror','-err_detect','explode','-threads','1','-i',str(path),'-map','0:a:0','-f','f32le','-acodec','pcm_f32le','-']
    ff=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    logdir.mkdir(parents=True,exist_ok=True)
    (logdir/'ffmpeg-stderr.log').write_bytes(ff.stderr)
    diagnostics=dict(argv=argv,returncode=ff.returncode,bytes=len(ff.stdout),stderr_empty=not bool(ff.stderr.strip()))
    atomic_json(logdir/'decode.json',diagnostics)
    if ff.returncode or ff.stderr.strip() or len(ff.stdout)%(4*channels): raise ValueError('Strict independent decoder failed')
    ff_frames=len(ff.stdout)//(4*channels)
    if ff_frames!=frames: raise ValueError(f'Independent native EOF counts disagree: {frames} vs {ff_frames}')
    if not np.isfinite(np.frombuffer(ff.stdout,dtype='<f4')).all(): raise ValueError('Non-finite independent decoder PCM')
    ffhash=hashlib.sha256(ff.stdout).hexdigest();del ff
    if sha256(path)!=expected_hash: raise ValueError('Source changed while decoding')
    return dict(frames=frames,sample_rate=rate,channels=channels,duration_seconds=frames/rate,
        float32_sample_sha256=digest,stream_float32_sha256=h.hexdigest(),stream_frames=count,
        ffmpeg_frames=ff_frames,ffmpeg_float32_sha256=ffhash,header_frames=header_frames,
        header_minus_actual_frames=header_frames-frames,rms=rms,bulk_stream_exact=True,
        independent_decoder_count_equal=True,local_time_origin_seconds=0,source_song_offset_seconds=None,
        original_song_completeness_verified=False,header_gate_overridden=False,
        timing_contract='separate_native_eof_v1',timeline_scaled=False)


def reused_native(path,cert):
    n=verify_native(path,cert)
    return dict(frames=n['frames'],sample_rate=n['sample_rate'],channels=n['channels'],duration_seconds=n['duration_seconds'],
        float32_sample_sha256=n['float32_sha256'],stream_float32_sha256=n['float32_sha256'],stream_frames=n['frames'],
        ffmpeg_frames=cert['ffmpeg_frames'],ffmpeg_float32_sha256=cert['ffmpeg_float32_sha256'],header_frames=n['header_frames'],
        header_minus_actual_frames=n['header_frames']-n['frames'],bulk_stream_exact=True,
        independent_decoder_count_equal=True,local_time_origin_seconds=0,source_song_offset_seconds=None,
        original_song_completeness_verified=False,header_gate_overridden=False,timing_contract='separate_native_eof_v1',timeline_scaled=False)


def pcm_key(row):
    n=row.get('native_audio') or {}
    if all(k in n for k in ('sample_rate','channels','float32_sample_sha256')):
        return (n['sample_rate'],n['channels'],n['float32_sample_sha256'])
    return None


def identities(checks):
    refs=references();index=MatchIndex();indexed=[];links=[];pcm={}
    for row in refs:
        if row.get('fingerprint'): index.add(reference_values(row));indexed.append(row['song_id'])
        key=pcm_key(row)
        if key: pcm.setdefault(key,[]).append(row['song_id'])
    reference_fingerprints=len(indexed)
    for row in checks:
        if row.get('fingerprint'):
            values=reference_values(row)
            links += [dict(song_id=row['song_id'],matched_song_id=indexed[i],kind='chromaprint',**detail) for i,detail in index.matches(values).items()]
            index.add(values);indexed.append(row['song_id'])
        key=pcm_key(row)
        if key:
            links += [dict(song_id=row['song_id'],matched_song_id=sid,kind='exact_native_pcm') for sid in pcm.get(key,[])]
            pcm.setdefault(key,[]).append(row['song_id'])
    # Only artist/identity metadata fields are used; no final-test labels or predictions.
    main_rows=[{k:r.get(k,'') for k in ('song_id','artist')} for r in (json.loads(line) for line in MAIN.read_text(encoding='utf-8').splitlines())]
    screened=screen(checks,refs,links,main_rows)
    by_id={r['song_id']:r for r in checks}
    for row in screened:
        if not by_id[row['song_id']].get('fingerprint'):
            row.update(annotation_candidate=False,disposition='fingerprint_unavailable_quarantined')
        row.update(human_quality_review_performed=False,formal_evaluation_admitted=False,
            browser_decoder_alignment_verified=False,original_song_completeness_verified=False)
    return screened,links,reference_fingerprints


def run(config):
    start=time.monotonic();chash=sha256(DIRECTORY/'launch.json');checks=[]
    certificates={read_json(ROOT/r['path'])['audio_sha256']:r for r in config['certificates']}
    cache=ROOT/config['fingerprint_cache'];cache.mkdir(parents=True,exist_ok=True)
    (DIRECTORY/'records').mkdir(exist_ok=True)
    for candidate in read_json(ROOT/config['candidates'])['candidates']:
        choice=candidate['selection'];digest=choice['audio_sha256'];path=ROOT/choice['audio_path'];saved=DIRECTORY/'records'/(digest+'.json')
        if saved.exists():
            row=read_json(saved)
            if row['config_sha256']!=chash or row['audio_sha256']!=digest: raise ValueError('Resume receipt mismatch')
            verify_hashes(row['diagnostic_files'])
            if row.get('fingerprint'): reference_values(row)
        else:
            logs=DIRECTORY/'native_logs'/digest
            row=dict(song_id='fabio-pilot:'+digest,audio_sha256=digest,audio_path=choice['audio_path'],
                annotation_song_ids=choice['annotation_song_ids'],config_sha256=chash,passed=False,
                metadata=candidate['embedded_metadata'],training_admitted=False)
            try:
                if digest in certificates:
                    row['reused_certificate']=certificates[digest]
                    native=reused_native(path,read_json(ROOT/certificates[digest]['path']))
                else: native=decode_eof(path,digest,logs)
                row.update(passed=True,native_audio=native)
            except Exception as exc: row['native_error']=f'{type(exc).__name__}: {exc}'
            try: row['fingerprint'],_=fingerprint(path,digest,cache)
            except Exception as exc: row['fingerprint_error']=f'{type(exc).__name__}: {exc}'
            row['diagnostic_files']={rel(p):sha256(p) for p in logs.glob('*') if p.is_file()}
            atomic_json(saved,row)
        checks.append(row);state('native_and_fingerprint',done=len(checks),total=60,passed=sum(r['passed'] for r in checks))
    state('screening_recording_identities',references=8887)
    screened,links,reference_count=identities(checks)
    atomic_json(DIRECTORY/'screened-candidates.json',screened);atomic_json(DIRECTORY/'recording-links.json',links)
    verify_hashes(config['protected_hashes'])
    outputs={rel(p):sha256(p) for p in list((DIRECTORY/'records').glob('*.json'))+[DIRECTORY/'screened-candidates.json',DIRECTORY/'recording-links.json']}
    report=dict(job_id=JOB,status='completed_native_identity_audit_pending_verification',completed_at=now(),seconds=time.monotonic()-start,
        songs=60,native_passed=sum(r['passed'] for r in checks),reused_certificates=sum('reused_certificate' in r for r in checks),
        native_failures=[dict(song_id=r['song_id'],error=r.get('native_error')) for r in checks if not r['passed']],
        candidate_fingerprints=sum(bool(r.get('fingerprint')) for r in checks),reference_identities=8887,reference_fingerprints=reference_count,
        identity_links=len(links),dispositions=dict(Counter(r['disposition'] for r in screened)),
        annotation_candidates=sum(r['annotation_candidate'] for r in screened),outputs=outputs,
        training_admitted=False,formal_evaluation_admitted=False,final_test_targets_evaluated=0,
        limitations=['Missing reference fingerprints remain a coverage limitation; no exhaustive identity exclusion.',
            'Unknown artists remain conservatively grouped; aliases/full-song completeness require actual review.',
            'Header estimates do not define the audio clock; prior failed audits are retained.',
            'No human review, completion flags, labels, fitted features or model results are generated.'])
    atomic_json(DIRECTORY/'report.json',report);state(report['status'],native_passed=report['native_passed'],annotation_candidates=report['annotation_candidates'])
    record_event(dict(event='E_pilot_native_completed',job_id=JOB,native_passed=report['native_passed'],annotation_candidates=report['annotation_candidates']),run=dict(run_id=JOB,stage='E',status='completed_pending_verification',report=rel(DIRECTORY/'report.json')))


def verify(config):
    report=read_json(DIRECTORY/'report.json');verify_hashes(report['outputs'])
    rows=[read_json(p) for p in sorted((DIRECTORY/'records').glob('*.json'))]
    expected={r['selection']['audio_sha256'] for r in read_json(ROOT/config['candidates'])['candidates']}
    if len(rows)!=60 or {r['audio_sha256'] for r in rows}!=expected: raise ValueError('Incomplete receipt coverage')
    for row in rows:
        if row['config_sha256']!=sha256(DIRECTORY/'launch.json'): raise ValueError('Receipt config changed')
        verify_hashes(row['diagnostic_files'])
        if row.get('fingerprint'): reference_values(row)
        if row['passed']:
            n=row['native_audio']
            if not n['frames']==n['stream_frames']==n['ffmpeg_frames'] or n['float32_sample_sha256']!=n['stream_float32_sha256']: raise ValueError('PCM/count disagreement')
            if n['duration_seconds']!=n['frames']/n['sample_rate'] or n['timeline_scaled']: raise ValueError('Time axis altered')
    saved=read_json(DIRECTORY/'screened-candidates.json')
    if any(r['training_admitted'] or r['formal_evaluation_admitted'] or r['final_split_assigned'] for r in saved): raise ValueError('Unjustified admission')
    if sum(r['passed'] for r in rows)!=report['native_passed'] or sum(r['annotation_candidate'] for r in saved)!=report['annotation_candidates']: raise ValueError('Summary mismatch')
    result=dict(status='passed',verified_at=now(),songs=60,report_sha256=sha256(DIRECTORY/'report.json'),
        native_passed=report['native_passed'],annotation_candidates=report['annotation_candidates'],
        failed_checks_preserved=True,training_admitted=False,formal_evaluation_admitted=False)
    atomic_json(DIRECTORY/'verification.json',result);state('completed_verified',native_passed=report['native_passed'])
    record_event(dict(event='E_pilot_native_verified',job_id=JOB,**result),run=dict(run_id=JOB,stage='E',status='completed_verified',verification=rel(DIRECTORY/'verification.json')))
    print(json.dumps(result))


def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--launch',type=Path);g.add_argument('--verify',action='store_true');g.add_argument('--status',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.prepare: prepare();return
    if args.status: print(json.dumps(status()));return
    config=read_json(DIRECTORY/'launch.json');verify_hashes(config['protected_hashes'])
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(config['python']).resolve(): raise ValueError('Wrong cwd/runtime')
    if any(os.environ.get(k)!=v for k,v in config['environment'].items()): raise ValueError('Environment changed')
    previous=status()
    if previous['process_alive'] or previous['os_lease_active']: raise ValueError('Job still active')
    if args.launch and args.launch.resolve()!=(DIRECTORY/'launch.json').resolve(): raise ValueError('Wrong launch')
    if args.launch and (DIRECTORY/'report.json').exists(): raise ValueError('Completed run: verify only')
    if args.launch and (DIRECTORY/'process.json').exists() and not args.resume: raise ValueError('Inspect prior attempt and use --resume')
    with run_lease(JOB):
        atomic_json(DIRECTORY/'process.json',dict(process_identity=process_identity(),argv=sys.argv,started_at=now()))
        record_event(dict(event='E_pilot_native_started',job_id=JOB,verification=args.verify,process_identity=process_identity()),run=dict(run_id=JOB,stage='E',status='running'))
        try:
            if args.verify: verify(config)
            else: run(config)
        except BaseException:
            state('failed',error=traceback.format_exc())
            record_event(dict(event='E_pilot_native_failed',job_id=JOB),run=dict(run_id=JOB,stage='E',status='failed'))
            raise


if __name__=='__main__': main()
