"""Versioned, resumable baseline features for the admitted full-track pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import soundfile as sf
import torch

from tools.train_next.core import atomic_json, sha256
from tools.train_next.features import FeatureStore, local_clap_snapshot
from tools.train_next.runtime import run_lease, lock_path, is_running
from tools.train_next_v2.common import (BASE, ROOT, PLAN, GPU_LEASE, now, read_json,
    record_event, verify_hashes, process_identity, assert_gpu_available)
from .contract import validate_candidates, feature_record, validate_feature_arrays
from .protocol import PROTOCOL, SOURCE, AUDIT, B_PROTOCOL, rel

JOB = 'd-jamendo-clap-features-v1'
DIRECTORY = BASE/'feature_jobs'/JOB
CACHE = BASE/'feature_caches'/JOB


def read_cached(row, cache, feature_key):
    """Require paired receipts; do not silently admit an orphaned cached shard."""
    record = feature_record(row)
    npz = Path(cache)/(record['audio_sha256']+'.npz')
    receipt = npz.with_suffix('.json')
    if not npz.exists() and not receipt.exists():
        return None
    if not npz.is_file() or not receipt.is_file():
        raise ValueError('Incomplete feature shard/receipt; preserve and diagnose before recovery')
    entry = read_json(receipt)
    if (entry['song_id'] != row['song_id'] or entry['audio_sha256'] != record['audio_sha256']
            or entry['feature_key'] != feature_key or entry['path'] != rel(npz)
            or entry['sha256'] != sha256(npz) or entry['native_duration_seconds'] != record['duration_seconds']
            or entry['native_float32_sha256'] != row['audio']['native_audio']['float32_sample_sha256']):
        raise ValueError('Cached feature receipt identity/hash differs')
    with np.load(npz, allow_pickle=False) as saved:
        if set(saved.files) != {'times', 'features', 'feature_key'} or str(saved['feature_key'].item()) != feature_key:
            raise ValueError('Cached features contain unexpected fields/targets or version')
        times, values = saved['times'], saved['features']
        validate_feature_arrays(times, values, record)
        if entry['frames'] != len(times) or entry['dimensions'] != 531:
            raise ValueError('Cached feature frame/dimension receipt differs')
    return dict(entry, receipt=rel(receipt), receipt_sha256=sha256(receipt))


def register():
    if DIRECTORY.exists() or CACHE.exists() or is_running(GPU_LEASE):
        raise RuntimeError('Preserve existing feature job; GPU must be idle for registration')
    assert_gpu_available()
    protocol, audit = read_json(PROTOCOL), read_json(AUDIT/'report.json')
    verify_hashes({rel(SOURCE):protocol['auxiliary_candidates_sha256'],rel(B_PROTOCOL):protocol['source_B_protocol_sha256'],
        rel(AUDIT/'report.json'):protocol['audio_audit_sha256']})
    verify_hashes(audit['outputs'])
    summary = validate_candidates(read_json(SOURCE))
    if summary != protocol['candidate_summary'] or audit['status'] != 'completed_native_audio_gate':
        raise ValueError('Pool is not the frozen audio-admitted pilot')
    store = FeatureStore(torch.device('cpu'))
    if store.key != protocol['feature_key'] or store.spec != protocol['feature_spec']:
        raise ValueError('Baseline representation changed')
    source_launch = read_json(AUDIT/'launch.json')
    protected = dict(source_launch['protected_hashes'])
    files = [Path(__file__), Path(__file__).parent/'contract.py', Path(__file__).parent/'protocol.py',
        Path(__file__).parent/'__init__.py',PROTOCOL,SOURCE,AUDIT/'report.json',
        ROOT/'tools/train_next/features.py',ROOT/'tools/train_next/core.py',ROOT/'tools/train_next/runtime.py',
        ROOT/'tools/train_next_v2/common.py']
    snapshot=local_clap_snapshot()
    files += [snapshot/name for name in ('config.json','preprocessor_config.json','model.safetensors')]
    protected.update({rel(p):sha256(p) for p in files})
    verify_hashes(protected)
    env=dict(source_launch['environment'],CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    argv=[str(Path(sys.executable).resolve()),'-u','-m','tools.train_next_v2.jamendo_data.features','--run']
    launch=dict(job_id=JOB,stage='D',registered_at=now(),original_argv=argv,recovery_argv=argv+['--resume'],
        cwd=str(ROOT),python=argv[0],environment=env,protected_hashes=protected,feature_key=store.key,feature_spec=store.spec,
        songs=summary['songs'],groups=summary['groups'],native_audio_hours=summary['hours'],batch_size=8,
        source=rel(SOURCE),protocol=rel(PROTOCOL),cache=rel(CACHE),os_lock=str(lock_path(JOB)),gpu_lock=str(lock_path(GPU_LEASE)),
        log=rel(DIRECTORY/'console.log'),status_path=rel(DIRECTORY/'status.json'),report=rel(DIRECTORY/'report.json'),
        labels_used=False,training_started=False,
        timeline='Full native tracks; .5second grid, centered2second overlapping feature windows. No timepoint targets are created.',
        recovery='After diagnosed exit and inactive job/GPU leases use original recovery_argv --resume. Reverify configuration/code/source and paired NPZ/receipt hashes. Orphaned shards stop for inspection; no deletion, relaxed checks or automatic command retries.')
    DIRECTORY.mkdir(parents=True);CACHE.mkdir(parents=True)
    atomic_json(DIRECTORY/'launch.json',launch)
    atomic_json(DIRECTORY/'configuration-sha256.json',dict(launch_sha256=sha256(DIRECTORY/'launch.json')))
    with run_lease('next-va-v2-d-ledger'):
        stages=read_json(PLAN)['stages']
        stages['D'].update(phase='Jamendo_baseline_feature_extraction',jamendo_feature_job=dict(job_id=JOB,status='registered',
            launch=rel(DIRECTORY/'launch.json'),songs=summary['songs'],true_status_module='tools.train_next_v2.jamendo_data.features --status'))
        record_event(dict(event='D_Jamendo_features_registered',launch=rel(DIRECTORY/'launch.json'),songs=summary['songs']),updates=dict(stages=stages))
    print(json.dumps(dict(launch=rel(DIRECTORY/'launch.json'),songs=summary['songs'])))


def run(resume):
    launch=read_json(DIRECTORY/'launch.json')
    if read_json(DIRECTORY/'configuration-sha256.json') != dict(launch_sha256=sha256(DIRECTORY/'launch.json')):
        raise ValueError('Feature launch configuration changed')
    verify_hashes(launch['protected_hashes'])
    if Path(sys.executable).resolve()!=Path(launch['python']) or Path.cwd().resolve()!=ROOT:
        raise ValueError('Wrong Python or working directory')
    if any(os.environ.get(k)!=v for k,v in launch['environment'].items()):
        raise ValueError('Frozen feature environment changed')
    if (DIRECTORY/'report.json').exists():
        raise FileExistsError('Completed features cannot be restarted')
    if (DIRECTORY/'process.json').exists() and not resume:
        raise ValueError('Existing feature attempt requires diagnosed --resume')
    assert_gpu_available()
    if not torch.cuda.is_available():
        raise RuntimeError('Independent training environment has no CUDA')
    identity=process_identity(); started=time.monotonic(); completed=[]; last_frame_log=0.
    atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=identity,resumed=resume,
        actual_argv=launch['recovery_argv'] if resume else launch['original_argv']))
    with run_lease('next-va-v2-d-ledger'):
        stages=read_json(PLAN)['stages']
        stages['D']['jamendo_feature_job'].update(status='running',process_identity=identity)
        record_event(dict(event='D_Jamendo_features_started',process_identity=identity,resumed=resume),updates=dict(stages=stages))
    rows=read_json(SOURCE);validate_candidates(rows)
    torch.set_num_threads(8)
    with (DIRECTORY/'console.log').open('a',encoding='utf-8') as log:
        def publish(**row):
            state=dict(updated_at=now(),process_identity=identity,completed=len(completed),total=len(rows),**row)
            atomic_json(DIRECTORY/'status.json',state)
            line=json.dumps(state);log.write(line+'\n');log.flush();print(line,flush=True)
        def progress(value):
            nonlocal last_frame_log
            if time.monotonic()-last_frame_log>=15:
                publish(stage='feature_frames',**value);last_frame_log=time.monotonic()
        try:
            store=FeatureStore(torch.device('cuda'),batch_size=launch['batch_size'],progress=progress)
            store.directory=CACHE
            if store.key!=launch['feature_key'] or store.spec!=launch['feature_spec']:
                raise ValueError('Runtime feature specification changed')
            torch.cuda.reset_peak_memory_stats()
            publish(stage='starting',feature_key=store.key)
            for row in rows:
                record=feature_record(row)
                if sha256(ROOT/record['audio_path'])!=record['audio_sha256']:
                    raise ValueError('Source audio changed')
                cached=read_cached(row,CACHE,store.key)
                reused=cached is not None
                if cached is None:
                    # Prove that actual feature inputs match the approved native samples.
                    audio,rate=sf.read(ROOT/record['audio_path'],dtype='float32',always_2d=True)
                    if (rate!=record['sample_rate'] or audio.shape!=(record['decoded_frames'],record['channels'])
                            or hashlib.sha256(audio.tobytes(order='C')).hexdigest()!=row['audio']['native_audio']['float32_sample_sha256']):
                        raise ValueError('Native feature input differs from audited sample clock/hash')
                    del audio
                    before=time.monotonic(); times,values,unexpected_reuse=store.get(record,ROOT)
                    if unexpected_reuse:
                        raise ValueError('Unexpected unreceipted cache appeared during extraction')
                    validate_feature_arrays(times,values,record)
                    npz=CACHE/(record['audio_sha256']+'.npz')
                    entry=dict(song_id=row['song_id'],audio_sha256=record['audio_sha256'],feature_key=store.key,
                        path=rel(npz),sha256=sha256(npz),frames=len(times),dimensions=531,
                        native_duration_seconds=record['duration_seconds'],native_float32_sha256=row['audio']['native_audio']['float32_sample_sha256'],
                        extraction_seconds=time.monotonic()-before,raw_labels_copied_to_timepoints=False)
                    atomic_json(npz.with_suffix('.json'),entry)
                    cached=read_cached(row,CACHE,store.key)
                completed.append(dict(cached,reused=reused))
                publish(stage='song_feature_verified',song_id=row['song_id'],frames=cached['frames'],reused=reused)
            store.unload();verify_hashes(launch['protected_hashes'])
            report=dict(job_id=JOB,status='completed_features_verified',completed_at=now(),process_identity=identity,
                songs=len(completed),frames=sum(x['frames'] for x in completed),feature_key=launch['feature_key'],
                elapsed_seconds=time.monotonic()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                feature_manifest=completed,source_sha256=sha256(SOURCE),launch_sha256=sha256(DIRECTORY/'launch.json'),
                labels_used=False,training_started=False,next_action='Verify paired training implementation/recovery and freeze complete feature-input receipts before six-run comparison.')
            atomic_json(DIRECTORY/'report.json',report)
            with run_lease('next-va-v2-d-ledger'):
                stages=read_json(PLAN)['stages']
                stages['D']['phase']='Jamendo_features_ready_training_implementation_pending'
                stages['D']['jamendo_feature_job'].update(status=report['status'],report=rel(DIRECTORY/'report.json'),completed_songs=len(completed))
                record_event(dict(event='D_Jamendo_features_completed',songs=len(completed),report=rel(DIRECTORY/'report.json')),updates=dict(stages=stages))
            publish(stage=report['status'])
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}'
            log.write(traceback.format_exc()+'\n');log.flush()
            failure=dict(failed_at=now(),error=error,process_identity=identity,resumed=resume)
            atomic_json(DIRECTORY/('failure-'+str(time.time_ns())+'.json'),failure)
            atomic_json(DIRECTORY/'failure.json',failure)
            publish(stage='failed',error=error)
            record_event(dict(event='D_Jamendo_features_failed',**failure))
            raise


def status():
    process=read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity=process.get('process_identity')
    report=read_json(DIRECTORY/'report.json') if (DIRECTORY/'report.json').exists() else {}
    state=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else {}
    print(json.dumps(dict(checked_at=now(),job_id=JOB,os_lease_held=is_running(JOB),gpu_lease_held=is_running(GPU_LEASE),
        process_identity=identity,process_identity_matches=bool(identity and process_identity(identity['pid'])==identity),
        npz_files=len(list(CACHE.glob('*.npz'))),receipts=len(list(CACHE.glob('*.json'))),progress=state,report_status=report.get('status'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--register',action='store_true');m.add_argument('--run',action='store_true');m.add_argument('--status',action='store_true')
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.status:
        status()
    else:
        with run_lease(JOB):
            if args.register:
                register()
            else:
                with run_lease(GPU_LEASE):
                    run(args.resume)
