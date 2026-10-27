"""Extract the unchanged baseline representation for screened static auxiliary clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from tools.train_next.core import atomic_json, sha256
from tools.train_next.features import FeatureStore, local_clap_snapshot
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, GPU_LEASE, now, process_identity, read_json, record_event, verify_hashes, assert_gpu_available

JOB='d-merge-clap-features-v2'
FOLDER=BASE/'feature_jobs'/JOB
CACHE=BASE/'feature_caches'/JOB
SOURCE=ROOT/'data/datasets/merge/prepared/read-consistent-v1/eligible-quadrant-candidates.json'
PROTOCOL=BASE/'protocols/d-merge-quadrant-v2.json'


def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def register():
    if FOLDER.exists():raise FileExistsError('Preserve existing feature job')
    if is_running(GPU_LEASE):raise RuntimeError('GPU already leased')
    assert_gpu_available()
    protocol=read_json(PROTOCOL);rows=read_json(SOURCE)
    if sha256(SOURCE)!=protocol['auxiliary_candidates_sha256']:
        raise ValueError('Candidate version differs from frozen protocol')
    if not rows or any(not x['strict_decode']['passed'] or not x['native_stderr_guard']['passed'] or not x['bulk_stream_parity']['passed'] for x in rows):
        raise ValueError('Both decoder gates must pass')
    baseline=read_json(BASE/'protocols/b-v1.json')
    store=FeatureStore(torch.device('cpu'))
    if store.key!=baseline['feature_key'] or store.spec!=baseline['feature_spec']:
        raise ValueError('Baseline feature definition or environment changed')
    snapshot=local_clap_snapshot()
    files=[Path(__file__),SOURCE,PROTOCOL,BASE/'protocols/b-v1.json',
           BASE/'analyses/d-merge-native-decoder-v1/report.json', BASE/'analyses/d-merge-read-parity-v1/report.json',ROOT/'tools/train_next/features.py',
           ROOT/'tools/train_next/core.py',ROOT/'tools/train_next/runtime.py',ROOT/'tools/train_next_v2/common.py']
    files += [snapshot/name for name in ('config.json','preprocessor_config.json','model.safetensors')]
    hashes={rel(p):sha256(p) for p in files}
    hashes['models/emotion_model.pth']=read_json(PLAN)['baseline']['legacy_model_sha256']
    verify_hashes(hashes)
    prior=read_json(BASE/'analyses/d-merge-native-decoder-v1/launch.json')
    env=dict(prior['environment'],CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    argv=[str(Path(sys.executable).resolve()),'-u','-m','tools.train_next_v2.auxiliary_data.merge_features_v2','--run']
    launch=dict(job_id=JOB,stage='D',registered_at=now(),original_argv=argv,recovery_argv=argv+['--resume'],
        cwd=str(ROOT),python=argv[0],environment=env,protected_hashes=hashes,feature_spec=store.spec,
        feature_key=store.key,songs=len(rows),batch_size=8,source=rel(SOURCE),cache=rel(CACHE),
        os_lock=str(lock_path(JOB)),gpu_lock=str(lock_path(GPU_LEASE)),
        status_path=rel(FOLDER/'status.json'),log=rel(FOLDER/'console.log'),report=rel(FOLDER/'report.json'),
        labels_used=False,timeline='Native full clip; 0.5s grid with centered 2s windows. No timepoint ground truth is created.',
        recovery='Confirm no active job/GPU lease, diagnose exit, use recovery_argv with --resume and unchanged hashes; cached shard receipts are checked before reuse. Never edit frozen baseline code.')
    FOLDER.mkdir(parents=True);CACHE.mkdir(parents=True)
    atomic_json(FOLDER/'launch.json',launch)
    stages=read_json(PLAN)['stages']
    stages['D'].update(phase='MERGE_baseline_feature_extraction',auxiliary_protocol=rel(PROTOCOL),
        feature_job=dict(job_id=JOB,status='registered',launch=rel(FOLDER/'launch.json'),songs=len(rows)))
    record_event(dict(event='D_MERGE_features_registered',songs=len(rows),launch=rel(FOLDER/'launch.json')),updates=dict(stages=stages))


def run(resume):
    launch=read_json(FOLDER/'launch.json');verify_hashes(launch['protected_hashes'])
    if (FOLDER/'report.json').exists():raise FileExistsError('Completed features must not be restarted')
    if (FOLDER/'process.json').exists() and not resume:raise ValueError('Prior execution requires --resume')
    assert_gpu_available()
    identity=process_identity();atomic_json(FOLDER/'process.json',dict(started_at=now(),process_identity=identity,
        actual_argv=launch['recovery_argv'] if resume else launch['original_argv']))
    rows=read_json(SOURCE);completed=[];started=time.monotonic();last_frame_log=0.
    torch.set_num_threads(8)
    if not torch.cuda.is_available():raise RuntimeError('Independent GPU Python has no available CUDA device')
    with (FOLDER/'console.log').open('a',encoding='utf-8') as log:
        def publish(value):
            value=dict(**value,completed=len(completed),total=len(rows),process_identity=identity,updated_at=now())
            atomic_json(FOLDER/'status.json',value);line=json.dumps(value);log.write(line+'\n');log.flush();print(line,flush=True)
        def progress(value):
            nonlocal last_frame_log
            if time.monotonic()-last_frame_log>=15:
                publish(dict(stage='feature_frames',**value));last_frame_log=time.monotonic()
        try:
            store=FeatureStore(torch.device('cuda'),batch_size=launch['batch_size'],progress=progress)
            if store.key!=launch['feature_key'] or store.spec!=launch['feature_spec']:
                raise ValueError('Feature specification changed')
            store.directory=CACHE
            torch.cuda.reset_peak_memory_stats()
            publish(dict(stage='starting',feature_key=store.key))
            for row in rows:
                audio=row['audio'];native=audio['native_audio'];digest=audio['audio_sha256']
                receipt=CACHE/(digest+'.json');npz=CACHE/(digest+'.npz')
                if receipt.exists():
                    previous=read_json(receipt)
                    if previous['audio_sha256']!=digest or previous['feature_key']!=store.key:
                        raise ValueError('Cache receipt identity mismatch')
                    verify_hashes({previous['path']:previous['sha256']})
                record=dict(song_id=row['song_id'],audio_path=audio['audio_path'],audio_sha256=digest,
                    duration_seconds=native['duration_seconds'],decoded_frames=native['frames'],
                    sample_rate=native['sample_rate'],channels=native['channels'])
                one_started=time.monotonic();times,values,reused=store.get(record,ROOT)
                result=dict(song_id=row['song_id'],audio_sha256=digest,feature_key=store.key,
                    path=rel(npz),sha256=sha256(npz),frames=len(times),dimensions=values.shape[1],
                    native_duration_seconds=native['duration_seconds'],extraction_seconds=time.monotonic()-one_started,
                    reused=reused,raw_labels_copied_to_timepoints=False)
                if not receipt.exists():atomic_json(receipt,result)
                completed.append(result)
                publish(dict(stage='song_feature_verified',song_id=row['song_id'],frames=len(times)))
            store.unload();verify_hashes(launch['protected_hashes'])
            report=dict(job_id=JOB,status='completed_features_verified',completed_at=now(),process_identity=identity,
                songs=len(completed),frames=sum(x['frames'] for x in completed),feature_key=launch['feature_key'],
                elapsed_seconds=time.monotonic()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                feature_manifest=completed,source_sha256=sha256(SOURCE),launch_sha256=sha256(FOLDER/'launch.json'),
                labels_used=False,training_started=False)
            atomic_json(FOLDER/'report.json',report)
            stages=read_json(PLAN)['stages']
            stages['D'].update(phase='MERGE_features_ready_auxiliary_implementation_pending')
            stages['D']['feature_job'].update(status=report['status'],report=rel(FOLDER/'report.json'),completed_songs=len(completed))
            record_event(dict(event='D_MERGE_feature_extraction_completed',songs=len(completed),report=rel(FOLDER/'report.json')),updates=dict(stages=stages))
            publish(dict(stage=report['status']))
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}';log.write(traceback.format_exc()+'\n');log.flush()
            publish(dict(stage='failed',error=error))
            atomic_json(FOLDER/('failure-'+now().replace(':','-')+'.json'),dict(error=error,time=now(),process_identity=identity))
            record_event(dict(event='D_MERGE_features_failed',error=error))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true');mode.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    with run_lease(JOB):
        if args.register:register()
        else:
            with run_lease(GPU_LEASE):run(args.resume)
