"""Registered finite loader/encoder parity job; never reads emotion targets."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from tools.train_next.core import atomic_json, collate_songs, sha256
from tools.train_next.runtime import is_running, run_lease
from tools.train_next_v2.common import (ROOT, PLAN, GPU_LEASE, now, read_json,
    record_event, verify_hashes, process_identity, assert_gpu_available)
from tools.train_next_v2.research_inference.benchmark import process_alive
from tools.train_next_v2.run import Tee
from .adapters import load_legacy_source, V2Encoder, V2Head, SequenceHeads
from .contract import JOB, DIRECTORY, PROTOCOL, rel, state, tests, prepare


def npz_receipt(name,arrays,protocol_hash,measurements):
    path=DIRECTORY/'outputs'/(name+'.npz'); receipt=DIRECTORY/'receipts'/(name+'.json')
    if path.exists() or receipt.exists(): raise ValueError('Preserve pre-existing artifacts: '+name)
    temporary=path.with_suffix('.tmp')
    if temporary.exists(): raise ValueError('Unfinished output requires inspection')
    with temporary.open('wb') as stream: np.savez_compressed(stream,**arrays)
    temporary.replace(path)
    atomic_json(receipt,dict(protocol_sha256=protocol_hash,files={rel(path):sha256(path)},measurements=measurements))


def checked_step(name,protocol_hash):
    receipt=DIRECTORY/'receipts'/(name+'.json')
    if not receipt.exists():
        if (DIRECTORY/'outputs'/(name+'.npz')).exists(): raise ValueError('Unreceipted output requires inspection')
        return None
    result=read_json(receipt)
    if result['protocol_sha256']!=protocol_hash: raise ValueError('Step protocol changed')
    verify_hashes(result['files'])
    return result


def run(launch,p,resume):
    if (DIRECTORY/'report.json').exists(): raise ValueError('Completed report exists; verify it')
    if (DIRECTORY/'process.json').exists() and not resume: raise ValueError('Existing run requires --resume')
    assert_gpu_available()
    with run_lease(GPU_LEASE):
        if not torch.cuda.is_available(): raise ValueError('GPU unavailable')
        atomic_json(DIRECTORY/'process.json',dict(process_identity=process_identity(),started_at=now(),resumed=resume))
        record_event(dict(event='F_cross_version_preflight_started',resumed=resume),run=dict(run_id=JOB,stage='F',status='running'))
        torch.set_num_threads(p['threads']); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        ph=sha256(PROTOCOL)
        if not checked_step('legacy_encoder',ph):
            state('legacy_original_cpu_encoder_parity')
            features,source=load_legacy_source()
            song=p['probe_songs'][0]
            y,sr=features.load_audio(str(ROOT/song['audio_path']))
            if sr!=22050: raise ValueError('Legacy sampling changed')
            y=y[:int(p['encoder_probe']['duration_seconds']*sr)]
            torch.manual_seed(42); np.random.seed(42)
            start=time.perf_counter()
            original=source.extract_clap_embeddings(y,sr,clap_source=ROOT/p['v2_features']['snapshot'],batch_size=16)
            cpu_seconds=time.perf_counter()-start
            cpu_head=source.EmotionInterface.load(str(ROOT/p['models']['v2']['checkpoint']))
            expected=np.stack([r['emotion'] for r in cpu_head.predict_sequence(original)])
            state('legacy_device_port_parity')
            encoder=V2Encoder(p,'cuda'); start=time.perf_counter()
            times,ported=encoder.extract(y,42); torch.cuda.synchronize()
            gpu_seconds=time.perf_counter()-start
            limits=p['encoder_probe']
            np.testing.assert_allclose(ported,original,rtol=limits['feature_rtol'],atol=limits['feature_atol'])
            head=V2Head.load(p,'cuda'); actual=head.predict(ported)
            np.testing.assert_allclose(actual,expected,rtol=limits['head_rtol'],atol=limits['head_atol'])
            repeat_times,repeat=encoder.extract(y,42)
            np.testing.assert_array_equal(times,repeat_times); np.testing.assert_array_equal(ported,repeat)
            encoder.unload()
            npz_receipt('legacy_encoder',dict(audio_22050=y,times=times,cpu_features=original,gpu_features=ported,
                cpu_predictions=expected,gpu_predictions=actual),ph,dict(frames=len(times),audio_seconds=len(y)/sr,
                cpu_seconds=cpu_seconds,gpu_seconds=gpu_seconds,feature_maximum_difference=float(np.max(np.abs(ported-original))),
                head_maximum_difference=float(np.max(np.abs(actual-expected))),gpu_repeat_bitwise_equal=True,labels_used=0))
            del head,cpu_head
        for kind in ('mixed','b01'):
            state('sequence_checkpoint_parity',kind=kind)
            adapter=SequenceHeads(p,kind,'cuda')
            for index,song in enumerate(p['probe_songs']):
                name=f'{kind}-{index}'
                if checked_step(name,ph): continue
                cache=ROOT/p['probe_cache']/(song['audio_sha256']+'.npz')
                with np.load(cache,allow_pickle=False) as data:
                    times=data['times'].copy(); values=data['features'].copy(); key=str(data['feature_key'].item())
                members,mean=adapter.predict(times,values,key,song['duration_seconds'])
                npz_receipt(name,dict(times=times,members=members,mean=mean),ph,
                    dict(song_id=song['song_id'],frames=len(times),member_count=len(members),labels_used=0))
            del adapter; torch.cuda.empty_cache()
        verify_hashes(p['pinned_hashes'])
        receipts={path.stem:read_json(path) for path in sorted((DIRECTORY/'receipts').glob('*.json'))}
        if len(receipts)!=7: raise ValueError('Missing preflight steps')
        report=dict(job_id=JOB,status='completed_pending_independent_verification',completed_at=now(),protocol_sha256=ph,
            steps=receipts,device=torch.cuda.get_device_name(),labels_used=0,accuracy_evaluated=False,
            protected_models_unchanged=True,application_defaults_changed=False)
        atomic_json(DIRECTORY/'report.json',report)
        state('completed_pending_independent_verification',completed_steps=7)
        record_event(dict(event='F_cross_version_preflight_finished',report=rel(DIRECTORY/'report.json')),
            run=dict(run_id=JOB,stage='F',status='pending_verification',report=rel(DIRECTORY/'report.json')))


@torch.inference_mode()
def verify(p):
    if (DIRECTORY/'verification.json').exists(): raise ValueError('Preserve completed verification')
    atomic_json(DIRECTORY/'verification-process.json',dict(process_identity=process_identity(),started_at=now()))
    torch.set_num_threads(p['threads'])
    report=read_json(DIRECTORY/'report.json'); ph=sha256(PROTOCOL)
    if report['protocol_sha256']!=ph: raise ValueError('Report protocol changed')
    for name,receipt in report['steps'].items():
        if checked_step(name,ph)!=receipt: raise ValueError('Step receipt changed')
    _,source=load_legacy_source()
    legacy=source.EmotionInterface.load(str(ROOT/p['models']['v2']['checkpoint']))
    with np.load(DIRECTORY/'outputs/legacy_encoder.npz',allow_pickle=False) as data:
        actual=np.stack([r['emotion'] for r in legacy.predict_sequence(data['cpu_features'])])
        np.testing.assert_array_equal(actual,data['cpu_predictions'])
    maximum=0.
    for kind in ('mixed','b01'):
        adapter=SequenceHeads(p,kind,'cpu')
        for index,song in enumerate(p['probe_songs']):
            with np.load(ROOT/p['probe_cache']/(song['audio_sha256']+'.npz'),allow_pickle=False) as data:
                times,features,key=data['times'].copy(),data['features'].copy(),str(data['feature_key'].item())
            members,mean=adapter.predict(times,features,key,song['duration_seconds'])
            raw=dict(x=features,y=np.zeros((len(times),2),np.float32),mask=np.zeros((len(times),2),bool))
            tensor,_,_,lengths=collate_songs([raw],*adapter.scaler)
            direct=np.stack([model(tensor,lengths).numpy()[0] for model in adapter.models])
            np.testing.assert_array_equal(members,direct)
            with np.load(DIRECTORY/'outputs'/(f'{kind}-{index}.npz'),allow_pickle=False) as saved:
                np.testing.assert_array_equal(times,saved['times'])
                np.testing.assert_allclose(members,saved['members'],rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(mean,saved['mean'],rtol=1e-5,atol=1e-6)
                maximum=max(maximum,float(np.max(np.abs(members-saved['members']))))
    verify_hashes(p['pinned_hashes'])
    evidence=dict(job_id=JOB,status='passed',verified_at=now(),report_sha256=sha256(DIRECTORY/'report.json'),protocol_sha256=ph,
        maximum_sequence_cpu_gpu_difference=maximum,fresh_cpu_model_loads=5,steps_verified=7,
        independent_training_collator_equal=True,original_v2_cpu_head_equal=True,labels_used=0,accuracy_evaluated=False,
        protected_models_unchanged=True)
    atomic_json(DIRECTORY/'verification.json',evidence)
    stages=read_json(PLAN)['stages']; stages['F'].update(phase='cross_version_loaders_verified',comparison_protocol=rel(PROTOCOL),
        cross_version_preflight=rel(DIRECTORY/'report.json'),cross_version_verification=rel(DIRECTORY/'verification.json'))
    record_event(dict(event='F_cross_version_loaders_independently_verified',**evidence),
        run=dict(run_id=JOB,status='completed_verified',stage='F',report=rel(DIRECTORY/'report.json'),verification=rel(DIRECTORY/'verification.json')),
        updates=dict(stages=stages,next_action='Implement and register the frozen 584-song V2/mixed/B01 descriptive comparison in a separate new evaluation module; reuse verified cross_version adapters unchanged. Extract separate V2 features; enforce common-axis/time masks, per-song receipts, per-dataset/dimension metrics and paired song-group bootstrap. Keep final-test labels sealed; do not tune or promote from this comparison.'))
    state('completed_verified',**evidence)


def status():
    process_path=DIRECTORY/('verification-process.json' if (DIRECTORY/'verification-process.json').exists() else 'process.json')
    process=read_json(process_path) if process_path.exists() else None
    held=is_running(JOB); verified=False
    if (DIRECTORY/'verification.json').exists() and (DIRECTORY/'report.json').exists():
        check=read_json(DIRECTORY/'verification.json')
        verified=check['status']=='passed' and check['report_sha256']==sha256(DIRECTORY/'report.json') and check['protocol_sha256']==sha256(PROTOCOL)
    log=DIRECTORY/'console.log'; tail=[]
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-4096)); tail=stream.read().decode('utf-8',errors='replace').splitlines()[-3:]
    return dict(job_id=JOB,checked_at=now(),os_lease_active=held,gpu_lease_active=is_running(GPU_LEASE),
        process_alive=process_alive(process['process_identity']) if process else None,
        outcome='active' if held else 'completed_verified' if verified else 'inactive_requires_inspection',
        completed_step_receipts=len(list((DIRECTORY/'receipts').glob('*.json'))),
        progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None,bounded_log_tail=tail)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--tests',action='store_true'); modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--status',action='store_true'); modes.add_argument('--launch',type=Path)
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.status: print(json.dumps(status(),ensure_ascii=False)); return
    with run_lease(JOB):
        if args.tests: tests(); return
        for folder in (DIRECTORY,DIRECTORY/'outputs',DIRECTORY/'receipts'): folder.mkdir(parents=True,exist_ok=True)
        if args.prepare: prepare(); return
        if args.launch.resolve()!=DIRECTORY/'launch.json': raise ValueError('Wrong launch')
        launch=read_json(args.launch)
        if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(launch['python']): raise ValueError('Wrong independent environment')
        if any(os.environ.get(k)!=v for k,v in launch['environment'].items()): raise ValueError('Frozen environment changed')
        if sha256(PROTOCOL)!=launch['protocol_sha256']: raise ValueError('Frozen protocol changed')
        p=read_json(PROTOCOL); verify_hashes(p['pinned_hashes']); verify_hashes(launch['code'])
        with (DIRECTORY/'console.log').open('a',encoding='utf-8',buffering=1) as stream:
            with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
                try:
                    if args.verify: verify(p)
                    else: run(launch,p,args.resume)
                except Exception as exc:
                    failure=dict(at=now(),mode='verify' if args.verify else 'preflight',error=f'{type(exc).__name__}: {exc}')
                    atomic_json(DIRECTORY/('failure-'+str(time.time_ns())+'.json'),failure)
                    record_event(dict(event='F_cross_version_preflight_failed',**failure),run=dict(run_id=JOB,status='failed',stage='F'))
                    traceback.print_exc(); raise


if __name__=='__main__': main()
