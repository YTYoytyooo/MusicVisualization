"""Verify CPU-faithful V2 features and independently load mixed/B01 heads."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from tools.train_next.core import atomic_json, collate_songs, sha256
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import ROOT, BASE, PLAN, GPU_LEASE, now, read_json, record_event, verify_hashes, process_identity, assert_gpu_available
from tools.train_next_v2.research_inference.benchmark import process_alive
from tools.train_next_v2.cross_version.adapters import load_legacy_source, V2Encoder, V2Head, SequenceHeads
from tools.train_next_v2.cross_version.contract import DIRECTORY as OLD_DIRECTORY, PROTOCOL as OLD_PROTOCOL, JOB as OLD_JOB
from tools.train_next_v2.run import Tee

JOB='f-cross-version-cpu-preflight-v2'
DIRECTORY=BASE/'comparison_jobs'/JOB
PROTOCOL=BASE/'protocols/f-v2-mixed-b01-validation-cpu-v2.json'


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def state(stage,**values):
    data={**values,'job_id':JOB,'stage':stage,'updated_at':now()}
    atomic_json(DIRECTORY/'status.json',data); print(json.dumps(data,ensure_ascii=False),flush=True)


def prepare():
    if PROTOCOL.exists() or (DIRECTORY/'launch.json').exists(): raise ValueError('Preserve existing frozen CPU contract')
    previous=read_json(OLD_DIRECTORY/'process.json')
    if is_running(OLD_JOB) or is_running(GPU_LEASE) or process_alive(previous['process_identity']) is not False:
        raise ValueError('Previous run is not confirmed exited')
    failures=list(OLD_DIRECTORY.glob('failure-*.json'))
    if len(failures)!=1 or 'AssertionError' not in read_json(failures[0])['error']: raise ValueError('Expected diagnosed parity failure')
    old=read_json(OLD_PROTOCOL); verify_hashes(old['pinned_hashes'])
    p=copy.deepcopy(old)
    p.update(id='f-v2-mixed-b01-validation-cpu-v2',frozen_at=now(),status='frozen_pending_cpu_preflight',
        previous_protocol=rel(OLD_PROTOCOL),previous_protocol_sha256=sha256(OLD_PROTOCOL),
        gpu_legacy_port_accepted=False,previous_gpu_failure=rel(failures[0]),
        execution_revision='Use original V2 CPU arithmetic. Reusable CPU feature extraction must match unmodified V2 extraction bitwise; no tolerance increase. All songs, checkpoints, score rules and frozen GPU rejection remain unchanged.')
    p['v2_features']['execution_device']='cpu'
    p['v2_features']['feature_parity_requirement']='bitwise equality with original V2 CPU extraction'
    p['v2_features']['head_execution']='unchanged V2 EmotionInterface.load(...).predict_sequence(...) on CPU'
    p['pinned_hashes'].update({rel(path):sha256(path) for path in [OLD_PROTOCOL,OLD_DIRECTORY/'launch.json',OLD_DIRECTORY/'console.log',failures[0],*Path(__file__).parent.glob('*.py')]})
    atomic_json(PROTOCOL,p)
    old_launch=read_json(OLD_DIRECTORY/'launch.json')
    argv=[old_launch['python'],'-u','-m',__package__+'.job','--launch',str(DIRECTORY/'launch.json')]
    launch=dict(job_id=JOB,registered_at=now(),original_argv=argv,resume_argv=argv+['--resume'],verification_argv=argv+['--verify'],
        cwd=str(ROOT),python=argv[0],environment=old_launch['environment'],protocol=rel(PROTOCOL),protocol_sha256=sha256(PROTOCOL),
        pinned_hashes=p['pinned_hashes'],os_lock=str(lock_path(JOB)),gpu_lock=str(lock_path(GPU_LEASE)),
        log=rel(DIRECTORY/'console.log'),receipts=rel(DIRECTORY/'receipts'),
        recovery='Preserve all previous failed artifacts. Confirm process exited and leases released, diagnose, then original argv + --resume. Only hash-verified complete step receipts are reusable; completed report requires --verify.',
        status_module=__package__+'.job --status')
    atomic_json(DIRECTORY/'launch.json',launch)
    stages=read_json(PLAN)['stages']; stages['F'].update(phase='cross_version_original_cpu_preflight_registered',comparison_protocol=rel(PROTOCOL),
        rejected_gpu_port=dict(job_id=OLD_JOB,reason='1 of 160 raw outputs exceeded frozen tolerance; preserve rejection and all code/config',failure=rel(failures[0])))
    record_event(dict(event='F_exact_cpu_legacy_preflight_registered',job_id=JOB,prior_failure=rel(failures[0]),validation_songs=584),
        run=dict(run_id=JOB,status='registered',stage='F',launch=rel(DIRECTORY/'launch.json')),
        updates=dict(stages=stages,next_action='Verify exact original-CPU V2 feature/head behavior and mixed/B01 checkpoint reloads. Keep GPU-port failure rejected and tolerances unchanged, then proceed with the frozen descriptive validation comparison.'))
    print(json.dumps(dict(job_id=JOB,protocol=rel(PROTOCOL),launch=rel(DIRECTORY/'launch.json'))))


def check(name,ph):
    path=DIRECTORY/'outputs'/(name+'.npz'); receipt=DIRECTORY/'receipts'/(name+'.json')
    if not receipt.exists():
        if path.exists() or path.with_suffix('.tmp').exists(): raise ValueError('Unreceipted artifact requires inspection: '+name)
        return None
    row=read_json(receipt)
    if row['protocol_sha256']!=ph: raise ValueError('Receipt protocol mismatch')
    verify_hashes(row['files']); return row


def save(name,arrays,ph,metrics):
    if check(name,ph): raise ValueError('Completed output already exists')
    path=DIRECTORY/'outputs'/(name+'.npz'); temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream: np.savez_compressed(stream,**arrays)
    temporary.replace(path)
    atomic_json(DIRECTORY/'receipts'/(name+'.json'),dict(protocol_sha256=ph,files={rel(path):sha256(path)},measurements=metrics))


def compare(a,b,p):
    limit=p['encoder_probe']; error=np.abs(a-b)
    bad=error>limit['head_atol']+limit['head_rtol']*np.abs(b)
    return dict(maximum_difference=float(error.max()),maximum_per_axis=error.max(axis=0).tolist(),
        violations=int(bad.sum()),violations_per_axis=bad.sum(axis=0).tolist())


def run(p,resume):
    if (DIRECTORY/'report.json').exists(): raise ValueError('Completed report exists; verify')
    if (DIRECTORY/'process.json').exists() and not resume: raise ValueError('Existing run requires --resume')
    assert_gpu_available()
    with run_lease(GPU_LEASE):
        if not torch.cuda.is_available(): raise ValueError('GPU unavailable for sequence-head checks')
        atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=process_identity(),resumed=resume))
        record_event(dict(event='F_exact_cpu_legacy_preflight_started',resumed=resume),run=dict(run_id=JOB,status='running',stage='F'))
        torch.set_num_threads(p['threads']); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
        ph=sha256(PROTOCOL)
        if not check('legacy_cpu',ph):
            state('original_cpu_feature_equivalence')
            features,source=load_legacy_source(); song=p['probe_songs'][0]
            y,sr=features.load_audio(str(ROOT/song['audio_path']))
            if sr!=22050: raise ValueError('Original decoder changed')
            y=y[:int(p['encoder_probe']['duration_seconds']*sr)]
            torch.manual_seed(42); np.random.seed(42)
            original=source.extract_clap_embeddings(y,sr,clap_source=ROOT/p['v2_features']['snapshot'],batch_size=16)
            interface=source.EmotionInterface.load(str(ROOT/p['models']['v2']['checkpoint']))
            expected=np.stack([r['emotion'] for r in interface.predict_sequence(original)])
            encoder=V2Encoder(p,'cpu'); times,reusable=encoder.extract(y,42)
            np.testing.assert_array_equal(reusable,original)
            repeat_times,repeat=encoder.extract(y,42)
            np.testing.assert_array_equal(reusable,repeat); np.testing.assert_array_equal(times,repeat_times)
            actual=np.stack([r['emotion'] for r in interface.predict_sequence(reusable)])
            np.testing.assert_array_equal(actual,expected); encoder.unload()
            state('isolating_previous_gpu_difference')
            gpu_encoder=V2Encoder(p,'cuda'); _,gpu_features=gpu_encoder.extract(y,42)
            gpu_head=V2Head.load(p,'cuda')
            gpu_head_only=gpu_head.predict(original)
            gpu_encoder_only=np.stack([r['emotion'] for r in interface.predict_sequence(gpu_features)])
            gpu_both=gpu_head.predict(gpu_features)
            diagnosis=dict(gpu_head_with_original_cpu_features=compare(gpu_head_only,expected,p),
                cpu_head_with_gpu_features=compare(gpu_encoder_only,expected,p),gpu_end_to_end=compare(gpu_both,expected,p),
                feature_maximum_difference=float(np.max(np.abs(gpu_features-original))),
                decision='GPU port remains rejected; exact CPU extraction and original CPU head selected',
                original_frozen_rtol=p['encoder_probe']['head_rtol'],original_frozen_atol=p['encoder_probe']['head_atol'])
            save('legacy_cpu',dict(audio_22050=y,times=times,cpu_features=original,cpu_predictions=expected,
                gpu_features=gpu_features,gpu_head_only=gpu_head_only,gpu_encoder_only=gpu_encoder_only,gpu_both=gpu_both),ph,
                dict(frames=len(times),cpu_features_bitwise_equal=True,cpu_repeat_bitwise_equal=True,cpu_head_bitwise_equal=True,diagnosis=diagnosis,labels_used=0))
            gpu_encoder.unload(); del gpu_head,interface
        for kind in ('mixed','b01'):
            state('sequence_checkpoint_parity',kind=kind)
            adapter=SequenceHeads(p,kind,'cuda')
            for index,song in enumerate(p['probe_songs']):
                name=f'{kind}-{index}'
                if check(name,ph): continue
                with np.load(ROOT/p['probe_cache']/(song['audio_sha256']+'.npz'),allow_pickle=False) as data:
                    times,values,key=data['times'].copy(),data['features'].copy(),str(data['feature_key'].item())
                members,mean=adapter.predict(times,values,key,song['duration_seconds'])
                save(name,dict(times=times,members=members,mean=mean),ph,dict(song_id=song['song_id'],frames=len(times),members=len(members),labels_used=0))
            del adapter; torch.cuda.empty_cache()
        verify_hashes(p['pinned_hashes'])
        receipts={f.stem:read_json(f) for f in sorted((DIRECTORY/'receipts').glob('*.json'))}
        if len(receipts)!=7: raise ValueError('Missing parity evidence')
        report=dict(job_id=JOB,status='completed_pending_independent_verification',completed_at=now(),protocol_sha256=ph,
            steps=receipts,accuracy_evaluated=False,labels_used=0,legacy_gpu_port_accepted=False,protected_models_unchanged=True)
        atomic_json(DIRECTORY/'report.json',report); state('completed_pending_independent_verification',steps=7)
        record_event(dict(event='F_exact_cpu_legacy_preflight_finished',report=rel(DIRECTORY/'report.json')),
            run=dict(run_id=JOB,status='pending_verification',stage='F',report=rel(DIRECTORY/'report.json')))


@torch.inference_mode()
def verify(p):
    if (DIRECTORY/'verification.json').exists(): raise ValueError('Preserve verified result')
    atomic_json(DIRECTORY/'verification-process.json',dict(started_at=now(),process_identity=process_identity()))
    torch.set_num_threads(p['threads']); ph=sha256(PROTOCOL); report=read_json(DIRECTORY/'report.json')
    if report['protocol_sha256']!=ph: raise ValueError('Report contract changed')
    for name,receipt in report['steps'].items():
        if check(name,ph)!=receipt: raise ValueError('Changed completion receipt')
    state('fresh_cpu_verification')
    _,source=load_legacy_source(); interface=source.EmotionInterface.load(str(ROOT/p['models']['v2']['checkpoint']))
    with np.load(DIRECTORY/'outputs/legacy_cpu.npz',allow_pickle=False) as saved:
        actual=np.stack([r['emotion'] for r in interface.predict_sequence(saved['cpu_features'])])
        np.testing.assert_array_equal(actual,saved['cpu_predictions'])
        encoder=V2Encoder(p,'cpu'); times,features=encoder.extract(saved['audio_22050'],42)
        np.testing.assert_array_equal(features,saved['cpu_features']); np.testing.assert_array_equal(times,saved['times']); encoder.unload()
    maximum=0.
    for kind in ('mixed','b01'):
        adapter=SequenceHeads(p,kind,'cpu')
        for index,song in enumerate(p['probe_songs']):
            with np.load(ROOT/p['probe_cache']/(song['audio_sha256']+'.npz'),allow_pickle=False) as data:
                times,values,key=data['times'].copy(),data['features'].copy(),str(data['feature_key'].item())
            members,mean=adapter.predict(times,values,key,song['duration_seconds'])
            row=dict(x=values,y=np.zeros((len(times),2),np.float32),mask=np.zeros((len(times),2),bool))
            tensor,_,_,lengths=collate_songs([row],*adapter.scaler)
            direct=np.stack([m(tensor,lengths).numpy()[0] for m in adapter.models])
            np.testing.assert_array_equal(members,direct)
            with np.load(DIRECTORY/'outputs'/(f'{kind}-{index}.npz'),allow_pickle=False) as saved:
                np.testing.assert_array_equal(times,saved['times'])
                np.testing.assert_allclose(members,saved['members'],rtol=1e-5,atol=1e-6)
                np.testing.assert_allclose(mean,saved['mean'],rtol=1e-5,atol=1e-6)
                maximum=max(maximum,float(np.max(np.abs(members-saved['members']))))
    verify_hashes(p['pinned_hashes'])
    result=dict(job_id=JOB,status='passed',verified_at=now(),report_sha256=sha256(DIRECTORY/'report.json'),protocol_sha256=ph,
        fresh_cpu_model_loads=5,steps_verified=7,original_v2_cpu_features_bitwise_equal=True,original_v2_cpu_head_bitwise_equal=True,
        independent_sequence_training_collator_equal=True,sequence_cpu_gpu_maximum_difference=maximum,legacy_gpu_port_accepted=False,
        labels_used=0,accuracy_evaluated=False,protected_models_unchanged=True)
    atomic_json(DIRECTORY/'verification.json',result)
    stages=read_json(PLAN)['stages']; stages['F'].update(phase='cross_version_cpu_loaders_verified',comparison_protocol=rel(PROTOCOL),
        cross_version_preflight=rel(DIRECTORY/'report.json'),cross_version_verification=rel(DIRECTORY/'verification.json'))
    record_event(dict(event='F_cross_version_cpu_loaders_independently_verified',**result),
        run=dict(run_id=JOB,status='completed_verified',stage='F',report=rel(DIRECTORY/'report.json'),verification=rel(DIRECTORY/'verification.json')),
        updates=dict(stages=stages,next_action='Implement/register the frozen 584-song descriptive V2/mixed/B01 comparison in a new evaluation module. Reuse frozen adapters unchanged: V2Encoder on CPU and original V2 EmotionInterface CPU head, mixed/B01 own scalers and whole-song heads. Preserve masks, terminal exclusion counts, per-song recovery, domain/axis metrics and paired group bootstrap. Final-test labels sealed; no retuning or promotion.'))
    state('completed_verified',**result)


def status():
    selected=DIRECTORY/('verification-process.json' if (DIRECTORY/'verification-process.json').exists() else 'process.json')
    process=read_json(selected) if selected.exists() else None
    held=is_running(JOB); verified=False
    if (DIRECTORY/'verification.json').exists() and (DIRECTORY/'report.json').exists():
        v=read_json(DIRECTORY/'verification.json')
        verified=v['status']=='passed' and v['report_sha256']==sha256(DIRECTORY/'report.json') and v['protocol_sha256']==sha256(PROTOCOL)
    tail=[]; log=DIRECTORY/'console.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-4096)); tail=stream.read().decode('utf-8',errors='replace').splitlines()[-3:]
    return dict(job_id=JOB,checked_at=now(),os_lease_active=held,gpu_lease_active=is_running(GPU_LEASE),
        process_alive=process_alive(process['process_identity']) if process else None,
        outcome='active' if held else 'completed_verified' if verified else 'inactive_requires_inspection',
        completed_step_receipts=len(list((DIRECTORY/'receipts').glob('*.json'))),
        progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None,bounded_log_tail=tail)


def main():
    parser=argparse.ArgumentParser(description=__doc__); modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true'); modes.add_argument('--status',action='store_true'); modes.add_argument('--launch',type=Path)
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    if args.status: print(json.dumps(status(),ensure_ascii=False)); return
    with run_lease(JOB):
        for folder in (DIRECTORY,DIRECTORY/'outputs',DIRECTORY/'receipts'): folder.mkdir(parents=True,exist_ok=True)
        if args.prepare: prepare(); return
        if args.launch.resolve()!=DIRECTORY/'launch.json': raise ValueError('Wrong launch path')
        launch=read_json(args.launch)
        if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(launch['python']): raise ValueError('Wrong workspace/Python')
        if any(os.environ.get(k)!=v for k,v in launch['environment'].items()): raise ValueError('Frozen environment changed')
        if sha256(PROTOCOL)!=launch['protocol_sha256']: raise ValueError('Protocol changed')
        p=read_json(PROTOCOL); verify_hashes(p['pinned_hashes'])
        with (DIRECTORY/'console.log').open('a',encoding='utf-8',buffering=1) as stream:
            with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
                try:
                    if args.verify: verify(p)
                    else: run(p,args.resume)
                except Exception as exc:
                    failure=dict(at=now(),error=f'{type(exc).__name__}: {exc}',mode='verify' if args.verify else 'run')
                    atomic_json(DIRECTORY/('failure-'+str(time.time_ns())+'.json'),failure)
                    record_event(dict(event='F_exact_cpu_legacy_preflight_failed',**failure),run=dict(run_id=JOB,status='failed',stage='F'))
                    traceback.print_exc(); raise


if __name__=='__main__': main()
