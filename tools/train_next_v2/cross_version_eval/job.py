"""Registered 584-song V2/mixed/B01 comparison with per-song recovery."""
import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import unittest

import numpy as np
import torch

from tools.train_next.core import atomic_json, sha256, collate_songs
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import (
    ROOT, BASE, PLAN, GPU_LEASE, now, read_json, record_event,
    verify_hashes, process_identity, assert_gpu_available)
from tools.train_next_v2.research_inference.benchmark import process_alive
from tools.train_next_v2.cross_version.adapters import load_legacy_source, V2Encoder, SequenceHeads
from tools.train_next_v2.cross_version_cpu.job import PROTOCOL, DIRECTORY as PREFLIGHT
from tools.train_next_v2.run import Tee
from .methods import align_labels, song_scores, summarize, SYSTEMS, AXES
from .storage import ArtifactStore

JOB='f-validation-v2-mixed-b01-cpu-v1'
DIRECTORY=BASE/'comparison_jobs'/JOB
ADDENDUM=BASE/'protocols/f-v2-mixed-b01-validation-cpu-v2-data-access-note.json'
TEST_DIRECTORY=BASE/'analyses/f-cross-version-evaluation-tests-v1'


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def code_hashes():
    files=list(Path(__file__).parent.glob('*.py')) + [
        ROOT/'tools/train_next_v2/tests/test_cross_version_eval.py',
        ROOT/'tools/train_next_v2/evaluation.py', ROOT/'tools/train_next/diagnostics.py']
    return {rel(f):sha256(f) for f in sorted(files)}


def state(stage, **values):
    data=dict(job_id=JOB,stage=stage,updated_at=now(),**values)
    atomic_json(DIRECTORY/'status.json',data)
    print(json.dumps(data,ensure_ascii=False,allow_nan=False),flush=True)


def tests():
    if (TEST_DIRECTORY/'report.json').exists(): raise ValueError('Preserve registered test evidence')
    TEST_DIRECTORY.mkdir(parents=True,exist_ok=True)
    argv=[sys.executable,'-u','-m',__package__+'.job','--tests']
    command=dict(original_argv=argv,cwd=str(ROOT),python=sys.executable,code=code_hashes(),
                 registered_at=now(),process_identity=process_identity(),
                 initial_red='unittest test_cross_version_eval failed before implementation with ModuleNotFoundError cross_version_eval; seven focused checks then passed.')
    atomic_json(TEST_DIRECTORY/'launch.json',command)
    record_event(dict(event='F_validation_scoring_tests_registered',**command))
    with (TEST_DIRECTORY/'console.log').open('w',encoding='utf-8') as stream:
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(name) for name in (
            'tools.train_next_v2.tests.test_cross_version_eval','tools.train_next_v2.tests.test_cross_version')])
        result=unittest.TextTestRunner(stream=Tee(sys.stdout,stream),verbosity=2).run(suite)
    report=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),code=code_hashes(),completed_at=now())
    atomic_json(TEST_DIRECTORY/'report.json',report)
    record_event(dict(event='F_validation_scoring_tests_finished',**report))
    if not result.wasSuccessful(): raise ValueError('Scoring/recovery contracts failed')


def inactive_previous(path):
    if path.exists():
        p=read_json(path)
        if process_alive(p['process_identity']) is not False:
            raise ValueError('Previous process not confirmed exited: '+str(path))


def prepare():
    if (DIRECTORY/'launch.json').exists(): raise ValueError('Preserve frozen job configuration')
    prior=read_json(PREFLIGHT/'verification.json')
    if (is_running('f-cross-version-cpu-preflight-v2') or is_running(GPU_LEASE)
            or prior['status']!='passed' or prior['report_sha256']!=sha256(PREFLIGHT/'report.json')
            or prior['protocol_sha256']!=sha256(PROTOCOL)):
        raise ValueError('Previous CPU preflight not verified/inactive')
    inactive_previous(PREFLIGHT/'verification-process.json')
    test=read_json(TEST_DIRECTORY/'report.json')
    if test['status']!='passed' or test['tests']!=16 or test['skipped']: raise ValueError('Fresh 16-test contract report required')
    verify_hashes(test['code'])
    protocol=read_json(PROTOCOL); verify_hashes(protocol['pinned_hashes'])
    rows=protocol['validation_songs']
    if (len(rows)!=584 or len({r['song_id'] for r in rows})!=584
            or any(r['split']!='validation' for r in rows) or Counter(r['source_dataset'] for r in rows)!=protocol['counts']):
        raise ValueError('Fixed validation membership changed')
    if read_json(ADDENDUM)['protocol_sha256']!=sha256(PROTOCOL): raise ValueError('Data access note is not bound to the contract')
    verification_songs=sorted({min([r for r in rows if r['source_dataset']==d],key=lambda r:(r['duration_seconds'],r['song_id']))['song_id'] for d in protocol['counts']}
        | {max(rows,key=lambda r:(r['duration_seconds'],r['song_id']))['song_id']})
    pinned={**protocol['pinned_hashes'],**code_hashes()}
    for path in (PROTOCOL,ADDENDUM,PREFLIGHT/'verification.json',PREFLIGHT/'report.json',TEST_DIRECTORY/'report.json'):
        pinned[rel(path)]=sha256(path)
    old=read_json(PREFLIGHT/'launch.json')
    argv=[old['python'],'-u','-m',__package__+'.job','--launch',str(DIRECTORY/'launch.json')]
    launch=dict(job_id=JOB,registered_at=now(),python=argv[0],cwd=str(ROOT),environment=old['environment'],
        original_argv=argv,resume_argv=argv+['--resume'],verification_argv=argv+['--verify'],
        protocol=rel(PROTOCOL),protocol_sha256=sha256(PROTOCOL),data_access_note=rel(ADDENDUM),
        pinned_hashes=pinned,os_lock=str(lock_path(JOB)),gpu_lock=str(lock_path(GPU_LEASE)),
        log=rel(DIRECTORY/'console.log'),features=rel(DIRECTORY/'v2_features'),predictions=rel(DIRECTORY/'predictions'),
        status_module=__package__+'.job --status',sequence_device='cuda',legacy_device='cpu',
        validation_songs=584,final_test_songs_used=0,verification_songs=verification_songs,
        original_encoder_recheck_song=min(rows,key=lambda r:(r['duration_seconds'],r['song_id']))['song_id'],
        free_bytes_at_registration=shutil.disk_usage(ROOT).free,
        recovery='After proving process exit and both leases released, diagnose failure and use original argv + --resume. Completed hash-verified feature and prediction receipts are reusable; orphan/partial artifacts fail closed and require explicit inspection. Never delete configuration or relax hashes. A report alone requires --verify.',
        acceptance='All 584 prediction receipts, equal masks, six-cell metrics, paired 2000 group bootstrap, fresh checkpoint reloads and raw-array rechecks, protected original SHA unchanged. Descriptive results cannot tune, promote, or open final test labels.')
    atomic_json(DIRECTORY/'launch.json',launch)
    stages=read_json(PLAN)['stages']; stages['F'].update(phase='cross_version_validation_registered',comparison_job=JOB,
        comparison_launch=rel(DIRECTORY/'launch.json'),comparison_status_module=launch['status_module'])
    record_event(dict(event='F_full_descriptive_validation_registered',job_id=JOB,songs=584,launch=rel(DIRECTORY/'launch.json')),
        run=dict(run_id=JOB,status='registered',stage='F',launch=rel(DIRECTORY/'launch.json')),
        updates=dict(stages=stages,next_action='Run and monitor the registered CPU-faithful V2/mixed/B01 comparison on the fixed 584 validation songs. Reuse only hash-checked receipts after confirmed exit; independently verify the full result before conclusions. Preserve failed release/calibration gates and sealed final tests.'))
    print(json.dumps(dict(job_id=JOB,launch=rel(DIRECTORY/'launch.json'),validation_songs=584,verification_songs=verification_songs)))


def key_for(index,row): return f'{index:04d}-{row["audio_sha256"][:16]}'


def sequence_cache(row,protocol):
    path=ROOT/row['sequence_cache']
    verify_hashes({row['sequence_cache']:protocol['pinned_hashes'][row['sequence_cache']]})
    with np.load(path,allow_pickle=False) as data:
        times,features,key=data['times'].copy(),data['features'].copy(),str(data['feature_key'].item())
    return times,features,key


def labels(row,times,v2_times,protocol):
    if row['split']!='validation': raise ValueError('Only frozen validation label paths may be opened')
    verify_hashes({row['labels_path']:protocol['pinned_hashes'][row['labels_path']]})
    with np.load(ROOT/row['labels_path'],allow_pickle=False) as data:
        return align_labels(times,v2_times,data['times_seconds'],data['targets'],data['mask'],row['duration_seconds'])


def configure(protocol):
    torch.set_num_threads(protocol['threads']); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42); np.random.seed(42)


def stores():
    binding=sha256(DIRECTORY/'launch.json')
    return ArtifactStore(DIRECTORY/'v2_features',binding),ArtifactStore(DIRECTORY/'predictions',binding)


def summary_row(row,receipt):
    return dict(**{k:row[k] for k in ('song_id','source_dataset','group_id','audio_sha256','duration_seconds')},
                **receipt['measurements'])


def collect(protocol,outputs):
    collected=[]
    for i,row in enumerate(protocol['validation_songs']):
        saved=outputs.read(key_for(i,row),row['song_id'])
        if saved is None: raise ValueError('Missing prediction receipt: '+row['song_id'])
        collected.append(summary_row(row,saved[0]))
    return collected


@torch.inference_mode()
def run(protocol,launch,resume):
    if (DIRECTORY/'report.json').exists(): raise ValueError('Completed report exists; verify without restarting inference')
    if (DIRECTORY/'process.json').exists() and not resume: raise ValueError('Existing attempt requires --resume')
    inactive_previous(DIRECTORY/'process.json')
    assert_gpu_available()
    with run_lease(GPU_LEASE):
        if not torch.cuda.is_available(): raise ValueError('Registered sequence GPU unavailable')
        atomic_json(DIRECTORY/'process.json',dict(process_identity=process_identity(),started_at=now(),resumed=resume))
        stages=read_json(PLAN)['stages']; stages['F']['phase']='cross_version_validation_running'
        record_event(dict(event='F_full_descriptive_validation_started',job_id=JOB,resumed=resume),
            run=dict(run_id=JOB,status='running',stage='F'),updates=dict(stages=stages))
        configure(protocol); torch.cuda.reset_peak_memory_stats()
        features,outputs=stores(); all_rows=protocol['validation_songs']; completed=0; completed_frames=0
        start=time.perf_counter(); state('loading_frozen_models',songs_total=len(all_rows),resumed=resume)
        legacy_features,source=load_legacy_source()
        interface=source.EmotionInterface.load(str(ROOT/protocol['models']['v2']['checkpoint']))
        encoder=V2Encoder(protocol,'cpu')
        mixed=SequenceHeads(protocol,'mixed',launch['sequence_device'])
        b01=SequenceHeads(protocol,'b01',launch['sequence_device'])
        startup=time.perf_counter()-start
        for i,row in enumerate(all_rows):
            key=key_for(i,row); existing=outputs.read(key,row['song_id'])
            if existing is not None:
                prior=features.read(key,row['song_id'])
                if prior is None or existing[0]['measurements']['v2_feature_sha256']!=prior[0]['sha256']:
                    raise ValueError('Prediction is missing its bound feature cache')
                completed+=1; completed_frames+=prior[0]['measurements']['feature_frames']; continue
            state('checking_song',songs_completed=completed,songs_total=len(all_rows),song_id=row['song_id'],completed_feature_frames=completed_frames)
            times,x,feature_key=sequence_cache(row,protocol)
            saved=features.read(key,row['song_id'])
            if saved is None:
                if shutil.disk_usage(ROOT).free<2*1024**3: raise ValueError('Less than 2 GiB free inside Vm')
                verify_hashes({row['audio_path']:row['audio_sha256']})
                before=time.perf_counter(); y,sr=legacy_features.load_audio(str(ROOT/row['audio_path']))
                decoding=time.perf_counter()-before
                if sr!=22050: raise ValueError('Unexpected unchanged V2 decoder rate')
                def progress(done,total):
                    if done%160==0 or done==total:
                        state('extracting_v2_cpu_features',song_id=row['song_id'],songs_completed=completed,songs_total=len(all_rows),
                              current_song_frames=done,current_song_total_frames=total,completed_feature_frames=completed_frames)
                cold_load=encoder.model is None
                before=time.perf_counter(); vt,embeddings=encoder.extract(y,42,progress)
                encoding=time.perf_counter()-before
                receipt=features.save(key,row['song_id'],dict(times=vt,features=embeddings),
                    dict(feature_frames=len(vt),decode_seconds=decoding,encoder_seconds=encoding,
                         decoded_22050_samples=len(y),source_audio_sha256=row['audio_sha256'],seed=42,device='cpu',
                         cold_encoder_load_included=cold_load))
                del y
            else:
                receipt,cache=saved; vt,embeddings=cache['times'],cache['features']
            if (embeddings.dtype!=np.float32 or embeddings.shape!=(len(vt),512)
                    or not np.isfinite(embeddings).all()): raise ValueError('Invalid saved V2 embedding')
            completed_frames+=len(vt)
            state('predicting_song',song_id=row['song_id'],songs_completed=completed,songs_total=len(all_rows))
            before=time.perf_counter(); v2raw=np.stack([v['emotion'] for v in interface.predict_sequence(embeddings)])
            v2_time=time.perf_counter()-before
            if v2raw.shape!=(len(vt),5) or not np.isfinite(v2raw).all() or np.any(np.abs(v2raw)>1):
                raise ValueError('Invalid legacy five-output prediction')
            torch.cuda.synchronize(); before=time.perf_counter()
            mixed_raw,_=mixed.predict(times,x,feature_key,row['duration_seconds'])
            b01_raw,ensemble=b01.predict(times,x,feature_key,row['duration_seconds'])
            torch.cuda.synchronize(); seq_time=time.perf_counter()-before
            aligned=labels(row,times,vt,protocol)
            v2_aligned=np.full((len(times),2),np.nan)
            supported=aligned['v2_indices']>=0
            v2_aligned[supported]=v2raw[aligned['v2_indices'][supported],:2]
            pred=np.concatenate([v2_aligned[None],mixed_raw,b01_raw,ensemble[None]],axis=0)
            scores=song_scores(times,aligned['y'],aligned['common_mask'],pred,aligned['native_seconds'])
            arrays=dict(times=times,v2_times=vt,v2_raw_five=v2raw,predictions=pred,
                        y=aligned['y'],common_mask=aligned['common_mask'],original_mask=aligned['original_mask'],
                        v2_indices=aligned['v2_indices'],native_seconds=np.array(aligned['native_seconds']))
            measurements=dict(scores=scores,excluded_values=aligned['excluded_values'].tolist(),
                original_values=aligned['original_mask'].sum(0).tolist(),common_values=aligned['common_mask'].sum(0).tolist(),
                sequence_feature_frames=len(times),v2_feature_frames=len(vt),v2_feature_sha256=receipt['sha256'],
                decoder_seconds=receipt['measurements']['decode_seconds'],encoder_seconds=receipt['measurements']['encoder_seconds'],
                legacy_head_seconds=v2_time,sequence_heads_seconds=seq_time,existing_sequence_cache_reused=True)
            outputs.save(key,row['song_id'],arrays,measurements); completed+=1
            state('song_completed',song_id=row['song_id'],songs_completed=completed,songs_total=len(all_rows),
                  completed_feature_frames=completed_frames,excluded_values=measurements['excluded_values'],
                  elapsed_this_invocation_seconds=time.perf_counter()-start)
        encoder.unload(); del encoder,interface,mixed,b01; torch.cuda.empty_cache()
        state('aggregating_paired_bootstrap',songs_completed=completed,songs_total=len(all_rows))
        rows=collect(protocol,outputs)
        aggregate=summarize(rows,protocol['scoring']['bootstrap'],protocol['scoring']['metrics'])
        verify_hashes(launch['pinned_hashes'])
        report=dict(job_id=JOB,status='completed_pending_independent_verification',completed_at=now(),
            launch_sha256=sha256(DIRECTORY/'launch.json'),protocol_sha256=sha256(PROTOCOL),data_access_note=read_json(ADDENDUM),
            validation_songs=len(rows),final_test_songs_used=0,training_performed=False,parameter_search=False,
            application_defaults_changed=False,promoted_model=False,protected_models_unchanged=True,
            systems=list(SYSTEMS),aggregate=aggregate,songs=rows,
            original_label_values=np.sum([r['original_values'] for r in rows],axis=0).tolist(),
            shared_label_values=np.sum([r['common_values'] for r in rows],axis=0).tolist(),
            excluded_terminal_values=np.sum([r['excluded_values'] for r in rows],axis=0).tolist(),
            total_audio_seconds=sum(r['duration_seconds'] for r in rows),total_v2_frames=completed_frames,
            compute=dict(startup_seconds_this_invocation=startup,elapsed_this_invocation_seconds=time.perf_counter()-start,
                completed_song_encoder_seconds=sum(r['encoder_seconds'] for r in rows),
                peak_torch_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_torch_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                legacy_device='cpu',sequence_device=launch['sequence_device'],timing_excludes_failed_attempts=True),
            limits=protocol['limitations']+[
                'Historical validation curves and summaries were previously parsed; see the pinned data-access addendum.',
                'V2 historical training provenance is incomplete; this is not a proven unseen evaluation for V2.',
                'B01 remains below its pre-registered internal release gate; this descriptive comparison cannot promote or retune it.',
                'Primary B01 is mean per-seed metric, not the secondary fixed arithmetic curve ensemble.',
                'Lag peaks are diagnostic only; no label or prediction shift was applied.',
                'No confidence calibration or display smoothing; target-domain audio and human review remain pending.'])
        atomic_json(DIRECTORY/'report.json',report)
        state('completed_pending_independent_verification',songs_completed=completed,songs_total=len(all_rows))
        record_event(dict(event='F_full_descriptive_validation_finished',job_id=JOB,report=rel(DIRECTORY/'report.json')),
            run=dict(run_id=JOB,status='pending_verification',stage='F',report=rel(DIRECTORY/'report.json')),
            updates=dict(next_action='Independently verify all 584 receipts/metrics, reload five checkpoint heads, and recheck the fixed unmodified V2 extraction. Then render the 18 fixed comparison curves and report descriptive findings without promotion or test-set tuning.'))


@torch.inference_mode()
def verify(protocol,launch):
    if (DIRECTORY/'verification.json').exists(): raise ValueError('Preserve verified report')
    inactive_previous(DIRECTORY/'process.json'); inactive_previous(DIRECTORY/'verification-process.json')
    atomic_json(DIRECTORY/'verification-process.json',dict(process_identity=process_identity(),started_at=now()))
    configure(protocol); features,outputs=stores(); report=read_json(DIRECTORY/'report.json')
    if report['launch_sha256']!=sha256(DIRECTORY/'launch.json') or report['protocol_sha256']!=sha256(PROTOCOL):
        raise ValueError('Report provenance mismatch')
    state('fresh_process_verification',songs_total=584)
    source_features,source=load_legacy_source()
    interface=source.EmotionInterface.load(str(ROOT/protocol['models']['v2']['checkpoint']))
    mixed=SequenceHeads(protocol,'mixed','cpu'); b01=SequenceHeads(protocol,'b01','cpu')
    maximum=0.; verified=[]; model_rechecked=[]
    for i,row in enumerate(protocol['validation_songs']):
        key=key_for(i,row); feature=features.read(key,row['song_id']); output=outputs.read(key,row['song_id'])
        if feature is None or output is None: raise ValueError('Missing completed song')
        fr,fc=feature; rr,arrays=output
        if fr['sha256']!=rr['measurements']['v2_feature_sha256']: raise ValueError('Output/feature binding mismatch')
        aligned=labels(row,arrays['times'],fc['times'],protocol)
        for field in ('y','original_mask','common_mask','v2_indices'):
            np.testing.assert_array_equal(aligned[field],arrays[field])
        if float(arrays['native_seconds'])!=aligned['native_seconds']: raise ValueError('Native cadence changed')
        scores=song_scores(arrays['times'],arrays['y'],arrays['common_mask'],arrays['predictions'],aligned['native_seconds'])
        if scores!=rr['measurements']['scores']: raise ValueError('Per-song metrics changed')
        if row['song_id'] in launch['verification_songs']:
            v2raw=np.stack([v['emotion'] for v in interface.predict_sequence(fc['features'])])
            np.testing.assert_array_equal(v2raw,arrays['v2_raw_five'])
            supported=aligned['v2_indices']>=0
            np.testing.assert_array_equal(v2raw[aligned['v2_indices'][supported],:2],arrays['predictions'][0,supported])
            times,x,key_=sequence_cache(row,protocol)
            np.testing.assert_array_equal(times,arrays['times'])
            for adapter,offset in ((mixed,1),(b01,2)):
                # Independent training collator, rather than the inference wrapper.
                song=dict(x=x,y=np.zeros((len(x),2),np.float32),mask=np.zeros((len(x),2),bool))
                tensor,_,_,lengths=collate_songs([song],*adapter.scaler)
                actual=np.stack([model(tensor,lengths).numpy()[0] for model in adapter.models])
                saved=arrays['predictions'][offset:offset+len(actual)]
                np.testing.assert_allclose(actual,saved,rtol=1e-5,atol=1e-6)
                maximum=max(maximum,float(np.max(np.abs(actual-saved))))
            np.testing.assert_array_equal(arrays['predictions'][2:5].mean(0),arrays['predictions'][5])
            model_rechecked.append(row['song_id'])
        if row['song_id']==launch['original_encoder_recheck_song']:
            state('unchanged_v2_full_song_encoder_recheck',song_id=row['song_id'])
            y,sr=source_features.load_audio(str(ROOT/row['audio_path'])); torch.manual_seed(42); np.random.seed(42)
            original=source.extract_clap_embeddings(y,sr,clap_source=ROOT/protocol['v2_features']['snapshot'],batch_size=16)
            np.testing.assert_array_equal(original,fc['features'])
        verified.append(summary_row(row,rr))
        if (i+1)%100==0: state('verifying_song_receipts',songs_completed=i+1,songs_total=584)
    aggregate=summarize(verified,protocol['scoring']['bootstrap'],protocol['scoring']['metrics'])
    if aggregate!=report['aggregate'] or verified!=report['songs']: raise ValueError('Aggregate/saved rows changed')
    verify_hashes(launch['pinned_hashes'])
    result=dict(job_id=JOB,status='passed',verified_at=now(),report_sha256=sha256(DIRECTORY/'report.json'),
        protocol_sha256=sha256(PROTOCOL),launch_sha256=sha256(DIRECTORY/'launch.json'),verified_songs=len(verified),
        fresh_checkpoint_loads=5,model_recheck_songs=model_rechecked,sequence_cpu_gpu_maximum_difference=maximum,
        original_v2_full_song_features_bitwise_equal=True,original_v2_head_outputs_bitwise_equal=True,
        masks_labels_metrics_and_bootstrap_reproduced=True,protected_models_unchanged=True,final_test_songs_used=0)
    atomic_json(DIRECTORY/'verification.json',result)
    stages=read_json(PLAN)['stages']; stages['F'].update(phase='cross_version_validation_verified',
        comparison_report=rel(DIRECTORY/'report.json'),comparison_verification=rel(DIRECTORY/'verification.json'))
    record_event(dict(event='F_full_descriptive_validation_verified',**result),
        run=dict(run_id=JOB,status='completed_verified',stage='F',verification=rel(DIRECTORY/'verification.json')),
        updates=dict(stages=stages,next_action='Render the frozen 18 representative truth/V2/mixed/B01 curves, means, amplitudes and diagnostic lags from verified raw outputs. Report all-domain descriptive scores and seed/group intervals. Keep no promotion, sealed tests, and human/audio blockers explicit.'))
    state('completed_verified',verified_songs=len(verified))


def status():
    held=is_running(JOB); process_path=DIRECTORY/('verification-process.json' if (DIRECTORY/'verification-process.json').exists() else 'process.json')
    process=read_json(process_path) if process_path.exists() else None
    alive=process_alive(process['process_identity']) if process else None
    verified=False
    if (DIRECTORY/'verification.json').exists():
        v=read_json(DIRECTORY/'verification.json')
        verified=(v['status']=='passed' and v['report_sha256']==sha256(DIRECTORY/'report.json')
                  and v['protocol_sha256']==sha256(PROTOCOL) and v['launch_sha256']==sha256(DIRECTORY/'launch.json'))
    tail=[]; log=DIRECTORY/'console.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-6144)); tail=stream.read().decode('utf-8',errors='replace').splitlines()[-4:]
    outcome=('active' if held and alive is True else 'active_identity_requires_inspection' if held
             else 'completed_verified' if verified else 'pending_verification' if (DIRECTORY/'report.json').exists()
             else 'registered' if process is None and (DIRECTORY/'launch.json').exists() else 'inactive_requires_inspection')
    return dict(job_id=JOB,checked_at=now(),os_lease_active=held,gpu_lease_active=is_running(GPU_LEASE),
        process_alive=alive,process_identity=process['process_identity'] if process else None,outcome=outcome,
        completed_feature_receipts=len(list((DIRECTORY/'v2_features').glob('*.json'))),
        completed_prediction_receipts=len(list((DIRECTORY/'predictions').glob('*.json'))),
        progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None,bounded_log_tail=tail)


def main():
    parser=argparse.ArgumentParser(description=__doc__); modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--tests',action='store_true'); modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--status',action='store_true'); modes.add_argument('--launch',type=Path)
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    if args.status: print(json.dumps(status(),ensure_ascii=False)); return
    with run_lease(JOB):
        for folder in (DIRECTORY,DIRECTORY/'v2_features',DIRECTORY/'predictions'): folder.mkdir(parents=True,exist_ok=True)
        if args.tests: tests(); return
        if args.prepare: prepare(); return
        if args.launch.resolve()!=DIRECTORY/'launch.json': raise ValueError('Wrong registered launch file')
        with (DIRECTORY/'console.log').open('a',encoding='utf-8',buffering=1) as stream:
            with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
                try:
                    launch=read_json(args.launch)
                    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(launch['python']).resolve():
                        raise ValueError('Wrong workspace or independent GPU Python')
                    if any(os.environ.get(k)!=v for k,v in launch['environment'].items()): raise ValueError('Frozen environment changed')
                    verify_hashes(launch['pinned_hashes'])
                    protocol=read_json(PROTOCOL)
                    if args.verify: verify(protocol,launch)
                    else: run(protocol,launch,args.resume)
                except Exception as exc:
                    failure=dict(at=now(),error=f'{type(exc).__name__}: {exc}',mode='verify' if args.verify else 'run',traceback=traceback.format_exc())
                    atomic_json(DIRECTORY/('failure-'+str(time.time_ns())+'.json'),failure)
                    state('failed',error=failure['error'])
                    record_event(dict(event='F_full_descriptive_validation_failed',job_id=JOB,**failure),run=dict(run_id=JOB,status='failed',stage='F'))
                    traceback.print_exc(); raise


if __name__=='__main__': main()
