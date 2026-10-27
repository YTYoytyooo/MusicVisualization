"""Metadata-only registration of the fixed, descriptive validation comparison."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest

import torch

from tools.datasets.deam_dataset import DeamSequences, contained_path
from tools.train_next.core import atomic_json, sha256
from tools.train_next.features import local_clap_snapshot, feature_spec
from tools.train_next.runtime import is_running, lock_path
from tools.train_next_v2.common import ROOT, BASE, PLAN, GPU_LEASE, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.calibration.job_v2 import PROTECTED, verified_checkpoint_hash
from tools.train_next_v2.run import Tee

JOB='f-cross-version-preflight-v1'
DIRECTORY=BASE/'comparison_jobs'/JOB
PROTOCOL=BASE/'protocols/f-v2-mixed-b01-validation-v1.json'
TEST_DIRECTORY=BASE/'analyses/f-cross-version-contract-tests-v1'
B_PROTOCOL=BASE/'protocols/b-v1.json'
PRIOR=BASE/'inference_jobs/f-research-reference-benchmark-v1'


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def source_hashes():
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md',
        ROOT/'tools/train_next_v2/tests/test_cross_version.py']
    return {rel(p):sha256(p) for p in files}


def state(stage, **values):
    data={**values,'job_id':JOB,'stage':stage,'updated_at':now()}
    atomic_json(DIRECTORY/'status.json',data)
    print(json.dumps(data,ensure_ascii=False),flush=True)


def tests():
    report=TEST_DIRECTORY/'report.json'
    if report.exists(): raise ValueError('Preserve existing tests')
    TEST_DIRECTORY.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','-m',__package__+'.preflight','--tests']
    record_event(dict(event='F_cross_version_tests_registered',argv=command,cwd=str(ROOT),code=source_hashes()))
    atomic_json(TEST_DIRECTORY/'launch.json',dict(original_argv=command,cwd=str(ROOT),
        process_identity=process_identity(),code=source_hashes(),
        initial_red='Before implementation, test import failed with ModuleNotFoundError for cross_version.'))
    with (TEST_DIRECTORY/'console.log').open('w',encoding='utf-8') as stream:
        result=unittest.TextTestRunner(stream=Tee(sys.stdout,stream),verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName('tools.train_next_v2.tests.test_cross_version'))
    value=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),code=source_hashes(),at=now())
    atomic_json(report,value)
    record_event(dict(event='F_cross_version_tests_finished',**value))
    if not result.wasSuccessful(): raise ValueError('Contract tests failed')


def prepare():
    if PROTOCOL.exists() or (DIRECTORY/'launch.json').exists(): raise ValueError('Preserve frozen comparison')
    test=read_json(TEST_DIRECTORY/'report.json')
    if test['status']!='passed' or test['tests']!=7 or test['skipped']: raise ValueError('Fresh tests required')
    verify_hashes(test['code']); verify_hashes(PROTECTED)
    b=read_json(B_PROTOCOL)
    if sha256(B_PROTOCOL)!='5ee479f2740f9fd2ddff0f11ed9bd381e70477ecaccb9493d686875f8bf09e6b':
        raise ValueError('B contract changed')
    hashes={**PROTECTED,**source_hashes()}
    previous=read_json(PRIOR/'launch.json')
    verify_hashes(previous['pinned_hashes']); hashes.update(previous['pinned_hashes'])
    pv=read_json(PRIOR/'verification.json')
    if pv['status']!='passed' or pv['report_sha256']!=sha256(PRIOR/'report.json'): raise ValueError('Prior inference audit invalid')
    members=[]
    for seed in (20261003,20261004,20261005):
        rid=f'b01-branch-fusion-s{seed}'
        directory=BASE/'runs'/rid
        config,report=read_json(directory/'config.json'),read_json(directory/'report.json')
        check_path=BASE/'reports'/(rid+'-verification.json')
        check=read_json(check_path)
        checkpoint=ROOT/'models/next_va_v2'/rid/'best.pth'
        if (is_running(rid) or report['status']!='completed' or report['completed_steps']!=20000
                or config['protocol_sha256']!=sha256(B_PROTOCOL) or check['report_sha256']!=sha256(directory/'report.json')
                or verified_checkpoint_hash(check)!=sha256(checkpoint) or report['checkpoint_sha256']!=sha256(checkpoint)
                or not check.get('fresh_process_best_load_verified',check.get('fresh_process_checkpoint_load_verified',False))
                or any(report[k] for k in ('original_validation_songs_used','test_songs_used','calibration_songs_used'))):
            raise ValueError('B01 checkpoint provenance incomplete: '+rid)
        verify_hashes(config['code']); hashes.update(config['code'])
        for path in (directory/'config.json',directory/'report.json',check_path,checkpoint): hashes[rel(path)]=sha256(path)
        members.append(dict(run_id=rid,seed=seed,checkpoint=rel(checkpoint),checkpoint_sha256=sha256(checkpoint),config=rel(directory/'config.json')))
    mixed=torch.load(ROOT/'models/next_va/mixed-20261003-recovery/best.pth',map_location='cpu',weights_only=True)
    hashes.update({'tools/train_next/'+k:v for k,v in mixed['config']['code'].items()})
    spec=feature_spec(local_clap_snapshot())
    if spec!=b['feature_spec'] or mixed['feature_spec']!=spec: raise ValueError('Frozen sequence features differ')
    dataset=DeamSequences(Path(b['source_manifest']).parent,split='validation')
    if len(dataset)!=584 or sha256(b['source_manifest'])!=b['manifest_sha256']: raise ValueError('Validation identities changed')
    record_event(dict(event='F_cross_version_metadata_freeze_started',original_argv=[sys.executable,'-u','-m',__package__+'.preflight','--prepare'],
        cwd=str(ROOT),code=source_hashes(),process_identity=process_identity(),labels_loaded=False,os_lock=str(lock_path(JOB))))
    rows=[]
    for i,row in enumerate(sorted(dataset.records,key=lambda r:r['song_id']),1):
        label=contained_path(dataset.root,row['labels_path'])
        cache=ROOT/'data/training/next_va/features'/b['feature_key']/(row['audio_sha256']+'.npz')
        audio=contained_path(dataset.root,row['audio_path'])
        if sha256(label)!=mixed['config']['labels_sha256'][row['song_id']]: raise ValueError('Historical validation label changed')
        if not cache.exists(): raise ValueError('Missing sequence feature cache')
        row=dict(row,audio_path=rel(audio),labels_path=rel(label),sequence_cache=rel(cache))
        hashes[rel(label)]=mixed['config']['labels_sha256'][row['song_id']]
        hashes[rel(cache)]=sha256(cache); hashes[rel(audio)]=row['audio_sha256']; rows.append(row)
        if i%100==0 or i==len(dataset): state('freezing_validation_metadata',done=i,total=len(dataset))
    diagnostics=ROOT/'data/training/next_va/diagnostics/mixed-20261003-curves/diagnostics.json'
    representatives=read_json(diagnostics)['representatives']
    probe=read_json(ROOT/previous['protocol'])['songs']
    for path in (B_PROTOCOL,Path(b['source_manifest']),Path(b['source_manifest']).parent/'config.json',
                 PRIOR/'report.json',PRIOR/'verification.json',ROOT/previous['protocol'],diagnostics,
                 TEST_DIRECTORY/'report.json',ROOT/'versions/v2.0/emotion_model.py',ROOT/'versions/v2.0/feature_extraction.py',
                 ROOT/'versions/v2.0/studio/pipeline.py',ROOT/'tools/train_next_v2/research_inference/adapter.py',
                 ROOT/'tools/train_next_v2/research_inference/benchmark.py',
                 ROOT/'data/training/next_va/env/Lib/site-packages/transformers/models/clap/feature_extraction_clap.py'):
        hashes[rel(path)]=sha256(path)
    for path in local_clap_snapshot().iterdir():
        if path.is_file() and path.suffix in ('.json','.txt','.safetensors'): hashes[rel(path)]=sha256(path)
    for row in probe:
        cache=BASE/'inference_features/research-reference-v1'/(row['audio_sha256']+'.npz')
        hashes[rel(cache)]=sha256(cache); hashes[rel(cache.with_suffix('.receipt.json'))]=sha256(cache.with_suffix('.receipt.json'))
    verify_hashes(hashes)
    protocol=dict(id='f-v2-mixed-b01-validation-v1',frozen_at=now(),status='frozen_pending_loader_preflight',
        B_protocol=rel(B_PROTOCOL),B_protocol_sha256=sha256(B_PROTOCOL),
        models=dict(v2=dict(checkpoint='models/emotion_model.pth',outputs=['valence','arousal','energy','tension','brightness']),
            mixed=dict(checkpoint='models/next_va/mixed-20261003-recovery/best.pth'),b01=dict(members=members)),
        v2_features=dict(snapshot=rel(local_clap_snapshot()),seed=42,batch_size=16,decode='unchanged V2 load_audio: librosa.load mono float32 at 22050',
            hop_seconds=.1,window_seconds=2,anchor='start',padding='unchanged processor padding=True; original is_longer flags',
            model_window_frames=10,temporal_head='left-zero-pad history, last-window-frame prediction',raw_clap_features=True),
        sequence_features=dict(key=b['feature_key'],spec=spec),validation_songs=rows,counts=dict(Counter(r['source_dataset'] for r in rows)),
        label_values_loaded_during_preparation=False,final_test_labels_used=0,
        scoring=dict(axes=['valence','arousal'],units='[-1,1]',primary_b01='mean of the three independently computed per-seed metrics; report each seed and sample std',
            ensemble='fixed arithmetic mean raw curve as secondary result; never select best seed',
            clock='exact label timestamps joined to 0.5/0.1 s model clocks; independent-axis common masks; unsupported V2 terminal points excluded equally from all scores and counted',
            aggregation='song macro per dataset/axis, then equal weight six dataset/axis cells',
            metrics=['ccc','mae','rmse','mean_absolute_bias','centered_rmse','derivative_mae','truth_std','prediction_std','std_ratio','truth_span90','prediction_span90','constant_prediction_mae','dynamics_beats_constant'],
            derivative='real elapsed seconds on adjacent valid native labels only; never bridge gaps',
            lag='diagnostic Pearson y(t) vs p(t+lag), same central valid points, native cadence, +/-8 s capped at quarter observed length; flag ambiguity; do not correct labels/predictions',
            bootstrap=dict(replicates=2000,seed=20261004,confidence=.95,unit='transitive group_id/audio_sha256 recording groups; paired draws shared across systems and seeds; stratify by dataset if no cross-dataset components',
                cross_dataset_policy='single joint cluster draw if any component crosses datasets; never split a shared cluster'),
            display_smoothing=False,calibration=False,parameter_search=False),representatives=representatives,
        limitations=['Validation was previously used to select mixed checkpoints; this is descriptive comparison, not untouched generalization evidence.',
            'V2 checkpoint lacks complete training-song provenance; absence of training overlap cannot be established.',
            'B01 failed the frozen internal non-inferiority screen; comparison cannot reverse that screen or authorize deployment.',
            'The existing 18 illustrative songs were selected using mixed diagnostics; use all-584 metrics for aggregate comparisons.',
            'Comparison concerns raw V/A in the registered independent environment, not all historical V2 caches or five-dimensional application behavior.'],
        probe_songs=probe,probe_cache=rel(BASE/'inference_features/research-reference-v1'),
        encoder_probe=dict(song_id=probe[0]['song_id'],start_seconds=0,duration_seconds=3.23,selection='first 3.23 seconds of fixed shortest unannotated benchmark song',
            feature_rtol=1e-4,feature_atol=1e-4,head_rtol=1e-5,head_atol=1e-6),threads=4,deterministic=True,pinned_hashes=hashes)
    atomic_json(PROTOCOL,protocol)
    argv=[str(Path(read_json(PLAN)['python']).resolve()),'-u','-m',__package__+'.preflight','--launch',str(DIRECTORY/'launch.json')]
    launch=dict(job_id=JOB,registered_at=now(),original_argv=argv,resume_argv=argv+['--resume'],verification_argv=argv+['--verify'],
        cwd=str(ROOT),python=argv[0],environment=previous['environment'],protocol=rel(PROTOCOL),protocol_sha256=sha256(PROTOCOL),code=source_hashes(),
        os_lock=str(lock_path(JOB)),gpu_lock=str(lock_path(GPU_LEASE)),log=rel(DIRECTORY/'console.log'),receipts=rel(DIRECTORY/'receipts'),
        recovery='Confirm process exited and leases released. Preserve and inspect partial outputs. Use exact original argv + --resume; only accept hash-verified step receipts. Completed report requires --verify.',
        status_module=__package__+'.preflight --status')
    atomic_json(DIRECTORY/'launch.json',launch)
    stages=read_json(PLAN)['stages']; stages['F'].update(phase='cross_version_loader_preflight_registered',comparison_protocol=rel(PROTOCOL))
    record_event(dict(event='F_cross_version_preflight_registered',job_id=JOB,protocol=rel(PROTOCOL),validation_songs=584),
        run=dict(run_id=JOB,status='registered',stage='F',launch=rel(DIRECTORY/'launch.json')),
        updates=dict(stages=stages,next_action='Run the fixed V2/mixed/B01 loader/encoder preflight and independent verification, then implement the frozen 584-song descriptive comparison. No final-test labels, tuning or app replacement.'))
    print(json.dumps(dict(protocol=rel(PROTOCOL),songs=len(rows),counts=protocol['counts'],launch=rel(DIRECTORY/'launch.json'))))
