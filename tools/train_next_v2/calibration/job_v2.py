"""Register, execute and independently verify the first untouched calibration audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from tools.datasets.deam_dataset import DeamSequences, contained_path
from tools.train_next.core import atomic_json, attach_targets, collate_songs, sha256
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import (AXES, BASE, GPU_LEASE, PLAN, ROOT,
    assert_gpu_available, code_hashes, now, paths, process_identity, read_json, record_event, verify_hashes)
from tools.train_next_v2.data import CachedFeatureStore
from tools.train_next_v2.evaluation import metrics_for_axis
from tools.train_next_v2.model import SequenceModel
from tools.train_next_v2.run import Tee
from .methods import (audit_intervals, axis_evidence, calibration_split, fit_intervals, readiness)

JOB = 'e-b00-grouped-calibration-v2'
DIRECTORY = BASE/'calibration_jobs'/JOB
PROTOCOL = BASE/'protocols/e-b00-grouped-calibration-v2.json'
B_PROTOCOL = BASE/'protocols/b-v1.json'
TEST_REPORT = BASE/'analyses/e-calibration-method-tests-v3/report.json'
PROTECTED = {
    'models/emotion_model.pth': 'f0c304aec8120ba09e21b2540cc5ae0de23a1536c449673edd0388b896e31667',
    'models/next_va/teacher-20261003-recovery/best.pth': '73aedc4c76b6add3eedbfc9b4c56ccec60fb01956865b5dd042b86173d20d67b',
    'models/next_va/mixed-20261003-recovery/best.pth': 'bcc942e4955b90125c75dd0a6f4f5fa927f2119cc07da7d5ecb54423288e337d',
}


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def source_hashes():
    files = list(Path(__file__).parent.glob('*.py')) + [
        ROOT/'tools/train_next_v2/analysis/paired_confirmation.py',
        ROOT/'tools/train_next_v2/tests/test_calibration_v2.py']
    return {**code_hashes(), **{relative(p):sha256(p) for p in files}}


def state(stage, **extra):
    value = dict(job_id=JOB, stage=stage, updated_at=now(), **extra)
    atomic_json(DIRECTORY/'status.json', value)
    print(json.dumps(value), flush=True)


def verified_checkpoint_hash(check):
    values = [check[k] for k in ('checkpoint_sha256','best_checkpoint_sha256') if k in check]
    if not values or len(set(values)) != 1 or len(values[0]) != 64:
        raise ValueError('Missing or conflicting independently verified checkpoint hash')
    return values[0]


def require_input_membership(inputs, split):
    if (set(inputs['labels_sha256']) != {r['song_id'] for r in split}
            or set(inputs['feature_hashes']) != {r['audio_sha256'] for r in split}):
        raise ValueError('Calibration-only input hash membership differs from the frozen split')
    if any(len(v)!=64 for mapping in (inputs['labels_sha256'],inputs['feature_hashes']) for v in mapping.values()):
        raise ValueError('Invalid calibration input digest')


def freeze_calibration_inputs(b, split):
    """B deliberately excludes calibration hashes; pin their historical labels here."""
    baseline=ROOT/read_json(PLAN)['baseline']['checkpoint']
    if sha256(baseline)!=b['baseline_checkpoint_sha256']:
        raise ValueError('Historical label-provenance checkpoint changed')
    metadata=torch.load(baseline,map_location='cpu',weights_only=True)
    historical_hashes=metadata['config']['labels_sha256']
    dataset=DeamSequences(Path(b['source_manifest']).parent,split='train')
    records={r['song_id']:r for r in dataset.records}
    store=CachedFeatureStore(torch.device('cpu'))
    if store.spec!=b['feature_spec'] or store.key!=b['feature_key']:
        raise ValueError('Frozen feature specification changed')
    features,labels={},{}
    for done,row in enumerate(split,1):
        original=records[row['song_id']]
        if b['split']['assignments'][row['song_id']]!='calibration':
            raise ValueError('Non-calibration input requested')
        label=contained_path(dataset.root,row['labels_path'])
        expected=historical_hashes[row['song_id']]
        if sha256(label)!=expected:raise ValueError('Historical calibration label changed')
        labels[row['song_id']]=expected
        path=store.directory/(row['audio_sha256']+'.npz')
        if not path.is_file():raise ValueError('Calibration cache absent; extraction forbidden')
        digest=sha256(path)
        # Validates source audio SHA, feature key, dimensions, finite values and native audio clock.
        times,values,cached=store.get(original,dataset.root)
        if not cached or digest!=sha256(path):raise ValueError('Cache mutated during audit')
        features[row['audio_sha256']]=digest
        if done%100==0 or done==len(split):
            print(json.dumps(dict(stage='freeze_calibration_inputs',songs=done,total=len(split))),flush=True)
    store.unload()
    inputs=dict(feature_hashes=features,labels_sha256=labels,
        historical_label_hash_source=relative(baseline),historical_label_hash_source_sha256=b['baseline_checkpoint_sha256'],
        feature_validation='Existing cache only; unchanged B feature spec; source audio SHA, exact native feature clock, shape and finite values checked before pinning file hash.',
        label_values_loaded=False,new_extraction=False)
    require_input_membership(inputs,split)
    return inputs


def prepare():
    if PROTOCOL.exists() or (DIRECTORY/'launch.json').exists():
        raise ValueError('Preserve the registered calibration protocol and launch')
    tests = read_json(TEST_REPORT)
    if tests['status'] != 'passed' or tests['failures'] or tests['errors'] or tests['skipped']:
        raise ValueError('Calibration tests must pass before freezing')
    verify_hashes(tests['code'])
    verify_hashes(PROTECTED)
    b = read_json(B_PROTOCOL)
    verify_hashes({relative(B_PROTOCOL):'5ee479f2740f9fd2ddff0f11ed9bd381e70477ecaccb9493d686875f8bf09e6b',
        relative(b['source_manifest']):b['manifest_sha256']})
    records = [r for r in map(json.loads, Path(b['source_manifest']).read_text(encoding='utf-8').splitlines())
               if r['split'] == 'train']
    split = calibration_split(records, b['split']['assignments'], 20261004)
    if len(split) != 383 or len({r['connected_group'] for r in split}) != 266:
        raise ValueError('Frozen calibration membership changed')
    inputs=freeze_calibration_inputs(b,split)
    members, hashes = [], dict(PROTECTED)
    for seed in (20261003, 20261004, 20261005):
        rid = f'b00-reference-s{seed}'
        run_dir, model_dir = paths(rid)
        if is_running(rid):
            raise ValueError('Reference model is still active')
        config, report = [read_json(run_dir/n) for n in ('config.json', 'report.json')]
        verification_path = BASE/'reports'/f'{rid}-verification.json'
        check = read_json(verification_path)
        if (report['status'] != 'completed' or not report['checkpoint_reload_verified']
                or report['initialization'] != 'random; no previous trained weights'
                or config['protocol_sha256'] != sha256(B_PROTOCOL)
                or report['protocol_sha256'] != sha256(B_PROTOCOL)
                or report['completed_steps'] != 20000 or report['train_songs'] != 3070
                or any(report[k] != 0 for k in ('calibration_songs_used', 'original_validation_songs_used', 'test_songs_used'))
                or check['report_sha256'] != sha256(run_dir/'report.json')
                or not check.get('fresh_process_best_load_verified', check.get('fresh_process_checkpoint_load_verified', False))
                or verified_checkpoint_hash(check) != sha256(model_dir/'best.pth')
                or report['checkpoint_sha256'] != verified_checkpoint_hash(check)):
            raise ValueError('Non-leaking, completed reference provenance missing: '+rid)
        verify_hashes(config['code'])
        for path in [run_dir/'config.json', run_dir/'report.json', verification_path, model_dir/'best.pth']:
            hashes[relative(path)] = sha256(path)
        members.append(dict(run_id=rid, checkpoint=relative(model_dir/'best.pth'),
            checkpoint_sha256=report['checkpoint_sha256'], config=relative(run_dir/'config.json'), seed=seed))
    hashes.update(source_hashes())
    hashes[relative(B_PROTOCOL)] = sha256(B_PROTOCOL)
    hashes[relative(TEST_REPORT)] = sha256(TEST_REPORT)
    manifest = Path(b['source_manifest'])
    hashes[relative(manifest)] = b['manifest_sha256']
    hashes[relative(manifest.parent/'config.json')] = b['prepared_config_sha256']
    counts = {}
    for role in ('interval_fit', 'interval_audit'):
        subset = [r for r in split if r['role'] == role]
        counts[role] = dict(songs=len(subset), groups=len({r['connected_group'] for r in subset}),
            domain_songs=dict(Counter(r['source_dataset'] for r in subset)),
            domain_groups={d:len({r['connected_group'] for r in subset if r['source_dataset'] == d})
                           for d in ('deam','memo2496','pmemo')})
    protocol = dict(id=JOB, schema_version=1, frozen_at=now(), labels_opened_at_freeze=False,
        source_B_protocol=relative(B_PROTOCOL), source_B_protocol_sha256=sha256(B_PROTOCOL),calibration_inputs=inputs,
        members=members, split_seed=20261004, songs=split, counts=counts,
        grouping='Transitive group_id and exact-audio identity over ALL original train; original B roles must be disjoint. Half the groups per domain by deterministic SHA256 order for interval fitting, the remainder for audit.',
        prediction='Fixed arithmetic mean of three retained B00 seeds. No retraining, normalization refit, bias correction, time shift or smoothing.',
        uncertainty=dict(primary='scaled', coverage=.90, std_ddof=1, scale_floor=.02,
            formula='scale=max(std_across_three_seeds,0.02); score=max(abs(target-ensemble_mean)/scale) over all observed timestamps and songs in a connected group, independently per dataset/axis.',
            quantile='ceil((number_of_fit_groups+1)*0.90)-th ordered group score; if rank exceeds count, finite interval unavailable.',
            interval='Unclipped prediction +/- q*scale; width and coverage reported before any display clipping.',
            constant_control='Same group-max procedure with scale=1. Diagnostic only, cannot replace primary based on audit scores.'),
        gates=dict(minimum_fit_groups=9, minimum_audit_groups=9, minimum_audit_group_coverage=.90,
            maximum_song_macro_mae=.15, maximum_point_p90_halfwidth=.25),
        gate_status='First conservative engineering screen on [-1,1] scale, fixed before opening calibration labels; not a promise of target-domain accuracy.',
        inference=dict(device='cuda', batch_size=4, threads=8, deterministic=True, arithmetic='float32; ensemble aggregation float64'),
        bootstrap=dict(seed=20261004, replicates=2000, confidence=.95, resampling='connected song groups within each audit dataset; conditional on fixed ensemble'),
        protected_hashes=hashes, code=source_hashes(),
        pseudo_training_enabled=False,
        additional_pseudo_requirements=['Independent target-pool deduplication against fit/development/calibration/validation/test and all duplicate recordings.',
            'Versioned pool-specific reliability evidence, multiple-model disagreement checks and REAL human review before any pseudo-label training.'],
        limitations=['Group-level exchangeability within the same dataset is an assumption; small group counts give coarse and uncertain coverage.',
            'Known identity groups are not proof of absence of near duplicates, covers or unidentified cross-role artists.',
            'No distribution-free guarantee for new datasets, genres, languages, unlabeled portions of a song or future songs.',
            'Three seeds of one architecture may agree while all are wrong. Audit data cannot be reused to tune this protocol.',
            'CCC excludes near-constant ground truth as in B; report number of valid songs.'],
        reference_method='https://arxiv.org/html/2107.07511v6 (normalized residual split-conformal quantile; group-max construction here is a project-specific adaptation)',
        validation_test_targets_to_read=0)
    previous_path=BASE/'protocols/e-b00-grouped-calibration-v1.json'
    previous=read_json(previous_path)
    for key in ('songs','counts','members','prediction','uncertainty','gates','inference','bootstrap'):
        if previous[key]!=protocol[key]:raise ValueError('Scientific protocol changed during input-contract repair: '+key)
    protocol.update(supersedes_failed_input_contract=relative(previous_path),previous_protocol_sha256=sha256(previous_path),
        repair='Version 1 failed on the first sealed B calibration cache before loading label values or predicting. Identical scientific protocol; v2 explicitly freezes the previously sealed calibration-only inputs.')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    atomic_json(PROTOCOL, protocol)
    python = str(Path(read_json(PLAN)['python']).resolve())
    environment = read_json(BASE/'queues/d-jamendo-paired-training-v1/launch.json')['environment']
    launch_path = DIRECTORY/'launch.json'
    argv = [python,'-u','-m','tools.train_next_v2.calibration.job_v2','--launch',str(launch_path)]
    launch = dict(job_id=JOB, registered_at=now(), original_argv=argv, resume_argv=argv+['--resume'],
        verification_argv=argv+['--verify'], cwd=str(ROOT), python=python, environment=environment,
        protocol=relative(PROTOCOL), protocol_sha256=sha256(PROTOCOL), protected_hashes=hashes,
        os_lock=str(lock_path(JOB)), gpu_lock=str(lock_path(GPU_LEASE)), log=relative(DIRECTORY/'console.log'),
        recovery='Confirm lease released and process creation identity; inspect bounded log. Use original argv plus --resume to reuse only hash-verified per-song prediction receipts. Preserve protocol/config/failed logs; no threshold adjustment or blind retry.',
        checkpoint_directory=relative(DIRECTORY/'predictions'), status_module='tools.train_next_v2.calibration.job_v2 --status')
    atomic_json(launch_path, launch)
    stages = read_json(PLAN)['stages']
    stages['E'].update(phase='calibration_registered', protocol=relative(PROTOCOL), calibration_job=JOB,
        calibration_counts=counts, pseudo_training_enabled=False)
    record_event(dict(event='E_calibration_registered_before_labels',job_id=JOB,counts=counts),
        run=dict(run_id=JOB,status='registered',stage='E',launch=relative(launch_path)),
        updates=dict(stages=stages,current_stage='E',next_action='Run the frozen group-disjoint B00 calibration audit, then fresh checkpoint/prediction verification.'))
    print(json.dumps(dict(job_id=JOB, counts=counts, launch=relative(launch_path))))


def load_calibration(protocol):
    b = read_json(ROOT/protocol['source_B_protocol'])
    manifest = Path(b['source_manifest'])
    dataset = DeamSequences(manifest.parent, split='train')
    chosen = {r['song_id']:r for r in protocol['songs']}
    inputs=protocol['calibration_inputs'];require_input_membership(inputs,protocol['songs'])
    recomputed = calibration_split(dataset.records, b['split']['assignments'], protocol['split_seed'])
    if recomputed != protocol['songs']:
        raise ValueError('Calibration group split changed')
    indices = {r['song_id']:i for i,r in enumerate(dataset.records)}
    store = CachedFeatureStore(torch.device('cpu'))
    if store.spec != b['feature_spec'] or store.key != b['feature_key']:
        raise ValueError('Feature specification or environment changed')
    songs = []
    # No original validation/test or internal fit/dev label file is opened.
    for done, (song_id, row) in enumerate(chosen.items(), 1):
        original = dataset.records[indices[song_id]]
        cache = store.directory/(row['audio_sha256']+'.npz')
        if (sha256(cache) != inputs['feature_hashes'][row['audio_sha256']]
                or sha256(contained_path(dataset.root,row['labels_path'])) != inputs['labels_sha256'][song_id]):
            raise ValueError('Calibration cache/label changed: '+song_id)
        times, features, cached = store.get(original, dataset.root)
        if not cached:
            raise ValueError('New feature extraction forbidden')
        item = dataset[indices[song_id]]
        observed = item['mask'].any(axis=1)
        target, mask = attach_targets(times,item['times_seconds'][observed],item['targets'][observed],item['mask'][observed])
        native = float(np.median(np.diff(item['times_seconds'])))
        if native not in (.5,1.) or not mask.any() or not np.isfinite(features).all():
            raise ValueError('Invalid native clock or supervision')
        if np.any(np.abs(target[mask]) > 1.000001):
            raise ValueError('Targets outside frozen [-1,1] scale')
        songs.append(dict(**row,x=features,y=target,mask=mask,times=times,native_seconds=native))
        if done%100 == 0 or done == len(chosen):
            state('loading_calibration_cache',songs=done,total=len(chosen))
    store.unload()
    return songs


def load_models(protocol, device):
    models, scaler = [], None
    for member in protocol['members']:
        config = read_json(ROOT/member['config'])
        saved = torch.load(ROOT/member['checkpoint'],map_location='cpu',weights_only=True)
        if (saved['provenance'] != config or saved['settings'] != config['settings']
                or saved['label_order'] != list(AXES) or saved['architecture'] != 'whole-song-va-experiment-v2'):
            raise ValueError('Checkpoint provenance or architecture changed')
        mean,std = saved['mean'].numpy(),saved['std'].numpy()
        if mean.shape != (531,) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError('Invalid frozen training scaler')
        if scaler is not None:
            np.testing.assert_array_equal(scaler[0],mean); np.testing.assert_array_equal(scaler[1],std)
        scaler = (mean,std)
        model = SequenceModel(len(mean),config['settings']).to(device)
        model.load_state_dict(saved['state_dict'],strict=True); model.eval(); models.append(model)
    return models, scaler


def prediction_paths(song_id):
    name = hashlib.sha256(song_id.encode()).hexdigest()
    return DIRECTORY/'predictions'/f'{name}.npz', DIRECTORY/'predictions'/f'{name}.json'


def read_prediction(song, protocol_hash):
    path, receipt_path = prediction_paths(song['song_id'])
    receipt = read_json(receipt_path)
    if (receipt['song_id'] != song['song_id'] or receipt['protocol_sha256'] != protocol_hash
            or receipt['audio_sha256'] != song['audio_sha256'] or receipt['sha256'] != sha256(path)):
        raise ValueError('Prediction receipt or hash mismatch')
    with np.load(path,allow_pickle=False) as data:
        values = data['members'].copy()
        np.testing.assert_array_equal(data['times'],song['times'])
    if values.shape != (3,len(song['times']),2) or not np.isfinite(values).all():
        raise ValueError('Prediction checkpoint shape/values changed')
    return values


@torch.inference_mode()
def predictions(protocol, songs, models, scaler, resume):
    folder = DIRECTORY/'predictions'; folder.mkdir(exist_ok=True)
    outputs = {}; reused = 0; start = time.monotonic(); h = sha256(PROTOCOL)
    pending = []
    for song in songs:
        path, receipt = prediction_paths(song['song_id'])
        if path.exists() or receipt.exists():
            if not resume:
                raise ValueError('Existing predictions require --resume')
            outputs[song['song_id']] = read_prediction(song,h); reused += 1
        else:
            pending.append(song)
    torch.cuda.reset_peak_memory_stats()
    for offset in range(0,len(pending),protocol['inference']['batch_size']):
        batch = pending[offset:offset+protocol['inference']['batch_size']]
        x,_,_,lengths = collate_songs(batch,*scaler)
        members = np.stack([model(x.to('cuda'),lengths).cpu().numpy() for model in models])
        for i,song in enumerate(batch):
            values = members[:,i,:len(song['times'])].copy()
            if not np.isfinite(values).all():
                raise ValueError('Non-finite GPU prediction')
            path, receipt_path = prediction_paths(song['song_id'])
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,members=values,times=song['times']);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
            atomic_json(receipt_path,dict(song_id=song['song_id'],audio_sha256=song['audio_sha256'],
                protocol_sha256=h,sha256=sha256(path),shape=list(values.shape),written_at=now()))
            outputs[song['song_id']] = values
        if (offset//protocol['inference']['batch_size'])%10 == 0 or offset+len(batch)==len(pending):
            state('three_seed_inference',songs=len(outputs),total=len(songs),reused=reused)
    return outputs, dict(seconds=time.monotonic()-start,predicted_songs=len(pending),reused_songs=reused,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(),
        scope='Cached-feature sequential three-member inference including atomic prediction writes; excludes feature extraction and original audio decoding.')


def macro(rows):
    if not rows:
        return dict(songs=0,ccc=None,ccc_songs=0,mae=None)
    result = {}
    for name in rows[0]:
        if name == 'values':continue
        valid = [row[name] for row in rows if row[name] is not None]
        result[name] = float(np.mean(valid)) if valid else None
    return dict(**result,songs=len(rows),ccc_songs=sum(r['ccc'] is not None for r in rows))


def bootstrap_audit(per_song, settings):
    rng = np.random.default_rng(settings['seed']); result = {}
    for domain in ('deam','memo2496','pmemo'):
        rows = [r for r in per_song if r['source_dataset']==domain and r['role']=='interval_audit']
        keys = sorted({r['connected_group'] for r in rows}); index={key:i for i,key in enumerate(keys)}
        weights = rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=settings['replicates'])
        result[domain] = {}
        for axis in AXES:
            result[domain][axis] = {}
            for metric in ('ccc','mae','rmse','mean_absolute_bias','centered_rmse','derivative_mae','dynamics_beats_constant'):
                sums=np.zeros(len(keys));counts=np.zeros(len(keys))
                for row in rows:
                    value=row['ensemble'][axis]
                    if value is not None and value[metric] is not None:
                        sums[index[row['connected_group']]]+=value[metric];counts[index[row['connected_group']]]+=1
                num,den=weights@sums,weights@counts; good=den>0
                draws=num[good]/den[good]
                result[domain][axis][metric]=dict(ci95=np.quantile(draws,[.025,.975]).tolist() if len(draws) else None,
                    valid_replicates=int(good.sum()),groups=len(keys))
    return result


def analyze(protocol,songs,outputs):
    evidence=defaultdict(list); per_song=[]
    for song in songs:
        members=outputs[song['song_id']]; ensemble=members.astype(float).mean(axis=0)
        row={k:song[k] for k in ('song_id','source_dataset','group_id','connected_group','role')}
        row['ensemble']={a:metrics_for_axis(ensemble,song,i) for i,a in enumerate(AXES)}
        row['members']=[{a:metrics_for_axis(p,song,i) for i,a in enumerate(AXES)} for p in members]
        per_song.append(row)
        for i,axis in enumerate(AXES):
            item=axis_evidence(members,song['y'],song['mask'],i,protocol['uncertainty']['scale_floor'])
            if item is not None:
                evidence[(song['source_dataset'],axis)].append(dict(**item,role=song['role'],connected_group=song['connected_group']))
    fitted={}
    for (domain,axis),rows in sorted(evidence.items()):
        fitted.setdefault(domain,{})[axis]=fit_intervals([r for r in rows if r['role']=='interval_fit'],protocol['uncertainty']['coverage'])
    parameters=dict(protocol_sha256=sha256(PROTOCOL),method=protocol['uncertainty'],fitted=fitted)
    parameter_path=DIRECTORY/'interval-parameters.json'
    if parameter_path.exists():
        if read_json(parameter_path)!=parameters:raise ValueError('Resumed interval parameters changed')
    else:
        atomic_json(parameter_path,parameters)
    summaries={}; gates={}; coverage={}
    for domain in ('deam','memo2496','pmemo'):
        summaries[domain]={};gates[domain]={};coverage[domain]={}
        for role in ('interval_fit','interval_audit'):
            subset=[r for r in per_song if r['source_dataset']==domain and r['role']==role]
            summaries[domain][role]=dict(
                ensemble={a:macro([r['ensemble'][a] for r in subset if r['ensemble'][a] is not None]) for a in AXES},
                members=[{a:macro([r['members'][i][a] for r in subset if r['members'][i][a] is not None]) for a in AXES} for i in range(3)])
        for axis in AXES:
            audit_rows=[r for r in evidence[(domain,axis)] if r['role']=='interval_audit']
            coverage[domain][axis]={method:audit_intervals(audit_rows,fitted[domain][axis][method],method)
                for method in ('scaled','constant')}
            gates[domain][axis]=readiness(fitted[domain][axis]['scaled'],coverage[domain][axis]['scaled'],
                summaries[domain]['interval_audit']['ensemble'][axis]['mae'],protocol['gates'])
    eligible=all(g['passes'] for d in gates.values() for g in d.values())
    return dict(summaries=summaries,coverage=coverage,per_axis_dataset_readiness=gates,
        all_statistical_gates_pass=eligible,interval_parameters=relative(parameter_path),
        interval_parameters_sha256=sha256(parameter_path),per_song=per_song,
        audit_song_group_bootstrap=bootstrap_audit(per_song,protocol['bootstrap']),
        decision='statistical_screen_passed_human_and_pool_gates_pending' if eligible else 'pseudo_labels_not_eligible',
        pseudo_training_enabled=False,real_human_review_completed=False,
        uncertainty_fitting_uses_audit_labels=False,raw_predictions_modified=False)


def validate_launch(launch_path):
    if launch_path.resolve()!=DIRECTORY/'launch.json':raise ValueError('Use the registered launch path')
    launch=read_json(launch_path)
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(launch['python']).resolve():
        raise ValueError('Wrong working directory or independent Python')
    for key,value in launch['environment'].items():
        if os.environ.get(key)!=value:raise ValueError('Frozen environment changed: '+key)
    verify_hashes(launch['protected_hashes']);verify_hashes({launch['protocol']:launch['protocol_sha256']})
    return launch,read_json(PROTOCOL)


def run(launch,protocol,resume):
    if (DIRECTORY/'report.json').exists():raise ValueError('Completed audit exists; verify it, do not rerun')
    if (DIRECTORY/'process.json').exists() and not resume:raise ValueError('Existing job requires --resume')
    assert_gpu_available()
    with run_lease(GPU_LEASE):
        if not torch.cuda.is_available():raise ValueError('Independent GPU runtime unavailable')
        torch.set_num_threads(protocol['inference']['threads']);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=process_identity(),mode='inference',resumed=resume))
        record_event(dict(event='E_calibration_started',resume=resume),run=dict(run_id=JOB,status='running',stage='E'))
        start=time.monotonic();models,scaler=load_models(protocol,torch.device('cuda'))
        songs=load_calibration(protocol)
        outputs,cost=predictions(protocol,songs,models,scaler,resume)
        result=analyze(protocol,songs,outputs)
        verify_hashes(launch['protected_hashes'])
        result.update(status='completed_pending_independent_verification',job_id=JOB,completed_at=now(),
            protocol=relative(PROTOCOL),protocol_sha256=sha256(PROTOCOL),members=protocol['members'],counts=protocol['counts'],
            feature_songs=len(songs),calibration_labels_used=383,original_validation_labels_used=0,test_labels_used=0,
            cost=cost,total_seconds=time.monotonic()-start,protected_models_unchanged=True,
            limitations=protocol['limitations'],gates=protocol['gates'])
        atomic_json(DIRECTORY/'report.json',result)
        state('completed_pending_independent_verification',decision=result['decision'],songs=len(songs))
        record_event(dict(event='E_calibration_audit_finished',decision=result['decision'],report=relative(DIRECTORY/'report.json')),
            run=dict(run_id=JOB,status='pending_verification',stage='E',report=relative(DIRECTORY/'report.json')))


@torch.inference_mode()
def verify(protocol):
    if (DIRECTORY/'verification.json').exists():raise ValueError('Independent verification already exists')
    atomic_json(DIRECTORY/'verification-process.json',dict(started_at=now(),process_identity=process_identity(),mode='verification_cpu'))
    torch.set_num_threads(2)
    report=read_json(DIRECTORY/'report.json')
    if report['protocol_sha256']!=sha256(PROTOCOL) or report['feature_songs']!=383:
        raise ValueError('Audit report provenance mismatch')
    verify_hashes({report['interval_parameters']:report['interval_parameters_sha256']})
    models,scaler=load_models(protocol,torch.device('cpu'))
    # Recheck all prediction receipts; fresh CPU inference for one song per dataset.
    songs=load_calibration(protocol); outputs={s['song_id']:read_prediction(s,sha256(PROTOCOL)) for s in songs}
    difference=0.;probes=[]
    for domain in ('deam','memo2496','pmemo'):
        song=next(s for s in songs if s['source_dataset']==domain)
        x,_,_,lengths=collate_songs([song],*scaler)
        fresh=np.stack([m(x,lengths).numpy()[0] for m in models])
        old=outputs[song['song_id']]
        np.testing.assert_allclose(fresh,old,rtol=1e-5,atol=1e-6)
        difference=max(difference,float(np.abs(fresh-old).max()));probes.append(song['song_id'])
    recomputed=analyze(protocol,songs,outputs)
    if any(report[key]!=value for key,value in recomputed.items()):
        raise ValueError('Fresh full calibration audit reproduction changed')
    verify_hashes(PROTECTED)
    evidence=dict(status='passed',verified_at=now(),job_id=JOB,report_sha256=sha256(DIRECTORY/'report.json'),
        protocol_sha256=sha256(PROTOCOL),fresh_process_identity=process_identity(),
        fresh_cpu_checkpoint_loads=3,prediction_receipts_verified=383,probe_songs=probes,
        maximum_cpu_gpu_prediction_difference=difference,all_audit_results_reproduced=True,
        protected_models_unchanged=True,original_validation_and_test_labels_used=0)
    atomic_json(DIRECTORY/'verification.json',evidence)
    stages=read_json(PLAN)['stages']
    stages['E'].update(phase='calibration_audit_complete_annotation_preparation_pending',calibration_report=relative(DIRECTORY/'report.json'),
        calibration_verification=relative(DIRECTORY/'verification.json'),calibration_decision=report['decision'],pseudo_training_enabled=False,
        human_review_status='not_completed')
    record_event(dict(event='E_calibration_independently_verified',decision=report['decision']),
        run=dict(run_id=JOB,status='completed_verified',stage='E',verification=relative(DIRECTORY/'verification.json')),
        updates=dict(stages=stages,current_stage='E',next_action='Prepare an honest target-song annotation pilot inventory and human review checklist; do not generate training pseudo-labels unless all calibration/pool/human gates pass.'))
    state('completed_verified',decision=report['decision'],fresh_cpu_gpu_difference=difference)


def status():
    lease=is_running(JOB);gpu=is_running(GPU_LEASE)
    process=read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else None
    if (DIRECTORY/'verification-process.json').exists():process=read_json(DIRECTORY/'verification-process.json')
    identity=process_identity(process['process_identity']['pid']) if process else None
    held_status=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None
    verified=False
    if (DIRECTORY/'verification.json').exists() and (DIRECTORY/'report.json').exists():
        v=read_json(DIRECTORY/'verification.json')
        verified=v['status']=='passed' and v['report_sha256']==sha256(DIRECTORY/'report.json') and v['protocol_sha256']==sha256(PROTOCOL)
    log=DIRECTORY/'console.log'
    tail=[]
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-16384));tail=stream.read().decode('utf-8',errors='replace').splitlines()[-3:]
    result=dict(job_id=JOB,checked_at=now(),os_lease_active=lease,gpu_lease_active=gpu,
        process_identity_matches=bool(process and identity==process['process_identity']),
        outcome='active' if lease else 'completed_verified' if verified else 'inactive_requires_inspection',
        progress=held_status,prediction_files=len(list((DIRECTORY/'predictions').glob('*.npz'))),
        receipt_files=len(list((DIRECTORY/'predictions').glob('*.json'))),bounded_log_tail=tail)
    print(json.dumps(result,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--status',action='store_true')
    mode.add_argument('--launch',type=Path);parser.add_argument('--resume',action='store_true');parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.status:status();return
    with run_lease(JOB):
        if args.prepare:prepare();return
        launch,protocol=validate_launch(args.launch)
        with (DIRECTORY/'console.log').open('a',encoding='utf-8',buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                try:
                    if args.verify:verify(protocol)
                    else:run(launch,protocol,args.resume)
                except BaseException as exc:
                    traceback.print_exc();error=f'{type(exc).__name__}: {exc}'
                    atomic_json(DIRECTORY/'failure.json',dict(failed_at=now(),error=error,mode='verification' if args.verify else 'inference'))
                    record_event(dict(event='E_calibration_failed',error=error),run=dict(run_id=JOB,status='failed',stage='E'))
                    raise


if __name__=='__main__':main()
