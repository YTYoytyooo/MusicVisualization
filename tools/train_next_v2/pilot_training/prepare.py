"""Audit all C feature shards and freeze identical training target views."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from tools.train_next.core import ROOT, atomic_json, sha256
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import BASE, PLAN, now, process_identity, read_json, record_event, verify_hashes
from tools.train_next_v2.feature_pilot.preflight import PROTOCOL, PROTOCOL_SHA, relative
from .data import CONTRACT, array_fingerprint, read_feature, read_targets, source_dataset, validate_membership

JOB_ID='c-pilot-input-verification-v1'
DIRECTORY=BASE/'analyses'/JOB_ID
JOBS={'encoder':'c-pilot-encoder-features-v2','acoustic':'c-pilot-rhythm-harmony-v2'}


def register():
    if sha256(PROTOCOL)!=PROTOCOL_SHA:
        raise ValueError('Frozen C protocol changed')
    protocol=read_json(PROTOCOL)
    protected={relative(PROTOCOL):PROTOCOL_SHA,protocol['source_B_protocol']:protocol['source_B_protocol_sha256'],
        protocol['pilot_songs']:protocol['pilot_songs_sha256']}
    for job in JOBS.values():
        folder=BASE/'feature_jobs'/job
        launch,report=read_json(folder/'launch.json'),read_json(folder/'report.json')
        if is_running(job) or report['status']!='completed' or not report['protected_hashes_verified']:
            raise ValueError('Feature job is active or incomplete: '+job)
        if report['completed_shards']!=launch['expected_shards'] or sha256(folder/'shards.json')!=report['shard_index_sha256']:
            raise ValueError('Feature index completion mismatch')
        for field in ('protected_hashes','code_hashes','runtime_source_hashes'):
            verify_hashes(launch[field])
            protected.update(launch[field])
        for name in ('launch.json','report.json','shards.json'):
            protected[relative(folder/name)]=sha256(folder/name)
    verify_hashes(protected)
    code={relative(path):sha256(path) for path in Path(__file__).parent.glob('*.py')}
    environment=dict(read_json(BASE/'feature_jobs'/JOBS['encoder']/'launch.json')['environment'])
    environment.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    if (DIRECTORY/'launch.json').exists() or CONTRACT.exists():
        raise FileExistsError('C inputs already registered/frozen; inspect existing verification')
    argv=[sys.executable,'-u','-m','tools.train_next_v2.pilot_training.prepare','--launch',str(DIRECTORY/'launch.json')]
    launch=dict(job_id=JOB_ID,registered_at=now(),original_argv=argv,cwd=str(ROOT),python=sys.executable,
        environment=environment,protected_hashes=protected,code_hashes=code,jobs=JOBS,
        os_lock=str(lock_path(JOB_ID)),log=str(DIRECTORY/'console.log'),output=str(CONTRACT),
        recovery='Finite read-only audit plus new immutable contract. Preserve failure report; never weaken data/code checks.',gpu_used=False)
    atomic_json(DIRECTORY/'launch.json',launch)
    p=read_json(PLAN)
    p['stages']['C']['input_verification']=dict(job_id=JOB_ID,status='registered',launch=relative(DIRECTORY/'launch.json'))
    record_event(dict(event='C_all_feature_target_verification_registered',job_id=JOB_ID),updates=dict(stages=p['stages']))
    print(DIRECTORY/'launch.json')


def run(path):
    launch=read_json(path)
    if path.resolve()!=(DIRECTORY/'launch.json').resolve() or Path.cwd().resolve()!=ROOT:
        raise ValueError('Unexpected C input verification launch')
    for key,value in launch['environment'].items():
        if os.environ.get(key)!=value:
            raise ValueError('Audit environment mismatch: '+key)
    verify_hashes(launch['code_hashes']);verify_hashes(launch['protected_hashes'])
    if CONTRACT.exists() or (DIRECTORY/'report.json').exists():
        raise FileExistsError('Verification result already exists')
    with run_lease(JOB_ID),(DIRECTORY/'console.log').open('x',encoding='utf-8',buffering=1) as log:
        with redirect_stdout(log),redirect_stderr(log):
            identity=process_identity()
            atomic_json(DIRECTORY/'process.json',identity)
            report=dict(job_id=JOB_ID,started_at=now(),status='running',gpu_used=False)
            try:
                c=read_json(PROTOCOL);b=read_json(ROOT/c['source_B_protocol'])
                songs=read_json(ROOT/c['pilot_songs'])['songs']
                validate_membership(songs,b['split']['assignments'])
                dataset,indices=source_dataset(b)
                jobs={key:read_json(BASE/'feature_jobs'/job/'launch.json') for key,job in JOBS.items()}
                indexes={key:read_json(BASE/'feature_jobs'/job/'shards.json')['shards'] for key,job in JOBS.items()}
                protected=dict(launch['protected_hashes'])
                protected[relative(Path(b['source_manifest']))]=b['manifest_sha256']
                protected[relative(Path(b['source_manifest']).parent/'config.json')]=b['prepared_config_sha256']
                counts=defaultdict(lambda:np.zeros(2,dtype=int));varying=defaultdict(lambda:np.zeros(2,dtype=int))
                costs=defaultdict(lambda:dict(seconds=0.,bytes=0,frames=0,peak_gpu_bytes=0))
                checked=[]
                for number,row in enumerate(songs,1):
                    times=np.arange(int(np.floor(row['duration_seconds']/.5))+1)*.5
                    original=ROOT/jobs['encoder']['existing_feature_cache']/(row['audio_sha256']+'.npz')
                    reference=dict(path=relative(original),sha256=row['original_feature_sha256'],feature_key=b['feature_key'],dimensions=531)
                    base=read_feature(reference,times,row['audio_sha256'])
                    target,mask,native=read_targets(dataset,indices[row['song_id']],row,times)
                    fingerprint=array_fingerprint(times,target,mask)
                    features={'c00-current-2s':reference}
                    protected[row['labels_path']]=row['expected_labels_sha256']
                    for variant in c['experiments'][1:]:
                        source='acoustic' if variant['id']=='c05-rhythm-harmony' else 'encoder'
                        index_key=row['song_id'] if source=='acoustic' else variant['id']+'/'+row['song_id']
                        item=indexes[source][index_key]
                        feature_path=ROOT/item['output'];metadata_path=feature_path.with_suffix('.json')
                        if sha256(metadata_path)!=item['sidecar_sha256']:
                            raise ValueError('Feature timing metadata changed')
                        metadata=read_json(metadata_path)
                        key=jobs[source]['feature_key'] if source=='acoustic' else jobs[source]['feature_specs'][variant['id']]['key']
                        entry=dict(path=item['output'],sha256=item['output_sha256'],feature_key=key,dimensions=variant['feature_dimensions'],
                            sidecar=relative(metadata_path),sidecar_sha256=item['sidecar_sha256'])
                        matrix=read_feature(entry,times,row['audio_sha256'])
                        np.testing.assert_array_equal(matrix[:,:531] if source=='acoustic' else matrix[:,-19:],
                                                      base if source=='acoustic' else base[:,-19:])
                        # Rejoin on this cache's clock, independently of the c00 view.
                        with np.load(feature_path,allow_pickle=False) as cached:
                            y,m,native_check=read_targets(dataset,indices[row['song_id']],row,cached['times'])
                        if array_fingerprint(times,y,m)!=fingerprint or native_check!=native:
                            raise ValueError('Features would change target times or independent V/A masks')
                        if metadata['song_id']!=row['song_id'] or metadata['role']!=row['role'] or metadata['variant']!=variant['id']:
                            raise ValueError('Feature shard membership mismatch')
                        if source=='encoder':
                            if len(metadata['timing'])!=len(times):
                                raise ValueError('Missing physical window timing')
                            for t,window in zip(times,metadata['timing'],strict=True):
                                if window['timestamp']!=t:
                                    raise ValueError('Feature window timestamp mismatch')
                                if variant['encoder']=='mert95m':
                                    if not t-.25<=window['central_first_seconds']<=window['central_last_seconds']<t+.25:
                                        raise ValueError('MERT central frame timing violation')
                        else:
                            if metadata['timing']['last_source_frame_seconds']>=row['duration_seconds']:
                                raise ValueError('Acoustic source frame lies beyond recording')
                        costs[variant['id']]['seconds']+=metadata['extraction_seconds']
                        costs[variant['id']]['bytes']+=metadata['bytes']
                        costs[variant['id']]['frames']+=len(times)
                        costs[variant['id']]['peak_gpu_bytes']=max(costs[variant['id']]['peak_gpu_bytes'],metadata.get('peak_allocated_bytes',0))
                        features[variant['id']]=entry
                    counts[row['role']+'/'+row['source_dataset']]+=mask.sum(0)
                    varying[row['role']+'/'+row['source_dataset']]+=np.array([np.var(target[mask[:,axis],axis])>1e-6 if mask[:,axis].any() else False for axis in range(2)])
                    checked.append(dict(row,features=features,target_fingerprint=fingerprint,native_seconds=native,valid_values=mask.sum(0).tolist()))
                    if number%10==0 or number==90:
                        progress=dict(state='verifying',songs=number,total=90,feature_views=number*6,process_identity=identity,updated_at=now())
                        atomic_json(DIRECTORY/'status.json',progress);print(json.dumps(progress),flush=True)
                if len(indexes['encoder'])!=360 or len(indexes['acoustic'])!=90:
                    raise ValueError('Unexpected feature job membership')
                if any(np.any(varying['development/'+domain]==0) for domain in ('deam','memo2496','pmemo')):
                    raise ValueError('Pilot development has a domain/axis without varying targets')
                verify_hashes(protected)
                contract=dict(schema_version=1,id='c-pilot-training-inputs-v1',created_at=now(),status='frozen_verified',
                    protocol=relative(PROTOCOL),protocol_sha256=PROTOCOL_SHA,source_B_protocol=c['source_B_protocol'],
                    protected_hashes=protected,builder_code_hashes=launch['code_hashes'],songs=checked,
                    variants=[row['id'] for row in c['experiments']],counts=c['counts'],
                    independent_mask_timestamp_parity_verified=True,original_heldout_labels_used=0,calibration_labels_used=0,
                    extraction_costs=dict(costs),labels_used_only_from_existing_training_partition=True)
                atomic_json(CONTRACT,contract)
                report.update(status='completed_verified',contract=relative(CONTRACT),contract_sha256=sha256(CONTRACT),
                    source_songs=90,feature_views=540,new_shards=450,valid_values={k:v.tolist() for k,v in counts.items()},
                    varying_songs={k:v.tolist() for k,v in varying.items()},extraction_costs=dict(costs),
                    all_features_finite=True,all_targets_masks_clocks_identical=True,
                    original_feature_columns_bitwise_unchanged=True,original_validation_test_calibration_labels_used=0)
            except Exception as error:
                traceback.print_exc();report.update(status='failed',error=repr(error))
            report['completed_at']=now()
            atomic_json(DIRECTORY/'report.json',report)
            atomic_json(DIRECTORY/'status.json',dict(state=report['status'],process_identity=identity,updated_at=now()))
            print(json.dumps(report),flush=True)
    p=read_json(PLAN)
    p['stages']['C']['input_verification'].update(status=report['status'],report=relative(DIRECTORY/'report.json'))
    if report['status']=='completed_verified':
        p['stages']['C']['phase']='pilot_features_targets_verified_ready_for_training'
        for key in ('feature_extraction','rhythm_harmony'):
            p['stages']['C'][key]['status']='completed_verified'
            p['stages']['C'][key]['actual_npz_count']=p['stages']['C'][key]['expected_shards']
    record_event(dict(event='C_all_feature_target_verification_finished',job_id=JOB_ID,status=report['status']),updates=dict(stages=p['stages']))
    return 0 if report['status']=='completed_verified' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--register',action='store_true')
    group.add_argument('--launch',type=Path)
    args=parser.parse_args()
    if args.register:
        register()
    else:
        raise SystemExit(run(args.launch))
