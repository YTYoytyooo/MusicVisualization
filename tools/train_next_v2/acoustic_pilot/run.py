"""CPU-only c05 preparation, with a three-song repeatability gate."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from tools.train_next.core import ROOT, atomic_json, sha256
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import BASE, PLAN, now, process_identity, read_json, record_event, verify_hashes
from tools.train_next_v2.feature_pilot.cache import read_shard, write_shard
from tools.train_next_v2.feature_pilot.preflight import PACKAGES, PROTOCOL, PROTOCOL_SHA, relative
from tools.train_next_v2.feature_pilot.resampling import sample_clock_metadata
from .features import FEATURE_NAMES, IMPLEMENTATION, extract_rhythm_harmony


JOB_ID = 'c-pilot-rhythm-harmony-v1'
DIRECTORY = BASE / 'feature_jobs' / JOB_ID
CACHE = BASE / 'feature_caches' / JOB_ID / 'c05-rhythm-harmony'


def register():
    if is_running(JOB_ID):
        raise RuntimeError('Rhythm/harmony preparation already active')
    protocol = read_json(PROTOCOL)
    if sha256(PROTOCOL) != PROTOCOL_SHA or protocol['rhythm_harmony']['features'] != FEATURE_NAMES:
        raise ValueError('Frozen C feature definition mismatch')
    p = read_json(PLAN)
    pilot = ROOT / protocol['pilot_songs']
    if sha256(pilot) != protocol['pilot_songs_sha256']:
        raise ValueError('Pilot membership changed')
    songs = read_json(pilot)['songs']
    protected = {relative(PROTOCOL): PROTOCOL_SHA, relative(pilot): protocol['pilot_songs_sha256'],
        p['baseline']['legacy_model']: p['baseline']['legacy_model_sha256']}
    for row in songs:
        protected[row['audio_path']] = row['audio_sha256']
        path = ROOT / p['paths']['existing_feature_cache'] / (row['audio_sha256']+'.npz')
        protected[relative(path)] = row['original_feature_sha256']
    verify_hashes(protected)
    preflight_launch = read_json(BASE/'preflights/c-encoder-preflight-v2/launch.json')
    libraries = {name: version(name) for name in PACKAGES}
    if libraries != preflight_launch['libraries']:
        raise ValueError('Pinned runtime packages changed')
    code = {relative(path): sha256(path) for path in Path(__file__).parent.glob('*.py')}
    for path in ['tools/train_next_v2/feature_pilot/cache.py', 'tools/train_next_v2/feature_pilot/resampling.py',
                 'tools/train_next_v2/tests/test_rhythm_harmony.py', 'tools/train_next_v2/feature_pilot/status.py']:
        code[path] = sha256(ROOT/path)
    library_sources = {}
    for file in ['core/audio.py', 'core/spectrum.py', 'feature/spectral.py', 'decompose.py', 'filters.py', 'util/utils.py']:
        path = ROOT / 'data/training/next_va/env/Lib/site-packages/librosa' / file
        library_sources[relative(path)] = sha256(path)
    specification = dict(protocol_sha256=PROTOCOL_SHA, implementation=IMPLEMENTATION,
        libraries=libraries, code_hashes=code, runtime_source_hashes=library_sources,
        source_features=p['paths']['existing_feature_cache'], dimensions=540,
        normalization='None in feature cache; future scaler fit only on the fixed 60 fit songs.')
    feature_key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:20]
    environment = dict(preflight_launch['environment'])
    environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', NUMBA_NUM_THREADS='2')
    preflight_ids = [next(row['song_id'] for row in songs if row['role']=='fit' and row['source_dataset']==dataset)
                     for dataset in ('deam', 'memo2496', 'pmemo')]
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    if (DIRECTORY/'launch.json').exists():
        raise FileExistsError('Rhythm/harmony job already registered')
    argv = [sys.executable, '-u', '-m', 'tools.train_next_v2.acoustic_pilot.run', '--launch', str(DIRECTORY/'launch.json')]
    launch = dict(job_id=JOB_ID, registered_at=now(), cwd=str(ROOT), python=sys.executable,
        original_argv=argv, resume_argv=argv+['--resume'], environment=environment, songs=songs,
        protected_hashes=protected, code_hashes=code, runtime_source_hashes=library_sources,
        libraries=libraries, feature_spec=specification, feature_key=feature_key,
        preflight_ids=preflight_ids, preflight_requirement='All three whole-song feature arrays finite and exactly equal on repeated independent extraction before proceeding to the other songs.',
        cache_dir=str(CACHE.parent), expected_shards=90, existing_feature_cache=p['paths']['existing_feature_cache'],
        os_lock=str(lock_path(JOB_ID)), process_path=str(DIRECTORY/'process.json'), status_path=str(DIRECTORY/'status.json'),
        report_path=str(DIRECTORY/'report.json'), checkpoint_path=str(DIRECTORY/'shards.json'), log_pattern=str(DIRECTORY/'attempt-*/console.log'),
        status_argv=[sys.executable,'-m','tools.train_next_v2.feature_pilot.status','--job-id',JOB_ID],
        recovery='Confirm this CPU job OS lease is inactive, diagnose, then use resume_argv. Preserve exact code/data checks and verify completed shards before reuse. No optimizer or model checkpoint is involved.',
        cpu_threads=2, gpu_used=False, labels_read=False, heldout_songs_used=0)
    atomic_json(DIRECTORY/'launch.json', launch)
    p['stages']['C']['rhythm_harmony'] = dict(job_id=JOB_ID, status='registered', launch=relative(DIRECTORY/'launch.json'),
        status_path=relative(DIRECTORY/'status.json'), report_path=relative(DIRECTORY/'report.json'),
        expected_shards=90, completed_shards=0, status_argv=launch['status_argv'], gpu_used=False)
    record_event(dict(event='C_rhythm_harmony_CPU_job_registered', job_id=JOB_ID, songs=90, preflight_songs=preflight_ids),
        updates=dict(stages=p['stages']))
    print(DIRECTORY/'launch.json')


def run(path, resume=False):
    import librosa
    import soundfile as sf
    import torch
    if path.resolve() != (DIRECTORY/'launch.json').resolve():
        raise ValueError('Unknown CPU launch')
    launch = read_json(path)
    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve()!=Path(launch['python']).resolve():
        raise ValueError('CPU launch path/interpreter mismatch')
    for name,value in launch['environment'].items():
        if os.environ.get(name)!=value:
            raise ValueError('CPU launch environment mismatch: '+name)
    for field in ('protected_hashes','code_hashes','runtime_source_hashes'):
        verify_hashes(launch[field])
    if {name:version(name) for name in PACKAGES}!=launch['libraries']:
        raise ValueError('Runtime packages changed')
    with run_lease(JOB_ID):
        attempts = sorted(DIRECTORY.glob('attempt-*'))
        if attempts and not resume:
            raise RuntimeError('Existing CPU job requires --resume after diagnosis')
        if (DIRECTORY/'report.json').exists() and read_json(DIRECTORY/'report.json')['status']=='completed':
            raise RuntimeError('CPU extraction already completed')
        attempt = DIRECTORY/f'attempt-{len(attempts)+1:03d}'
        attempt.mkdir()
        identity = process_identity()
        atomic_json(DIRECTORY/'process.json',dict(**identity,argv=sys.argv,started_at=now(),log=str(attempt/'console.log')))
        with (attempt/'console.log').open('x',encoding='utf-8',buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            index = {}
            report = dict(job_id=JOB_ID,status='running',started_at=now(),expected_shards=90,preflight=[],gpu_used=False,labels_read=False)
            started = time.monotonic()
            def status(**progress):
                data=dict(job_id=JOB_ID,state='running',updated_at=now(),process_identity=identity,
                    completed_shards=len(index),expected_shards=90,**progress)
                atomic_json(DIRECTORY/'status.json',data)
                print(json.dumps(data),flush=True)
            status(phase='preflight')
            try:
                torch.set_num_threads(2)
                if torch.cuda.is_available():
                    raise RuntimeError('CPU acoustic job unexpectedly sees CUDA')
                ordered = [next(s for s in launch['songs'] if s['song_id']==song_id) for song_id in launch['preflight_ids']]
                ordered += [s for s in launch['songs'] if s['song_id'] not in launch['preflight_ids']]
                if len(ordered)!=90 or len({s['song_id'] for s in ordered})!=90:
                    raise ValueError('Unexpected pilot membership')
                for number,row in enumerate(ordered):
                    if number==3:
                        if len(report['preflight'])!=3 or not all(x['passed'] for x in report['preflight']):
                            raise ValueError('Rhythm/harmony runtime gate failed')
                        atomic_json(DIRECTORY/'preflight.json',dict(status='passed',completed_at=now(),cases=report['preflight']))
                    times=np.arange(int(np.floor(row['duration_seconds']/.5))+1)*.5
                    output=CACHE/(row['audio_sha256']+'.npz')
                    metadata=read_shard(output,times,launch['feature_key'],row['audio_sha256'],540)
                    if metadata is None or number<3:
                        status(phase='preflight' if number<3 else 'extracting',song_id=row['song_id'],frames=len(times))
                        song_started=time.monotonic()
                        audio,rate=sf.read(ROOT/row['audio_path'],dtype='float32',always_2d=True)
                        if audio.shape!=(row['decoded_frames'],row['channels']) or rate!=row['sample_rate']:
                            raise ValueError('Prepared source audio changed')
                        waveform=librosa.resample(audio.mean(1),orig_sr=rate,target_sr=24000,res_type='soxr_hq')
                        clock_metadata=sample_clock_metadata(len(audio),rate,24000,len(waveform))
                        values,timing=extract_rhythm_harmony(waveform,len(audio)/rate,times)
                        first_seconds=time.monotonic()-song_started
                        if number<3:
                            repeated,_=extract_rhythm_harmony(waveform,len(audio)/rate,times)
                            np.testing.assert_array_equal(values,repeated)
                            report['preflight'].append(dict(song_id=row['song_id'],passed=True,frames=len(times),
                                dimensions=9,repeat_max_absolute_difference=float(np.max(np.abs(values-repeated))),
                                first_extraction_seconds=first_seconds))
                        original=ROOT/launch['existing_feature_cache']/(row['audio_sha256']+'.npz')
                        with np.load(original,allow_pickle=False) as cache:
                            np.testing.assert_array_equal(cache['times'],times)
                            base=cache['features'].copy()
                        combined=np.concatenate([base,values],axis=1)
                        if metadata is None:
                            metadata=write_shard(output,times,combined,launch['feature_key'],row['audio_sha256'],dict(
                                completed_at=now(),song_id=row['song_id'],source_dataset=row['source_dataset'],role=row['role'],
                                variant='c05-rhythm-harmony',output=relative(output),extraction_seconds=first_seconds,
                                audio_seconds=row['duration_seconds'],sample_clock=clock_metadata,timing=timing,
                                source_feature_sha256=row['original_feature_sha256'],gpu_used=False))
                        else:
                            with np.load(output,allow_pickle=False) as cached:
                                np.testing.assert_array_equal(cached['features'],combined)
                    index[row['song_id']]=dict(output=relative(output),output_sha256=metadata['output_sha256'],
                        sidecar_sha256=sha256(output.with_suffix('.json')),frames=len(times),dimensions=540)
                    atomic_json(DIRECTORY/'shards.json',dict(job_id=JOB_ID,feature_key=launch['feature_key'],shards=index))
                    status(phase='shard_complete',song_id=row['song_id'])
                if len(index)!=90:
                    raise ValueError('Incomplete acoustic pilot')
                for field in ('protected_hashes','code_hashes','runtime_source_hashes'):
                    verify_hashes(launch[field])
                report.update(status='completed',protected_hashes_verified=True,shard_index_sha256=sha256(DIRECTORY/'shards.json'))
            except Exception as error:
                traceback.print_exc()
                report.update(status='failed',error_type=type(error).__name__,error=str(error))
            report.update(completed_at=now(),completed_shards=len(index),elapsed_seconds=time.monotonic()-started)
            atomic_json(attempt/'report.json',report)
            atomic_json(DIRECTORY/'report.json',report)
            atomic_json(DIRECTORY/'status.json',dict(job_id=JOB_ID,state=report['status'],updated_at=now(),
                process_identity=identity,completed_shards=len(index),expected_shards=90,report=str(DIRECTORY/'report.json')))
            print(json.dumps(report),flush=True)
    p=read_json(PLAN)
    p['stages']['C']['rhythm_harmony'].update(status=report['status'],completed_shards=len(index),process_identity=identity)
    record_event(dict(event='C_rhythm_harmony_CPU_job_finished',job_id=JOB_ID,status=report['status'],completed_shards=len(index)),updates=dict(stages=p['stages']))
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--register',action='store_true')
    group.add_argument('--launch',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.register:
        register()
    else:
        raise SystemExit(run(args.launch,args.resume))
