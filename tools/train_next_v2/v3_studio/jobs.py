"""Finite subprocess jobs with persisted argv, provenance and PID creation time."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import uuid
from . import store


def directory(job_id):
    return store.DATA / 'jobs' / store.identifier(job_id)


def status(job_id):
    path = directory(job_id)
    launch = store.read(path/'launch.json')
    state = store.read(path/'status.json') if (path/'status.json').exists() else {'status':'registered'}
    ident = state.get('process_identity')
    if not ident and (path/'process.json').exists():ident=store.read(path/'process.json')
    alive = bool(ident and store.identity(ident['pid']) == ident)
    if (state['status'] in ('running','cancelling')
            or state['status']=='registered' and (path/'process.json').exists()) and not alive:
        state = {**state,'status':'interrupted','stage':'进程已退出，可从已验证缓存恢复'}
    return {**state,'id':job_id,'project_id':launch['project_id'],'kind':launch['kind'],
            'process_alive':alive,'log':str(path/'console.log')}


def all_jobs():
    states=[status(p.parent.name) for p in (store.DATA/'jobs').glob('*/launch.json')]
    return sorted(states,key=lambda s:s.get('updated_at',''),reverse=True)[:50]


def environment():
    return {'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1',
            'TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
            'HF_HOME':str(store.ROOT/'data/v3.0/runtime/huggingface'),
            'TORCH_HOME':str(store.ROOT/'data/v3.0/runtime/torch'),
            'NUMBA_CACHE_DIR':str(store.ROOT/'data/v3.0/runtime/numba'),
            'MPLCONFIGDIR':str(store.ROOT/'data/v3.0/runtime/matplotlib'),
            'TEMP':str(store.ROOT/'data/v3.0/runtime/tmp'),
            'TMP':str(store.ROOT/'data/v3.0/runtime/tmp')}


def submit(kind, arguments, project_id=None, start=True):
    if kind not in ('analyze','export'):
        raise ValueError('Unsupported job type')
    with store.lease('job-submit'):
        if any(j['process_alive'] or j['status']=='registered' for j in all_jobs()):
            raise RuntimeError('已有任务正在执行，请完成或取消后再启动')
        if kind == 'analyze':
            row = store.audio_entry(arguments['audio_id'])
            if arguments.get('device','cuda') not in ('cpu','cuda'):
                raise ValueError('Invalid device')
            if type(arguments.get('include_v2',False)) is not bool:
                raise ValueError('include_v2 must be boolean')
            project_id = 'p-' + row['sha256'][:12] + '-' + uuid.uuid4().hex[:8]
            arguments = {**arguments, 'audio_path':row['path'],'audio_sha256':row['sha256']}
        else:
            store.load_project(project_id)
        job_id = 'j-' + uuid.uuid4().hex[:16]
        path = directory(job_id); path.mkdir(parents=True)
        argv = [str(store.PYTHON),'-u','-m','tools.train_next_v2.v3_studio.worker','--job',job_id]
        code = {p.relative_to(store.ROOT).as_posix():store.digest(p)
                for package in ['v3_studio','v3_inference']
                for p in (store.ROOT/'tools/train_next_v2'/package).glob('*.py')}
        launch = {'id':job_id,'kind':kind,'arguments':arguments,'project_id':project_id,
                  'registered_at':store.now(),'original_argv':argv,'resume_argv':argv+['--resume'],
                  'cwd':str(store.ROOT),'environment':environment(),'code_hashes':code,
                  'manifest_sha256':store.digest(store.MANIFEST),'log':str(path/'console.log'),
                  'os_lock':str(store.DATA/'locks'/f'{job_id}.lock'),
                  'recovery':'Same argv plus --resume only after PID identity is inactive; code, manifest and audio must match.'}
        store.atomic(path/'launch.json',launch)
        if kind == 'analyze':
            # Publish UI metadata before spawning. The inference pipeline writes
            # its own authoritative project.json only after decoding native EOF.
            store.atomic(store.project_path(project_id)/'pending.json',{
                'schema_version':3,'id':project_id,'status':'registered',
                'created_at':launch['registered_at'],'manifest_sha256':launch['manifest_sha256'],
                'audio':{key:row[key] for key in ('path','sha256','name','duration_seconds')},
                'duration_source':'library metadata; native decoded duration pending',
                'job_id':job_id})
        store.atomic(path/'status.json',{'status':'registered','stage':'已登记','updated_at':store.now()})
        if start:
            spawn(launch, False)
        return {'job_id':job_id,'project_id':project_id}


def spawn(launch, resume):
    env = {**os.environ,**launch['environment']}
    for key in ['TEMP','TMP','HF_HOME','TORCH_HOME','NUMBA_CACHE_DIR','MPLCONFIGDIR']:
        Path(env[key]).mkdir(parents=True,exist_ok=True)
    path=directory(launch['id'])
    try:
        with (path/'console.log').open('ab') as log:
            proc = subprocess.Popen(launch['resume_argv'] if resume else launch['original_argv'],
                cwd=launch['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        store.atomic(path/'process.json',store.identity(proc.pid))
    except OSError as exc:
        store.atomic(path/'status.json',{'status':'failed','stage':'启动失败','error':str(exc),'updated_at':store.now()})
        raise


def cancel(job_id):
    s=status(job_id)
    if s['status'] not in ('registered','running','cancelling') or not s['process_alive']:
        raise ValueError('任务当前没有活动进程')
    store.atomic(directory(job_id)/'cancel.json',{'requested_at':store.now()})
    return {'status':'cancel_requested','job_id':job_id}


def resume(job_id):
    with store.lease('job-submit'):
        s=status(job_id)
        if s['process_alive'] or s['status'] not in ('interrupted','failed','cancelled'):
            raise ValueError('Only an inactive interrupted/failed/cancelled job can resume')
        if any(j['process_alive'] for j in all_jobs()):
            raise RuntimeError('Another job is active')
        path=directory(job_id)
        if (path/'cancel.json').exists():
            (path/'cancel.json').rename(path/('cancel-'+uuid.uuid4().hex+'.json'))
        store.atomic(path/'status.json',{'status':'registered','stage':'恢复已登记','updated_at':store.now()})
        spawn(store.read(path/'launch.json'),True)
        return {'job_id':job_id}
