"""Single-worker local queue, durable state and cooperative cancellation."""
from copy import deepcopy
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from .store import ROOT, atomic_json, now, read_json, resolve_project, SAFE_ID, object_digest
from .paths import WORKSPACE, PYTHON

ACTIVE = {'running', 'queued', 'cancelling'}
TERMINAL = {'succeeded', 'failed', 'cancelled', 'interrupted'}
CONSUMER_ID = re.compile(r'^[a-zA-Z0-9_.:-]{1,200}$')


def launch_record(projects, job):
    """Freeze command, code and immutable inputs before publishing a queue item."""
    from tools.train_next_v2.v3_studio.jobs import environment
    from tools.train_next_v2.v3_studio.store import digest, MANIFEST
    payload = job['payload']
    files = []
    for key in ('audio_path',):
        if payload.get(key):
            files.append(Path(payload[key]).resolve())
    for key in ('analysis_dir', 'source_dir'):
        if payload.get(key):
            source = Path(payload[key]).resolve()
            files.extend(p for p in source.rglob('*') if p.is_file()
                         and p.suffix.lower() in ('.json', '.npz', '.wav', '.mp3', '.flac', '.m4a'))
    frozen_project = None
    if payload.get('project_id'):
        from .store import load_project, load_revision, _contained_file
        project = resolve_project(projects, payload['project_id'])
        revision = payload.get('revision') or payload.get('base_revision')
        pinned = load_revision(project, revision) if revision else {}
        if payload.get('snapshot_id'):
            snapshot = project / 'previews' / payload['snapshot_id'] / 'snapshot.json'
            pinned = read_json(snapshot)
            files.append(snapshot)
        selected = payload.get('source_system') or pinned.get('source_system')
        meta = load_project(project, system=selected)
        frozen_meta = {k:v for k,v in meta.items() if k not in ('current_revision','updated_at')}
        frozen_project = {'path':str(project), 'source_system':meta['source_system'],
                          'sha256':object_digest(frozen_meta)}
        files.append(project / meta['audio_file'])
        files.extend(_contained_file(project, record['file']) for record in meta['systems'].values())
        if revision:
            files.extend((project / 'revisions' / revision).glob('*.json'))
    code = [p for package in ('v3_workbench', 'v3_inference', 'v3_studio')
            for p in (WORKSPACE / 'tools/train_next_v2' / package).rglob('*.py')]
    argv = [str(PYTHON), '-u', '-m', 'tools.train_next_v2.v3_workbench.main',
            '_worker', '--projects', str(Path(projects).resolve()), '--job', job['id']]
    return {'original_argv': argv, 'cwd': str(WORKSPACE), 'environment': environment(),
            'code_hashes': {str(p): digest(p) for p in code},
            'input_hashes': {str(p): digest(p) for p in sorted(set(files))},
            'payload_sha256':object_digest(payload), 'project_metadata':frozen_project,
            'manifest_path': str(MANIFEST), 'manifest_sha256': digest(MANIFEST),
            'log': str(Path(projects) / '.jobs' / (job['id'] + '.log')),
            'run_lock': str(WORKSPACE / 'data/v3.0/locks' / ('workbench-' + job['id'] + '.lock')),
            'recovery': 'Retry only after PID identity is inactive and frozen inputs/code match. '
                        'New job retains the same payload; render checkpoints verify full configuration.'}


def verify_launch(job):
    from tools.train_next_v2.v3_studio.store import digest
    launch = job['launch']
    if object_digest(job['payload']) != launch['payload_sha256']:
        raise ValueError('已登记任务配置已变化')
    if launch.get('project_metadata'):
        from .store import load_project
        pinned = launch['project_metadata']
        meta = load_project(pinned['path'],system=pinned['source_system'])
        if object_digest({k:v for k,v in meta.items() if k not in ('current_revision','updated_at')}) != pinned['sha256']:
            raise ValueError('已登记作品的音频、模型或时间元数据已变化')
    for key in ('code_hashes', 'input_hashes'):
        for path, expected in launch[key].items():
            if not Path(path).is_file() or digest(path) != expected:
                raise ValueError(f'已登记任务的文件已变化，请创建新任务：{path}')
    if digest(launch['manifest_path']) != launch['manifest_sha256']:
        raise ValueError('已冻结模型清单已变化')


def validate_consumer(consumer_id, purpose='temporary'):
    if purpose not in ('temporary', 'confirmed'):
        raise ValueError('规划用途必须为 temporary / confirmed')
    if consumer_id is None and purpose == 'confirmed':
        return
    if not isinstance(consumer_id, str) or not CONSUMER_ID.fullmatch(consumer_id):
        raise ValueError('临时规划必须提供有效 consumer_id')


def validate_consumer_seq(consumer_id, consumer_seq):
    if consumer_seq is not None:
        validate_consumer(consumer_id)
        if isinstance(consumer_seq, bool) or not isinstance(consumer_seq, int) or consumer_seq < 0:
            raise ValueError('consumer_seq 必须是非负整数')


def sanitize_log(text):
    """Display local diagnostic text, not terminal escapes or common credentials."""
    text = re.sub(r'\x1b\][^\x07]*(?:\x07|\x1b\\)', '', text)
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'(?i)(authorization\s*:\s*(?:bearer|basic)\s+)\S+', r'\1[REDACTED]', text)
    text = re.sub(r'(?i)((?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*)[^\s,;]+',
                  r'\1[REDACTED]', text)
    text = re.sub(r'\bhf_[a-zA-Z0-9]{8,}\b', '[REDACTED]', text)
    return text.replace('\r\n', '\n').replace('\r', '\n')


def process_identity(pid):
    """Read Windows process creation time. Never signal an arbitrary PID."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                return None
            return str(times[0].dwHighDateTime << 32 | times[0].dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    try:
        return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except OSError:
        return None


def recover_interrupted(job, path):
    """Recover a dead worker without overwriting a final record published meanwhile."""
    if job['status'] not in ('running', 'cancelling'):
        return job
    identity = process_identity(job.get('pid', 0))
    if identity and identity == job.get('process_identity'):
        return job
    # The worker may have published success and exited after the scheduler's
    # snapshot was read. Once it is dead, its final on-disk record is authoritative.
    latest = read_json(path)
    if latest['status'] not in ('running', 'cancelling'):
        return latest
    if (latest.get('pid'), latest.get('process_identity')) != (
            job.get('pid'), job.get('process_identity')):
        return latest
    latest.update(status='interrupted',
                  error='工作进程已退出，原始结果和已有视频未改动', completed_at=now())
    atomic_json(path, latest)
    return latest


class JobManager:
    def __init__(self, projects):
        self.projects = Path(projects).resolve()
        self.folder = self.projects / '.jobs'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.processes = {}
        self.refs_path = self.folder / 'plan_references.json'
        self.plan_refs = (read_json(self.refs_path) if self.refs_path.is_file()
                          else {'consumers': {}, 'pinned': []})
        if (not isinstance(self.plan_refs, dict)
                or not isinstance(self.plan_refs.get('consumers'), dict)
                or not isinstance(self.plan_refs.get('pinned'), list)):
            raise ValueError('规划引用记录损坏，请保留现场并检查')
        self.plan_refs.setdefault('sequences', {})
        if not isinstance(self.plan_refs['sequences'], dict):
            raise ValueError('规划请求序号记录损坏，请保留现场并检查')
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def path(self, job_id):
        if not isinstance(job_id, str) or not SAFE_ID.fullmatch(job_id):
            raise ValueError('非法任务 ID')
        return self.folder / f'{job_id}.json'

    def list(self):
        result = []
        for p in self.folder.glob('j-*.json'):
            try:
                job = read_json(p)
                if job['status'] in ('running', 'cancelling') and self.cancel_path(job['id']).is_file():
                    job['status'] = 'cancelling'
                result.append(job)
            except (ValueError, OSError):
                pass
        return sorted(result, key=lambda j: j['created_at'], reverse=True)

    def cancel_path(self, job_id):
        self.path(job_id)  # Validate before building any path.
        return self.folder / f'{job_id}.cancel'

    def _save_refs(self):
        atomic_json(self.refs_path, self.plan_refs)

    def _submit_locked(self, kind, payload, dedupe_key=None, parent_job_id=None, force_new=False):
        if kind not in ('analyze', 'render', 'plan', 'preview', 'import_analysis', 'import_v2'):
            raise ValueError('未知任务类型')
        if dedupe_key is not None and not force_new:
            for existing in self.list():
                if (existing['kind'] == kind and existing.get('dedupe_key') == dedupe_key
                        and existing['status'] in ACTIVE
                        and not self.cancel_path(existing['id']).exists()):
                    return existing
        job = {'id': 'j-' + uuid.uuid4().hex[:12], 'kind': kind, 'payload': deepcopy(payload),
               'status': 'queued', 'stage': 'queued', 'progress': 0., 'created_at': now()}
        if dedupe_key is not None:
            job['dedupe_key'] = dedupe_key
        if parent_job_id is not None:
            job['parent_job_id'] = parent_job_id
        job['launch'] = launch_record(self.projects, job)
        atomic_json(self.path(job['id']), job)
        return job

    def submit(self, kind, payload, dedupe_key=None):
        with self.lock:
            job = self._submit_locked(kind, payload, dedupe_key)
            if kind == 'plan' and job['id'] not in self.plan_refs['pinned']:
                self.plan_refs['pinned'].append(job['id'])
                self._save_refs()
        return job

    def _cancel_locked(self, job_id):
        path = self.path(job_id)
        job = read_json(path)
        if job['status'] == 'queued':
            job.update(status='cancelled', completed_at=now())
            atomic_json(path, job)
        elif job['status'] in ('running', 'cancelling'):
            # Worker alone owns its running JSON; display status is derived.
            self.cancel_path(job_id).touch()
            job['status'] = 'cancelling'
        return job

    def _cancel_unreferenced_locked(self, job_id):
        if (job_id in self.plan_refs['pinned']
                or job_id in self.plan_refs['consumers'].values()):
            return False
        path = self.path(job_id)
        if not path.is_file():
            return False
        job = read_json(path)
        if job['kind'] != 'plan' or job['status'] not in ACTIVE:
            return False
        self._cancel_locked(job_id)
        return True

    def _stale_request(self, consumer_id, consumer_seq, dedupe_key=None, releasing=False):
        if consumer_seq is None:
            return False
        previous = self.plan_refs['sequences'].get(consumer_id)
        if previous is None:
            return False
        if consumer_seq < previous['seq']:
            return True
        return (not releasing and consumer_seq == previous['seq']
                and (previous.get('released', False) or previous.get('key') != dedupe_key))

    @staticmethod
    def _superseded(consumer_id, consumer_seq):
        return {'status': 'superseded', 'consumer_id': consumer_id, 'consumer_seq': consumer_seq}

    def _remember_request(self, consumer_id, consumer_seq, dedupe_key):
        if consumer_seq is not None:
            self.plan_refs['sequences'][consumer_id] = {
                'seq': consumer_seq, 'key': dedupe_key, 'released': False}

    def request_plan(self, payload, dedupe_key, consumer_id=None, purpose='confirmed', consumer_seq=None):
        validate_consumer(consumer_id, purpose)
        validate_consumer_seq(consumer_id, consumer_seq)
        with self.lock:
            if self._stale_request(consumer_id, consumer_seq, dedupe_key):
                return self._superseded(consumer_id, consumer_seq)
            job = self._submit_locked('plan', payload, dedupe_key)
            previous = self.plan_refs['consumers'].get(consumer_id) if consumer_id is not None else None
            if consumer_id is not None:
                self.plan_refs['consumers'][consumer_id] = job['id']
            if purpose == 'confirmed' and job['id'] not in self.plan_refs['pinned']:
                self.plan_refs['pinned'].append(job['id'])
            self._remember_request(consumer_id, consumer_seq, dedupe_key)
            self._save_refs()
            if previous is not None and previous != job['id']:
                self._cancel_unreferenced_locked(previous)
            return job

    def release_plan(self, consumer_id, consumer_seq=None):
        validate_consumer(consumer_id)
        validate_consumer_seq(consumer_id, consumer_seq)
        with self.lock:
            if self._stale_request(consumer_id, consumer_seq, releasing=True):
                return self._superseded(consumer_id, consumer_seq)
            previous = self.plan_refs['consumers'].pop(consumer_id, None)
            sequence = self.plan_refs['sequences'].get(consumer_id)
            if consumer_seq is not None or sequence is not None:
                self.plan_refs['sequences'][consumer_id] = {
                    'seq': consumer_seq if consumer_seq is not None else sequence['seq'],
                    'key': sequence.get('key') if sequence is not None else None, 'released': True}
            self._save_refs()
            cancelled = previous is not None and self._cancel_unreferenced_locked(previous)
            return {'consumer_id': consumer_id, 'released': [previous] if previous else [],
                    'cancelled': [previous] if cancelled else []}

    def accept_cached_plan(self, dedupe_key, consumer_id=None, purpose='confirmed', consumer_seq=None):
        """A published cache needs no consumer, but confirmation still pins its producer."""
        validate_consumer(consumer_id, purpose)
        validate_consumer_seq(consumer_id, consumer_seq)
        with self.lock:
            if self._stale_request(consumer_id, consumer_seq, dedupe_key):
                return self._superseded(consumer_id, consumer_seq)
            if purpose == 'confirmed':
                for job in self.list():
                    if (job['kind'] == 'plan' and job.get('dedupe_key') == dedupe_key
                            and job['status'] in ACTIVE
                            and not self.cancel_path(job['id']).exists()
                            and job['id'] not in self.plan_refs['pinned']):
                        self.plan_refs['pinned'].append(job['id'])
            previous = self.plan_refs['consumers'].pop(consumer_id, None)
            self._remember_request(consumer_id, consumer_seq, dedupe_key)
            self._save_refs()
            if previous is not None:
                self._cancel_unreferenced_locked(previous)
            return {'status': 'ready'}

    def cancel(self, job_id):
        # Explicit user cancellation is distinct from releasing temporary interest.
        with self.lock:
            return self._cancel_locked(job_id)

    def retry(self, job_id):
        with self.lock:
            old = read_json(self.path(job_id))
            if old['status'] not in TERMINAL:
                raise ValueError('只有已结束的任务可以重试；运行中任务请先等待或取消')
            if process_identity(old.get('pid', 0)) == old.get('process_identity') and old.get('process_identity'):
                raise ValueError('原任务进程仍活动，不能重试')
            verify_launch(old)
            job = self._submit_locked(old['kind'], old['payload'], old.get('dedupe_key'),
                                      parent_job_id=old['id'], force_new=True)
            if job['kind'] == 'plan':
                self.plan_refs['pinned'].append(job['id'])
                self._save_refs()
            return job

    def log(self, job_id, limit=16384):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 65536:
            raise ValueError('日志 limit 必须为 1 到 65536 字节')
        job_path = self.path(job_id)
        if not job_path.is_file():
            raise FileNotFoundError('任务不存在')
        job = read_json(job_path)
        log_path = (self.folder / f'{job_id}.log').resolve()
        if log_path.parent != self.folder.resolve():
            raise ValueError('日志路径越界')
        if log_path.is_file():
            with open(log_path, 'rb') as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                start = max(0, size - limit - 4096)
                stream.seek(start)
                content = stream.read(limit + 4096)
            if start:
                # Drop an incomplete first line: it may start inside a secret
                # whose key/prefix is before the bounded tail window.
                content = content.partition(b'\n')[2]
            text = content.decode('utf-8', errors='replace')
            truncated = size > limit
        else:
            content = str(job.get('error') or '暂无日志；任务可能尚未开始。').encode('utf-8')
            truncated = len(content) > limit
            text = content.decode('utf-8', errors='replace')
        clean = sanitize_log(text).encode('utf-8')
        truncated = truncated or len(clean) > limit
        return {'job_id': job_id, 'text': clean[-limit:].decode('utf-8', errors='ignore'),
                'truncated': truncated}

    def _loop(self):
        while not self.stopping.wait(.5):
            try:
                with self.lock:
                    jobs = self.list()
                    running = False
                    for job in jobs:
                        if job['status'] not in ('running', 'cancelling'):
                            continue
                        latest = recover_interrupted(job, self.path(job['id']))
                        if latest['status'] in ('running', 'cancelling'):
                            running = True
                    if running:
                        continue
                    queued = sorted((j for j in jobs if j['status'] == 'queued'), key=lambda j: j['created_at'])
                    if not queued:
                        continue
                    job = queued[0]
                    # Worker waits for the started flag before reading its JSON.
                    log = open(self.folder / f"{job['id']}.log", 'ab')
                    try:
                        verify_launch(job)
                        launch = job['launch']
                        env = {**os.environ, **launch['environment']}
                        for key in ('TEMP', 'TMP', 'HF_HOME', 'TORCH_HOME', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR'):
                            Path(env[key]).mkdir(parents=True, exist_ok=True)
                        proc = subprocess.Popen(launch['original_argv'],
                                                cwd=launch['cwd'], env=env, stdout=log, stderr=log,
                                                stdin=subprocess.DEVNULL,
                                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    except Exception as exc:
                        job.update(status='failed', error=str(exc), completed_at=now())
                        atomic_json(self.path(job['id']), job)
                        continue
                    finally:
                        log.close()
                    job.update(status='running', stage='starting', pid=proc.pid,
                               process_identity=process_identity(proc.pid), started_at=now())
                    atomic_json(self.path(job['id']), job)
                    (self.folder / f"{job['id']}.started").touch()
                    self.processes[job['id']] = proc
            except Exception:
                traceback.print_exc()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=2)


def run_worker(projects, job_id):
    from tools.train_next_v2.v3_studio.store import lease
    if not SAFE_ID.fullmatch(job_id):
        raise ValueError('非法任务 ID')
    with lease('workbench-' + job_id):
        return _run_worker(projects, job_id)


def _run_worker(projects, job_id):
    from .pipeline import analyze, render, prepare_visual_plan, Cancelled
    folder = Path(projects) / '.jobs'
    if not SAFE_ID.fullmatch(job_id):
        raise ValueError('非法任务 ID')
    ready = folder / f'{job_id}.started'
    for _ in range(100):
        if ready.exists():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('任务启动握手超时')
    path = folder / f'{job_id}.json'
    job = read_json(path)
    # Windows venv python.exe is a redirector: Popen observes its PID, while
    # Python code executes in its direct child. Verify the registered launcher
    # and relationship, then publish the actual worker identity under the OS lock.
    launcher_pid = job.get('pid')
    if (job.get('status') != 'running' or launcher_pid not in (os.getpid(),os.getppid())
            or not job.get('process_identity')
            or job['process_identity'] != process_identity(launcher_pid)):
        raise ValueError('任务进程身份与登记不符')
    job.update(launcher_pid=launcher_pid, launcher_identity=job['process_identity'],
               pid=os.getpid(), process_identity=process_identity(os.getpid()))
    atomic_json(path,job)
    def cancel():
        return (folder / f'{job_id}.cancel').exists()
    def progress(stage, value, details=None):
        job.update(stage=stage, progress=value, updated_at=now())
        if details is not None:
            job['details'] = deepcopy(details)
        else:
            job.pop('details', None)
        atomic_json(path, job)
    progress.accepts_details = True
    try:
        verify_launch(job)
        payload = job['payload']
        if cancel():
            raise Cancelled('用户取消')
        if job['kind'] == 'analyze':
            from .bridge import analyze_existing
            from tools.train_next_v2.v3_studio.store import lease
            from tools.train_next.runtime import run_lease
            with lease('compute'), run_lease('next-va-v2-gpu'):
                project = analyze_existing(payload['audio_path'], root=projects,
                              name=payload.get('name'), device=payload.get('device', 'cpu'),
                              include_v2=payload.get('include_v2', False), progress=progress, cancel=cancel,
                              seed=payload.get('seed',42))
            result = {'project_id': project.name}
        elif job['kind'] == 'import_analysis':
            from .bridge import import_analysis
            project = import_analysis(payload['analysis_dir'], root=projects,
                                      name=payload.get('name'), source_system=payload.get('source_system', 'b01'),
                                      seed=payload.get('seed',42))
            result = {'project_id': project.name}
        elif job['kind'] == 'import_v2':
            from .bridge import import_v2
            project = import_v2(payload['source_dir'], root=projects, name=payload.get('name'))
            result = {'project_id': project.name}
        elif job['kind'] == 'plan':
            project = resolve_project(projects, payload['project_id'])
            result = prepare_visual_plan(project, payload['base_revision'], payload['edits'], progress, cancel,
                                         smoothing=payload.get('smoothing'), source_system=payload.get('source_system'))
        elif job['kind'] == 'preview':
            from .pipeline import render_preview
            project = resolve_project(projects, payload['project_id'])
            result = render_preview(project, payload['snapshot_id'], payload.get('options'), progress, cancel)
        else:
            project = resolve_project(projects, payload['project_id'])
            result = render(project, payload['revision'], payload.get('options'), progress, cancel)
        if cancel():
            raise Cancelled('用户取消；已完成的文件保留')
        job.update(status='succeeded', result=result, progress=1.)
    except BaseException as exc:
        traceback.print_exc()
        job.update(status='cancelled' if isinstance(exc, (Cancelled, InterruptedError)) else 'failed', error=str(exc))
    finally:
        job['completed_at'] = now()
        atomic_json(path, job)
    return 0 if job['status'] == 'succeeded' else 1
