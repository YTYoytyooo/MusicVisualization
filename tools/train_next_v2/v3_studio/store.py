"""Small, auditable persistence boundary for the independent V3 application."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/v3.0'
MANIFEST = ROOT / 'models/next_va_v2/v3-preview-v1/manifest.json'
PYTHON = ROOT / 'data/training/next_va/env/Scripts/python.exe'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    # Windows readers/antivirus can briefly deny replace while holding an open
    # destination handle. Retry only sharing/access conflicts, keeping atomic
    # publication and the exact temporary bytes; permanent failures still raise.
    for attempt in range(8):
        try:
            temp.replace(path)
            break
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt==7:
                raise
            time.sleep(min(.01*2**attempt,.2))


def inside(path, root=ROOT):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path is outside the authorized workspace')
    return path


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,96}', value):
        raise ValueError('Invalid identifier')
    return value


def project_path(project_id):
    return inside(DATA / 'projects' / identifier(project_id), DATA / 'projects')


def library():
    path = DATA / 'library.json'
    if not path.exists():
        return []
    value = read(path)
    return value if isinstance(value, list) else value['songs']


def audio_entry(audio_id):
    row = next((r for r in library() if r['id'] == audio_id), None)
    if row is None:
        raise ValueError('Audio is not in the frozen existing-audio library')
    path = inside(ROOT / row['path'])
    if not path.is_file() or digest(path) != row['sha256']:
        raise ValueError('Source audio is missing or has changed')
    return row


def load_project(project_id):
    directory = project_path(project_id)
    if not (directory/'project.json').exists() and (directory/'pending.json').exists():
        pending=read(directory/'pending.json')
        launch=read(DATA/'jobs'/identifier(pending['job_id'])/'launch.json')
        if (pending['id']!=project_id or launch['project_id']!=project_id or launch['kind']!='analyze'
                or pending['audio']['sha256']!=launch['arguments']['audio_sha256']
                or pending['audio']['path']!=launch['arguments']['audio_path']
                or pending['manifest_sha256']!=launch['manifest_sha256']):
            raise ValueError('Pending project identity mismatch')
        return {'project':pending,'result':None}
    project = read(directory / 'project.json')
    result_path = directory / 'analysis/result.json'
    result = None
    if result_path.exists():
        receipt = read(directory / 'analysis/receipt.json')
        expected = receipt.get('result_sha256')
        if not expected or digest(result_path) != expected:
            raise ValueError('Raw prediction receipt mismatch; preserve and inspect the project')
        if receipt.get('project_sha256') != digest(directory / 'project.json'):
            raise ValueError('Project receipt mismatch; preserve and inspect the project')
        result = read(result_path)
        if result['project_id'] != project_id or result['audio'] != project['audio']:
            raise ValueError('Project identity mismatch')
        if (result['manifest_sha256'] != project['manifest_sha256']
                or receipt.get('manifest_sha256') != result['manifest_sha256']
                or receipt.get('audio_sha256') != result['audio']['sha256']):
            raise ValueError('Prediction provenance mismatch')
    elif (directory / 'analysis/receipt.json').exists():
        raise ValueError('Receipt has no raw result')
    return {'project': project, 'result': result}


def projects():
    result = []
    paths={p.parent:p for name in ('pending.json','project.json') for p in (DATA/'projects').glob('*/'+name)}
    for path in sorted(paths.values(), reverse=True):
        try:
            loaded = load_project(path.parent.name); p=loaded['project'];ready=loaded['result'] is not None
            result.append({'id':path.parent.name,'name':p['audio']['name'],
                           'status':'ready' if ready else 'pending','has_result':ready,
                           'created_at':p.get('created_at',''),
                           'includes_v2':ready and 'v2' in loaded['result']['systems']})
        except (ValueError,KeyError,OSError) as exc:
            result.append({'id':path.parent.name,'name':path.parent.name,'status':'invalid',
                           'has_result':False,'error':str(exc)})
    return sorted(result,key=lambda p:p.get('created_at',''),reverse=True)


@contextmanager
def lease(name):
    path = DATA / 'locks' / (identifier(name) + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        if path.stat().st_size == 0:
            f.write(b'0'); f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            lock = lambda: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: (f.seek(0), msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            lock = lambda: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(f, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise RuntimeError('An operation already owns ' + name) from exc
        try:
            yield
        finally:
            unlock()


def identity(pid=None):
    pid = os.getpid() if pid is None else int(pid)
    if os.name != 'nt':
        try:
            return {'pid': pid, 'created': Path(f'/proc/{pid}/stat').read_text().split()[21]}
        except OSError:
            return None
    import ctypes
    from ctypes import wintypes as w
    dll = ctypes.WinDLL('kernel32', use_last_error=True)
    dll.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]; dll.OpenProcess.restype = w.HANDLE
    dll.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)]*4
    dll.CloseHandle.argtypes = [w.HANDLE]
    handle = dll.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        times = [w.FILETIME() for _ in range(4)]
        if not dll.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        code = w.DWORD()
        dll.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
        if not dll.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
            return None
        return {'pid': pid, 'created': str(times[0].dwLowDateTime | times[0].dwHighDateTime << 32)}
    finally:
        dll.CloseHandle(handle)
