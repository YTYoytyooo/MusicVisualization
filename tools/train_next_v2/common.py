from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from tools.train_next.core import ROOT, atomic_json, sha256
from tools.train_next.runtime import is_running, run_lease, lock_path

BASE = ROOT / 'data/training/next_va/optimization'
PLAN = BASE / 'plan.json'
GPU_LEASE = 'next-va-v2-gpu'
AXES = ('valence', 'arousal')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paths(run_id):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', run_id):
        raise ValueError('Invalid run ID')
    return BASE / 'runs' / run_id, ROOT / 'models/next_va_v2' / run_id


def code_hashes():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT / 'tools/train_next' / name for name in ('core.py', 'features.py', 'runtime.py', 'run.py')]
    files += [ROOT / 'tools/datasets/deam_dataset.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)}


def verify_hashes(mapping):
    for relative, expected in mapping.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != expected:
            raise ValueError(f'Version/hash mismatch: {relative}')


def process_identity(pid=None):
    """Query PID creation identity without requiring WMI administrator access."""
    pid = os.getpid() if pid is None else int(pid)
    if os.name != 'nt':
        return {'pid': pid, 'creation_filetime': None}
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        stamps = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(stamp) for stamp in stamps)):
            return None
        created = stamps[0].dwLowDateTime | (stamps[0].dwHighDateTime << 32)
        return dict(pid=pid, creation_filetime=str(created))
    finally:
        kernel.CloseHandle(handle)


def record_event(event, run=None, updates=None):
    """Serialize orchestrator/worker ledger updates and preserve other fields."""
    with run_lease('next-va-v2-plan'):
        plan = read_json(PLAN)
        if updates:
            plan.update(updates)
        if run:
            old = next((row for row in plan['experiments'] if row['run_id'] == run['run_id']), {})
            row = {**old, **run}
            plan['experiments'] = [item for item in plan['experiments'] if item['run_id'] != row['run_id']] + [row]
            plan['active_jobs'] = [item for item in plan['active_jobs'] if item['run_id'] != row['run_id']]
            if row.get('status') in ('registered', 'running'):
                plan['active_jobs'].append(row)
        plan['updated_at'] = now()
        atomic_json(PLAN, plan)
        with (BASE / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': plan['updated_at'], **event}, ensure_ascii=False) + '\n')


def assert_gpu_available():
    historical = read_json(ROOT / 'data/training/next_va/active-training.json')
    for run_id in (historical['teacher'], historical['mixed'], 'queue-' + historical['mixed']):
        if is_running(run_id):
            raise RuntimeError(f'Historical GPU pipeline is active: {run_id}')

