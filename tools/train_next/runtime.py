"""OS-held run locks: process death releases ownership even if status is stale."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import os

from .core import ROOT, output_paths


def _lock(stream, unlock=False):
    stream.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB)


def lock_path(run_id):
    output_paths(run_id)
    return ROOT / 'data/training/next_va/locks' / (run_id + '.lock')


@contextmanager
def run_lease(run_id):
    path = lock_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        try:
            _lock(stream)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise RuntimeError(f'Run already active: {run_id}') from exc
            raise
        try:
            yield
        finally:
            _lock(stream, unlock=True)


def is_running(run_id):
    path = lock_path(run_id)
    if not path.exists():
        return False
    with path.open('r+b') as stream:
        try:
            _lock(stream)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return True
            raise
        _lock(stream, unlock=True)
        return False
