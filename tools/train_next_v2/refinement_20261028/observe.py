"""Read-only status with native Windows termination checks, separate from frozen inputs.

Windows may retain an exited process object while a launcher owns its handle.
A readable creation timestamp alone is not proof that a worker is alive.
"""
from __future__ import annotations

import argparse
import os
import json

from .run import actual_status


def observe_process(pid, expected_identity=None):
    if os.name != 'nt':
        return dict(state='unknown', reason='Native identity observer currently supports Windows only')
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    if not isinstance(pid, int) or pid <= 0:
        return dict(state='missing', reason='No valid PID')
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        return dict(state='missing' if error in (87, 1168) else 'unknown', winerror=error)
    try:
        stamps = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(stamp) for stamp in stamps)):
            return dict(state='unknown', reason='Creation identity query failed', winerror=ctypes.get_last_error())
        created = stamps[0].dwLowDateTime | (stamps[0].dwHighDateTime << 32)
        identity = dict(pid=pid, creation_filetime=str(created))
        if expected_identity is not None and identity != expected_identity:
            return dict(state='identity_mismatch', identity=identity)
        waited = kernel.WaitForSingleObject(handle, 0)
        if waited == 258:  # WAIT_TIMEOUT: the process object is not signaled.
            return dict(state='alive', identity=identity)
        if waited == 0:
            code = wintypes.DWORD()
            ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            return dict(state='exited', identity=identity, exit_code=int(code.value) if ok else None)
        return dict(state='unknown', identity=identity, wait_result=int(waited), winerror=ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def observe(arm, seed):
    result = actual_status(arm, seed)
    identity = result['process_identity']
    worker = observe_process(identity['pid'], identity) if identity else dict(state='not_started')
    alive = worker['state'] == 'alive'
    held = result['run_lock_held']
    result.update(worker_process=worker, same_process_alive=alive,
                  active=bool(alive and held),
                  indeterminate=worker['state'] == 'unknown' or bool(alive != held))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=('control', 'mean_loss', 'tone_residual'))
    parser.add_argument('--seed', required=True, type=int, choices=(20261003, 20261004, 20261005))
    args = parser.parse_args()
    print(json.dumps(observe(args.arm, args.seed), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
