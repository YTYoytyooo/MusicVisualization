"""Hold only a thread execution request while the bound evaluation is alive."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import ROOT, BASE, PLAN, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.research_inference.benchmark import process_alive
from tools.train_next_v2.run import Tee

JOB='f-validation-ac-guard-66428-v1'
TARGET='f-validation-v2-mixed-b01-cpu-v1'
TARGET_DIRECTORY=BASE/'comparison_jobs'/TARGET
DIRECTORY=BASE/'execution_guards'/JOB
CONTINUOUS=0x80000000
SYSTEM_REQUIRED=0x00000001


def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()


class PowerStatus(ctypes.Structure):
    _fields_=[('ac',wintypes.BYTE),('battery_flags',wintypes.BYTE),('battery_percent',wintypes.BYTE),
              ('battery_saver',wintypes.BYTE),('battery_seconds',wintypes.DWORD),('battery_full_seconds',wintypes.DWORD)]


class NativePower:
    def __init__(self):
        if os.name!='nt': raise RuntimeError('Windows only')
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.GetSystemPowerStatus.argtypes=[ctypes.POINTER(PowerStatus)]
        self.api.GetSystemPowerStatus.restype=wintypes.BOOL
        self.api.SetThreadExecutionState.argtypes=[wintypes.DWORD]
        self.api.SetThreadExecutionState.restype=wintypes.DWORD

    def query(self):
        power=PowerStatus()
        if not self.api.GetSystemPowerStatus(ctypes.byref(power)):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(ac=int(power.ac),battery_percent=int(power.battery_percent),battery_saver=int(power.battery_saver))

    def request(self, enabled):
        # Deliberately omit DISPLAY_REQUIRED and AWAYMODE_REQUIRED.
        previous=int(self.api.SetThreadExecutionState(CONTINUOUS | (SYSTEM_REQUIRED if enabled else 0)))
        if previous==0: raise ctypes.WinError(ctypes.get_last_error())
        return previous


def should_hold(alive, lease, power):
    return alive is True and lease is True and power['ac']==1 and power['battery_saver']==0


def files():
    return {rel(p):sha256(p) for p in [*Path(__file__).parent.glob('*.py'),ROOT/'tools/train_next_v2/tests/test_execution_guard.py']}


def state(stage, **values):
    data=dict(job_id=JOB,stage=stage,updated_at=now(),**values)
    atomic_json(DIRECTORY/'status.json',data)
    print(json.dumps(data,ensure_ascii=False),flush=True)


def tests():
    path=DIRECTORY/'tests.json'
    if path.exists(): raise ValueError('Preserve test evidence')
    run=dict(original_argv=[sys.executable,'-u','-m',__package__+'.guard','--tests'],cwd=str(ROOT),code=files(),started_at=now())
    atomic_json(DIRECTORY/'tests-launch.json',run)
    record_event(dict(event='F_ac_execution_guard_tests_registered',**run))
    with (DIRECTORY/'tests.log').open('w',encoding='utf-8') as stream:
        result=unittest.TextTestRunner(stream=Tee(sys.stdout,stream),verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromName('tools.train_next_v2.tests.test_execution_guard'))
    if not result.wasSuccessful(): raise ValueError('Guard policy tests failed')
    native=NativePower(); before=native.query()
    if not should_hold(True,True,before): raise ValueError('AC with battery saver off required for the brief native request/release test')
    previous=None; released=None
    try:
        previous=native.request(True)
    finally:
        released=native.request(False)
    if not released & SYSTEM_REQUIRED: raise ValueError('Native API did not retain the temporary thread request')
    report=dict(status='passed',tests=result.testsRun,code=files(),completed_at=now(),native_power=before,
                previous_thread_flags=previous,flags_before_release=released,native_request_then_release_verified=True)
    atomic_json(path,report)
    record_event(dict(event='F_ac_execution_guard_tests_finished',**report))


def prepare():
    if (DIRECTORY/'launch.json').exists(): raise ValueError('Preserve frozen guard configuration')
    test=read_json(DIRECTORY/'tests.json')
    if test['status']!='passed' or test['tests']!=3 or not test['native_request_then_release_verified']:
        raise ValueError('Guard tests required')
    verify_hashes(test['code'])
    launch=read_json(TARGET_DIRECTORY/'launch.json'); process=read_json(TARGET_DIRECTORY/'process.json')
    identity=process['process_identity']
    if identity['pid']!=66428 or process_alive(identity) is not True or not is_running(TARGET):
        raise ValueError('Expected target process identity and actual lease required')
    command=[launch['python'],'-u','-m',__package__+'.guard','--launch',str(DIRECTORY/'launch.json')]
    pinned=files()
    verify_hashes(launch['pinned_hashes'])
    for path in (DIRECTORY/'tests.json',TARGET_DIRECTORY/'launch.json',TARGET_DIRECTORY/'process.json',
                 ROOT/'tools/train_next/core.py',ROOT/'tools/train_next/runtime.py',ROOT/'tools/train_next_v2/common.py',
                 ROOT/'tools/train_next_v2/research_inference/benchmark.py',ROOT/'tools/train_next_v2/run.py'):
        pinned[rel(path)]=sha256(path)
    value=dict(job_id=JOB,registered_at=now(),python=launch['python'],cwd=str(ROOT),environment=launch['environment'],
        original_argv=command,resume_argv=command+['--resume'],pinned_hashes=pinned,target_job=TARGET,
        target_process_identity=identity,target_launch=rel(TARGET_DIRECTORY/'launch.json'),
        os_lock=str(lock_path(JOB)),log=rel(DIRECTORY/'console.log'),status_module=__package__+'.guard --status',
        poll_seconds=15,policy='On AC with battery saver off, only while the exact target PID creation identity is alive and target OS lease is held. Clear on battery/unknown power and on target exit. No display/away-mode flags or persistent power-plan edits.',
        recovery='If this helper exits unexpectedly, first confirm its PID/lease inactive. Use original argv + --resume only if the exact original target identity remains active; never silently rebind to a replacement PID.',
        docs=['https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate',
              'https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-getsystempowerstatus'])
    atomic_json(DIRECTORY/'launch.json',value)
    stages=read_json(PLAN)['stages']; stages['F']['execution_guard']=dict(job_id=JOB,launch=rel(DIRECTORY/'launch.json'),target_job=TARGET,temporary=True)
    record_event(dict(event='F_ac_execution_guard_registered',job_id=JOB,target=TARGET,identity=identity),
        run=dict(run_id=JOB,stage='F',kind='temporary_ac_execution_guard',status='registered',launch=rel(DIRECTORY/'launch.json')),
        updates=dict(stages=stages))
    print(json.dumps(dict(launch=rel(DIRECTORY/'launch.json'),target_process_identity=identity)))


def run(launch,resume):
    previous=DIRECTORY/'process.json'
    if previous.exists():
        if not resume: raise ValueError('Existing helper requires --resume')
        if process_alive(read_json(previous)['process_identity']) is not False: raise ValueError('Previous guard not confirmed exited')
    if (DIRECTORY/'report.json').exists(): raise ValueError('Guard already completed')
    identity=launch['target_process_identity']
    if process_alive(identity) is not True or not is_running(TARGET): raise ValueError('Bound target not active')
    atomic_json(previous,dict(process_identity=process_identity(),started_at=now(),resumed=resume))
    record_event(dict(event='F_ac_execution_guard_started',job_id=JOB,target=TARGET),run=dict(run_id=JOB,status='running',stage='F'))
    native=NativePower(); held=False; last=None; reason=None; checks=0
    try:
        while True:
            alive=process_alive(identity); lease=is_running(TARGET); power=native.query()
            desired=should_hold(alive,lease,power)
            if desired!=held:
                native.request(desired); held=desired
            checks+=1
            current=(held,alive,lease,power['ac'],power['battery_saver'])
            values=dict(target_process_alive=alive,target_lease_active=lease,request_held=held,power=power,checks=checks)
            if current!=last or checks%4==0:
                state('guarding_on_ac' if held else 'request_released',**values); last=current
            else:
                atomic_json(DIRECTORY/'status.json',dict(job_id=JOB,stage='guarding_on_ac' if held else 'request_released',updated_at=now(),**values))
            if alive is not True or not lease:
                reason='target_exited_or_lease_released' if alive is False or not lease else 'target_identity_unverifiable'
                break
            time.sleep(launch['poll_seconds'])
    finally:
        native.request(False)
    if reason=='target_identity_unverifiable': raise RuntimeError('Target process query became unverifiable; request cleared')
    report=dict(job_id=JOB,status='completed',completed_at=now(),target=TARGET,reason=reason,request_released=True,
                persistent_power_settings_changed=False,checks=checks,launch_sha256=sha256(DIRECTORY/'launch.json'))
    atomic_json(DIRECTORY/'report.json',report); state('completed',reason=reason,request_held=False)
    record_event(dict(event='F_ac_execution_guard_completed',**report),run=dict(run_id=JOB,status='completed',stage='F'))


def status():
    process=read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else None
    return dict(job_id=JOB,checked_at=now(),lease_active=is_running(JOB),
                process_alive=process_alive(process['process_identity']) if process else None,
                process_identity=process['process_identity'] if process else None,
                progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__); modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--tests',action='store_true'); modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--status',action='store_true'); modes.add_argument('--launch',type=Path)
    parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    if args.status: print(json.dumps(status(),ensure_ascii=False)); return
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    with run_lease(JOB):
        if args.tests: tests(); return
        if args.prepare: prepare(); return
        if args.launch.resolve()!=DIRECTORY/'launch.json': raise ValueError('Wrong guard launch path')
        launch=read_json(args.launch)
        with (DIRECTORY/'console.log').open('a',encoding='utf-8',buffering=1) as stream:
            with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
                try:
                    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(launch['python']).resolve():
                        raise ValueError('Wrong workspace or isolated Python')
                    if any(os.environ.get(k)!=v for k,v in launch['environment'].items()): raise ValueError('Guard environment changed')
                    verify_hashes(launch['pinned_hashes']); run(launch,args.resume)
                except Exception as exc:
                    failure=dict(error=f'{type(exc).__name__}: {exc}',at=now(),traceback=traceback.format_exc())
                    atomic_json(DIRECTORY/f'failure-{time.time_ns()}.json',failure)
                    record_event(dict(event='F_ac_execution_guard_failed',job_id=JOB,**failure),run=dict(run_id=JOB,status='failed',stage='F'))
                    raise


if __name__=='__main__': main()
