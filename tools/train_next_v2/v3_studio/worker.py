"""Registered foreground worker; app launch does not modify legacy application state."""
from __future__ import annotations
import argparse
import os
import traceback
from . import store,jobs


def _run(job_id, resume=False):
    path=jobs.directory(job_id);launch=store.read(path/'launch.json')
    previous=store.read(path/'status.json')
    if previous['status']=='completed':raise ValueError('Completed jobs are immutable; create a separate operation')
    with store.lease(job_id),store.lease('compute'):
        ident=store.identity()
        def update(value):
            state={'status':'running','stage':'初始化','progress':None,'process_identity':ident,
                   'updated_at':store.now(),**value}
            store.atomic(path/'status.json',state)
            print(state,flush=True)
        def cancelled(): return (path/'cancel.json').exists()
        update({})
        try:
            for key,value in launch['environment'].items(): os.environ[key]=value
            for p,expected in launch['code_hashes'].items():
                if store.digest(store.ROOT/p)!=expected: raise ValueError('Registered code changed: '+p)
            if store.digest(store.MANIFEST)!=launch['manifest_sha256']: raise ValueError('Registered manifest changed')
            if cancelled(): raise InterruptedError('Cancelled before execution')
            if launch['kind']=='analyze':
                args=launch['arguments'];row=store.audio_entry(args['audio_id'])
                if row['sha256']!=args['audio_sha256']: raise ValueError('Registered source changed')
                from tools.train_next.runtime import run_lease
                from tools.train_next_v2.v3_inference import analyze
                # Honor the same GPU lease as all previous research tasks.
                with run_lease('next-va-v2-gpu'):
                    analyze(store.ROOT/row['path'],store.project_path(launch['project_id']),
                            device=args.get('device','cuda'),progress=update,cancelled=cancelled,
                            include_v2=args.get('include_v2',False))
            else:
                from .render import export
                args=launch['arguments']
                export(launch['project_id'],args.get('smoothing_seconds',0),args['start'],args['end'],update,cancelled)
            if cancelled(): raise InterruptedError('Cancelled; completed receipts retained')
            update({'status':'completed','stage':'完成','progress':1})
        except InterruptedError as exc:
            update({'status':'cancelled','stage':'已取消，已验证缓存保留','error':str(exc)})
        except Exception as exc:
            traceback.print_exc()
            update({'status':'failed','stage':'失败','error':str(exc)})
            raise


def run(job_id,resume=False):
    try:
        return _run(job_id,resume)
    except Exception as exc:
        path=jobs.directory(job_id)
        current=store.read(path/'status.json')
        if current.get('status') not in ('completed','failed'):
            ident=current.get('process_identity')
            # A competing live worker owns its own status; never overwrite it.
            if not ident or ident==store.identity() or store.identity(ident['pid'])!=ident:
                store.atomic(path/'status.json',{'status':'failed','stage':'执行前检查失败',
                    'error':str(exc),'updated_at':store.now(),'process_identity':store.identity()})
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.job,a.resume)


if __name__=='__main__': main()
