"""V3 entry points share the same registered worker as the application."""
import argparse
import json
from . import store,jobs


def main():
    p=argparse.ArgumentParser(description='V3 离线音乐情绪研究预览')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('serve');s.add_argument('--port',type=int,default=8768)
    sub.add_parser('prepare');sub.add_parser('status');sub.add_parser('doctor')
    a=sub.add_parser('analyze');a.add_argument('audio_id');a.add_argument('--device',choices=['cpu','cuda'],default='cuda');a.add_argument('--include-v2',action='store_true')
    r=sub.add_parser('resume');r.add_argument('job_id')
    e=sub.add_parser('export');e.add_argument('project_id');e.add_argument('--start',type=float,default=0);e.add_argument('--end',type=float,default=15);e.add_argument('--smoothing',type=float,default=0)
    args=p.parse_args()
    if args.command=='prepare':
        from tools.train_next_v2.v3_inference import prepare_manifest,build_inventory
        print(prepare_manifest());print(json.dumps(build_inventory(),ensure_ascii=False))
    elif args.command=='doctor':
        checks={'python':str(store.PYTHON),'python_exists':store.PYTHON.is_file(),'manifest':store.MANIFEST.is_file(),
                'existing_library_songs':len(store.library()),'ffmpeg':(store.ROOT/'versions/v2.0/.runtime/bin/ffmpeg.exe').is_file(),
                'training_enabled':False,'promoted_model':False}
        print(json.dumps(checks,ensure_ascii=False,indent=2))
    elif args.command=='serve':
        if not store.MANIFEST.exists() or not store.library():
            raise RuntimeError('V3 model/library not prepared; run the explicit prepare command first')
        from .server import make_server
        server=make_server(args.port)
        print(f'V3 research preview: http://127.0.0.1:{server.server_port}',flush=True)
        try:server.serve_forever()
        finally:server.server_close()
    elif args.command=='status':print(json.dumps(jobs.all_jobs(),ensure_ascii=False,indent=2))
    else:
        from .worker import run
        if args.command=='resume':
            state=jobs.status(args.job_id)
            if state['process_alive'] or state['status'] not in ('failed','cancelled','interrupted'):raise ValueError('Job cannot resume')
            cancel=jobs.directory(args.job_id)/'cancel.json'
            if cancel.exists():cancel.rename(cancel.with_name('cancel-acknowledged-'+str(__import__('time').time_ns())+'.json'))
            run(args.job_id,True)
        else:
            if args.command=='analyze':item=jobs.submit('analyze',{'audio_id':args.audio_id,'device':args.device,'include_v2':args.include_v2},start=False)
            else:item=jobs.submit('export',{'start':args.start,'end':args.end,'smoothing_seconds':args.smoothing},args.project_id,start=False)
            print(json.dumps(item),flush=True);run(item['job_id'])


if __name__=='__main__':main()
