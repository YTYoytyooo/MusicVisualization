"""Finite, registered integration acceptance on existing audio; no training."""
import argparse
import csv
import hashlib
import http.client
import json
import os
from pathlib import Path
import threading
import time

from tools.train_next_v2.v3_workbench.studio import server, store, pipeline
from tools.train_next_v2.v3_workbench.studio.paths import WORKSPACE, DEFAULT_PROJECTS

REPORT = WORKSPACE/'data/v3.0/reports/workbench/integration.json'


def run():
    import sys
    from tools.train_next_v2.v3_studio.jobs import environment
    store.atomic_json(REPORT.parent/'integration-launch.json',{
        'registered_at':store.now(),'original_argv':[sys.executable,*sys.argv],
        'cwd':str(WORKSPACE),'environment':environment(),
        'code_hashes':{str(p.relative_to(WORKSPACE)):store.digest(p)
                       for p in (WORKSPACE/'tools/train_next_v2/v3_workbench').rglob('*.py')},
        'script_sha256':store.digest(__file__), 'training_enabled':False,
        'log':str(REPORT.parent/'integration-console.log')})
    http_server=server.make_server(DEFAULT_PROJECTS,0)
    thread=threading.Thread(target=http_server.serve_forever,daemon=True);thread.start()
    checks=[]
    report={'status':'running','started_at':store.now(),'checks':checks,'training_performed':False}
    def checkpoint(label,details):
        checks.append({'check':label,'time':store.now(),**details})
        store.atomic_json(REPORT,report)
        print(json.dumps(checks[-1],ensure_ascii=False),flush=True)
    token=None
    def api(path,body=None):
        c=http.client.HTTPConnection('127.0.0.1',http_server.server_port,timeout=120)
        try:
            c.request('POST' if body is not None else 'GET',path,
                      json.dumps(body) if body is not None else None,
                      {'Content-Type':'application/json','X-Studio-Token':token or ''})
            r=c.getresponse();value=json.loads(r.read())
            if r.status>=400:raise RuntimeError(f'{path}: {r.status} {value}')
            return value
        finally:c.close()
    def wait(job,expect='succeeded',cancel_running=False):
        job_id=job.get('id') or job['job_id'];started=time.monotonic();announced=None;sent=False
        while time.monotonic()-started<1800:
            state=next(j for j in api('/api/jobs')['jobs'] if j['id']==job_id)
            stage=(state['status'],state.get('stage'))
            if stage!=announced:
                print(json.dumps({'job':job_id,'status':stage,'progress':state.get('progress')},ensure_ascii=False),flush=True)
                announced=stage
            if cancel_running and state['status']=='running' and not sent:
                assert state.get('process_identity')
                api('/api/jobs/'+job_id+'/cancel',{});sent=True
            if state['status'] in ('succeeded','failed','cancelled','interrupted'):
                if state['status']!=expect:raise RuntimeError(json.dumps({k:state.get(k) for k in ('id','kind','status','error')},ensure_ascii=False))
                from tools.train_next_v2.v3_workbench.studio.jobs import process_identity
                for _ in range(50):
                    if process_identity(state.get('pid',0)) != state.get('process_identity') or not state.get('process_identity'):break
                    time.sleep(.05)
                return state,time.monotonic()-started
            time.sleep(.5)
        raise TimeoutError(job_id)
    try:
        token=api('/api/bootstrap')['token']
        checkpoint('environment',pipeline.doctor())
        sources=api('/api/sources')
        source_id='p-22cc95f60fa8-d1d8b430'
        assert any(s['id']==source_id for s in sources['analyses'])
        source=WORKSPACE/'data/v3.0/projects'/source_id
        source_hashes={str(p):store.digest(p) for p in source.rglob('*') if p.is_file()}
        state,elapsed=wait(api('/api/import-analysis',{'analysis_id':source_id,
                'name':'V3 完整制作验收 · 614','source_system':'b01','seed':73}))
        work=DEFAULT_PROJECTS/state['result']['project_id'];route='/api/projects/'+work.name
        p=api(route)
        assert p['project']['seed']==73
        assert p['project']['tempo']>0 and p['project']['beat_times']
        assert {r['id'] for r in p['systems']}=={'b01','b00','v2'}
        raw_hashes={r['id']:r['raw_sha256'] for r in p['systems']}
        checkpoint('real_import',{'project_id':work.name,'seconds':elapsed,
                   'duration':p['project']['duration'],'systems':list(raw_hashes),'tempo':p['project']['tempo']})
        report['project_id']=work.name
        # Alternate source is a view until an explicit revision save.
        v2=api(route+'?source_system=v2')
        assert v2['timeline']['step_us']==100000 and v2['revision']['source_system']=='b01'
        edits=[dict(id='accept-va',layer='emotion',field='arousal',operation='interval_set',
                    start_us=2_000_000,end_us=6_000_000,value=.45,
                    transition_in_us=1_000_000,transition_out_us=1_000_000,interpolation='smootherstep')]
        base=p['revision']['id']
        plan=api(route+'/visual-plan',{'base_revision':base,'edits':edits,'source_system':'b01'})
        if plan['status']=='queued':wait(plan)
        edits.append(dict(id='accept-color',layer='visual',field='hue_base',operation='interval_set',
                         start_us=4_000_000,end_us=8_000_000,value=210,
                         transition_in_us=500_000,transition_out_us=500_000,interpolation='smoothstep'))
        motion={'engine':'flow-v1','sensitivity':1.,'min_hold_seconds':6.,'transition_seconds':2.,'overrides':[]}
        revised=api(route+'/revisions',{'base_revision':base,'edits':edits,'source_system':'b01','motion':motion})
        checkpoint('emotion_visual_motion_revision',{'revision':revised['id'],'source_system':revised['source_system']})
        # Full song, real mux, default 720p30 with the original audio.
        full,elapsed=wait(api(route+'/renders',{'revision':revised['id'],'mode':'analysis'}))
        result=full['result']; folder=work/'renders'/result['id']
        metadata=store.read_json(folder/'render.json')
        assert metadata['status']=='succeeded' and metadata['video_file']=='output.mp4'
        assert metadata['settings']['end']==p['project']['duration']
        assert metadata['frames']==len(__import__('numpy').arange(0,p['project']['duration'],1/30))
        with (folder/'frame_values.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        assert len(rows)==metadata['frames']
        checkpoint('full_song_720p30_with_audio',{'render_id':result['id'],'seconds':elapsed,
                   'frames':metadata['frames'],'duration':p['project']['duration'],
                   'video_bytes':(folder/'output.mp4').stat().st_size,'video':str(folder/'output.mp4')})
        report['full_video']=str(folder/'output.mp4')
        # Selected 1080p presentation and late-offset checkpoint execution.
        hi,elapsed=wait(api(route+'/renders',{'revision':revised['id'],'mode':'presentation',
                   'width':1920,'height':1080,'fps':30,'start':20.,'end':21.}))
        checkpoint('1080p_late_excerpt',{'render_id':hi['result']['id'],'seconds':elapsed})
        # Preview is pinned even when a later saved revision switches driver.
        queued=api(route+'/previews',{'base_revision':revised['id'],'edits':edits,
                   'source_system':'b01','motion':motion,'edit_id':'accept-color','start':3.,'end':4.})
        switched=api(route+'/revisions',{'base_revision':revised['id'],'edits':[],
                    'source_system':'v2','motion':{'engine':'legacy'}})
        preview,elapsed=wait(queued)
        assert preview['result']['source_system']=='b01'
        checkpoint('snapshot_pinned_after_driver_switch',{'render_id':preview['result']['id'],
                   'saved_driver':switched['source_system'],'preview_driver':preview['result']['source_system']})
        # V2 native clock's final partial interval can render without stretching.
        end=p['project']['duration']
        tail,_=wait(api(route+'/renders',{'revision':switched['id'],'mode':'presentation',
                   'width':640,'height':360,'fps':30,'start':end-.4,'end':end}))
        checkpoint('v2_native_tail',{'render_id':tail['result']['id'],'source_system':tail['result']['source_system']})
        # Cancel a real identified worker; retry the unchanged short job.
        candidate=api(route+'/renders',{'revision':switched['id'],'mode':'presentation',
                     'width':640,'height':360,'fps':30,'start':0.,'end':1.})
        cancelled,_=wait(candidate,expect='cancelled',cancel_running=True)
        retry=api('/api/jobs/'+cancelled['id']+'/retry',{})
        retried,_=wait(retry)
        assert retried['payload']==cancelled['payload']
        checkpoint('actual_cancel_and_same_configuration_retry',{'cancelled':cancelled['id'],'retried':retried['id']})
        legacy=WORKSPACE/'data/v2.0/projects/p-a1bf51ef7a59'
        legacy_hashes={str(p):store.digest(p) for p in legacy.rglob('*') if p.is_file() and p.name!='.write.lock'}
        migrated,_=wait(api('/api/import-v2',{'path':str(legacy),'name':'V2 历史修订兼容验收（合成演示）'}))
        imported=DEFAULT_PROJECTS/migrated['result']['project_id']
        migration=store.read_json(imported/'migration.json')
        assert len(migration['revision_mapping'])>=3
        assert store.load_project(imported)['source_system']=='v2'
        assert all(store.digest(k)==v for k,v in legacy_hashes.items())
        checkpoint('v2_revision_archive_migration',{'project_id':imported.name,'revision_mapping':migration['revision_mapping']})
        assert {r['id']:r['raw_sha256'] for r in api(route)['systems']}==raw_hashes
        assert all(store.digest(k)==v for k,v in source_hashes.items())
        protected=store.read_json(WORKSPACE/'data/v3.0/reports/workbench/protected-before.json')
        assert all(store.digest(WORKSPACE/k)==v for k,v in protected.items())
        checkpoint('protected_sources_unchanged',{'protected_files':len(protected),
                     'original_analysis_files':len(source_hashes),'v2_project_files':len(legacy_hashes)})
        report.update(status='passed',completed_at=store.now())
    except BaseException as exc:
        report.update(status='failed',error=str(exc),completed_at=store.now())
        raise
    finally:
        store.atomic_json(REPORT,report)
        http_server.shutdown();http_server.manager.close();http_server.server_close();thread.join()


if __name__=='__main__':
    run()
