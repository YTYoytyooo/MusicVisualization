"""Finite real-work API acceptance for historical restore and manual motion."""
import http.client
import json
import time
from tools.train_next_v2.v3_workbench.studio import store
from tools.train_next_v2.v3_workbench.studio.paths import WORKSPACE

token=''
def api(path,body=None):
    conn=http.client.HTTPConnection('127.0.0.1',8768,timeout=30)
    try:
        conn.request('POST' if body is not None else 'GET',path,
            json.dumps(body) if body is not None else None,
            {'Content-Type':'application/json','X-Studio-Token':token})
        response=conn.getresponse();result=json.loads(response.read())
        assert response.status<400, result
        return result
    finally:conn.close()

def wait(job):
    jid=job.get('id') or job['job_id']
    start=time.monotonic()
    while time.monotonic()-start<300:
        item=next(j for j in api('/api/jobs')['jobs'] if j['id']==jid)
        if item['status'] in ('succeeded','failed','cancelled','interrupted'):
            assert item['status']=='succeeded',item.get('error')
            return item
        time.sleep(.5)
    raise TimeoutError(jid)

token=api('/api/bootstrap')['token']
route='/api/projects/p-51315e7a7fa2'
current=api(route)
old=api(route+'?revision=r000001')
assert current['project']['current_revision']=='r000003', 'Do not repeat this acceptance after it has completed.'
before={s['id']:s['raw_sha256'] for s in current['systems']}
restored=api(route+'/revisions',{'base_revision':current['project']['current_revision'],
    'edits':old['revision']['edits'],'smoothing':old['revision']['smoothing'],
    'motion':old['revision']['motion'],'source_system':old['revision']['source_system']})
assert restored['id']=='r000004' and restored['edits']==old['revision']['edits']
assert api(route+'?revision=r000003')['revision']['edits']==[]
motion={'engine':'flow-v1','sensitivity':1.,'min_hold_seconds':6.,'transition_seconds':2.,
    'overrides':[{'id':'manual-orbit','mode':'orbit','start_us':1_000_000,'end_us':3_000_000,
        'transition_in_us':500_000,'transition_out_us':500_000,'interpolation':'smootherstep',
        'params':{'speed':.2,'radius':.25,'rotation':-1,'trail_seconds':.8}}]}
body={'base_revision':restored['id'],'edits':restored['edits'],
      'smoothing':restored['smoothing'],'source_system':'b01','motion':motion}
plan=api(route+'/visual-plan',body)
if plan['status']=='queued':wait(plan)
api(route+'/motion-plan',body)
saved=api(route+'/revisions',body)
started=time.monotonic()
job=wait(api(route+'/renders',{'revision':saved['id'],'mode':'presentation',
         'start':1.,'end':2.,'width':640,'height':360,'fps':30}))
after=api(route)
assert before=={s['id']:s['raw_sha256'] for s in after['systems']}
store.atomic_json(WORKSPACE/'data/v3.0/reports/workbench/revision-motion-acceptance.json',{
    'status':'passed','completed_at':store.now(),'project_id':current['project']['id'],
    'restore_from':'r000001','restore_parent':'r000003','restored_as':restored['id'],
    'motion_revision':saved['id'],'manual_motion':saved['motion'],
    'render':job['result'],'render_wall_seconds':time.monotonic()-started,
    'original_predictions_unchanged':True,'browser_click_acceptance':False,
    'scope':'Actual loopback API and worker, supplemental to browser edits/driver switches.'})
print(json.dumps({'status':'passed','restored':restored['id'],'motion_revision':saved['id'],'render':job['result']['id']}))
