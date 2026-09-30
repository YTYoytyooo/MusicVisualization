import copy
import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
from unittest.mock import patch
import numpy as np
import soundfile as sf

from studio.demo import create_demo
from studio.store import digest, load_project, load_revision, load_raw, save_revision
from studio.pipeline import (timeline_payload, validate_preview, create_preview_snapshot,
                             render, render_preview, planning_key)
from studio.server import make_server


class MotionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.project=create_demo(self.root,4.)
        self.config={'engine':'flow-v1','overrides':[dict(id='motion-a',mode='rise',
            start_us=1000000,end_us=3000000,transition_in_us=1000000,transition_out_us=1000000,
            params={'speed':.1,'turbulence':0.,'coherence':1.})]}

    def test_revision_keeps_legacy_and_raw_and_inherits_motion_if_omitted(self):
        original=digest(self.project/'analysis/predictions_raw.npz')
        self.assertEqual(timeline_payload(self.project)['motion']['config']['engine'],'legacy')
        before_key=planning_key(self.project,[],iterations=1)
        revision=save_revision(self.project,'r000000',[],motion=self.config)
        payload=timeline_payload(self.project,revision=revision['id'])
        self.assertEqual(payload['motion']['config']['engine'],'flow-v1')
        self.assertEqual(payload['motion']['effective'][20]['mode'],'rise')
        self.assertEqual(load_revision(self.project,'r000000')['motion']['engine'],'legacy')
        inherited=save_revision(self.project,revision['id'],[])
        self.assertEqual(inherited['motion'],revision['motion'])
        self.assertEqual(planning_key(self.project,[],iterations=1),before_key)
        self.assertEqual(digest(self.project/'analysis/predictions_raw.npz'),original)

    def test_motion_snapshot_is_frozen_and_selects_its_own_support(self):
        snapshot=create_preview_snapshot(self.project,'r000000',[],motion=self.config,motion_edit_id='motion-a')
        saved=copy.deepcopy(snapshot['motion'])
        self.config['overrides'][0]['mode']='fall'
        save_revision(self.project,'r000000',[],motion=self.config)
        with patch('studio.pipeline.render',return_value={}) as execute:
            render_preview(self.project,snapshot['id'])
        self.assertEqual(execute.call_args.kwargs['_snapshot']['motion'],saved)
        self.assertEqual(snapshot['options']['start'],0)
        self.assertEqual(snapshot['options']['end'],4)

    def test_http_plan_validation_and_saved_motion_roundtrip(self):
        server=make_server(self.root,0,start_queue=False)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base=f'http://127.0.0.1:{server.server_port}'
        with urllib.request.urlopen(base+'/api/bootstrap') as result:token=json.load(result)['token']
        def post(suffix,body):
            req=urllib.request.Request(base+'/api/projects/'+self.project.name+'/'+suffix,
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Studio-Token':token})
            with urllib.request.urlopen(req) as result:return result.status,json.load(result)
        body={'base_revision':'r000000','edits':[],'motion':self.config}
        status,plan=post('motion-plan',body)
        self.assertEqual(status,200);self.assertTrue(plan['segments'])
        self.assertIn('scores',plan['auto'][0])
        status,preview=post('preview',body)
        self.assertEqual(preview['motion']['config']['engine'],'flow-v1')
        status,revision=post('revisions',body)
        self.assertEqual(status,201)
        with urllib.request.urlopen(base+'/api/projects/'+self.project.name) as result:detail=json.load(result)
        self.assertEqual(detail['revision']['motion'],revision['motion'])
        with urllib.request.urlopen(base+'/motion-math.js') as result:self.assertEqual(result.status,200)

    def test_actual_frame_motion_matches_csv_and_does_not_use_legacy_update(self):
        from renderer import VideoRenderer, ParticleSystem
        revision=save_revision(self.project,'r000000',[],motion=self.config)
        seen=[];original=VideoRenderer.render_frame
        def capture(renderer,*args,**kwargs):
            seen.append(copy.deepcopy(kwargs['motion']))
            return original(renderer,*args,**kwargs)
        with patch('feature_extraction.load_audio',side_effect=lambda path:sf.read(path,dtype='float32')), \
                patch.object(VideoRenderer,'render_frame',capture), \
                patch.object(ParticleSystem,'update',side_effect=AssertionError('legacy engine entered')):
            result=render(self.project,revision['id'],dict(fps=4,width=320,height=180,mode='presentation'),allow_silent=True,iterations=1)
        folder=self.project/'renders'/result['id']
        with open(folder/'frame_values.csv',encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),len(seen));self.assertEqual(len(rows),16)
        for row,frame in zip(rows,seen):
            self.assertEqual(row['motion_mode'],frame['mode'])
            self.assertEqual(json.loads(row['motion_components_json']),frame['components'])
            self.assertIn('audio_rms',row)
        self.assertTrue((folder/'motion_plan.csv').is_file())
        self.assertEqual(result['motion_engine'],'flow-v1')

    def test_explicit_null_motion_inherits_consistently_across_http_endpoints(self):
        revision=save_revision(self.project,'r000000',[],motion=self.config)
        with patch('studio.jobs.threading.Thread'):
            server=make_server(self.root,0,start_queue=True)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.manager.close);self.addCleanup(server.shutdown)
        base=f'http://127.0.0.1:{server.server_port}'
        with urllib.request.urlopen(base+'/api/bootstrap') as result:token=json.load(result)['token']
        def post(suffix,body):
            req=urllib.request.Request(base+'/api/projects/'+self.project.name+'/'+suffix,
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Studio-Token':token})
            with urllib.request.urlopen(req) as result:return json.load(result)
        body={'base_revision':revision['id'],'edits':[],'motion':None}
        self.assertEqual(post('motion-plan',body)['config'],revision['motion'])
        self.assertEqual(post('preview',body)['motion']['config'],revision['motion'])
        job=post('previews',body)
        from studio.store import read_json
        frozen=read_json(self.project/'previews'/job['payload']['snapshot_id']/'snapshot.json')
        self.assertEqual(frozen['motion'],revision['motion'])
        self.assertEqual(post('revisions',body)['motion'],revision['motion'])

    def test_pipeline_flow_checkpoint_restores_clip_pixels_and_frame_clock(self):
        from renderer import VideoRenderer
        project=create_demo(self.root,10.6)
        revision=save_revision(project,'r000000',[],motion={'engine':'flow-v1'})
        original=VideoRenderer.render_frame
        full_frames=[];clip_frames=[];target=full_frames
        def capture(renderer,*args,**kwargs):
            frame=original(renderer,*args,**kwargs);target.append(frame.copy());return frame
        options={'width':320,'height':180,'fps':2,'mode':'analysis'}
        with patch('feature_extraction.load_audio',side_effect=lambda path:sf.read(path,dtype='float32')), \
                patch.object(VideoRenderer,'render_frame',capture):
            full=render(project,revision['id'],options,allow_silent=True,iterations=1)
            target=clip_frames
            clip=render(project,revision['id'],{**options,'start':10},allow_silent=True,iterations=1)
        self.assertEqual(clip['resumed_from_frame'],20)
        self.assertEqual(len(clip_frames),len(full_frames)-20)
        for a,b in zip(clip_frames,full_frames[20:]):np.testing.assert_array_equal(a,b)
        with open(project/'renders'/clip['id']/'frame_values.csv',encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(rows[0]['frame_index'],'20');self.assertEqual(rows[0]['time_us'],'10000000')

    def test_frontend_backend_manual_math_known_vectors(self):
        import subprocess
        from motion_schema import mode_params
        from studio.motion import apply_overrides
        base={'times_us':[0,2000000,4000000],'auto':[
            dict(mode=mode,source='auto',reason=mode,components=[dict(mode=mode,weight=1.,params=mode_params(mode))])
            for mode in ('rise','orbit','fall')]}
        script="const m=require('./web/motion-math.js');let x=JSON.parse(process.argv[1]);process.stdout.write(JSON.stringify(m.applyOverrides(x.base,x.config,4000000)))"
        completed=subprocess.run(['node','-e',script,json.dumps({'base':base,'config':self.config})],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,encoding='utf-8',check=True,timeout=10)
        js=json.loads(completed.stdout);py=apply_overrides(base,self.config,4000000)
        self.assertEqual(js['times_us'],py['times_us'])
        self.assertEqual(js['config'],py['config'])
        for a,b in zip(js['effective'],py['effective']):
            self.assertEqual(a['mode'],b['mode']);self.assertEqual(a['source'],b['source'])
            for c,d in zip(a['components'],b['components']):
                self.assertEqual(c['mode'],d['mode']);self.assertAlmostEqual(c['weight'],d['weight'],places=12)
                for key in c['params']:self.assertAlmostEqual(c['params'][key],d['params'][key],places=12)


if __name__=='__main__':unittest.main()
