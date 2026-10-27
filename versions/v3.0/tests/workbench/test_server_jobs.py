"""Real loopback API and durable queue registration contracts."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave
import numpy as np
from tools.train_next_v2.v3_workbench.studio import store, jobs, server
from tools.train_next_v2.v3_workbench.studio.paths import WORKSPACE, PYTHON


class ServerJobTests(unittest.TestCase):
    def setUp(self):
        base = WORKSPACE / 'data/v3.0/test-workbench'
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        audio = self.root / 'tone.wav'
        with wave.open(str(audio), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000)
            f.writeframes(np.zeros(8000, dtype='<i2').tobytes())
        self.project = store.create_project(self.root/'works', 'contract', audio, None, None,
            {'duration':1., 'tempo':120., 'beat_times':[0., .5]}, systems={
                'b01': {'values':[[.2,.3],[.4,.5]], 'times_us':[0,500000], 'step_us':500000,
                        'model_sha256':'one'},
                'v2': {'values':np.zeros((10,2)), 'times_us':np.arange(10)*100000,
                       'step_us':100000, 'model_sha256':'two'}}, source_system='b01',
            provenance={'kind':'synthetic'})
        self.http = server.make_server(self.root/'works', 0, start_queue=False)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_http)
        self.token = self.request('/api/bootstrap')[1]['token']

    def close_http(self):
        self.http.shutdown(); self.http.server_close(); self.thread.join()

    def request(self, route, body=None, method=None, headers=None):
        c = http.client.HTTPConnection('127.0.0.1', self.http.server_port, timeout=10)
        try:
            h = dict(headers or {})
            if body is not None:
                h.setdefault('Content-Type','application/json')
                h.setdefault('X-Studio-Token',getattr(self,'token',''))
            c.request(method or ('POST' if body is not None else 'GET'), route,
                      json.dumps(body) if body is not None else None, h)
            r = c.getresponse(); data = r.read()
            return r.status, json.loads(data) if data and 'application/json' in r.getheader('Content-Type','') else data, dict(r.getheaders())
        finally:
            c.close()

    def test_driver_fetch_does_not_change_current_revision(self):
        current = store.load_revision(self.project)
        code, body, _ = self.request('/api/projects/'+self.project.name+'?source_system=v2')
        self.assertEqual(code,200)
        self.assertEqual(body['timeline']['source_system'],'v2')
        self.assertEqual(body['timeline']['step_us'],100000)
        self.assertEqual(len(body['timeline']['source_raw']),10)
        self.assertEqual(body['revision']['source_system'],'b01')
        self.assertEqual(store.load_revision(self.project),current)

    def test_driver_save_is_new_revision_and_stale_write_rejected(self):
        route='/api/projects/'+self.project.name+'/revisions'
        base=store.load_revision(self.project)['id']
        code, result, _=self.request(route,{'base_revision':base,'edits':[],'source_system':'v2'})
        self.assertEqual(code,201,result)
        self.assertEqual(store.load_revision(self.project)['source_system'],'v2')
        self.assertEqual(self.request(route,{'base_revision':base,'edits':[]})[0],400)
        self.assertEqual(store.load_revision(self.project,base)['source_system'],'b01')

    def test_head_range_and_request_authority(self):
        route='/media/'+self.project.name+'/audio'
        code, body, headers=self.request(route,method='HEAD',headers={'Range':'bytes=0-15'})
        self.assertEqual((code,body,headers['Content-Length']),(206,b'', '16'))
        self.assertEqual(len(self.request(route,headers={'Range':'bytes=0-15'})[1]),16)
        self.assertEqual(self.request('/api/bootstrap',headers={'Host':'outside.invalid'})[0],403)
        self.assertEqual(self.request('/api/projects/'+self.project.name+'/preview',
            {'edits':[]},headers={'Origin':'https://outside.invalid'})[0],403)
        self.assertEqual(self.request('/api/projects/..')[0],400)

    def test_reset_driver_does_not_apply_incompatible_old_offsets(self):
        route='/api/projects/'+self.project.name
        base=store.load_revision(self.project)['id']
        edits=[dict(id='offset',layer='emotion',field='valence',operation='interval_offset',
                    start_us=0,end_us=900000,value=.9,transition_in_us=0,transition_out_us=0)]
        code,saved,_=self.request(route+'/revisions',{'base_revision':base,'source_system':'v2','edits':edits})
        self.assertEqual(code,201,saved)
        body={'base_revision':saved['id'],'source_system':'b01','edits':edits}
        self.assertEqual(self.request(route+'/preview',body)[0],400)
        body['edits']=[]
        self.assertEqual(self.request(route+'/preview',body)[0],200)
        self.assertEqual(self.request(route+'/revisions',body)[0],201)
        self.assertEqual(store.load_revision(self.project)['source_system'],'b01')

    def test_state_boundary_module_is_served_with_ui(self):
        code,body,headers=self.request('/workbench-state.js')
        self.assertEqual(code,200)
        self.assertIn(b'prepareSwitch',body)
        self.assertIn('javascript',headers['Content-Type'])

    def test_audio_geometry_counts_pcm_eof_instead_of_container_estimate(self):
        import soundfile as sf
        original = sf.SoundFile
        class EstimatedHeader:
            frames=9_999_999
            def __init__(self,*args,**kwargs):self.stream=original(*args,**kwargs);self.samplerate=self.stream.samplerate
            def __enter__(self):return self
            def __exit__(self,*args):self.stream.close()
            def read(self,*args,**kwargs):return self.stream.read(*args,**kwargs)
        audio=self.project/store.load_project(self.project)['audio_file']
        with patch.object(sf,'SoundFile',EstimatedHeader):
            p=store.create_project(self.root/'works','estimated header',audio,None,np.zeros((10,2)),
                                   {'duration':1.},provenance={'kind':'synthetic'})
        self.assertEqual(store.load_project(p)['duration_us'],1_000_000)

    def test_original_csv_uses_requested_native_clock_without_revision_mutation(self):
        code,body,_=self.request('/api/projects/'+self.project.name+'/raw?source_system=v2')
        self.assertEqual(code,200)
        rows=body.decode('utf-8-sig').splitlines()
        self.assertEqual(len(rows),11)
        self.assertTrue(rows[2].startswith('0.1,'))
        self.assertEqual(store.load_revision(self.project)['source_system'],'b01')

    def manager(self):
        with patch.object(jobs.threading,'Thread'):
            m=jobs.JobManager(self.root/'works')
        self.addCleanup(m.close)
        return m

    def test_registered_command_inputs_cancel_retry_and_changed_input_rejected(self):
        m=self.manager()
        payload={'project_id':self.project.name,'revision':store.load_revision(self.project)['id'],'options':{}}
        item=m.submit('render',payload)
        launch=item['launch']
        self.assertEqual(launch['original_argv'][0],str(PYTHON))
        self.assertEqual(launch['cwd'],str(WORKSPACE))
        self.assertTrue(launch['input_hashes']); self.assertTrue(launch['code_hashes'])
        jobs.verify_launch(item)
        self.assertEqual(m.cancel(item['id'])['status'],'cancelled')
        retried=m.retry(item['id'])
        self.assertEqual(retried['payload'],payload)
        self.assertEqual(retried['parent_job_id'],item['id'])
        m.cancel(retried['id'])
        audio=self.project/store.load_project(self.project)['audio_file']
        with audio.open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'文件已变化'):
            m.retry(retried['id'])

    def test_shared_plans_respect_consumer_order(self):
        m=self.manager()
        payload={'project_id':self.project.name,'base_revision':store.load_revision(self.project)['id'],
                 'edits':[],'source_system':'b01'}
        first=m.request_plan(payload,'same','page-a','temporary',2)
        same=m.request_plan(payload,'same','page-b','temporary',1)
        self.assertEqual(first['id'],same['id'])
        stale=m.request_plan(payload,'old','page-a','temporary',1)
        self.assertEqual(stale['status'],'superseded')
        m.release_plan('page-a',3)
        self.assertEqual(store.read_json(m.path(first['id']))['status'],'queued')
        m.release_plan('page-b',2)
        self.assertEqual(store.read_json(m.path(first['id']))['status'],'cancelled')

    def test_pinned_job_allows_later_revision_but_rejects_configuration_or_metadata_changes(self):
        m=self.manager()
        original=store.load_revision(self.project)
        item=m.submit('render',{'project_id':self.project.name,'revision':original['id'],'options':{}})
        store.save_revision(self.project,original['id'],[],source_system='v2')
        jobs.verify_launch(item)
        item['payload']['options']['fps']=60
        with self.assertRaisesRegex(ValueError,'配置已变化'):
            jobs.verify_launch(item)
        item['payload']['options'].pop('fps')
        metadata=store.read_json(self.project/'project.json')
        metadata['seed']+=1
        store.atomic_json(self.project/'project.json',metadata)
        with self.assertRaisesRegex(ValueError,'元数据已变化'):
            jobs.verify_launch(item)

    def test_pid_identity_recovery_does_not_overwrite_final_success(self):
        m=self.manager()
        item=m.submit('render',{'project_id':self.project.name,'revision':store.load_revision(self.project)['id']})
        stale={**item,'status':'running','pid':99999999,'process_identity':'wrong'}
        store.atomic_json(m.path(item['id']),{**item,'status':'succeeded','result':{'kept':True}})
        recovered=jobs.recover_interrupted(stale,m.path(item['id']))
        self.assertEqual(recovered['status'],'succeeded')
        self.assertTrue(recovered['result']['kept'])


if __name__ == '__main__':
    unittest.main()
