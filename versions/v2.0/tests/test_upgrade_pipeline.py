import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
from studio.demo import create_demo
from studio.store import load_project, load_raw, save_revision, digest
from studio.smoothing import smooth_source, baseline_at, emotion_values, validate_smoothing
from studio.pipeline import (timeline_payload, planning_key, create_preview_snapshot,
                             render_preview, render)
from studio.fingerprints import stage_fingerprint


class UpgradePipelineTests(unittest.TestCase):
    def test_smoothing_known_answer_and_source_immutable(self):
        raw=np.zeros((5,5));raw[2,0]=1;raw[:,2]=.7
        original=raw.copy()
        config={'enabled':True,'window_seconds':.3}
        values=smooth_source(raw,100000,config)
        np.testing.assert_allclose(values[:,0],[0,1/3,1/3,1/3,0])
        self.assertAlmostEqual(baseline_at(raw,[50000],100000,config)[0,0],1/6)
        np.testing.assert_array_equal(raw,original)
        np.testing.assert_array_equal(values[:,2],raw[:,2])
        np.testing.assert_array_equal(baseline_at(raw,[150000]),raw[[1]])

    def test_hard_target_after_smoothing_and_disabled_compatibility(self):
        from studio.timeline import validate_edits, emotion_at
        raw=np.zeros((10,5));raw[3,1]=.8
        edit=validate_edits([dict(id='a',layer='emotion',field='arousal',operation='point_target',
            time_us=450000,value=.9,transition_in_us=200000,transition_out_us=200000,
            interpolation='smootherstep')],1000000)
        values=emotion_values(raw,[0,450000,900000],edit,100000,{'enabled':True})
        self.assertEqual(values[1,1],.9)
        np.testing.assert_array_equal(emotion_values(raw,[0,350000,450000],edit),emotion_at(raw,[0,350000,450000],edit))
        for config in ({'enabled':'true'},{'window_seconds':float('nan')},{'window_seconds':10},{'unknown':0}):
            with self.assertRaises(ValueError):validate_smoothing(config)

    def test_smoothing_javascript_python_parity(self):
        from studio.timeline import validate_edits
        raw=np.array([[np.sin(i)*.8,np.cos(i)*.6,0,0,0] for i in range(20)])
        edits=validate_edits([dict(id='a',layer='emotion',field='arousal',operation='point_target',
            time_us=1234567,value=.8,transition_in_us=200000,transition_out_us=300000,
            interpolation='smootherstep')],2000000)
        config={'enabled':True,'window_seconds':.5}
        script="const M=require('./web/timeline-math.js');let x=JSON.parse(process.argv[1]);process.stdout.write(JSON.stringify(M.emotionPreview(x.raw,100000,2000000,x.edits,x.config)));"
        process=subprocess.run(['node','-e',script,json.dumps({'raw':raw.tolist(),'edits':edits,'config':config})],
            cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True,timeout=10)
        result=json.loads(process.stdout)
        np.testing.assert_allclose(result['effective'],emotion_values(raw,result['times_us'],edits,100000,config),atol=1e-12)

    def test_revision_smoothing_preserved_and_plan_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=create_demo(tmp,1.)
            before=digest(project/'analysis/predictions_raw.npz')
            revision=save_revision(project,'r000000',[],smoothing={'enabled':True,'window_seconds':.5})
            payload=timeline_payload(project,revision=revision['id'])
            self.assertTrue(payload['smoothing']['enabled'])
            self.assertNotEqual(planning_key(project,[]),planning_key(project,[],smoothing=revision['smoothing']))
            self.assertEqual(digest(project/'analysis/predictions_raw.npz'),before)

    def test_render_changes_do_not_invalidate_features_or_prediction(self):
        from studio import fingerprints
        original=fingerprints.digest
        before={stage:stage_fingerprint(stage) for stage in ('features','prediction','planning','render')}
        def changed(path):
            return 'changed-renderer' if Path(path).name=='renderer.py' else original(path)
        with patch('studio.fingerprints.digest',side_effect=changed):
            after={stage:stage_fingerprint(stage) for stage in before}
        for stage in ('features','prediction','planning'):self.assertEqual(before[stage],after[stage])
        self.assertNotEqual(before['render'],after['render'])

    def test_model_constant_change_invalidates_prediction_fingerprint(self):
        original=Path.read_text
        before=stage_fingerprint('prediction')
        def changed(path,*args,**kwargs):
            source=original(path,*args,**kwargs)
            return source.replace('WINDOW_SIZE = 10','WINDOW_SIZE = 11') if path.name=='emotion_model.py' else source
        with patch.object(Path,'read_text',changed):
            self.assertNotEqual(before,stage_fingerprint('prediction'))

    def test_analysis_code_change_does_not_publish_mislabeled_cache(self):
        from studio.pipeline import analyze
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audio=root/'a.wav';model=root/'model.pth'
            sf.write(audio,np.zeros(22050),22050);model.write_bytes(b'test fixture only')
            seen=[]
            def fingerprint(stage):
                seen.append(stage)
                return 'old' if len(seen)<=2 else 'changed'
            def embeddings(y,sr,cache_path,**kwargs):
                value=np.zeros((10,512));np.save(cache_path,value);return value
            with patch('studio.fingerprints.stage_fingerprint',side_effect=fingerprint), \
                 patch('studio.fingerprints.clap_snapshot',return_value={'source':'local','sha256':'fixed'}), \
                 patch('feature_extraction.load_audio',return_value=(np.zeros(22050),22050)), \
                 patch('feature_extraction.extract_global_info',return_value={'duration':1.,'tempo':120.,'beat_times':[]}), \
                 patch('emotion_model.extract_clap_embeddings',side_effect=embeddings):
                with self.assertRaisesRegex(ValueError,'代码在分析过程中变化'):
                    analyze(audio,model,root/'projects')
            self.assertEqual(list((root/'projects').glob('p-*/project.json')),[])
            self.assertEqual(list((root/'projects').glob('.cache/*/metadata.json')),[])

    def test_preview_is_immutable_independent_and_bounds_include_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=create_demo(tmp,8.)
            edit=dict(id='a',layer='emotion',field='arousal',operation='point_target',time_us=4000000,
                value=.8,transition_in_us=1000000,transition_out_us=1000000)
            snapshot=create_preview_snapshot(project,'r000000',[edit],edit_id='a')
            self.assertEqual(snapshot['options']['start'],1)
            self.assertEqual(snapshot['options']['end'],7)
            save_revision(project,'r000000',[])
            with patch('studio.pipeline.render',return_value={}) as execute:
                render_preview(project,snapshot['id'])
            frozen=execute.call_args.kwargs['_snapshot']
            self.assertEqual(frozen['base_revision'],'r000000')
            self.assertEqual(frozen['edits'][0]['value'],.8)
            self.assertEqual(load_project(project)['current_revision'],'r000001')

    def test_insufficient_disk_fails_before_render_output(self):
        from collections import namedtuple
        usage=namedtuple('usage','total used free')
        with tempfile.TemporaryDirectory() as tmp:
            project=create_demo(tmp,1.)
            with patch('studio.pipeline.shutil.disk_usage',return_value=usage(10,9,1)):
                with self.assertRaisesRegex(ValueError,'空间不足'):render(project,allow_silent=True)
            self.assertEqual(list((project/'renders').iterdir()),[])

    def test_checkpoint_segment_matches_full_replay_actual_frames(self):
        import cv2
        from renderer import VideoRenderer
        with tempfile.TemporaryDirectory() as tmp:
            project=create_demo(tmp,10.6)
            options=dict(fps=2,width=320,height=180,mode='analysis')
            full_frames,clip_frames=[],[]
            target=full_frames
            original=VideoRenderer.render_frame
            def capture(renderer,*args,**kwargs):
                result=original(renderer,*args,**kwargs)
                target.append(result.copy())
                return result
            with patch('feature_extraction.load_audio',side_effect=lambda p:sf.read(p,dtype='float32')), patch.object(VideoRenderer,'render_frame',capture):
                full=render(project,options=options,allow_silent=True,iterations=1)
                target=clip_frames
                clip=render(project,options={**options,'start':10},allow_silent=True,iterations=1)
            self.assertEqual(clip['resumed_from_frame'],20)
            self.assertEqual(len(clip_frames),len(full_frames)-20)
            for frame,reference in zip(clip_frames,full_frames[20:]):
                np.testing.assert_array_equal(frame,reference)
            a=cv2.VideoCapture(str(project/'renders'/full['id']/full['video_file']))
            b=cv2.VideoCapture(str(project/'renders'/clip['id']/clip['video_file']))
            try:
                a.set(cv2.CAP_PROP_POS_FRAMES,20)
                while True:
                    ok,frame=b.read()
                    if not ok:break
                    success,reference=a.read();self.assertTrue(success)
                    # Separate lossy/inter-frame encoded files need not decode
                    # byte-identically; exact simulation pixels are checked above.
                    self.assertEqual(frame.shape,reference.shape)
            finally:a.release();b.release()


if __name__=='__main__':unittest.main()
