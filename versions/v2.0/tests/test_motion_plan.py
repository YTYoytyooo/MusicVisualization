import copy
import unittest
import numpy as np
from motion_schema import validate_config, mode_params
from studio.motion import (automatic_plan, apply_overrides, sample_frame, blend_frames,
                           override_weight)


class MotionPlanTests(unittest.TestCase):
    def test_legacy_default_and_input_immutability(self):
        value={'engine':'flow-v1','overrides':[]}
        original=copy.deepcopy(value)
        result=validate_config(value,10000000)
        self.assertEqual(validate_config()['engine'],'legacy')
        self.assertEqual(result['min_hold_seconds'],6)
        self.assertEqual(value,original)

    def test_bad_bounds_and_overlapping_motion_rejected(self):
        for config in ({'engine':'bogus'}, {'sensitivity':float('nan')},
                       {'overrides':[self.override(params={'trail_seconds':2.1})]},
                       {'overrides':[self.override(params={'rotation':0})]},
                       {'overrides':[self.override(),self.override(id='b')]}):
            with self.subTest(config=config),self.assertRaises(ValueError):
                validate_config(config,12000000)

    @staticmethod
    def override(**kwargs):
        return dict({'id':'m-one','mode':'orbit','start_us':4000000,'end_us':6000000,
                     'transition_in_us':2000000,'transition_out_us':2000000},**kwargs)

    @staticmethod
    def frame(mode):
        return {'mode':mode,'source':'auto','reason':mode,'components':[
            {'mode':mode,'weight':1.,'params':mode_params(mode)}]}

    def test_manual_target_and_return_to_current_auto_not_old_auto(self):
        plan={'times_us':[0,5000000,10000000],
              'auto':[self.frame('rise'),self.frame('rise'),self.frame('fall')]}
        config=validate_config({'engine':'flow-v1','overrides':[self.override()]},10000000)
        before=copy.deepcopy(plan)
        result=apply_overrides(plan,config,10000000)
        at_target=sample_frame(result,5000000)
        self.assertEqual(at_target['mode'],'orbit')
        self.assertEqual(at_target['source'],'manual')
        self.assertEqual(at_target['components'][0]['weight'],1)
        after=sample_frame(result,9000000)
        expected=sample_frame(plan,9000000,'auto')
        self.assertEqual([c['mode'] for c in after['components']],[c['mode'] for c in expected['components']])
        for actual,reference in zip(after['components'],expected['components']):
            self.assertAlmostEqual(actual['weight'],reference['weight'])
            for key in actual['params']:
                self.assertAlmostEqual(actual['params'][key],reference['params'][key])
        self.assertEqual(plan,before)

    def test_short_arc_and_normalized_weights(self):
        a=self.frame('meteor');b=self.frame('meteor')
        a['components'][0]['params']['direction_deg']=350
        b['components'][0]['params']['direction_deg']=10
        middle=blend_frames(a,b,.5)
        self.assertAlmostEqual(middle['components'][0]['params']['direction_deg'],0)
        self.assertAlmostEqual(sum(x['weight'] for x in middle['components']),1)

    def test_transition_is_zero_at_support_and_exact_in_interval(self):
        edit=validate_config({'overrides':[self.override()]},10000000)['overrides'][0]
        self.assertEqual(override_weight(edit,2000000),0)
        self.assertEqual(override_weight(edit,4000000),1)
        self.assertEqual(override_weight(edit,6000000),1)
        self.assertEqual(override_weight(edit,8000000),0)
        self.assertAlmostEqual(override_weight(edit,3000000),.5)

    def test_automatic_stable_input_stable_mode_and_silence_stops(self):
        times=np.arange(0,20000001,100000,dtype=np.int64)
        values=np.tile([.4,-.6,0,0,0],(len(times),1))
        features={'times_us':times,'activity':np.full(len(times),.3),
                  'rms':np.full(len(times),.03),'pulse_strength':np.zeros(len(times))}
        config=validate_config({'engine':'flow-v1'})
        frames=automatic_plan(times,values,features,config,42,[])
        self.assertEqual(len({f['mode'] for f in frames}),1)
        self.assertEqual(frames,automatic_plan(times,values,features,config,42,[]))
        features['rms']=np.zeros(len(times));features['activity']=np.zeros(len(times))
        silent=automatic_plan(times,values,features,config,42,[])
        self.assertTrue(all(c['params']['speed']==0 for f in silent for c in f['components']))

    def test_emotion_change_changes_form_and_respects_hold(self):
        times=np.arange(0,30000001,100000,dtype=np.int64)
        values=np.tile([.4,-.6,0,0,0],(len(times),1));values[120:,:2]=[-.6,.9]
        features={'times_us':times,'activity':np.full(len(times),.5),
                  'rms':np.full(len(times),.08),'pulse_strength':np.zeros(len(times))}
        frames=automatic_plan(times,values,features,validate_config({'engine':'flow-v1'}),42,[])
        self.assertNotEqual(frames[0]['mode'],frames[-1]['mode'])
        changes=[i for i in range(1,len(frames)) if frames[i]['mode']!=frames[i-1]['mode']]
        self.assertTrue(all((b-a)*.1>=4 for a,b in zip(changes,changes[1:])))

    def test_nine_artistic_regions_are_reachable_without_manual_overrides(self):
        # Explicit art-direction acceptance vectors, not learned class labels.
        regions={'rise':(.65,.35),'fall':(-.65,-.6),'orbit':(.35,-.65),
                 'spiral':(.05,.45),'expand':(.65,.85),'gather':(-.45,.05),
                 'meteor':(.35,.65),'wave':(-.05,-.2),'turbulent':(-.6,.75)}
        # Allow the declared six-second hold plus two-second transition; pulse
        # smoothing initially makes the meteor region favor a nearby spiral.
        times=np.arange(0,10000001,100000,dtype=np.int64)
        features={'times_us':times,'activity':np.full(len(times),.8),
                  'rms':np.full(len(times),.1),'pulse_strength':np.full(len(times),.6)}
        for mode,(v,a) in regions.items():
            with self.subTest(mode=mode):
                values=np.tile([v,a,0,0,0],(len(times),1))
                frames=automatic_plan(times,values,features,validate_config({'engine':'flow-v1'}))
                self.assertEqual(frames[-1]['mode'],mode)
                self.assertTrue(all(frame['source']=='auto' for frame in frames))


if __name__=='__main__':unittest.main()
