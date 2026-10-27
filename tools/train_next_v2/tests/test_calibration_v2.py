"""Scientific boundary tests before opening the untouched calibration labels."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.train_next_v2.calibration import job_v2 as job
from tools.train_next_v2.calibration.methods import (
    calibration_split, finite_quantile, axis_evidence, fit_intervals,
    audit_intervals, readiness, wilson_interval)


def identity(song_id, group=None, audio=None, domain='deam'):
    return dict(song_id=song_id,group_id=group or song_id,audio_sha256=audio or song_id,
        source_dataset=domain,split='train',labels_path=song_id+'.npz')


def evidence(group, score, role='interval_fit', error=None):
    error=np.array([score] if error is None else error,dtype=float)
    return dict(connected_group=group,role=role,group_score=score,constant_score=score,
        error=error,scale=np.ones_like(error),disagreement=np.zeros_like(error))


class CalibrationTests(unittest.TestCase):
    def test_calibration_only_input_contract_requires_every_frozen_identity(self):
        split=[identity('a',audio='audio-a'),identity('b',audio='audio-b')]
        contract=dict(labels_sha256={'a':'1'*64,'b':'2'*64},feature_hashes={'audio-a':'3'*64,'audio-b':'4'*64})
        job.require_input_membership(contract,split)
        bad=deepcopy(contract);bad['feature_hashes'].pop('audio-b')
        with self.assertRaises(ValueError):job.require_input_membership(bad,split)
        bad=deepcopy(contract);bad['labels_sha256']['fit-song']='5'*64
        with self.assertRaises(ValueError):job.require_input_membership(bad,split)
        bad=deepcopy(contract);bad['labels_sha256']['a']='missing'
        with self.assertRaises(ValueError):job.require_input_membership(bad,split)

    def test_both_historical_verification_schemas_require_exact_hash(self):
        digest='a'*64
        self.assertEqual(job.verified_checkpoint_hash({'checkpoint_sha256':digest}),digest)
        self.assertEqual(job.verified_checkpoint_hash({'best_checkpoint_sha256':digest}),digest)
        self.assertEqual(job.verified_checkpoint_hash({'checkpoint_sha256':digest,'best_checkpoint_sha256':digest}),digest)
        for record in ({},{'checkpoint_sha256':'bad'},{'checkpoint_sha256':digest,'best_checkpoint_sha256':'b'*64}):
            with self.assertRaises(ValueError):job.verified_checkpoint_hash(record)

    def test_split_is_deterministic_transitive_and_only_calibration(self):
        records=[identity('a','g','aa'),identity('b','g','bb'),identity('c','h','bb')]
        records += [identity(str(i)) for i in range(10)]
        roles={r['song_id']:'calibration' for r in records}; roles['9']='fit'
        a=calibration_split(records,roles,7)
        self.assertEqual(a,calibration_split(list(reversed(records)),roles,7))
        by_id={r['song_id']:r for r in a}
        self.assertNotIn('9',by_id)
        self.assertEqual(len({by_id[i]['connected_group'] for i in 'abc'}),1)
        self.assertEqual(len({by_id[i]['role'] for i in 'abc'}),1)
        left={r['connected_group'] for r in a if r['role']=='interval_fit'}
        right={r['connected_group'] for r in a if r['role']=='interval_audit'}
        self.assertFalse(left&right)

    def test_original_cross_role_or_cross_domain_identity_is_rejected(self):
        rows=[identity('a','g'),identity('b','g')]
        with self.assertRaisesRegex(ValueError,'crosses'):
            calibration_split(rows,{'a':'fit','b':'calibration'},7)
        rows[1]['source_dataset']='pmemo'
        with self.assertRaisesRegex(ValueError,'Cross-domain'):
            calibration_split(rows,{'a':'calibration','b':'calibration'},7)
        rows[1]['split']='test'
        with self.assertRaisesRegex(ValueError,'original-training'):
            calibration_split(rows,{'a':'calibration','b':'calibration'},7)

    def test_finite_sample_rank_and_small_sample_infinity(self):
        q=finite_quantile(np.arange(9),.9)
        self.assertEqual((q['rank'],q['q'],q['groups']),(9,8.,9))
        q=finite_quantile(np.arange(8),.9)
        self.assertFalse(q['finite']);self.assertIsNone(q['q'])
        self.assertEqual(finite_quantile(np.arange(19),.9)['q'],17.)
        for bad in ([],[np.nan],[np.inf],[-1]):
            with self.assertRaises(ValueError):finite_quantile(bad,.9)

    def test_axis_masks_and_disagreement_are_independent(self):
        p=np.array([[[-.1,0],[0,.1],[0,.2]],[[0,0],[0,.2],[0,.4]],[[.1,0],[0,.3],[0,.6]]])
        y=np.array([[.2,np.nan],[np.nan,.5],[np.nan,.6]])
        mask=np.isfinite(y)
        v=axis_evidence(p,y,mask,0,.02)
        self.assertEqual(v['error'].shape,(1,));self.assertAlmostEqual(v['group_score'],2.)
        a=axis_evidence(p,y,mask,1,.02)
        np.testing.assert_allclose(a['error'],[.3,.2]);np.testing.assert_allclose(a['scale'],[.1,.2])
        self.assertIsNone(axis_evidence(p,y,np.zeros_like(mask),0,.02))
        with self.assertRaises(ValueError):axis_evidence(p[:2],y,mask,0,.02)
        with self.assertRaises(ValueError):axis_evidence(p,y,mask.astype(float),0,.02)
        p[0,0,0]=np.nan
        with self.assertRaises(ValueError):axis_evidence(p,y,mask,0,.02)

    def test_zero_disagreement_has_positive_scale_floor(self):
        v=axis_evidence(np.zeros((3,2,2)),np.full((2,2),.1),np.ones((2,2),bool),0,.02)
        np.testing.assert_array_equal(v['scale'],[.02,.02]);self.assertAlmostEqual(v['group_score'],5.)

    def test_quantile_counts_connected_groups_not_songs(self):
        rows=[evidence(str(i),i+1.) for i in range(9)]
        rows.append(evidence('0',100.))
        fitted=fit_intervals(rows,.9)
        self.assertEqual(fitted['scaled']['groups'],9)
        self.assertEqual(fitted['scaled']['q'],100.)

    def test_audit_cannot_fit_or_select_primary(self):
        with self.assertRaisesRegex(ValueError,'interval_fit'):
            fit_intervals([evidence('a',1.,'interval_audit')],.9)
        with self.assertRaisesRegex(ValueError,'interval_audit'):
            audit_intervals([evidence('a',1.)],dict(finite=True,q=1),'scaled')
        fit=fit_intervals([evidence(str(i),.1) for i in range(9)],.9)
        before=deepcopy(fit)
        audit_intervals([evidence('x',10.,'interval_audit')],fit['scaled'],'scaled')
        self.assertEqual(before,fit)

    def test_coverage_is_group_simultaneous_with_unclipped_width(self):
        rows=[evidence('g',0.,'interval_audit',[.1,.2]),
              evidence('g',0.,'interval_audit',[.4]),
              evidence('h',0.,'interval_audit',[.1])]
        audit=audit_intervals(rows,dict(finite=True,q=.2),'scaled')
        self.assertEqual(audit['group_simultaneous_coverage'],.5)
        self.assertEqual(audit['point_coverage'],.75)
        wide=audit_intervals(rows,dict(finite=True,q=3.),'constant')
        self.assertEqual(wide['halfwidth_p90'],3.)
        self.assertEqual(audit_intervals(rows,dict(finite=False,q=None),'scaled')['finite'],False)

    def test_readiness_does_not_hide_single_failed_width_or_coverage(self):
        gates=dict(minimum_fit_groups=9,minimum_audit_groups=9,minimum_audit_group_coverage=.9,
            maximum_song_macro_mae=.15,maximum_point_p90_halfwidth=.25)
        fit=dict(groups=9,finite=True,q=1)
        audit=dict(groups=10,finite=True,group_simultaneous_coverage=.9,halfwidth_p90=.25)
        self.assertTrue(readiness(fit,audit,.15,gates)['passes'])
        self.assertFalse(readiness(fit,{**audit,'halfwidth_p90':.251},.15,gates)['passes'])
        self.assertFalse(readiness(fit,{**audit,'group_simultaneous_coverage':.899},.15,gates)['passes'])
        self.assertFalse(readiness(fit,audit,np.nan,gates)['passes'])
        self.assertFalse(readiness({**fit,'finite':False},audit,.15,gates)['passes'])

    def test_receipts_reject_wrong_protocol_clock_or_corrupted_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=job.BASE/'tmp') as tmp:
            root=Path(tmp);(root/'predictions').mkdir()
            song=dict(song_id='cal-song',audio_sha256='audio',times=np.array([0.,.5]))
            with patch.object(job,'DIRECTORY',root):
                path,receipt=job.prediction_paths(song['song_id'])
                np.savez(path,members=np.zeros((3,2,2)),times=song['times'])
                record=dict(song_id=song['song_id'],audio_sha256='audio',protocol_sha256='p',sha256=job.sha256(path))
                job.atomic_json(receipt,record)
                self.assertEqual(job.read_prediction(song,'p').shape,(3,2,2))
                with self.assertRaises(ValueError):job.read_prediction(song,'wrong')
                with self.assertRaises(AssertionError):job.read_prediction({**song,'times':np.array([1.,1.5])},'p')
                with path.open('ab') as stream:stream.write(b'bad')
                with self.assertRaises(ValueError):job.read_prediction(song,'p')

    def test_group_bootstrap_preserves_song_weighting(self):
        rows=[]
        metrics={'ccc':.2,'mae':.1,'rmse':.12,'mean_absolute_bias':.02,'centered_rmse':.05,'derivative_mae':.04,'dynamics_beats_constant':1.}
        for domain in ('deam','memo2496','pmemo'):
            for i in range(4):
                rows.append(dict(source_dataset=domain,connected_group=f'{domain}-{i//2}',role='interval_audit',
                    ensemble={a:dict(metrics) for a in ('valence','arousal')}))
        result=job.bootstrap_audit(rows,dict(seed=2,replicates=2000))
        for d in result.values():
            for axis in d.values():
                np.testing.assert_allclose(axis['ccc']['ci95'],[.2,.2])
                self.assertEqual(axis['ccc']['groups'],2)

    def test_wilson_retains_small_sample_uncertainty(self):
        lo,hi=wilson_interval(15,15)
        self.assertLess(lo,.9);self.assertAlmostEqual(hi,1.)
        with self.assertRaises(ValueError):wilson_interval(0,0)


if __name__=='__main__':unittest.main()
