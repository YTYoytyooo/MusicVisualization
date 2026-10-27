"""Scoring and recovery boundaries for the frozen descriptive comparison."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.train_next_v2.common import BASE
from tools.train_next_v2.cross_version_eval.methods import (
    align_labels, song_scores, recording_groups, summarize, SYSTEMS)
from tools.train_next_v2.cross_version_eval.storage import ArtifactStore


class ScoringTests(unittest.TestCase):
    def test_exact_join_axis_masks_and_equal_terminal_exclusion(self):
        t = np.arange(5) * .5
        lt = np.array([.5, 1., 2.])
        y = np.array([[.1, .2], [.3, .4], [.5, .6]], np.float32)
        m = np.array([[True, False], [False, True], [True, True]])
        a = align_labels(t, np.arange(20) * .1, lt, y, m, 2.1)
        np.testing.assert_array_equal(a['common_mask'].sum(0), [1, 1])
        np.testing.assert_array_equal(a['excluded_values'], [1, 1])
        np.testing.assert_array_equal(a['original_mask'].sum(0), [2, 2])
        self.assertEqual(a['v2_indices'][-1], -1)
        self.assertEqual(a['native_seconds'], .5)

    def test_reject_label_shift_stretch_and_invalid_values(self):
        t = np.arange(6) * .5
        for lt in (np.array([.1, .6]), np.array([0., .6]), np.array([.5, .5])):
            with self.assertRaises(ValueError):
                align_labels(t, np.arange(26)*.1, lt, np.zeros((2,2),np.float32), np.ones((2,2),bool), 2.6)
        with self.assertRaises(ValueError):
            align_labels(t, np.arange(26)*.1, np.array([0., .5]), np.full((2,2), 2.,np.float32), np.ones((2,2),bool), 2.6)

    def test_fully_missing_rows_do_not_create_labels_or_change_native_cadence(self):
        m=np.zeros((4,2),bool); m[[0,1],0]=True
        y=np.zeros((4,2),np.float32)
        a=align_labels(np.arange(5)*.5,np.arange(20)*.1,np.arange(4,dtype=float),y,m,2.1)
        self.assertEqual(a['native_seconds'],1.)
        np.testing.assert_array_equal(a['common_mask'].sum(0),[2,0])
        self.assertFalse(a['original_mask'][1].any())

    def test_delay_sign_uses_same_central_points_without_shifting_truth(self):
        from tools.train_next_v2.cross_version_eval.methods import lag_diagnostic
        t=np.arange(201)*.5
        truth=np.sin(t*.73)*.4 + np.sin(t*.29)*.2
        pred=np.sin((t-2.)*.73)*.4 + np.sin((t-2.)*.29)*.2
        frozen=truth.copy()
        result=lag_diagnostic(t,truth,pred,np.ones(len(t),bool),.5)
        self.assertEqual(result['best_seconds'],2.)
        self.assertTrue(result['credible'])
        np.testing.assert_array_equal(truth,frozen)

    def test_derivative_respects_one_second_cadence_and_missing_interval(self):
        t = np.arange(9)*.5
        y = np.zeros((9,2), np.float32)
        m = np.zeros((9,2),bool); m[[0,2,6,8],0] = True
        pred = np.zeros((len(SYSTEMS),9,2))
        pred[:,[0,2,6,8],0] = [0., .1, .8, .9]
        result = song_scores(t,y,m,pred,1.)
        self.assertAlmostEqual(result['v2']['valence']['derivative_mae'], .1)
        self.assertIsNone(result['v2']['arousal'])
        self.assertIsNone(result['v2']['valence']['ccc'])
        self.assertAlmostEqual(result['v2']['valence']['mae'], .45)

    def test_group_union_is_transitive_across_domains(self):
        rows=[dict(song_id=str(i), group_id=g, audio_sha256=a, source_dataset=d)
              for i,g,a,d in [(0,'g1','a1','deam'),(1,'g1','a2','deam'),
                              (2,'g2','a2','pmemo'),(3,'g3','a3','memo2496')]]
        ids, info = recording_groups(rows)
        self.assertEqual(ids.tolist(), [0,0,0,1])
        self.assertTrue(info['cross_dataset_components'])
        self.assertEqual(info['groups'],2)

    def test_paired_bootstrap_shared_groups_and_seed_mean_not_ensemble(self):
        rows=[]
        for d in ('deam','memo2496','pmemo'):
            for i in range(3):
                scores={s:{a:{'ccc': .1*i + .01*j, 'mae': .3} for a in ('valence','arousal')}
                        for j,s in enumerate(SYSTEMS)}
                rows.append(dict(song_id=f'{d}{i}',source_dataset=d,group_id=f'{d}{i}',
                                 audio_sha256=f'a{d}{i}',scores=scores))
        out=summarize(rows,dict(replicates=100,seed=42,confidence=.95),metrics=['ccc','mae'])
        c=out['paired_differences']['b01_mean_minus_v2']['overall']['ccc']
        self.assertAlmostEqual(c['point'], .03)
        self.assertAlmostEqual(c['lower'], .03)
        self.assertAlmostEqual(c['upper'], .03)
        self.assertAlmostEqual(out['systems']['b01_mean']['overall']['ccc'],.13)
        self.assertAlmostEqual(out['systems']['b01_ensemble']['overall']['ccc'],.15)
        self.assertAlmostEqual(out['b01_seed_sample_std']['overall']['ccc'],.01)

    def test_macro_weights_domains_equally_not_song_counts(self):
        rows=[]
        for domain,n,v in [('deam',1,.9),('pmemo',3,0.),('memo2496',2,0.)]:
            for i in range(n):
                scores={s:{a:{'ccc':v} for a in ('valence','arousal')} for s in SYSTEMS}
                rows.append(dict(song_id=f'{domain}{i}',group_id=f'{domain}{i}',audio_sha256=f'{domain}{i}',source_dataset=domain,scores=scores))
        out=summarize(rows,dict(replicates=20,seed=4,confidence=.95),metrics=['ccc'])
        self.assertAlmostEqual(out['systems']['v2']['overall']['ccc'],.3)


class StorageTests(unittest.TestCase):
    def test_receipt_hash_and_orphan_are_fail_closed(self):
        folder=BASE/'tmp'; folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            store=ArtifactStore(Path(temporary),'fixed-contract')
            self.assertIsNone(store.read('first','song1'))
            store.save('first','song1',dict(x=np.arange(4)),dict(purpose='test'))
            receipt, arrays=store.read('first','song1')
            np.testing.assert_array_equal(arrays['x'],np.arange(4))
            with self.assertRaises(ValueError): store.read('first','other-song')
            with self.assertRaises(ValueError): ArtifactStore(Path(temporary),'changed-contract').read('first','song1')
            (Path(temporary)/'first.npz').write_bytes(b'broken')
            with self.assertRaises(ValueError): store.read('first','song1')
            (Path(temporary)/'orphan.npz').write_bytes(b'incomplete')
            with self.assertRaises(ValueError): store.read('orphan','song2')


if __name__=='__main__': unittest.main()
