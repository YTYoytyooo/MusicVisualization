"""D selection must preserve group pairing, fair budgets, and every frozen gate."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from tools.train_next_v2.analysis.d_auxiliary import (selection_gates, check_pair_fairness, compare, analyze)
from tools.train_next_v2.analysis.paired_confirmation import METRICS, weighted_macro

SELECTION=dict(minimum_mean_CCC_gain=.02, maximum_dataset_axis_CCC_drop=.01,
    maximum_dataset_axis_MAE_increase=.01, minimum_dataset_axis_CCC_CI_lower=-.02,
    require_positive_gain_in_every_seed=True)


def gate_rows():
    return [dict(dataset=d,axis=a,metrics=dict(ccc=dict(mean_delta=.03,delta_ci95=[.01,.06]),
                    mae=dict(mean_delta=-.01)))
            for d in ('deam','memo2496','pmemo') for a in ('valence','arousal')]


def fixtures(gain=.03):
    groups={}; reports=[]
    for d in ('deam','memo2496','pmemo'):
        records=[dict(song_id=f'{d}-{i}',group_id=f'{d}-group-{0 if i<2 else 1}') for i in range(3)]
        groups[d]=[records[:2],records[2:]]
    for shift in (0.,gain):
        arm=[]
        for seed_index in range(3):
            by_domain={}
            for d in groups:
                records=[r for g in groups[d] for r in g]; per_song=[]
                for i,r in enumerate(records):
                    axes={}
                    for axis in ('valence','arousal'):
                        m={name:.2+i*.1 for name in METRICS}
                        m.update(values=5,ccc=.1+i*.1+seed_index*.01+shift,truth_std=.2,truth_span90=.4)
                        axes[axis]=m
                    per_song.append(dict(song_id=r['song_id'],group_id=r['group_id'],**axes))
                macro={a:{m:float(np.mean([r[a][m] for r in per_song])) for m in METRICS}
                       for a in ('valence','arousal')}
                by_domain[d]=dict(per_song=per_song,macro=macro)
            arm.append(dict(best_development=by_domain,best_development_score=.2+seed_index*.01+shift))
        reports.append(arm)
    return reports[0],reports[1],groups


class AuxiliaryAnalysisTests(unittest.TestCase):
    def test_average_gain_cannot_hide_negative_seed_or_axis(self):
        self.assertFalse(selection_gates([.05,.05,-.005],gate_rows(),SELECTION)['positive_gain_in_every_seed'])
        rows=gate_rows();rows[4]['metrics']['ccc']['mean_delta']=-.01001
        self.assertFalse(selection_gates([.04]*3,rows,SELECTION)['all_axis_dataset_ccc_point_bounds'])
        rows=gate_rows();rows[5]['metrics']['mae']['mean_delta']=.01001
        self.assertFalse(selection_gates([.04]*3,rows,SELECTION)['all_axis_dataset_mae_point_bounds'])

    def test_ci_lower_bound_is_required_and_strict(self):
        rows=gate_rows();rows[1]['metrics']['ccc']['delta_ci95'][0]=-.02
        result=selection_gates([.04]*3,rows,SELECTION)
        self.assertFalse(result['all_axis_dataset_ccc_ci_lower_bounds'])
        rows[1]['metrics']['ccc']['delta_ci95'][0]=-.019999
        self.assertTrue(all(selection_gates([.02]*3,rows,SELECTION).values()))
        self.assertFalse(selection_gates([.01999]*3,rows,SELECTION)['mean_ccc_gain'])

    def test_missing_duplicate_or_nonfinite_evidence_rejected(self):
        with self.assertRaises(ValueError):selection_gates([.03]*2,gate_rows(),SELECTION)
        rows=gate_rows();rows[5]=copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):selection_gates([.03]*3,rows,SELECTION)
        with self.assertRaises(ValueError):selection_gates([.03,.03,float('nan')],gate_rows(),SELECTION)

    def test_sampling_whole_groups_preserves_song_macro(self):
        # One cluster has two songs; it must retain twice the song count.
        sums=np.array([[[.6],[.9]]]);counts=np.array([[[2.],[1.]]])
        weights=np.array([[1.,1.],[2.,0.],[0.,2.]])
        np.testing.assert_allclose(weighted_macro(sums,counts,weights)[0,:,0],[.5,.3,.9])

    def test_paired_bootstrap_preserves_shared_shift_and_seed_spread(self):
        ref,cand,groups=fixtures()
        bootstrap=dict(seed=17,replicates=2000,confidence=.95,quantile_method='linear')
        result=compare(ref,cand,groups,[1,2,3],bootstrap,SELECTION)
        self.assertAlmostEqual(result['mean_paired_ccc_gain'],.03)
        np.testing.assert_allclose(result['overall_ccc_gain_ci95'],[.03,.03],atol=1e-15)
        self.assertTrue(result['passes_internal_development_gates'])
        self.assertTrue(all(r['groups']==2 and r['songs']==3 for r in result['rows']))
        cand[0]['best_development']['pmemo']['per_song'][0]['valence']['ccc']=None
        with self.assertRaises(AssertionError):compare(ref,cand,groups,[1,2,3],bootstrap,SELECTION)

    def test_pairing_rejects_target_or_group_changes(self):
        ref,cand,groups=fixtures()
        bootstrap=dict(seed=17,replicates=20,confidence=.95,quantile_method='linear')
        for field in ('truth_std','values'):
            changed=copy.deepcopy(cand)
            changed[1]['best_development']['deam']['per_song'][0]['valence'][field]+=1
            with self.assertRaisesRegex(ValueError,'Target'):compare(ref,changed,groups,[1,2,3],bootstrap,SELECTION)
        changed=copy.deepcopy(cand);changed[0]['best_development']['deam']['per_song'][0]['group_id']='wrong'
        with self.assertRaisesRegex(ValueError,'group'):compare(ref,changed,groups,[1,2,3],bootstrap,SELECTION)

    def test_same_seed_and_steps_are_insufficient_without_exposure_and_init_parity(self):
        config=dict(settings=dict(seed=7,max_steps=20000,quadrant_loss_weight=0.),contract_sha256='same',
            protocol_sha256='same',feature_dimensions=531,code={'source':'frozen'},runtime={'torch':'same'},arithmetic={'fp32':True})
        other=copy.deepcopy(config);other['settings']['quadrant_loss_weight']=.1
        report=dict(initial_state_sha256='initial',exposure_chain='exposures',samplers={'main':80000,'aux':80000},
            completed_steps=20000,train_songs=3070,development_songs=383,auxiliary_clips=1001)
        check_pair_fairness(report,report,config,other)
        for field in ('initial_state_sha256','exposure_chain'):
            with self.assertRaisesRegex(ValueError,'mismatch'):
                check_pair_fairness(report,dict(report,**{field:'different'}),config,other)
        other['settings']['seed']=8
        with self.assertRaisesRegex(ValueError,'configurations'):check_pair_fairness(report,report,config,other)

    def test_active_queue_cannot_be_analyzed(self):
        with patch('tools.train_next_v2.analysis.d_auxiliary.is_running',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'partial'):analyze({})


if __name__=='__main__':unittest.main()
