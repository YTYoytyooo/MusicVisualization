import copy
import unittest

from tools.train_next_v2.refinement_20261028.screen import compare, select_candidate, DOMAINS, AXES

SELECTION = dict(maximum_axis_domain_point_ccc_drop=.01, maximum_axis_domain_mae_increase=.01,
    screening_mean_delta_min_exclusive=0., dynamic_guardrails=dict(centered_rmse_max_relative_increase=.05,
    derivative_mae_max_relative_increase=.05, dynamics_beats_constant_max_absolute_drop=.03))


def fixture(ccc=.2):
    result = {}
    for domain in DOMAINS:
        values = dict(values=10, ccc=ccc, mae=.2, rmse=.25, mean_absolute_bias=.1,
                      truth_std=.1, truth_span90=.3, centered_rmse=.1, derivative_mae=.1,
                      std_ratio=1.2, constant_prediction_mae=.21, dynamics_beats_constant=.4,
                      prediction_span90=.35)
        result[domain] = dict(per_song=[dict(song_id=domain+'-1', group_id=domain+'-g1',
                                     **{a: copy.deepcopy(values) for a in AXES})],
            macro={a: {**{k:v for k,v in values.items() if k!='values'}, 'songs':1, 'ccc_songs':1} for a in AXES})
    return result


def change(data, metric, value, domain='pmemo', axis='arousal'):
    data[domain]['per_song'][0][axis][metric] = value
    data[domain]['macro'][axis][metric] = value


class ScreenChecks(unittest.TestCase):
    def test_positive_gain_is_only_screening_not_promotion(self):
        result = compare(fixture(), fixture(.25), SELECTION)
        self.assertTrue(result['eligible'])
        self.assertAlmostEqual(result['mean_ccc_delta'], .05)
        self.assertFalse(result['promotion_authorized'])

    def test_mean_and_each_cell_must_pass(self):
        self.assertFalse(compare(fixture(), fixture(.199), SELECTION)['eligible'])
        self.assertFalse(compare(fixture(), fixture(), SELECTION)['positive_mean_gain'])
        candidate = fixture(.5)
        change(candidate, 'ccc', .18)
        result = compare(fixture(), candidate, SELECTION)
        self.assertGreater(result['mean_ccc_delta'], .2)
        self.assertFalse(result['eligible'])
        self.assertFalse(result['rows'][-1]['gates']['ccc'])

    def test_mae_and_dynamic_guardrails_cannot_hide_behind_ccc(self):
        for metric, value in [('mae', .211), ('centered_rmse', .106),
                              ('derivative_mae', .106), ('dynamics_beats_constant', .369)]:
            with self.subTest(metric=metric):
                candidate = fixture(.3);change(candidate, metric, value)
                result = compare(fixture(), candidate, SELECTION)
                self.assertFalse(result['eligible'])
                self.assertFalse(result['rows'][-1]['gates'][metric])

    def test_inclusive_guardrail_boundaries(self):
        candidate = fixture(.3)
        for metric, value in [('ccc', .19), ('mae', .21), ('centered_rmse', .105),
                              ('derivative_mae', .105), ('dynamics_beats_constant', .37)]:
            change(candidate, metric, value)
        self.assertTrue(compare(fixture(), candidate, SELECTION)['eligible'])

    def test_missing_nonfinite_and_incorrect_aggregation_fail_closed(self):
        for value in (None, float('nan'), float('inf')):
            candidate = fixture(.3);change(candidate, 'derivative_mae', value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare(fixture(), candidate, SELECTION)
        candidate = fixture(.3);candidate['deam']['macro']['valence']['mae']=.19
        with self.assertRaisesRegex(ValueError, 'aggregation'):
            compare(fixture(), candidate, SELECTION)

    def test_support_identity_and_grouping_cannot_change(self):
        for mutation in ('id', 'group', 'mask', 'truth'):
            candidate=fixture(.3);row=candidate['pmemo']['per_song'][0]
            if mutation=='id':row['song_id']='another'
            elif mutation=='group':row['group_id']='another'
            elif mutation=='mask':row['arousal']['values']=9
            else:change(candidate, 'truth_std', .2)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare(fixture(), candidate, SELECTION)

    def test_incomplete_matrix_and_no_winner_do_not_enable_repeats(self):
        protocol={'budget':{'screening':['control','mean_loss','tone_residual']}}
        fail=compare(fixture(),fixture(.19),SELECTION)
        passed=compare(fixture(),fixture(.3),SELECTION)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            select_candidate({'mean_loss':passed},protocol)
        decision=select_candidate({'mean_loss':fail,'tone_residual':fail},protocol)
        self.assertFalse(decision['eligible']);self.assertIsNone(decision['selected_arm'])
        decision=select_candidate({'mean_loss':fail,'tone_residual':passed},protocol)
        self.assertEqual(decision['selected_arm'],'tone_residual')
        self.assertFalse(decision['promotion_authorized'])


if __name__ == '__main__':
    unittest.main()
