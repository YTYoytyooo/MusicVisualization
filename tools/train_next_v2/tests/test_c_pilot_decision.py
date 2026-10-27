"""The tiny-pilot point gates cannot be replaced by an average-only result."""
import unittest
from tools.train_next_v2.analysis.c_pilot import selection_gates


def rows(ccc=.03, mae=-.02):
    return [dict(metrics=dict(ccc=dict(mean_delta=ccc), mae=dict(mean_delta=mae))) for _ in range(6)]


class PilotDecisionTests(unittest.TestCase):
    def test_positive_average_does_not_hide_one_negative_seed(self):
        gates = selection_gates([.05, .05, -.01], rows())
        self.assertTrue(gates['mean_ccc_gain_at_least_002'])
        self.assertFalse(gates['positive_gain_in_every_seed'])

    def test_any_domain_axis_regression_blocks_expansion(self):
        data = rows()
        data[5]['metrics']['ccc']['mean_delta'] = -.0101
        self.assertFalse(selection_gates([.04, .04, .04], data)['no_dataset_axis_mean_ccc_drop_over_001'])
        data = rows()
        data[2]['metrics']['mae']['mean_delta'] = .0101
        self.assertFalse(selection_gates([.04, .04, .04], data)['no_dataset_axis_mean_mae_increase_over_001'])

    def test_all_seeds_and_axes_required_and_fixed_boundaries(self):
        with self.assertRaises(ValueError):
            selection_gates([.03, .03], rows())
        with self.assertRaises(ValueError):
            selection_gates([.03]*3, rows()[:5])
        self.assertTrue(all(selection_gates([.02]*3, rows(ccc=-.01, mae=.01)).values()))
        self.assertFalse(selection_gates([.019]*3, rows())['mean_ccc_gain_at_least_002'])


if __name__ == '__main__':
    unittest.main()
