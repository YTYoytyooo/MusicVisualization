import unittest

import numpy as np

from tools.train_next_v2.acoustic_pilot.features import pool_acoustics, pulse_features


class RhythmHarmonyTests(unittest.TestCase):
    def test_zero_variance_has_no_pulse(self):
        np.testing.assert_array_equal(pulse_features(np.zeros(800)), np.zeros(3))
        np.testing.assert_array_equal(pulse_features(np.ones(800)), np.zeros(3))

    def test_regular_two_hz_pulses_have_a_120_bpm_peak(self):
        pulse = np.tile(np.r_[1., np.zeros(49)], 16)
        result = pulse_features(pulse)
        np.testing.assert_allclose(result, [1, 1, 120], atol=1e-12)

    def test_energy_entropy_change_and_source_edge_validity(self):
        clock = np.arange(200) * .01
        chroma = np.zeros((12,200))
        chroma[0,:100] = 1
        chroma[1,100:] = 1
        values = pool_acoustics([0.,1.], clock, np.full(200,4.), np.ones(200), chroma, np.zeros(200))
        np.testing.assert_allclose(values[:, :3], np.tile([np.log(4), 0, .2], (2,1)), atol=1e-7)
        np.testing.assert_allclose(values[1,6:], [np.log(2)/np.log(12), 1, 1], atol=1e-7)
        np.testing.assert_array_equal(values[0,7:], [0,0])

    def test_silence_is_finite_and_is_not_a_valid_harmonic_change(self):
        clock = np.arange(200)*.01
        values = pool_acoustics([1.], clock, np.zeros(200), np.zeros(200), np.zeros((12,200)), np.zeros(200))
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(values[0,:2], np.log(1e-10), atol=1e-6)
        np.testing.assert_array_equal(values[0,2:], np.zeros(7))


if __name__ == '__main__':
    unittest.main()
