"""Behavioral checks for grouped, paired development-set confirmation."""
import unittest

import numpy as np

from tools.train_next_v2.analysis.paired_confirmation import connected_groups, weighted_macro


class GroupConfirmationTests(unittest.TestCase):
    def test_transitive_audio_and_artist_identity_stay_together(self):
        rows = [
            dict(song_id='a', group_id='artist1', audio_sha256='audio1', split='train'),
            dict(song_id='b', group_id='artist1', audio_sha256='audio2', split='train'),
            dict(song_id='c', group_id='artist2', audio_sha256='audio2', split='train'),
            dict(song_id='d', group_id='artist3', audio_sha256='audio3', split='train'),
        ]
        self.assertEqual([[r['song_id'] for r in g] for g in connected_groups(rows).values()], [['a', 'b', 'c'], ['d']])
        self.assertEqual(connected_groups(rows), connected_groups(list(reversed(rows))))

    def test_holdout_and_duplicate_song_identity_are_rejected(self):
        row = dict(song_id='a', group_id='artist', audio_sha256='audio', split='test')
        with self.assertRaises(ValueError):
            connected_groups([row])
        row['split'] = 'train'
        with self.assertRaises(ValueError):
            connected_groups([row, row.copy()])

    def test_unequal_clusters_keep_song_weights(self):
        # One group has two songs scoring 0 and 2; the other has one scoring 10.
        sums = np.array([[[2.], [10.]]])
        counts = np.array([[[2.], [1.]]])
        weights = np.array([[1., 1.], [2., 0.], [0., 2.], [2., 1.]])
        actual = weighted_macro(sums, counts, weights)[0, :, 0]
        np.testing.assert_allclose(actual, [4., 1., 10., 2.8])

    def test_paired_draws_preserve_constant_improvement_for_all_seeds(self):
        counts = np.array([[[2.], [1.]], [[2.], [1.]]])
        reference = np.array([[[2.], [10.]], [[4.], [8.]]])
        candidate = reference + 0.25 * counts
        weights = np.array([[2., 0.], [1., 1.], [0., 2.]])
        delta = weighted_macro(candidate, counts, weights) - weighted_macro(reference, counts, weights)
        np.testing.assert_allclose(delta, np.full_like(delta, 0.25))
        np.testing.assert_array_equal(weighted_macro(reference, counts, weights) - weighted_macro(reference, counts, weights), np.zeros_like(delta))

    def test_zero_valid_observations_fail_instead_of_imputing(self):
        with self.assertRaises(ValueError):
            weighted_macro(np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.ones((1, 1)))


if __name__ == '__main__':
    unittest.main()
