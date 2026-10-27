import unittest
import numpy as np

from tools.datasets.index_supplemental import MatchIndex


class RecordingOverlapTests(unittest.TestCase):
    def test_crop_with_reencoding_noise_is_found(self):
        rng = np.random.default_rng(17)
        original = rng.integers(0, 2**32, 1200, dtype=np.uint32)
        crop = original[273:673].copy()
        crop[::3] ^= np.uint32(1)
        index = MatchIndex()
        index.add(original)
        result = index.matches(crop)
        self.assertIn(0, result)
        self.assertEqual(result[0]['offset_frames'], 273)

    def test_unrelated_music_and_silence_do_not_match(self):
        rng = np.random.default_rng(22)
        index = MatchIndex()
        index.add(rng.integers(0, 2**32, 600, dtype=np.uint32))
        index.add(np.ones(600, dtype=np.uint32))
        self.assertEqual(index.matches(rng.integers(0, 2**32, 600, dtype=np.uint32)), {})
        self.assertEqual(index.matches(np.ones(600, dtype=np.uint32)), {})


if __name__ == '__main__':
    unittest.main()
