import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from prediction_overlay import PredictionOverlay, format_playback_time
from mcts import VisualState
from renderer import VideoRenderer


class PredictionOverlayTests(unittest.TestCase):
    def make_overlay(self, states=None):
        if states is None:
            states = [dict(valence=-1, arousal=-1),
                      dict(valence=0, arousal=0),
                      dict(valence=1, arousal=1)]
        return PredictionOverlay('Test song', 2.0, states, frame_duration=0.1)

    def test_time_is_media_time_and_carries_correctly(self):
        self.assertEqual(format_playback_time(0), '00:00.000')
        self.assertEqual(format_playback_time(61.234), '01:01.234')
        self.assertEqual(format_playback_time(59.9996), '01:00.000')

    def test_sample_hold_includes_boundary_but_not_future(self):
        overlay = self.make_overlay()
        self.assertEqual(overlay.sample_at(0), (0, -1.0, -1.0))
        self.assertEqual(overlay.sample_at(0.099), (0, -1.0, -1.0))
        self.assertEqual(overlay.sample_at(0.1), (1, 0.0, 0.0))
        self.assertEqual(overlay.sample_at(1.99), (2, 1.0, 1.0))

    def test_axes_use_fixed_range_and_arousal_points_up(self):
        overlay = self.make_overlay()
        low = overlay.point_for(-1, -1)
        high = overlay.point_for(1, 1)
        center = overlay.point_for(0, 0)
        self.assertLess(low[0], high[0])
        self.assertGreater(low[1], high[1])
        np.testing.assert_allclose(center, (low + high) / 2, atol=1)

    def test_future_predictions_do_not_change_current_image(self):
        a = self.make_overlay()
        b = self.make_overlay([dict(valence=-1, arousal=-1),
                               dict(valence=1, arousal=-1),
                               dict(valence=-1, arousal=1)])
        frame = np.zeros((720, 1280, 3), np.uint8)
        np.testing.assert_array_equal(a.draw(frame.copy(), 0.05),
                                      b.draw(frame.copy(), 0.05))
        self.assertFalse(np.array_equal(a.draw(frame.copy(), 0.2),
                                       a.draw(frame.copy(), 0.05)))

    def test_empty_nonfinite_and_invalid_timing_rejected(self):
        with self.assertRaises(ValueError):
            self.make_overlay([])
        with self.assertRaises(ValueError):
            self.make_overlay([dict(valence=np.nan, arousal=0)])
        with self.assertRaises(ValueError):
            PredictionOverlay('x', 2, [dict(valence=0, arousal=0)], 0)

    def test_long_title_fits_header(self):
        overlay = PredictionOverlay('Long song name ' * 100, 2,
                                    [dict(valence=0, arousal=0)])
        self.assertLessEqual(overlay.title_width, overlay.title_max_width)

    def test_real_renderer_writes_playable_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'test.avi')
            renderer = VideoRenderer(path, prediction_overlay=self.make_overlay())
            try:
                for _ in range(3):
                    frame = renderer.render_frame(VisualState(particle_count=5),
                                                  np.zeros(735), 0, 0, 0)
                    self.assertEqual(frame.shape, (720, 1280, 3))
                    self.assertEqual(frame.dtype, np.uint8)
                    renderer.write_frame(frame)
            finally:
                renderer.release()
            video = cv2.VideoCapture(path)
            try:
                self.assertTrue(video.isOpened())
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                ok, frame = video.read()
                self.assertTrue(ok)
                self.assertGreater(int(frame[100:560, 900:].sum()), 0)
            finally:
                video.release()


if __name__ == '__main__':
    unittest.main()
