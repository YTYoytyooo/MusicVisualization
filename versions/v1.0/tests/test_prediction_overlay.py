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

    def test_zoom_magnifies_small_changes_without_modifying_predictions(self):
        states = [dict(valence=.12, arousal=.30),
                  dict(valence=.16, arousal=.34),
                  dict(valence=.20, arousal=.38)]
        overlay = self.make_overlay(states)
        self.assertEqual(overlay.sample_at(.1), (1, .16, .34))
        np.testing.assert_array_equal(overlay.values, [[.12, .30], [.16, .34], [.20, .38]])
        # .08 units should cover most of a 240px zoom plot, not just ~10px.
        self.assertGreater(overlay.points[2, 0] - overlay.points[0, 0], 150)
        self.assertGreater(overlay.points[0, 1] - overlay.points[2, 1], 150)
        bounds = overlay.zoom_bounds
        overlay.draw(np.zeros((720, 1280, 3), np.uint8), .2)
        self.assertEqual(overlay.zoom_bounds, bounds)
        for (low, high), values in zip(bounds, overlay.values.T):
            self.assertLessEqual(low, values.min())
            self.assertGreaterEqual(high, values.max())
            self.assertGreaterEqual(high - low, .1 - 1e-9)

    def test_constant_and_boundary_predictions_have_safe_zoom_bounds(self):
        for value in (-1., 0., 1.):
            overlay = self.make_overlay([dict(valence=value, arousal=value)] * 3)
            for low, high in overlay.zoom_bounds:
                self.assertGreaterEqual(low, -1)
                self.assertLessEqual(high, 1)
                self.assertGreaterEqual(high - low, .1 - 1e-9)
            self.assertTrue(np.isfinite(overlay.points).all())
            np.testing.assert_array_equal(overlay.points[0], overlay.points[-1])

    def test_full_range_inset_retains_absolute_coordinates(self):
        overlay = self.make_overlay()
        np.testing.assert_array_equal(overlay.overview_points[0], [228, 485])
        np.testing.assert_array_equal(overlay.overview_points[1], [272, 441])
        np.testing.assert_array_equal(overlay.overview_points[2], [316, 397])

    def test_highlight_only_covers_last_two_seconds(self):
        states = [dict(valence=i / 100, arousal=i / 100) for i in range(41)]
        overlay = PredictionOverlay('Test', 4.1, states)
        self.assertEqual(overlay.recent_start_at(0), 0)
        self.assertEqual(overlay.recent_start_at(2), 0)
        self.assertEqual(overlay.recent_start_at(3), 10)
        self.assertEqual(overlay.recent_start_at(3.05), 11)
        self.assertEqual(overlay.recent_start_at(4), 20)

    def test_old_segments_are_dim_and_future_points_are_absent(self):
        states = [dict(valence=-.8, arousal=-.8),
                  dict(valence=0., arousal=-.8),
                  dict(valence=0., arousal=.8),
                  dict(valence=.8, arousal=.8),
                  dict(valence=.8, arousal=-.8)]
        overlay = PredictionOverlay('Test', 5., states, frame_duration=1.)
        frame = overlay.draw(np.zeros((720, 1280, 3), np.uint8), 3.)
        def pixel_at(v, a):
            x, y = overlay.point_for(v, a)
            return frame[118 + y, 924 + x].astype(int)
        # Earlier segment is still visible as history, but no longer cyan.
        self.assertLess(np.linalg.norm(pixel_at(-.4, -.8) - [120, 105, 85]), 30)
        self.assertLess(np.linalg.norm(pixel_at(.4, .8) - overlay.CYAN), 30)
        self.assertLess(np.linalg.norm(pixel_at(.8, .4) - overlay.BACKGROUND), 5)

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
