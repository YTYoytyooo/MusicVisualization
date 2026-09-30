"""Short deterministic engine contracts; no downloads or real model training."""
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from emotion_model import CLAP_DIM, WINDOW_SIZE, EmotionInterface, _build_windows
from mcts import MCTS, VisualState
from prediction_overlay import PredictionOverlay
from renderer import VideoRenderer


class LastEmbedding(torch.nn.Module):
    def forward(self, inputs):
        return inputs[:, -1, :5]


class EngineTests(unittest.TestCase):
    def test_inference_window_ends_at_current_embedding(self):
        embeddings = np.zeros((3, CLAP_DIM), dtype=np.float32)
        embeddings[:, :5] = np.array([.1, .2, .3])[:, None]
        results = EmotionInterface(LastEmbedding()).predict_sequence(embeddings)
        np.testing.assert_allclose([r['valence'] for r in results], [.1, .2, .3])

    def test_training_includes_last_window_and_rejects_short_input(self):
        embeddings = np.zeros((WINDOW_SIZE + 1, CLAP_DIM), dtype=np.float32)
        labels = np.arange((WINDOW_SIZE + 1) * 5).reshape(-1, 5)
        windows, targets = _build_windows(embeddings, labels)
        self.assertEqual(len(windows), 2)
        np.testing.assert_array_equal(targets[-1], labels[-1])
        with self.assertRaises(ValueError):
            _build_windows(embeddings[:1], labels[:1])

    def test_bad_checkpoint_window_is_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'incompatible.pth')
            torch.save({'window_size': WINDOW_SIZE + 1, 'state_dict': {}}, path)
            with self.assertRaisesRegex(ValueError, 'window_size'):
                EmotionInterface.load(path)

    def test_empty_inference_is_rejected(self):
        with self.assertRaises(ValueError):
            EmotionInterface(LastEmbedding()).predict_sequence(np.empty((0, CLAP_DIM)))

    def test_mcts_seed_reproduces_successive_searches(self):
        first, second = MCTS(n_iter=5, rng_seed=812), MCTS(n_iter=5, rng_seed=812)
        for v, a in [(0.1, 0.2), (-.3, .7)]:
            self.assertEqual(asdict(first.search(v, a)), asdict(second.search(v, a)))

    def test_renderer_seed_reproduces_real_frames_and_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            renderers = [VideoRenderer(str(Path(folder) / f'{i}.avi'), width=160,
                                       height=120, fps=30, rng_seed=123) for i in range(2)]
            try:
                state = VisualState(particle_count=50, trail_length=5)
                for i in range(3):
                    frames = [r.render_frame(state, np.zeros(32), 0, i / 600, 0)
                              for r in renderers]
                    np.testing.assert_array_equal(*frames)
                    for r, frame in zip(renderers, frames):
                        r.write_frame(frame)
                self.assertEqual(renderers[0].frame_idx, 3)
            finally:
                for r in renderers:
                    r.release()
            cap = cv2.VideoCapture(str(Path(folder) / '0.avi'))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            finally:
                cap.release()

    def test_failed_writer_is_rejected(self):
        with patch('renderer.cv2.VideoWriter') as writer:
            writer.return_value.isOpened.return_value = False
            with self.assertRaisesRegex(RuntimeError, 'video writer'):
                VideoRenderer('not-written.avi')
            writer.return_value.release.assert_called_once()

    def test_overlay_common_zoom_and_exact_frame_values(self):
        raw = [{'valence': -.6, 'arousal': -.5}] * 3
        effective = [{'valence': .4, 'arousal': .3}] * 3
        overlay = PredictionOverlay('Song', 1, effective, raw_emotion_states=raw,
                                    times=[0, .033, .067], source_label='MANUAL REVISION r002')
        np.testing.assert_allclose(overlay.raw_values[:, 0], -.6)
        np.testing.assert_allclose(overlay.values[:, 0], .4)
        self.assertLessEqual(overlay.zoom_bounds[0][0], -.6)
        self.assertGreaterEqual(overlay.zoom_bounds[0][1], .4)
        self.assertEqual(overlay.sample_at(.05), (1, .4, .3))
        first = np.zeros((720, 1280, 3), dtype=np.uint8)
        second = first.copy()
        with patch.object(overlay, '_text', wraps=overlay._text) as text:
            overlay.draw_values(first, .05, .2, -.1)
            labels = [args.args[1] for args in text.call_args_list]
            self.assertIn('V +0.200', labels)
            self.assertIn('A -0.100', labels)
        overlay.draw(second, .05)
        self.assertFalse(np.array_equal(first, second))
        # The exact marker is orange at the true V/A coordinate, not normalized.
        x, y = overlay.point_for(.2, -.1)
        np.testing.assert_array_equal(first[118 + y, 924 + x], overlay.ORANGE)

    def test_overlay_rejects_invalid_explicit_timeline_and_raw_values(self):
        values = [{'valence': 0., 'arousal': 0.}] * 2
        for times in ([0, 0], [0, float('nan')], [.1, .2], [0, 2], [0]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                PredictionOverlay('Song', 1, values, times=times)
        with self.assertRaises(ValueError):
            PredictionOverlay('Song', 1, values, raw_emotion_states=values[:1])


if __name__ == '__main__':
    unittest.main()
