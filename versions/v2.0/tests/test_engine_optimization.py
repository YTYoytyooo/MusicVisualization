"""No-download contracts for batched cancellation, incremental HUD and resume."""
from copy import deepcopy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from emotion_model import (AnalysisCancelled, CLAP_DIM, WINDOW_SIZE,
                           EmotionInterface, extract_clap_embeddings)
from mcts import VisualState
from prediction_overlay import PredictionOverlay
from renderer import VideoRenderer
from studio.checkpoints import find_checkpoint, load_checkpoint, save_checkpoint


class WindowProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, values):
        self.seen.append(values.clone())
        return values[:, -1, :5]


def make_hud(count=100, step=.1):
    states = [{'valence': float(.2 * np.sin(i / 9)), 'arousal': float(.2 * np.cos(i / 13))}
              for i in range(count)]
    raw = [{'valence': s['valence'] / 2, 'arousal': s['arousal'] / 2} for s in states]
    return PredictionOverlay('Synthetic checkpoint test', count * step, states,
                             frame_duration=step, raw_emotion_states=raw,
                             source_label='SYNTHETIC / MANUAL REVISION')


class EngineOptimizationTests(unittest.TestCase):
    def test_inference_working_windows_are_batched_and_time_aligned(self):
        source = np.random.default_rng(42).random((17, CLAP_DIM), dtype=np.float32)
        model, progress = WindowProbe(), []
        actual = EmotionInterface(model).predict_sequence(source, batch_size=4,
            progress=lambda done, total: progress.append((done, total)))
        self.assertEqual([len(batch) for batch in model.seen], [4, 4, 4, 4, 1])
        # Independent reference: each window ends at its assigned source frame.
        padded = np.pad(source, ((WINDOW_SIZE - 1, 0), (0, 0)))
        expected = np.stack([padded[i:i + WINDOW_SIZE] for i in range(len(source))])
        np.testing.assert_array_equal(torch.cat(model.seen).numpy(), expected)
        np.testing.assert_array_equal([row['valence'] for row in actual], source[:, 0])
        self.assertEqual(progress, [(0, 17), (4, 17), (8, 17), (12, 17), (16, 17), (17, 17)])

    def test_inference_cancel_stops_before_next_batch(self):
        model, stop = WindowProbe(), [False]
        def progress(done, total):
            if done >= 4:
                stop[0] = True
        with self.assertRaises(AnalysisCancelled):
            EmotionInterface(model).predict_sequence(np.zeros((17, CLAP_DIM), np.float32),
                batch_size=4, progress=progress, cancel=lambda: stop[0])
        self.assertEqual(len(model.seen), 1)

    def clap_fakes(self):
        batches = []
        def processor(audio, **kwargs):
            batches.append([np.asarray(win).copy() for win in audio])
            return {'input_features': torch.zeros((len(audio), 2))}
        def audio_model(input_features, is_longer=None):
            return SimpleNamespace(pooler_output=torch.full((len(input_features), CLAP_DIM), .25))
        return SimpleNamespace(audio_model=audio_model), processor, batches

    def test_clap_cancel_does_not_publish_partial_cache(self):
        model, processor, batches = self.clap_fakes()
        stopped, progress = [False], []
        def report(done, total):
            progress.append((done, total))
            if done >= 4:
                stopped[0] = True
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'cache.npy'
            with patch('emotion_model._check_transformers', return_value=True), \
                 patch('emotion_model._load_clap', return_value=(model, processor)), \
                 patch.dict(sys.modules, {'librosa': SimpleNamespace(resample=lambda y, **kwargs: y)}):
                with self.assertRaises(AnalysisCancelled):
                    extract_clap_embeddings(np.arange(100, dtype=np.float32), 100,
                        frame_duration=.1, window_duration=.2, clap_sr=100, batch_size=4,
                        cache_path=str(cache), progress=report, cancel=lambda: stopped[0])
            self.assertFalse(cache.exists())
            self.assertEqual(len(batches), 1)
            self.assertEqual(progress, [(0, 10), (4, 10)])

    def test_clap_progress_and_atomic_cache_use_pinned_source(self):
        model, processor, batches = self.clap_fakes()
        progress = []
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'cache.npy'
            with patch('emotion_model._check_transformers', return_value=True), \
                 patch('emotion_model._load_clap', return_value=(model, processor)) as loader, \
                 patch.dict(sys.modules, {'librosa': SimpleNamespace(resample=lambda y, **kwargs: y)}):
                result = extract_clap_embeddings(np.arange(100, dtype=np.float32), 100,
                    frame_duration=.1, window_duration=.2, clap_sr=100, batch_size=4,
                    cache_path=str(cache), progress=lambda *args: progress.append(args),
                    clap_source='pinned/local/snapshot')
            loader.assert_called_once_with('pinned/local/snapshot')
            self.assertEqual([len(batch) for batch in batches], [4, 4, 2])
            self.assertEqual(progress, [(0, 10), (4, 10), (8, 10), (10, 10)])
            np.testing.assert_array_equal(np.load(cache, allow_pickle=False), result)
            self.assertFalse(list(Path(folder).glob('.clap-pending-*')))
            np.testing.assert_array_equal(batches[-1][-1], np.r_[np.arange(90, 100), np.zeros(10)])

    def test_cancel_before_cached_clap_does_not_load_or_remove_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'cache.npy'
            np.save(cache, np.zeros((1, CLAP_DIM), np.float32))
            before = cache.read_bytes()
            with patch('emotion_model.np.load', side_effect=AssertionError('cache should not load')):
                with self.assertRaises(AnalysisCancelled):
                    extract_clap_embeddings(np.zeros(10), 100, cache_path=str(cache), cancel=lambda: True)
            self.assertEqual(cache.read_bytes(), before)

    def test_hud_sequential_seek_backward_and_exact_values_are_identical(self):
        sequential = make_hud()
        for t in [0, .1, .2, .5, 1., 3., 5., .4, .42, 4.]:
            actual = np.zeros((360, 640, 3), np.uint8)
            expected = actual.copy()
            sequential.draw_values(actual, t, .11, -.03)
            make_hud().draw_values(expected, t, .11, -.03)
            np.testing.assert_array_equal(actual, expected, err_msg=f'HUD at {t}')

    def test_hud_history_drawing_work_is_linear(self):
        count = 100
        hud = make_hud(count)
        original_line, original_polyline = cv2.line, cv2.polylines
        line_count, poly_points = [0], [0]
        def line(*args, **kwargs):
            line_count[0] += 1
            return original_line(*args, **kwargs)
        def polyline(image, points, *args, **kwargs):
            poly_points[0] += sum(len(p) for p in points)
            return original_polyline(image, points, *args, **kwargs)
        with patch('prediction_overlay.cv2.line', side_effect=line), \
             patch('prediction_overlay.cv2.polylines', side_effect=polyline):
            for i in range(count):
                hud.draw(np.zeros((180, 320, 3), np.uint8), i * .1)
        self.assertEqual(line_count[0], 4 * (count - 1))
        self.assertLessEqual(poly_points[0], 2 * 22 * count)

    def test_checkpoint_resumes_identical_frames_with_rng_trails_and_hud(self):
        def advance(renderer, index):
            state = VisualState(particle_count=50 + (index % 4) * 7,
                                hue_base=175 + index * 2, trail_length=12)
            return renderer.render_frame(state, np.sin(np.arange(20) / 10).astype(np.float32),
                                         float(index % 3 == 0), index * .005, index % 10 / 10)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            original = VideoRenderer(str(directory / 'first.avi'), width=160, height=120,
                                     fps=10, rng_seed=123, prediction_overlay=make_hud())
            resumed = VideoRenderer(str(directory / 'second.avi'), width=160, height=120,
                                    fps=10, rng_seed=999, prediction_overlay=make_hud())
            try:
                original.particles.life[:20] = 2  # Exercise rebirth RNG as well as beat RNG.
                for i in range(4):
                    advance(original, i)
                manifest = save_checkpoint(directory / 'checkpoints', 'same-config', original)
                self.assertIsNone(find_checkpoint(directory / 'checkpoints', 'same-config', 3))
                self.assertEqual(find_checkpoint(directory / 'checkpoints', 'same-config', 4), manifest)
                with self.assertRaisesRegex(ValueError, 'configuration key'):
                    load_checkpoint(manifest, 'other-config', resumed)
                self.assertEqual(resumed.frame_idx, 0)
                self.assertEqual(load_checkpoint(manifest, 'same-config', resumed), 4)
                for i in range(4, 10):
                    np.testing.assert_array_equal(advance(original, i), advance(resumed, i))
                self.assertEqual(original.frame_idx, resumed.frame_idx)
                np.testing.assert_array_equal(original.particles.pos, resumed.particles.pos)
                self.assertEqual(original.particles._rng.bit_generator.state,
                                 resumed.particles._rng.bit_generator.state)
            finally:
                original.release(); resumed.release()

    def test_checkpoint_corruption_and_bad_shape_fail_without_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            renderer = VideoRenderer(str(directory / 'test.avi'), width=160, height=120, rng_seed=123)
            try:
                initial = renderer.export_state()
                bad = deepcopy(initial)
                bad['arrays']['pos'] = np.zeros((1, 2), np.float32)
                with self.assertRaises(ValueError):
                    renderer.restore_state(bad)
                np.testing.assert_array_equal(renderer.particles.pos, initial['arrays']['pos'])
                self.assertEqual(renderer.frame_idx, 0)
                manifest = save_checkpoint(directory / 'checkpoints', 'test', renderer)
                data = json.loads(manifest.read_text(encoding='utf-8'))
                array_path = manifest.parent / data['arrays_file']
                with open(array_path, 'ab') as stream:
                    stream.write(b'corruption')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    load_checkpoint(manifest, 'test', renderer)
                np.testing.assert_array_equal(renderer.particles.pos, initial['arrays']['pos'])
            finally:
                renderer.release()


if __name__ == '__main__':
    unittest.main()
