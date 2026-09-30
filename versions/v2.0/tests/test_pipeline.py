"""Small real render loop with synthetic input; no ML, network or training."""
import csv
import tempfile
import subprocess
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import soundfile as sf

from renderer import VideoRenderer
from studio.pipeline import VISUAL_FIELDS, render, ffmpeg_path
from studio.demo import create_demo
from studio.store import create_project, digest, load_project, load_raw, save_revision


class PipelineTests(unittest.TestCase):
    @unittest.skipUnless(ffmpeg_path(), 'H.264/AAC encoder not installed')
    def test_local_encoder_writes_playable_audio_and_matching_frame_table(self):
        with tempfile.TemporaryDirectory() as folder:
            project = create_demo(folder, .63)
            with patch('feature_extraction.load_audio', side_effect=lambda path: sf.read(path, dtype='float32')):
                result = render(project, options=dict(fps=20, width=320, height=180), iterations=2)
            output = project / 'renders' / result['id']
            self.assertEqual(result['status'], 'succeeded')
            with open(output / 'frame_values.csv', encoding='utf-8-sig') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 13)
            cap = cv2.VideoCapture(str(output / 'output.mp4'))
            try:
                self.assertTrue(cap.read()[0])
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 13)
                self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 20)
            finally:
                cap.release()
            audio = subprocess.run([ffmpeg_path(), '-v', 'error', '-i', str(output / 'output.mp4'),
                                    '-map', '0:a:0', '-ar', '16000', '-ac', '1', '-f', 's16le', '-'],
                                   capture_output=True, check=True, timeout=10)
            self.assertAlmostEqual(len(audio.stdout) / 32000, 13 / 20, delta=.08)
            self.assertTrue(any(audio.stdout))

    def test_synthetic_revision_render_csv_and_visual_cache_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'sample.wav'
            with wave.open(str(source), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(22050)
                tone = np.sin(np.arange(44100) * (2 * np.pi * 220 / 22050)) * 1600
                stream.writeframes(tone.astype('<i2').tobytes())
            raw = np.zeros((20, 5))
            raw[:, 0] = np.linspace(-.1, .1, 20)
            project = create_project(Path(folder) / 'projects', 'Synthetic integration test', source,
                                     None, raw, {'duration': 2., 'tempo': 120., 'beat_times': [0., .5, 1., 1.5]},
                                     provenance={'kind': 'synthetic'})
            p = load_project(project)
            original_hash = digest(project / 'analysis/predictions_raw.npz')
            captured = []
            original_render = VideoRenderer.render_frame

            def capture(renderer, state, *args, **kwargs):
                captured.append([getattr(state, name) for name in VISUAL_FIELDS])
                return original_render(renderer, state, *args, **kwargs)

            options = {'fps': 5, 'width': 320, 'height': 180, 'mode': 'analysis'}

            def render_wav(*args, **kwargs):
                # This fixture already has the engine sample rate. Decode the
                # real WAV with soundfile to isolate slow first-use librosa JIT.
                def decode(path):
                    samples, rate = sf.read(path, dtype='float32')
                    self.assertEqual(rate, 22050)
                    return samples, rate
                with patch('feature_extraction.load_audio', side_effect=decode):
                    return render(*args, **kwargs)

            with patch.object(VideoRenderer, 'render_frame', capture):
                initial = render_wav(project, options=options, allow_silent=True, iterations=2)
            output = project / 'renders' / initial['id']
            with open(output / 'frame_values.csv', encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            self.assertEqual(initial['frames'], 10)
            self.assertEqual([int(r['time_us']) for r in rows], list(range(0, 2000000, 200000)))
            exported = [[float(row['visual_' + name]) for name in VISUAL_FIELDS] for row in rows]
            np.testing.assert_allclose(exported, captured)
            cap = cv2.VideoCapture(str(output / initial['video_file']))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
                self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 5)
                self.assertTrue(cap.read()[0])
            finally:
                cap.release()
            edit = {'id': 'speed', 'layer': 'visual', 'field': 'particle_speed',
                    'operation': 'interval_set', 'start_us': 400000, 'end_us': 1200000,
                    'value': 3., 'transition_in_us': 200000, 'transition_out_us': 200000}
            revised = save_revision(project, p['current_revision'], [edit])
            # A visual-only revision must not invoke the expensive search again.
            with patch('mcts.MCTS.search', side_effect=AssertionError('visual edit reran planning')):
                second = render_wav(project, revised['id'], options=options, allow_silent=True, iterations=2)
            self.assertEqual(initial['plan_key'], second['plan_key'])
            with open(project / 'renders' / second['id'] / 'frame_values.csv',
                      encoding='utf-8-sig', newline='') as stream:
                revised_rows = list(csv.DictReader(stream))
            self.assertEqual(float(revised_rows[3]['visual_particle_speed']), 3.)
            self.assertEqual(revised_rows[3]['revision'], revised['id'])
            self.assertEqual(original_hash, digest(project / 'analysis/predictions_raw.npz'))
            np.testing.assert_array_equal(load_raw(project), raw)


if __name__ == '__main__':
    unittest.main()
