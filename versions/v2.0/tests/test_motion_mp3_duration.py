"""Regression for VBR MP3 files whose reported length exceeds decoded audio."""
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from studio.motion_features import load_motion_features
from studio.pipeline import ffmpeg_path
from studio.store import atomic_json, digest


class Mp3DurationTests(unittest.TestCase):
    def test_vbr_without_xing_uses_actual_samples_and_keeps_source_unchanged(self):
        ffmpeg = ffmpeg_path()
        if not ffmpeg:
            self.skipTest('FFmpeg required for the real MP3 fixture')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rate = 48000
            t = np.arange(rate * 3 + 137) / rate
            signal = .2 * np.sin(2 * np.pi * 440 * t)
            signal[t < 1.0] = 0
            signal[t > 2.0] += np.random.default_rng(7).normal(0, .07, np.count_nonzero(t > 2.0))
            wav, mp3 = root / 'fixture.wav', root / 'audio.mp3'
            sf.write(wav, signal, rate, subtype='PCM_16')
            subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                            '-i', str(wav), '-c:a', 'libmp3lame', '-q:a', '4',
                            '-write_xing', '0', str(mp3)], check=True,
                           capture_output=True, timeout=15)
            decoded, sample_rate = sf.read(mp3, dtype='float64', always_2d=True)
            self.assertEqual(sample_rate, rate)
            self.assertNotEqual(sf.info(mp3).frames, len(decoded), 'Fixture must exercise an inaccurate MP3 header')
            atomic_json(root / 'project.json', {
                'schema_version': 1, 'audio_file': 'audio.mp3', 'audio_sha256': digest(mp3),
                'duration_us': round(len(decoded) / rate * 1e6), 'current_revision': 'r000000'})
            protected = [mp3, root / 'project.json']
            before = [digest(path) for path in protected]
            result = load_motion_features(root)
            expected = [np.sqrt(np.mean(decoded[left:left + rate // 10] ** 2))
                        for left in range(0, len(decoded), rate // 10)]
            # Independent MP3 decoders may differ by tiny sample-rounding noise.
            np.testing.assert_allclose(result['rms'], expected, rtol=2e-5, atol=2e-6)
            self.assertEqual(len(result['times_us']), len(expected))
            self.assertTrue(np.isfinite(result['pulse_strength']).all())
            self.assertEqual(before, [digest(path) for path in protected])
            self.assertEqual(list(root.glob('.motion-audio-*')), [])
            cached = load_motion_features(root)
            self.assertEqual(result['key'], cached['key'])
            np.testing.assert_array_equal(result['rms'], cached['rms'])


if __name__ == '__main__':
    unittest.main()
