import tempfile
import subprocess
import sys
from unittest.mock import patch
import unittest
import wave
from pathlib import Path
import numpy as np
from studio.store import create_project, load_project, load_raw, save_revision, load_revision, digest, project_lock, list_projects, read_json


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        audio = self.root / '测试 空格.wav'
        with wave.open(str(audio), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(8000)
            f.writeframes(b'\0\0' * 40000)
        self.project = create_project(self.root / 'projects', '测试', audio, None,
                                      np.zeros((50, 5)), dict(duration=5., tempo=120, beat_times=[]))

    def tearDown(self):
        self.temp.cleanup()

    def test_revisions_preserve_original_and_stale_save_rejected(self):
        before = digest(self.project / 'analysis/predictions_raw.npz')
        p = load_project(self.project)
        r = save_revision(self.project, p['current_revision'], [])
        self.assertEqual(r['id'], 'r000001')
        self.assertEqual(digest(self.project / 'analysis/predictions_raw.npz'), before)
        self.assertEqual(load_revision(self.project)['parent'], 'r000000')
        with self.assertRaises(ValueError):
            save_revision(self.project, 'r000000', [])

    def test_external_corruption_detected(self):
        target = self.project / 'analysis/predictions_raw.npz'
        target.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            load_raw(self.project)

    def test_lock_exclusion_and_reuse(self):
        with project_lock(self.project):
            with self.assertRaises(ValueError):
                with project_lock(self.project):
                    self.fail('simultaneous writer')
        with project_lock(self.project):
            pass

    def test_lock_released_after_exception(self):
        with self.assertRaises(RuntimeError):
            with project_lock(self.project):
                raise RuntimeError('simulated failure')
        with project_lock(self.project):
            pass

    def test_os_releases_lock_when_worker_crashes(self):
        code = ('from studio.store import project_lock; import os,sys; '
                'guard=project_lock(sys.argv[1]); guard.__enter__(); os._exit(7)')
        result = subprocess.run([sys.executable, '-c', code, str(self.project)],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 7, result.stderr)
        with project_lock(self.project):
            pass

    def test_failed_initial_revision_is_not_published(self):
        root = self.root / 'projects'
        with patch('studio.store.save_revision', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                create_project(root, 'incomplete', self.root / '测试 空格.wav', None,
                               np.zeros((50, 5)), dict(duration=5., tempo=120, beat_times=[]))
        self.assertEqual([p['id'] for p in list_projects(root)], [self.project.name])
        self.assertEqual(len(list(root.glob('.pending-p-*'))), 1)

    def test_waveform_is_streamed_and_preserves_channel_peaks(self):
        import soundfile as sf
        audio = self.root / 'stereo.wav'
        samples = np.zeros((825, 2), dtype=np.float32)
        samples[10, 1], samples[450, 0], samples[824, 1] = -.75, .5, -.25
        sf.write(audio, samples, 8000, subtype='FLOAT')
        with patch('soundfile.read', side_effect=AssertionError('full-file read')):
            project = create_project(self.root / 'projects', 'streaming', audio, None,
                                     np.zeros((2, 5)), dict(duration=825 / 8000, tempo=120, beat_times=[]))
        waveform = read_json(project / 'analysis/waveform.json')
        self.assertEqual(waveform['times'], [0., .05, .1])
        self.assertEqual(waveform['peaks'], [.75, .5, .25])


if __name__ == '__main__':
    unittest.main()
