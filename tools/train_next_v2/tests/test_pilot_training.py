"""Safety checks for pilot membership, clocks and genuine completion evidence."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.train_next.core import atomic_json, sha256
from tools.train_next_v2.common import BASE, ROOT
from tools.train_next_v2.pilot_training.data import array_fingerprint, read_feature, validate_membership
from tools.train_next_v2.pilot_training.execution import verification_matches


class PilotTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='pilot-training-test-', dir=BASE / 'tmp')
        self.folder = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.assertTrue(self.folder.is_relative_to((BASE / 'tmp').resolve()))
        self.temporary.cleanup()

    def test_calibration_and_linked_group_leakage_rejected(self):
        rows = []
        for domain in ('deam', 'memo2496', 'pmemo'):
            for role, count in [('fit', 20), ('development', 10)]:
                for index in range(count):
                    key = f'{domain}-{role}-{index}'
                    rows.append(dict(song_id=key, role=role, source_dataset=domain,
                                     audio_sha256=key, linked_component=key))
        assignments = {row['song_id']: row['role'] for row in rows}
        validate_membership(rows, assignments)
        assignments[rows[0]['song_id']] = 'calibration'
        with self.assertRaisesRegex(ValueError, 'calibration'):
            validate_membership(rows, assignments)
        assignments[rows[0]['song_id']] = 'fit'
        rows[-1]['linked_component'] = rows[0]['linked_component']
        with self.assertRaisesRegex(ValueError, 'linked'):
            validate_membership(rows, assignments)

    def test_feature_clock_and_hash_mismatch_rejected(self):
        file = self.folder / 'feature.npz'
        times = np.arange(5) * .5
        x = np.ones((5, 3), dtype=np.float32)
        np.savez(file, times=times, features=x, feature_key='key', audio_sha256='audio')
        entry = dict(path=file.relative_to(ROOT).as_posix(), sha256=sha256(file), dimensions=3, feature_key='key')
        np.testing.assert_array_equal(read_feature(entry, times, 'audio'), x)
        with self.assertRaises(AssertionError):
            read_feature(entry, times + .5, 'audio')
        with self.assertRaisesRegex(ValueError, 'identity'):
            read_feature(entry, times, 'different-recording')
        with file.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            read_feature(entry, times, 'audio')

    def test_independent_axis_masks_part_of_frozen_target_identity(self):
        times, y = np.arange(5)*.5, np.zeros((5, 2), dtype=np.float32)
        first = np.ones((5, 2), dtype=bool)
        second = first.copy()
        second[2, 0] = False
        self.assertNotEqual(array_fingerprint(times, y, first), array_fingerprint(times, y, second))
        third = first.copy()
        third[2, 1] = False
        self.assertNotEqual(array_fingerprint(times, y, second), array_fingerprint(times, y, third))

    def test_completion_requires_fresh_process_and_matching_checkpoints(self):
        run, models = self.folder / 'run', self.folder / 'models'
        run.mkdir()
        models.mkdir()
        (models / 'best.pth').write_bytes(b'best-fixture')
        (models / 'last.pth').write_bytes(b'last-fixture')
        atomic_json(run / 'config.json', dict(stage='C'))
        atomic_json(run / 'report.json', dict(status='completed', completed_steps=2000,
            checkpoint_sha256=sha256(models / 'best.pth'), last_checkpoint_sha256=sha256(models / 'last.pth')))
        checked = dict(status='completed_verified', separate_process=True, strict_state_load=True,
            full_resume_state_verified=True, fit_only_scaler_verified=True, legacy_model_unchanged=True,
            baseline_model_unchanged=True, calibration_validation_test_songs_used=0,
            report_sha256=sha256(run / 'report.json'), config_sha256=sha256(run / 'config.json'),
            best_sha256=sha256(models / 'best.pth'), last_sha256=sha256(models / 'last.pth'),
            fit_songs=60, development_songs=30)
        with patch('tools.train_next_v2.pilot_training.execution.paths', return_value=(run, models)):
            self.assertFalse(verification_matches('fixture'))
            atomic_json(run / 'verification.json', checked)
            self.assertTrue(verification_matches('fixture'))
            atomic_json(run / 'verification.json', {**checked, 'separate_process': False})
            self.assertFalse(verification_matches('fixture'))
            atomic_json(run / 'verification.json', checked)
            (models / 'last.pth').write_bytes(b'changed-last')
            self.assertFalse(verification_matches('fixture'))


if __name__ == '__main__':
    unittest.main()
