import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.train_next.core import (SequenceVA, atomic_json, attach_targets, collate_songs, fit_scaler,
                                   output_paths, regression_metrics, select_records, supervised_loss)
from tools.train_next.features import centered_window
from tools.train_next.run import train


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_time_join_retains_initial_gap_and_independent_mask(self):
        times = np.arange(33) * .5
        y, mask = attach_targets(times, np.array([15, 15.5]), np.array([[.4, 0], [0, -.3]], np.float32),
                                 np.array([[True, False], [False, True]]))
        self.assertFalse(mask[:30].any())
        np.testing.assert_array_equal(mask[30:32], [[True, False], [False, True]])
        self.assertAlmostEqual(float(y[31, 1]), -.3)
        with self.assertRaises(ValueError):
            attach_targets(times, np.array([15.1]), np.zeros((1, 2)), np.ones((1, 2), bool))

    def test_padding_and_missing_labels_do_not_change_loss(self):
        p = torch.tensor([[[.3, -.2], [.7, .8]]], requires_grad=True)
        y = torch.tensor([[[.1, -.4], [99., float('nan')]]])
        mask = torch.tensor([[[True, True], [False, False]]])
        loss = supervised_loss(p, y, mask)
        expected = supervised_loss(p[:, :1], y[:, :1], mask[:, :1])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        torch.testing.assert_close(p.grad[:, 1], torch.zeros((1, 2)))

    def test_centered_audio_windows_keep_zero_origin(self):
        audio = np.arange(6, dtype=np.float32)
        np.testing.assert_array_equal(centered_window(audio, 2, 0, 2), [0, 0, 0, 1])
        np.testing.assert_array_equal(centered_window(audio, 2, 1, 2), [0, 1, 2, 3])

    def test_scaler_fits_only_supplied_training_songs(self):
        mean, std = fit_scaler([{'x': np.array([[1, 10], [3, 30]], np.float32)}])
        np.testing.assert_array_equal(mean, [2, 20])
        np.testing.assert_array_equal(std, [1, 10])

    def test_test_split_cannot_be_selected_for_training(self):
        records = [dict(song_id=i, split=split) for i, split in enumerate(['train', 'validation', 'test'])]
        self.assertEqual([r['song_id'] for r in select_records(records, 'train', 0, 1)], [0])
        with self.assertRaises(ValueError):
            select_records(records, 'test', 0, 1)

    def test_output_cannot_overwrite_legacy_model(self):
        _, path = output_paths('pilot-test')
        self.assertEqual(path.parent.name, 'next_va')
        for identifier in ('../emotion_model.pth', 'a/b', '..', ''):
            with self.assertRaises(ValueError):
                output_paths(identifier)

    def test_packed_sequences_ignore_added_padding(self):
        torch.manual_seed(1)
        model = SequenceVA(3, hidden=4).eval()
        short = torch.randn(1, 3, 3)
        padded = torch.cat((short, torch.full((1, 8, 3), 12345.)), dim=1)
        with torch.no_grad():
            a = model(short, torch.tensor([3]))
            b = model(padded, torch.tensor([3]))
        torch.testing.assert_close(a, b[:, :3])

    def test_metrics_equal_for_perfect_prediction(self):
        y = np.array([[-.5, .4], [.1, -.3], [.7, .8]], np.float32)
        metrics = regression_metrics([y], [dict(song_id=1, y=y, mask=np.ones_like(y, bool))])
        for axis in ('valence', 'arousal'):
            self.assertEqual(metrics['macro'][axis]['mae'], 0)
            self.assertAlmostEqual(metrics['macro'][axis]['ccc'], 1)

    def test_real_optimizer_checkpoint_and_resume(self):
        torch.manual_seed(4)
        songs = []
        for i in range(6):
            x = np.linspace(-1, 1, 8 + i, dtype=np.float32)[:, None]
            y = np.concatenate((x * .6, x * -.4), axis=1)
            songs.append(dict(song_id=i, x=x, y=y, mask=np.ones_like(y, bool)))
        config = dict(stage='pilot', hidden=4, lr=.03, seed=5, epochs=3, patience=8, song_batch=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / 'models'
            models.mkdir()
            result = train(dict(train=songs[:4], validation=songs[4:]), config, root, models, torch.device('cpu'), {'test': True})
            self.assertEqual(result['test_songs_used'], 0)
            state = torch.load(models / 'last.pth', weights_only=True)
            self.assertEqual(state['epoch'], 3)
            self.assertEqual(state['label_order'], ['valence', 'arousal'])
            self.assertTrue(np.isfinite(result['best_validation']['macro']['valence']['mae']))
            again = train(dict(train=songs[:4], validation=songs[4:]), config, root, models, torch.device('cpu'), {'test': True}, resume=True)
            self.assertEqual(result['best_epoch'], again['best_epoch'])
            self.assertEqual(len(json.loads((root / 'metrics.json').read_text())), 3)

    def test_interrupted_training_resumes_same_optimizer_trajectory(self):
        songs = []
        for i in range(4):
            x = np.linspace(-1, 1, 9 + i, dtype=np.float32)[:, None]
            y = np.concatenate([x * .5, x * -.3], axis=1)
            songs.append(dict(song_id=i, x=x, y=y, mask=np.ones_like(y, bool)))
        data = dict(train=songs[:2], validation=songs[2:])
        config = dict(stage='pilot', hidden=4, lr=.02, seed=2, epochs=3, patience=8, song_batch=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('reference', 'interrupted'):
                (root / name / 'models').mkdir(parents=True)
            torch.manual_seed(17)
            train(data, config, root / 'reference', root / 'reference/models', torch.device('cpu'), {})
            def interrupt_after_epoch(path, value):
                atomic_json(path, value)
                if path.name == 'status.json' and value.get('epoch') == 1:
                    raise KeyboardInterrupt('simulated stop after completed epoch')
            torch.manual_seed(17)
            with patch('tools.train_next.run.write_status', side_effect=interrupt_after_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    train(data, config, root / 'interrupted', root / 'interrupted/models', torch.device('cpu'), {})
            train(data, config, root / 'interrupted', root / 'interrupted/models', torch.device('cpu'), {}, resume=True)
            reference = torch.load(root / 'reference/models/last.pth', weights_only=True)
            resumed = torch.load(root / 'interrupted/models/last.pth', weights_only=True)
            for key, value in reference['state_dict'].items():
                torch.testing.assert_close(value, resumed['state_dict'][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
