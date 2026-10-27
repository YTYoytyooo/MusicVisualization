from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.train_next.core import SequenceVA, atomic_json, fit_scaler
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import BASE, ROOT, process_identity
from tools.train_next_v2.data import StepSampler
from tools.train_next_v2.evaluation import evaluate, metrics_for_axis, assert_metrics_match
from tools.train_next_v2.model import SequenceModel, derivative_loss
from tools.train_next_v2.protocol import split_groups, experiment_matrix
from tools.train_next_v2.run import train_loop
from tools.train_next_v2.status import snapshot, recent_progress


SETTINGS = dict(hidden=4, layers=1, dropout=.15, fusion='concatenate', separate_heads=False,
    lr=3e-4, weight_decay=1e-4, ccc_weight=.2, derivative_weight=.05,
    sampling_alpha=.5, batch_size=3, max_steps=6, evaluate_every=2,
    checkpoint_every=2, seed=105, threads=1, clip_gradient=1.)


def songs_fixture():
    rng = np.random.default_rng(67)
    result = dict(fit=[], development=[])
    for role, n in [('fit', 4), ('development', 2)]:
        for domain in ('deam', 'pmemo', 'memo'):
            for i in range(n):
                t = np.arange(7 + i) * .5
                x = rng.normal(size=(len(t), 3)).astype(np.float32)
                y = np.stack((.3 * np.sin(t + i), .2 * np.cos(t + i)), axis=1).astype(np.float32)
                mask = np.ones(y.shape, bool)
                mask[2, 0] = False
                native = .5
                if domain == 'memo':
                    mask[1::2] = False
                    native = 1.
                if role == 'development' and i == 0:
                    mask[:, 1] = False
                sid = f'{role}-{domain}-{i}'
                result[role].append(dict(song_id=sid, group_id=sid, source_dataset=domain,
                    x=x, y=y, mask=mask, times=t, native_seconds=native))
    return result


def seed_all():
    torch.manual_seed(105)
    random.seed(105)
    np.random.seed(105)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        directory = BASE / 'tmp'
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='experiment-test-', dir=directory)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(ROOT))

    def tearDown(self):
        # Only this freshly created workspace-contained test directory is removed.
        self.assertTrue(self.root.is_relative_to((BASE / 'tmp').resolve()))
        self.temporary.cleanup()

    def test_group_split_is_deterministic_and_prevents_transitive_leakage(self):
        records = [dict(song_id=f'{d}-{i}', group_id=f'{d}-{i//2}', audio_sha256=f'audio-{d}-{i}',
                        source_dataset=d, split='train') for d in ('a', 'b', 'c') for i in range(100)]
        records[4]['audio_sha256'] = records[2]['audio_sha256']
        records[104]['audio_sha256'] = records[4]['audio_sha256']
        first = split_groups(records)
        self.assertEqual(first, split_groups(list(reversed(records))))
        assignment = first['assignments']
        for field in ('group_id', 'audio_sha256'):
            memberships = defaultdict(set)
            for row in records:
                memberships[row[field]].add(assignment[row['song_id']])
            self.assertTrue(all(len(roles) == 1 for roles in memberships.values()))
        self.assertEqual(len({assignment[s] for s in ('a-2', 'a-3', 'a-4', 'a-5', 'b-4', 'b-5')}), 1)
        self.assertEqual(set(assignment.values()), {'fit', 'development', 'calibration'})
        self.assertEqual(sum(sum(c.values()) for c in first['counts'].values()), 300)
        with self.assertRaisesRegex(ValueError, 'original training'):
            split_groups([{**records[0], 'split': 'test'}])

    def test_experiment_matrix_changes_one_factor(self):
        matrix = experiment_matrix()
        self.assertEqual(len(matrix), 8)
        self.assertEqual(matrix[0]['overrides'], {})
        self.assertTrue(all(len(row['overrides']) == 1 for row in matrix[1:]))

    def test_derivative_uses_native_time_and_independent_masks(self):
        for cadence in (.5, 1.):
            t = torch.arange(7).float() * .5
            target = torch.zeros(1, 7, 2)
            prediction = torch.zeros_like(target)
            prediction[0, :, 0] = 2 * t
            target[0, :, 0] = t
            mask = torch.zeros_like(target, dtype=torch.bool)
            mask[0, ::int(cadence / .5), 0] = True
            # A is completely missing, not treated as a zero target.
            prediction[0, :, 1] = 1e6 * t
            value = derivative_loss(prediction, target, mask, [cadence])
            self.assertAlmostEqual(float(value), .9, places=6)

    def test_derivative_does_not_bridge_missing_labels(self):
        prediction = torch.zeros(1, 6, 2, requires_grad=True)
        with torch.no_grad():
            prediction[0, 4:, 0] = 10
        target = torch.zeros_like(prediction)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[0, [0, 1, 4, 5], 0] = True
        value = derivative_loss(prediction, target, mask, [.5])
        self.assertEqual(float(value.detach()), 0.)
        value.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        mask.zero_()
        self.assertEqual(float(derivative_loss(prediction, target, mask, [.5]).detach()), 0.)

    def test_reference_matches_old_architecture(self):
        old = SequenceVA(3, hidden=4).eval()
        model = SequenceModel(3, SETTINGS).eval()
        model.load_state_dict(old.state_dict(), strict=True)
        values, lengths = torch.randn(2, 9, 3), torch.tensor([9, 4])
        with torch.inference_mode():
            self.assertTrue(torch.equal(old(values, lengths), model(values, lengths)))

    def test_alternative_heads_respect_whole_song_padding(self):
        for override in (dict(fusion='branches'), dict(separate_heads=True)):
            model = SequenceModel(531, {**SETTINGS, **override}).eval()
            values, lengths = torch.randn(2, 8, 531), torch.tensor([8, 5])
            with torch.inference_mode():
                batched = model(values, lengths)
                alone = model(values[1:2, :5], torch.tensor([5]))
            self.assertEqual(batched.shape, (2, 8, 2))
            torch.testing.assert_close(batched[1, :5], alone[0], rtol=1e-5, atol=1e-6)

    def test_step_sampler_resume_and_probabilities(self):
        songs = [dict(source_dataset=d) for d, n in [('a', 1), ('b', 4), ('c', 9)] for _ in range(n)]
        for alpha, expected in [(0., [1/3]*3), (.5, [1/6, 2/6, 3/6]), (1., [1/14, 4/14, 9/14])]:
            sampler = StepSampler(songs, alpha, 7)
            np.testing.assert_allclose(sampler.probabilities, expected)
            original = [sampler.indices(step, 100) for step in range(1, 8)]
            restored = StepSampler(songs, alpha, 7)
            self.assertEqual(restored.indices(7, 100), original[-1])

    def test_metrics_mask_constants_and_reload(self):
        songs = songs_fixture()
        song = songs['development'][0]
        exact = metrics_for_axis(song['y'], song, 0)
        self.assertAlmostEqual(exact['ccc'], 1.)
        self.assertEqual(exact['mae'], 0.)
        self.assertEqual(exact['derivative_mae'], 0.)
        self.assertIsNone(metrics_for_axis(song['y'], song, 1))
        mean, std = fit_scaler(songs['fit'])
        model = SequenceModel(3, SETTINGS)
        output = evaluate(model, songs['development'], mean, std, torch.device('cpu'))
        self.assertEqual(assert_metrics_match(output, output), 0.)

    def test_full_optimizer_resume_is_identical_and_provenance_strict(self):
        songs, provenance = songs_fixture(), dict(data='fixture-v1', code='frozen-fixture')
        dirs = {}
        for label in ('whole', 'resumed'):
            dirs[label] = (self.root / label / 'run', self.root / label / 'model')
            for directory in dirs[label]:
                directory.mkdir(parents=True)
        seed_all()
        full = train_loop(songs, SETTINGS, provenance, *dirs['whole'], torch.device('cpu'))
        def interrupt(step):
            if step == 2:
                raise InterruptedError('simulated process exit after full checkpoint')
        seed_all()
        with self.assertRaises(InterruptedError):
            train_loop(songs, SETTINGS, provenance, *dirs['resumed'], torch.device('cpu'), after_checkpoint=interrupt)
        before = (dirs['resumed'][1] / 'last.pth').read_bytes()
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            train_loop(songs, SETTINGS, {**provenance, 'code': 'changed'}, *dirs['resumed'], torch.device('cpu'), resume=True)
        self.assertEqual(before, (dirs['resumed'][1] / 'last.pth').read_bytes())
        seed_all()
        resumed = train_loop(songs, SETTINGS, provenance, *dirs['resumed'], torch.device('cpu'), resume=True)
        self.assertEqual(full['best_step'], resumed['best_step'])
        self.assertEqual(full['best_development_score'], resumed['best_development_score'])
        checkpoints = [torch.load(dirs[label][1] / 'last.pth', weights_only=True) for label in ('whole', 'resumed')]
        for key, value in checkpoints[0]['state_dict'].items():
            self.assertTrue(torch.equal(value, checkpoints[1]['state_dict'][key]), key)
        for key, state in checkpoints[0]['optimizer']['state'].items():
            for name, value in state.items():
                self.assertTrue(torch.equal(value, checkpoints[1]['optimizer']['state'][key][name]), name)
        self.assertEqual(assert_metrics_match(full['best_development'], resumed['best_development']), 0.)

    def test_status_uses_os_lock_and_creation_identity_not_stale_json(self):
        run_dir, model_dir = self.root / 'run', self.root / 'model'
        run_dir.mkdir()
        model_dir.mkdir()
        identity = process_identity()
        self.assertEqual(identity['pid'], os.getpid())
        atomic_json(run_dir / 'process.json', dict(process_identity=identity))
        atomic_json(run_dir / 'status.json', dict(stage='training', step=50))
        log = run_dir / 'console.log'
        log.write_text('x' * 70000 + '\n' + json.dumps(dict(stage='training', step=123)) + '\nwarning\n', encoding='utf-8')
        self.assertEqual(recent_progress(log)['step'], 123)
        run_id = 'next-va-v2-unittest-status'
        with patch('tools.train_next_v2.status.paths', return_value=(run_dir, model_dir)):
            stopped = snapshot(run_id)
            self.assertFalse(stopped['running'])
            self.assertEqual(stopped['outcome'], 'stopped_without_completion')
            with run_lease(run_id):
                active = snapshot(run_id)
                self.assertTrue(active['running'])
                self.assertTrue(active['process_identity_matches'])
            self.assertFalse(snapshot(run_id)['running'])


if __name__ == '__main__':
    unittest.main()
