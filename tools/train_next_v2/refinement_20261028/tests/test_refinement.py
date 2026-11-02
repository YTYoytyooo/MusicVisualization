from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.train_next_v2.model import SequenceModel, loss as old_loss
from tools.train_next_v2.run import train_loop as old_loop
from tools.train_next_v2.refinement_20261028.engine import train_loop
from tools.train_next_v2.refinement_20261028.model import build_model, observed_mean_loss, objective
from tools.train_next_v2.refinement_20261028 import run


def settings(**overrides):
    return dict(hidden=8, layers=1, dropout=.15, fusion='branches', separate_heads=False,
        lr=.0003, weight_decay=.0001, ccc_weight=.2, derivative_weight=0., sampling_alpha=0.,
        batch_size=2, max_steps=6, evaluate_every=3, checkpoint_every=2, seed=42, threads=1,
        clip_gradient=1., **{'architecture': 'branch_lstm', 'mean_weight': 0., **overrides})


def seed():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def synthetic_songs():
    rng = np.random.default_rng(17)
    result = {}
    for role in ('fit', 'development'):
        group = []
        for i, n in enumerate((5, 8, 11)):
            mask = np.ones((n, 2), bool)
            mask[1:3, 0] = False
            mask[::2, 1] = False
            group.append(dict(song_id=f'{role}-{i}', group_id=f'g{role}-{i}', source_dataset='synthetic',
                x=rng.normal(size=(n, 531)).astype(np.float32),
                y=rng.uniform(-.6, .6, (n, 2)).astype(np.float32), mask=mask,
                times=np.arange(n)*.5, native_seconds=.5))
        result[role] = group
    return result


class RefinementChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_control_initialization_forward_and_loss_are_identical(self):
        config = settings()
        seed(); old = SequenceModel(531, config)
        seed(); new = build_model(531, config)
        old.eval(); new.eval()
        x = torch.randn(2, 7, 531)
        lengths = torch.tensor([4, 7])
        torch.testing.assert_close(old(x, lengths), new(x, lengths), rtol=0, atol=0)
        target = torch.randn(2, 7, 2)
        mask = torch.rand(2, 7, 2) > .3
        p = new(x, lengths)
        torch.testing.assert_close(old_loss(p, target, mask, [.5, 1.], config),
                                   objective(p, target, mask, [.5, 1.], config), rtol=0, atol=0)

    def test_mean_loss_independent_masks_and_invalid_values(self):
        p = torch.tensor([[[.4, 99.], [.4, -.2], [99., -.2]]], requires_grad=True)
        y = torch.tensor([[[0., float('nan')], [0., 0.], [float('nan'), 0.]]])
        m = torch.tensor([[[True, False], [True, True], [False, True]]])
        value = observed_mean_loss(p, y, m)
        # SmoothL1(.4, beta=.2)=.3; SmoothL1(-.2)=.1; axes equally weighted.
        self.assertAlmostEqual(float(value.detach()), .2, places=6)
        value.backward()
        self.assertEqual(float(p.grad[0, 0, 1]), 0.)
        self.assertEqual(float(p.grad[0, 2, 0]), 0.)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_empty_mean_loss_is_differentiable_zero(self):
        p = torch.randn(2, 5, 2, requires_grad=True)
        value = observed_mean_loss(p, torch.full_like(p, float('nan')), torch.zeros_like(p, dtype=torch.bool))
        value.backward()
        self.assertEqual(float(value.detach()), 0.)
        self.assertEqual(float(p.grad.abs().sum()), 0.)

    def test_tone_model_bounds_centering_padding_and_gradient(self):
        seed(); model = build_model(531, settings(architecture='tone_residual'))
        model.eval()
        x = torch.randn(2, 9, 531)
        lengths = torch.tensor([4, 9])
        tone, residual = model.components(x, lengths)
        output = model(x, lengths)
        self.assertTrue((output.abs() <= 1.000001).all())
        for i, n in enumerate(lengths):
            torch.testing.assert_close(output[i, :n].mean(0), tone[i, 0], atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(residual[i, :n].mean(0), torch.zeros(2), atol=1e-7, rtol=0)
        changed = x.clone(); changed[0, 4:] = 10000
        torch.testing.assert_close(output[0, :4], model(changed, lengths)[0, :4], atol=0, rtol=0)
        torch.testing.assert_close(output[:1, :4], model(x[:1, :4], torch.tensor([4])), atol=1e-7, rtol=1e-6)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_new_control_training_matches_frozen_loop(self):
        with tempfile.TemporaryDirectory(dir=run.WINDOW) as temp:
            root = Path(temp)
            for name in ('old', 'new'):
                (root/name/'run').mkdir(parents=True); (root/name/'model').mkdir()
            config, songs, provenance = settings(), synthetic_songs(), {'synthetic_test': True}
            seed(); old = old_loop(songs, config, provenance, root/'old/run', root/'old/model', torch.device('cpu'))
            seed(); new = train_loop(songs, config, provenance, root/'new/run', root/'new/model', torch.device('cpu'))
            self.assertEqual(old['best_development'], new['best_development'])
            self.assertEqual(old['best_step'], new['best_step'])
            left = torch.load(root/'old/model/last.pth', weights_only=True)
            right = torch.load(root/'new/model/last.pth', weights_only=True)
            for key in left['state_dict']:
                torch.testing.assert_close(left['state_dict'][key], right['state_dict'][key], atol=0, rtol=0)
            torch.testing.assert_close(left['torch_rng'], right['torch_rng'], atol=0, rtol=0)

    def test_full_checkpoint_resume_equivalent_for_each_arm(self):
        for overrides in ({}, {'mean_weight': .1}, {'architecture': 'tone_residual'}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory(dir=run.WINDOW) as temp:
                root = Path(temp)
                for name in ('whole', 'resume'):
                    (root/name/'run').mkdir(parents=True); (root/name/'model').mkdir()
                config, songs, provenance = settings(**overrides), synthetic_songs(), {'test': True}
                seed(); whole = train_loop(songs, config, provenance, root/'whole/run', root/'whole/model', torch.device('cpu'))
                stopped = [False]
                def checkpoint(step):
                    if step == 2:
                        stopped[0] = True
                seed(); paused = train_loop(songs, config, provenance, root/'resume/run', root/'resume/model', torch.device('cpu'),
                    should_stop=lambda: stopped[0], after_checkpoint=checkpoint)
                self.assertEqual(paused['status'], 'paused_deadline')
                self.assertEqual(paused['completed_steps'], 2)
                torch.manual_seed(999)  # Resume must restore authoritative RNG, not caller seed.
                resumed = train_loop(songs, config, provenance, root/'resume/run', root/'resume/model', torch.device('cpu'), resume=True)
                self.assertEqual(whole['best_development'], resumed['best_development'])
                left = torch.load(root/'whole/model/last.pth', weights_only=True)
                right = torch.load(root/'resume/model/last.pth', weights_only=True)
                for key in left['state_dict']:
                    torch.testing.assert_close(left['state_dict'][key], right['state_dict'][key], atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, 'Resume code'):
                    train_loop(songs, config, {'changed': True}, root/'resume/run', root/'resume/model', torch.device('cpu'), resume=True)

    def test_expired_engine_never_evaluates_or_saves_weights(self):
        with tempfile.TemporaryDirectory(dir=run.WINDOW) as temp:
            root = Path(temp)
            (root/'run').mkdir(); (root/'model').mkdir()
            with patch('tools.train_next_v2.refinement_20261028.engine.evaluate', side_effect=AssertionError('No evaluation allowed')):
                result = train_loop(synthetic_songs(), settings(), {}, root/'run', root/'model', torch.device('cpu'), should_stop=lambda: True)
            self.assertEqual(result['completed_steps'], 0)
            self.assertEqual(list((root/'model').iterdir()), [])

    def test_time_boundary_and_budget_are_fail_closed(self):
        deadline = '2026-10-28T01:50:00+00:00'
        self.assertFalse(run.expired(deadline, datetime(2026, 10, 28, 1, 49, 59, tzinfo=timezone.utc)))
        self.assertTrue(run.expired(deadline, datetime(2026, 10, 28, 1, 50, tzinfo=timezone.utc)))
        with tempfile.TemporaryDirectory(dir=run.WINDOW) as temp, patch.object(run, 'WINDOW', Path(temp)):
            protocol = {'latest_new_training_utc': '2099-01-01T00:00:00+00:00', 'seeds': [20261003, 20261004], 'budget': {'max_runs': 7}}
            with self.assertRaisesRegex(ValueError, 'screening selection'):
                run.validate_budget(protocol, 'mean_loss', 20261004)
            for i in range(7):
                directory = Path(temp)/'runs'/str(i); directory.mkdir(parents=True)
                (directory/'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'budget exhausted'):
                run.validate_budget(protocol, 'control', 20261003)


if __name__ == '__main__':
    unittest.main()
