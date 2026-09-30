"""Small deterministic contracts for flow-v1, without ML/audio/network work."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mcts import VisualState
from motion_renderer import FlowParticleSystem, MODES, normalize_motion
from motion_schema import mode_params
from renderer import VideoRenderer
from studio.checkpoints import load_checkpoint, save_checkpoint


def motion(mode, **params):
    return {'mode': mode, 'source': 'manual', 'reason': 'test',
            'components': [{'mode': mode, 'weight': 1., 'params': mode_params(mode, params)}]}


def blend(parts):
    return {'mode': max(parts, key=lambda item: item[1])[0], 'source': 'transition', 'reason': 'test',
            'components': [{'mode': mode, 'weight': weight,
                            'params': mode_params(mode, {'speed': .1, 'coherence': 1., 'turbulence': 0., 'pulse': 0.})}
                           for mode, weight in parts]}


class MotionEngineTests(unittest.TestCase):
    def state(self):
        return VisualState(particle_count=50, field_turbulence=0., trail_length=5)

    def test_direction_works_without_turbulence(self):
        for mode, sign in (('rise', -1), ('fall', 1)):
            with self.subTest(mode=mode):
                system = FlowParticleSystem(320, 180, max_particles=64, rng_seed=12)
                config = motion(mode, coherence=1., turbulence=0., pulse=0.)
                for _ in range(30):
                    system.update(self.state(), config, 1 / 30)
                self.assertGreater(sign * np.median(system.vel[:50, 1]), system.short_side * .04)
                self.assertLess(np.max(np.abs(system.vel[:50, 0])), 1e-8)

    def test_speed_is_short_side_per_second_not_per_frame(self):
        speeds = []
        for fps in (15, 30, 60):
            system = FlowParticleSystem(320, 180, max_particles=64, rng_seed=2, fps=fps)
            config = motion('rise', speed=.1, coherence=1., turbulence=0., pulse=0.)
            for _ in range(fps * 2):
                system.update(self.state(), config, 1 / fps)
            speeds.append(np.median(np.linalg.norm(system.vel[:50], axis=1)))
        np.testing.assert_allclose(speeds, [.1 * 180] * 3, atol=.01)

    def test_orbit_keeps_radius_and_rotation_sign(self):
        for rotation in (-1, 1):
            with self.subTest(rotation=rotation):
                system = FlowParticleSystem(320, 240, max_particles=64, rng_seed=42)
                config = motion('orbit', speed=.1, radius=.3, coherence=1., turbulence=0.,
                                rotation=rotation, pulse=0.)
                for _ in range(150):
                    system.update(self.state(), config, 1 / 30)
                relative = system.pos[:50] - (160, 120)
                actual = np.linalg.norm(relative, axis=1)
                target = system.radius_factors[:50] * .3 * 240
                self.assertLess(np.max(np.abs(actual - target) / target), .05)
                cross = relative[:, 0] * system.vel[:50, 1] - relative[:, 1] * system.vel[:50, 0]
                self.assertTrue((cross * rotation > 0).all())

    def test_expand_gather_and_signed_spiral(self):
        cases = [('expand', 1., 1), ('gather', -1., -1), ('spiral', .4, 1), ('spiral', -.4, -1)]
        for mode, radial, sign in cases:
            with self.subTest(mode=mode, radial=radial):
                system = FlowParticleSystem(320, 240, max_particles=64, rng_seed=14)
                config = motion(mode, speed=.08, radial=radial, coherence=1., turbulence=0., pulse=0.)
                for _ in range(30):
                    system.update(self.state(), config, 1 / 30)
                relative = system.pos[:50] - (160, 120)
                radial_speed = np.sum(relative * system.vel[:50], axis=1)
                self.assertGreater(sign * np.median(radial_speed), 0)

    def test_opposing_modes_turn_without_vector_cancellation(self):
        system = FlowParticleSystem(640, 480, max_particles=64, rng_seed=15)
        for _ in range(30):
            system.update(self.state(), blend([('rise', 1.)]), 1 / 30)
        system.pos[0] = (320, 240)
        system.life[:50] = 100
        speeds = []
        for i in range(61):
            before = system.vel[0].copy()
            weight = i / 60
            system.update(self.state(), blend([('rise', 1 - weight), ('fall', weight)]), 1 / 30)
            speeds.append(np.linalg.norm(system.vel[0]))
            self.assertLessEqual(np.linalg.norm(system.vel[0] - before), 2.5 * 480 / 30 + 1e-8)
        self.assertGreater(min(speeds), .085 * 480)
        self.assertGreater(system.vel[0, 1], .07 * 480)

    def test_three_components_and_all_modes_are_finite_deterministic(self):
        a = FlowParticleSystem(320, 180, max_particles=64, rng_seed=123)
        b = FlowParticleSystem(320, 180, max_particles=64, rng_seed=123)
        for config in [motion(mode) for mode in MODES] + [blend([('rise', .25), ('fall', .25), ('wave', .5)])]:
            for _ in range(3):
                a.update(self.state(), config, 1 / 30, beat_impulse=.7)
                b.update(self.state(), config, 1 / 30, beat_impulse=.7)
            self.assertTrue(np.isfinite(a.pos).all())
            np.testing.assert_array_equal(a.pos, b.pos)
            np.testing.assert_array_equal(a.vel, b.vel)
            np.testing.assert_array_equal(a.generation, b.generation)

    def test_tail_duration_fades_and_rebirth_breaks_segments(self):
        system = FlowParticleSystem(320, 240, max_particles=64, rng_seed=16)
        config = motion('meteor', trail_seconds=2., coherence=1., turbulence=0.)
        for _ in range(100):
            system.update(self.state(), config, 1 / 30)
        self.assertLessEqual(len(system.trail_history), 62)
        self.assertGreaterEqual(system.trail_history[0][0], system.time - 2. - 1e-9)
        segments = list(system.trail_segments())
        self.assertTrue(segments)
        self.assertTrue(all(age <= 2. + 1e-9 for *_, age in segments))
        alphas = [segment[3] for segment in segments]
        self.assertTrue(all(b >= a for a, b in zip(alphas, alphas[1:])))
        generation = system.generation[0]
        system.life[0] = .001
        system.update(self.state(), config, 1 / 30)
        self.assertGreater(system.generation[0], generation)
        self.assertTrue(all(0 not in ids for _, _, ids, _, _ in system.trail_segments()))

    def test_zero_speed_and_no_trail_are_explicit_supported_controls(self):
        system = FlowParticleSystem(320, 180, max_particles=64, rng_seed=12)
        config = motion('rise', speed=0., trail_seconds=0.)
        system.update(self.state(), config, 1 / 30)
        original = system.pos.copy()
        system.update(self.state(), config, 1 / 30)
        np.testing.assert_array_equal(system.pos, original)
        self.assertEqual(list(system.trail_segments()), [])

    def test_flow_snapshot_restores_identical_future_positions_and_canvas(self):
        original = FlowParticleSystem(320, 180, max_particles=64, rng_seed=19)
        for _ in range(40):
            original.update(self.state(), motion('meteor'), 1 / 30, .5)
        restored = FlowParticleSystem.from_state(original.export_state(), 320, 180, 30, max_particles=64)
        for i in range(8):
            config = blend([('meteor', 1 - i / 8), ('orbit', i / 8)])
            for system in (original, restored):
                system.update(self.state(), config, 1 / 30, .7)
            actual = np.zeros((180, 320, 3), np.uint8)
            expected = actual.copy()
            original.render(actual); restored.render(expected)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(original.pos, restored.pos)

    def test_renderer_legacy_none_and_old_checkpoint_remain_compatible(self):
        with tempfile.TemporaryDirectory() as folder:
            a = VideoRenderer(str(Path(folder) / 'a.avi'), width=160, height=120, rng_seed=20)
            b = VideoRenderer(str(Path(folder) / 'b.avi'), width=160, height=120, rng_seed=20)
            try:
                for i in range(3):
                    args = (self.state(), np.zeros(10), 0., i / 600, 0.)
                    np.testing.assert_array_equal(a.render_frame(*args), b.render_frame(*args, motion=None))
                self.assertIsNone(a.flow_particles)
                old = a.export_state()
                old['metadata'].pop('flow'); old['metadata'].pop('flow_seed')
                b.restore_state(old)
                self.assertIsNone(b.flow_particles)
            finally:
                a.release(); b.release()

    def test_renderer_flow_checkpoint_roundtrip_and_atomic_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            a = VideoRenderer(str(Path(folder) / 'a.avi'), width=160, height=120, rng_seed=21)
            b = VideoRenderer(str(Path(folder) / 'b.avi'), width=160, height=120, rng_seed=99)
            try:
                for i in range(6):
                    a.render_frame(self.state(), np.zeros(10), .5, i / 600, 0., motion=motion('orbit'))
                manifest = save_checkpoint(Path(folder) / 'checkpoints', 'flow-config', a)
                self.assertEqual(load_checkpoint(manifest, 'flow-config', b), 6)
                for i in range(6, 9):
                    args = (self.state(), np.zeros(10), .7, i / 600, 0.)
                    np.testing.assert_array_equal(a.render_frame(*args, motion=motion('spiral')),
                                                  b.render_frame(*args, motion=motion('spiral')))
                before = b.flow_particles.pos.copy()
                invalid = deepcopy(a.export_state())
                invalid['arrays']['flow_pos'][0, 0] = np.nan
                with self.assertRaises(ValueError):
                    b.restore_state(invalid)
                np.testing.assert_array_equal(b.flow_particles.pos, before)
            finally:
                a.release(); b.release()

    def test_motion_validation_rejects_bad_weights_or_unknown_params(self):
        bad = motion('rise')
        bad['components'][0]['weight'] = .5
        with self.assertRaises(ValueError):
            normalize_motion(bad)
        bad = motion('rise')
        bad['components'][0]['params']['unknown'] = 1
        with self.assertRaises(ValueError):
            normalize_motion(bad)


if __name__ == '__main__':
    unittest.main()
