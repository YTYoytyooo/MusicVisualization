"""Deterministic particle-local motion fields, independent of the legacy engine.

Positions/velocities are pixels and pixels/second internally. Public speed and
radius use the canvas short side; all integration and trails use real seconds.
No full-screen vector field is constructed. Rotation +1 is screen-clockwise.
"""
from collections import deque
from copy import deepcopy
import math

import cv2
import numpy as np

from motion_schema import MODE_LABELS, MOTION_DEFAULTS, mode_params

MODES = tuple(MODE_LABELS)
TAU = 2. * np.pi


def normalize_motion(motion):
    """Validate a frame, fill preset params, normalize only roundoff in weights."""
    if not isinstance(motion, dict):
        raise ValueError('Motion frame must be an object')
    components = motion.get('components')
    if not isinstance(components, list) or not 1 <= len(components) <= 3:
        raise ValueError('A motion frame needs one to three components')
    result = []
    for component in components:
        if not isinstance(component, dict) or set(component) - {'mode', 'weight', 'params'}:
            raise ValueError('Invalid motion component')
        mode, weight = component.get('mode'), component.get('weight')
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not math.isfinite(weight) or weight < 0):
            raise ValueError('Motion weights must be finite and nonnegative')
        result.append({'mode': mode, 'weight': float(weight),
                       'params': mode_params(mode, component.get('params'))})
    total = sum(c['weight'] for c in result)
    if abs(total - 1.) > 1e-6:
        raise ValueError('Motion component weights must sum to one')
    for component in result:
        component['weight'] /= total
    mode = motion.get('mode', max(result, key=lambda c: c['weight'])['mode'])
    if mode not in MODES:
        raise ValueError('Unknown dominant motion mode')
    source = motion.get('source', 'auto')
    if source not in ('auto', 'manual', 'transition'):
        raise ValueError('Unknown motion source')
    return {'mode': mode, 'source': source, 'reason': str(motion.get('reason', ''))[:2000],
            'components': result}


def _unit(vectors, fallback=None):
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    result = vectors / np.maximum(lengths, 1e-12)
    zero = lengths[:, 0] < 1e-12
    if zero.any():
        result[zero] = (1., 0.) if fallback is None else fallback[zero]
    return result


class FlowParticleSystem:
    VERSION = 1
    ARRAY_FIELDS = ('pos', 'vel', 'heading', 'age', 'life', 'active', 'generation',
                    'radius_factors', 'phase_offsets', 'hue_offsets', 'sizes', 'color', 'birth_mode')

    def __init__(self, width=1280, height=720, max_particles=500, rng_seed=None, fps=30):
        if width < 2 or height < 2 or max_particles < 1 or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid flow canvas/particle count/fps')
        self.W, self.H, self.max_p, self.fps = int(width), int(height), int(max_particles), float(fps)
        self.short_side = float(min(width, height))
        self._rng = np.random.default_rng(rng_seed)
        self.pos = np.zeros((self.max_p, 2), np.float64)
        self.vel = np.zeros((self.max_p, 2), np.float64)
        self.heading = np.zeros(self.max_p, np.float64)
        self.age = np.zeros(self.max_p, np.float64)
        self.life = np.ones(self.max_p, np.float64)
        self.active = np.zeros(self.max_p, np.bool_)
        self.generation = np.zeros(self.max_p, np.int64)
        # Immutable particle traits: radius stays tied to this particle, never
        # sampled again each frame; no flickering circle sizes or hue offsets.
        self.radius_factors = self._rng.uniform(.9, 1.1, self.max_p)
        self.phase_offsets = self._rng.uniform(0, TAU, self.max_p)
        self.hue_offsets = self._rng.uniform(-.5, .5, self.max_p)
        self.sizes = self._rng.uniform(.8, 1.2, self.max_p)
        self.color = np.zeros((self.max_p, 3), np.uint8)
        self.birth_mode = np.zeros(self.max_p, np.int16)
        self._n_active = 0
        self.time = 0.
        self._started = False
        self._last_motion = None
        self.trail_history = deque()

    def _center(self, params):
        return np.array([params['center_x'] * self.W, params['center_y'] * self.H])

    def _spawn(self, indexes, motion, initial=False):
        indexes = np.asarray(indexes, dtype=np.int64)
        if not len(indexes):
            return
        components = motion['components']
        chosen = self._rng.choice(len(components), len(indexes), p=[c['weight'] for c in components])
        for component_index, component in enumerate(components):
            ids = indexes[chosen == component_index]
            if not len(ids):
                continue
            mode, params = component['mode'], component['params']
            self.birth_mode[ids] = MODES.index(mode)
            center = self._center(params)
            radius = params['radius'] * self.short_side * self.radius_factors[ids]
            radial_sign = params['radial'] or (1 if mode == 'expand' else -1 if mode == 'gather' else 0)
            angles = self._rng.uniform(0, TAU, len(ids))
            radial = np.column_stack([np.cos(angles), np.sin(angles)])
            if mode == 'orbit':
                self.pos[ids] = center + radial * radius[:, None]
            elif mode in ('gather', 'expand', 'spiral') and radial_sign < 0:
                self.pos[ids] = center + radial * radius[:, None] * self._rng.uniform(.85, 1.15, (len(ids), 1))
            elif mode in ('expand', 'gather', 'spiral'):
                self.pos[ids] = center + radial * self._rng.uniform(.015, .05, (len(ids), 1)) * self.short_side
            elif initial and mode != 'meteor' or mode == 'turbulent':
                self.pos[ids] = self._rng.random((len(ids), 2)) * (self.W, self.H)
            else:
                direction = np.deg2rad(params['direction_deg'])
                dx, dy = np.cos(direction), np.sin(direction)
                # Entry edges weighted by flux; horizontal or vertical motion
                # never spawns on the downstream edge and never wraps a trail.
                x_flux, y_flux = abs(dx) * self.H, abs(dy) * self.W
                x_edge = self._rng.random(len(ids)) < x_flux / max(1e-12, x_flux + y_flux)
                positions = self._rng.random((len(ids), 2)) * (self.W, self.H)
                positions[x_edge, 0] = .5 if dx >= 0 else self.W - .5
                positions[~x_edge, 1] = .5 if dy >= 0 else self.H - .5
                self.pos[ids] = positions
            self.life[ids] = self._rng.uniform(1.5, 3.2, len(ids)) if mode == 'meteor' else self._rng.uniform(12., 22., len(ids))
            self.age[ids] = 0.
            self.vel[ids] = 0.
            self.heading[ids] = np.deg2rad(params['direction_deg'])
            self.generation[ids] += 1
            self.active[ids] = True

    def _field(self, mode, params, ids):
        positions = self.pos[ids]
        center = self._center(params)
        relative = positions - center
        distance = np.linalg.norm(relative, axis=1)
        fallback = np.column_stack([np.cos(self.phase_offsets[ids]), np.sin(self.phase_offsets[ids])])
        radial = _unit(relative, fallback)
        tangent = np.column_stack([-radial[:, 1], radial[:, 0]]) * params['rotation']
        direction = np.deg2rad(params['direction_deg'])
        forward = np.array([np.cos(direction), np.sin(direction)])
        speed = max(params['speed'] * self.short_side, 1e-9)
        if mode in ('rise', 'fall', 'meteor'):
            field = np.broadcast_to(forward, (len(ids), 2)).copy()
        elif mode == 'orbit':
            target_radius = params['radius'] * self.short_side * self.radius_factors[ids]
            # Strong radial restoration plus the small Euler curvature term.
            correction = ((target_radius - distance) * 4. - speed * speed / self.fps
                          / np.maximum(2 * distance, self.short_side * .02)) / speed
            field = tangent + radial * np.clip(correction, -2., 2.)[:, None]
        elif mode == 'spiral':
            field = tangent + radial * params['radial']
        elif mode in ('expand', 'gather'):
            sign = np.sign(params['radial']) or (1 if mode == 'expand' else -1)
            field = radial * sign
        elif mode == 'wave':
            normal = np.array([-forward[1], forward[0]])
            phase = positions @ forward / self.short_side * TAU * 1.4 - self.time * 1.8
            phase += self.phase_offsets[ids] * (1 - params['coherence'])
            field = forward[None, :] + np.sin(phase)[:, None] * normal[None, :] * .65
        else:
            x, y = positions[:, 0] / self.short_side, positions[:, 1] / self.short_side
            field = np.column_stack([np.sin(y * 7 + self.time * .8) + .45 * np.cos(x * 3 - self.time),
                                     -np.sin(x * 7 - self.time * .7) + .45 * np.cos(y * 3 + self.time)])
        # Deterministic small perturbations; even with turbulence=0 the selected
        # mode's primary velocity still exists. No random kick is needed.
        noise_amount = .35 * (1 - params['coherence']) + .3 * params['turbulence']
        phase = self.phase_offsets[ids] + self.time
        noise = np.column_stack([np.sin(phase * 1.13), np.cos(phase * .91)])
        return _unit(field + noise_amount * noise, fallback)

    def _targets(self, ids, motion, beat_impulse):
        # Blend headings relative to current travel direction, not vectors.
        # Opposing rise/fall vectors must turn through a half-circle, not cancel.
        reference = self.heading[ids]
        delta_sum = np.zeros(len(ids))
        target_speed = np.zeros(len(ids))
        coherence = 0.
        for component in motion['components']:
            p, weight = component['params'], component['weight']
            if weight == 0:
                continue
            field = self._field(component['mode'], p, ids)
            angles = np.arctan2(field[:, 1], field[:, 0])
            delta = (angles - reference + np.pi) % TAU - np.pi
            opposite = np.isclose(np.abs(delta), np.pi, atol=1e-10, rtol=0)
            delta[opposite] = np.pi * p['rotation']
            delta_sum += weight * delta
            target_speed += weight * p['speed'] * self.short_side * (1 + .3 * p['pulse'] * np.clip(beat_impulse, 0, 1))
            coherence += weight * p['coherence']
        return reference + delta_sum, target_speed, coherence

    def update(self, visual_state, motion, dt, beat_impulse=0.):
        if not np.isfinite(dt) or dt <= 0 or dt > 1:
            raise ValueError('Flow timestep must lie in (0, 1] seconds')
        motion = normalize_motion(motion)
        count = int(np.clip(visual_state.particle_count, 1, self.max_p))
        newborn = np.zeros(self.max_p, bool)
        if count > self._n_active:
            ids = np.arange(self._n_active, count)
            self._spawn(ids, motion, initial=not self._started)
            newborn[ids] = True
        self.active[count:] = False
        self._n_active = count
        ids = np.arange(count)
        headings, speeds, coherence = self._targets(ids, motion, beat_impulse)
        self.heading[ids[newborn[ids]]] = headings[newborn[ids]]
        response = 1 - np.exp(-dt / (.12 + .25 * (1 - coherence)))
        turn = (headings - self.heading[ids] + np.pi) % TAU - np.pi
        # At precisely pi choose a consistent direction; it avoids sign flips.
        turn[np.isclose(turn, -np.pi, atol=1e-10, rtol=0)] = np.pi
        self.heading[ids] += np.clip(turn * response, -TAU * dt, TAU * dt)
        previous_speed = np.linalg.norm(self.vel[ids], axis=1)
        speed_delta = np.clip((speeds - previous_speed) * response,
                              -1.5 * self.short_side * dt, 1.5 * self.short_side * dt)
        next_speed = np.maximum(0., previous_speed + speed_delta)
        desired = np.column_stack([np.cos(self.heading[ids]), np.sin(self.heading[ids])]) * next_speed[:, None]
        # Limit full acceleration as well as turn rate. Reconstruct speed after
        # heading changes so opposed components never reduce it algebraically.
        acceleration = desired - self.vel[ids]
        norms = np.linalg.norm(acceleration, axis=1)
        max_acceleration = 2.5 * self.short_side * dt
        acceleration *= np.minimum(1., max_acceleration / np.maximum(norms, 1e-12))[:, None]
        self.vel[ids] += acceleration
        moving = np.linalg.norm(self.vel[ids], axis=1) > 1e-9
        self.heading[ids[moving]] = np.arctan2(self.vel[ids[moving], 1], self.vel[ids[moving], 0])
        self.pos[ids] += self.vel[ids] * dt
        self.age[ids] += dt
        self.time += dt
        dominant = max(motion['components'], key=lambda c: c['weight'])
        mode, params = dominant['mode'], dominant['params']
        distance = np.linalg.norm(self.pos[ids] - self._center(params), axis=1)
        dead = self.age[ids] >= self.life[ids]
        if mode != 'orbit':
            dead |= (self.pos[ids, 0] < -2) | (self.pos[ids, 0] > self.W + 2)
            dead |= (self.pos[ids, 1] < -2) | (self.pos[ids, 1] > self.H + 2)
        radial_sign = params['radial'] or (1 if mode == 'expand' else -1 if mode == 'gather' else 0)
        inward = mode in ('expand', 'gather', 'spiral') and radial_sign < 0
        if inward:
            dead |= distance < self.short_side * .02
        if mode == 'spiral' and params['radial'] >= 0:
            dead |= distance > params['radius'] * self.short_side * 1.8
        if dead.any():
            self._spawn(ids[dead], motion)
        # HSV colors are stable per-particle, while all current visual controls
        # (including brightness/saturation) remain effective every frame.
        hsv = np.zeros((count, 1, 3), np.float32)
        hsv[:, 0, 0] = (visual_state.hue_base + self.hue_offsets[ids] * visual_state.hue_range) % 360
        hsv[:, 0, 1] = np.clip(visual_state.saturation, 0, 1)
        hsv[:, 0, 2] = np.clip(visual_state.brightness * 1.5, .1, 1)
        self.color[ids] = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0] * 255, 0, 255).astype(np.uint8)
        self._last_motion = motion
        self._started = True
        self.trail_history.append((self.time, self.pos.copy(), self.generation.copy(), self.active.copy()))
        while self.trail_history and self.time - self.trail_history[0][0] > 2. + 1e-9:
            self.trail_history.popleft()

    def trail_segments(self):
        """Yield age-faded segments, never connecting different birth generations."""
        if self._last_motion is None:
            return
        trail = sum(c['weight'] * c['params']['trail_seconds'] for c in self._last_motion['components'])
        if trail <= 0:
            return
        meteor = sum(c['weight'] for c in self._last_motion['components'] if c['mode'] == 'meteor')
        history = list(self.trail_history)
        for before, after in zip(history, history[1:]):
            age = self.time - after[0]
            if age > trail + 1e-9:
                continue
            valid = (before[3] & after[3] & self.active & (before[2] == after[2])
                     & (after[2] == self.generation))
            alpha = max(0., 1 - age / trail) ** 2 * (.28 + .37 * meteor)
            if alpha > 0:
                yield before[1], after[1], np.flatnonzero(valid), alpha, age

    def render(self, frame):
        layer = np.zeros_like(frame)
        for before, after, ids, alpha, _ in self.trail_segments():
            for i in ids:
                a, b = tuple(np.rint(before[i]).astype(int)), tuple(np.rint(after[i]).astype(int))
                cv2.line(layer, a, b, tuple(int(c * alpha) for c in self.color[i]), 1, cv2.LINE_AA)
        cv2.add(frame, layer, dst=frame)
        for i in np.flatnonzero(self.active):
            point = tuple(np.rint(self.pos[i]).astype(int))
            if not (0 <= point[0] < self.W and 0 <= point[1] < self.H):
                continue
            fade = min(1., self.age[i] / .12, max(0., (self.life[i] - self.age[i]) / .3))
            radius = max(1, int(round(1.7 * self.short_side / 720 * self.sizes[i])))
            cv2.circle(frame, point, radius, tuple(int(c * fade) for c in self.color[i]), -1, cv2.LINE_AA)

    def export_state(self):
        arrays = {name: getattr(self, name).copy() for name in self.ARRAY_FIELDS}
        history = list(self.trail_history)
        arrays['trail_times'] = np.asarray([item[0] for item in history], np.float64)
        arrays['trail_pos'] = np.stack([item[1] for item in history]) if history else np.empty((0, self.max_p, 2), np.float64)
        arrays['trail_generation'] = np.stack([item[2] for item in history]) if history else np.empty((0, self.max_p), np.int64)
        arrays['trail_active'] = np.stack([item[3] for item in history]) if history else np.empty((0, self.max_p), np.bool_)
        return {'metadata': {'version': self.VERSION, 'width': self.W, 'height': self.H,
                             'max_particles': self.max_p, 'fps': self.fps, 'time': self.time,
                             'n_active': self._n_active, 'started': self._started,
                             'last_motion': deepcopy(self._last_motion),
                             'rng_state': deepcopy(self._rng.bit_generator.state)}, 'arrays': arrays}

    @classmethod
    def from_state(cls, snapshot, width, height, fps, max_particles=500):
        """Create a fully validated replacement; no existing state is mutated."""
        metadata, arrays = snapshot['metadata'], snapshot['arrays']
        if (metadata.get('version') != cls.VERSION or metadata.get('width') != width
                or metadata.get('height') != height or metadata.get('fps') != fps
                or metadata.get('max_particles') != max_particles):
            raise ValueError('Flow checkpoint configuration mismatch')
        n = metadata.get('n_active')
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= max_particles:
            raise ValueError('Invalid flow active count')
        time = metadata.get('time')
        if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0:
            raise ValueError('Invalid flow checkpoint time')
        if not isinstance(metadata.get('started'), bool):
            raise ValueError('Invalid flow started flag')
        replacement = cls(width, height, max_particles=max_particles, rng_seed=0, fps=fps)
        last_motion = normalize_motion(metadata['last_motion']) if metadata.get('last_motion') is not None else None
        if metadata['started'] != (last_motion is not None):
            raise ValueError('Flow checkpoint motion is incomplete')
        expected = set(cls.ARRAY_FIELDS) | {'trail_times', 'trail_pos', 'trail_generation', 'trail_active'}
        if set(arrays) != expected:
            raise ValueError('Flow checkpoint arrays are incomplete')
        for name in cls.ARRAY_FIELDS:
            value, template = arrays[name], getattr(replacement, name)
            if (not isinstance(value, np.ndarray) or value.shape != template.shape
                    or value.dtype != template.dtype or not np.isfinite(value).all()):
                raise ValueError(f'Invalid flow checkpoint array {name}')
        times = arrays['trail_times']
        if (not isinstance(times, np.ndarray) or times.dtype != np.float64 or times.ndim != 1
                or not np.isfinite(times).all() or len(times) > math.ceil(2 * fps) + 2
                or (np.diff(times) <= 0).any() or (times > time + 1e-9).any()
                or (times < max(0., time - 2.) - 1e-9).any()):
            raise ValueError('Invalid flow trail timestamps')
        for name, shape, dtype in (
                ('trail_pos', (len(times), max_particles, 2), np.float64),
                ('trail_generation', (len(times), max_particles), np.int64),
                ('trail_active', (len(times), max_particles), np.bool_)):
            value = arrays[name]
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f'Invalid flow checkpoint {name}')
        if (not arrays['active'][:n].all() or arrays['active'][n:].any()
                or (arrays['age'] < 0).any() or (arrays['life'] <= 0).any()
                or (arrays['generation'] < 0).any() or (arrays['birth_mode'] < 0).any()
                or (arrays['birth_mode'] >= len(MODES)).any()
                or (arrays['radius_factors'] < .9).any() or (arrays['radius_factors'] > 1.1).any()):
            raise ValueError('Invalid flow lifecycle or radius state')
        try:
            replacement._rng.bit_generator.state = deepcopy(metadata['rng_state'])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError('Invalid flow RNG checkpoint') from exc
        for name in cls.ARRAY_FIELDS:
            setattr(replacement, name, arrays[name].copy())
        replacement.trail_history = deque((float(t), arrays['trail_pos'][i].copy(),
            arrays['trail_generation'][i].copy(), arrays['trail_active'][i].copy()) for i, t in enumerate(times))
        replacement.time = float(time)
        replacement._n_active = n
        replacement._started = metadata['started']
        replacement._last_motion = last_motion
        return replacement
