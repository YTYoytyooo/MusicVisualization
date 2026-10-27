"""Offline cross-runtime motion/visual contract evidence; no inference or rendering.

Run from the repository root with --out <workspace evidence directory>, then
run check_motion_parity.cjs against the generated parity-fixture.json.
Synthetic values below are engineering fixtures, never emotional annotations.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.train_next_v2.v3_workbench.mcts import VisualState
from tools.train_next_v2.v3_workbench.motion_renderer import FlowParticleSystem
from tools.train_next_v2.v3_workbench.motion_schema import mode_params
from tools.train_next_v2.v3_workbench.studio.motion import apply_overrides, sample_frame
from tools.train_next_v2.v3_workbench.studio.pipeline import visual_at
from tools.train_next_v2.v3_workbench.studio.smoothing import emotion_values
from tools.train_next_v2.v3_workbench.studio.store import (
    atomic_json, digest, load_project, load_raw, load_revision, object_digest, read_json)


def queries(times):
    return sorted(set(times) | {(a + b) // 2 for a, b in zip(times, times[1:])})


def frame(mode, **params):
    return dict(mode=mode, source='auto', reason='engineering fixture',
                components=[dict(mode=mode, weight=1., params=mode_params(mode, params))])


def motion_case(name, plan, config, duration):
    before = object_digest(plan)
    result = apply_overrides(plan, config, duration)
    times = queries(result['times_us']) if result['times_us'] else []
    expected = [sample_frame(result, t) for t in times]
    assert object_digest(plan) == before, 'sampling mutated its source'
    for f in expected:
        assert abs(sum(c['weight'] for c in f['components']) - 1) < 1e-12
        assert all(c['params']['rotation'] in (-1, 1) for c in f['components'])
    return dict(name=name, plan=plan, config=config, duration_us=duration,
                times=times, expected_plan=result, expected=expected)


def verified_cache(path):
    value = read_json(path)
    assert value['sha256'] == object_digest({k: v for k, v in value.items() if k != 'sha256'})
    return value


def renderer_checks():
    # Test actual state integration without drawing frames or running an encoder.
    state = VisualState(particle_count=64)
    sequence = [frame('orbit', rotation=1 if i < 90 else -1, speed=.5)
                for i in range(240)]
    systems = [FlowParticleSystem(640, 360, max_particles=64, rng_seed=42) for _ in range(2)]
    max_ratio = 0.
    checked = 0
    respawns = 0
    for f in sequence:
        p = systems[0]
        velocity, generation, active = p.vel.copy(), p.generation.copy(), p.active.copy()
        for system in systems:
            system.update(state, f, 1 / 30)
            assert np.isfinite(system.pos).all() and np.isfinite(system.vel).all()
        surviving = active & p.active & (generation == p.generation)
        respawns += int(np.count_nonzero(active & (generation != p.generation)))
        delta = np.linalg.norm(p.vel[surviving] - velocity[surviving], axis=1)
        if len(delta):
            max_ratio = max(max_ratio, float(delta.max() / (2.5 * p.short_side / 30)))
            assert max_ratio <= 1 + 1e-10, 'surviving particle acceleration exceeded bound'
            checked += len(delta)
        for key in ('pos', 'vel', 'generation', 'color'):
            assert np.array_equal(getattr(systems[0], key), getattr(systems[1], key)), key
    assert checked > 10000 and max_ratio > .95, 'fixture must exercise the acceleration limiter'
    return dict(frames=240, fps=30, particles=64, seed=42,
                surviving_particle_steps=checked, respawns_excluded=respawns,
                max_acceleration_limit_ratio=max_ratio, deterministic_replay=True,
                limitation='Numeric integration only; no claim about perceived smoothness.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    output = args.out.resolve()
    assert output.is_relative_to(ROOT), 'Evidence must remain in Vm'
    started = time.perf_counter()
    inputs, motion, visual = {}, [], []
    works = ROOT / 'data/v3.0/works'
    for pid in ('p-51315e7a7fa2', 'p-25a57aae23c0'):
        path = works / pid
        revision = load_revision(path)
        revision_file = path / 'revisions' / revision['id'] / 'revision.json'
        inputs[str(revision_file.relative_to(ROOT))] = digest(revision_file)
        # The second project's current revision uses legacy. Its cached B01
        # flow plan belongs to the verified historical r000001, not r000002.
        source_revision = revision if pid == 'p-51315e7a7fa2' else load_revision(path, 'r000001')
        source_file = path / 'revisions' / source_revision['id'] / 'revision.json'
        inputs[str(source_file.relative_to(ROOT))] = digest(source_file)
        edits = source_revision['edits']
        candidates = []
        for cache in sorted((path / 'motion_plans').glob('*.json')):
            saved = verified_cache(cache)
            identity = saved['identity']
            assert saved['key'] == object_digest(identity) == cache.stem
            p = load_project(path, identity['source_system'])
            if (p['raw_sha256'] != identity['raw_sha256']
                    or identity['source_system'] != source_revision['source_system']):
                continue
            raw = load_raw(path, system=identity['source_system'])
            values = emotion_values(raw, np.array(identity['times_us']), edits,
                                    p['step_us'], source_revision['smoothing'])
            if object_digest(values.tolist()) == identity['effective_values_sha256']:
                candidates.append((cache, saved, p))
        assert len(candidates) == 1, f'{pid}: ambiguous or missing source-matched cache'
        cache, saved, p = candidates[0]
        inputs[str(cache.relative_to(ROOT))] = digest(cache)
        raw_file = path / p['systems'][p['source_system']]['file']
        inputs[str(raw_file.relative_to(ROOT))] = digest(raw_file)
        plan = dict(times_us=saved['identity']['times_us'], auto=saved['auto'])
        config = dict(saved['identity']['config'], overrides=[])
        motion.append(motion_case(pid + ':cached-auto', plan, config, p['duration_us']))
        if pid == 'p-51315e7a7fa2':
            motion.append(motion_case(pid + ':saved-manual', plan, revision['motion'], p['duration_us']))
        else:
            motion.append(motion_case(pid + ':saved-legacy', plan, revision['motion'], p['duration_us']))
    synthetic = dict(times_us=[0, 4000000], auto=[
        frame('orbit', direction_deg=359, rotation=1), frame('orbit', direction_deg=1, rotation=-1)])
    mid = sample_frame(synthetic, 2000000, 'auto')['components'][0]['params']
    assert abs(mid['direction_deg']) < 1e-12 and mid['rotation'] == 1
    assert sample_frame(synthetic, 2000001, 'auto')['components'][0]['params']['rotation'] == -1
    motion.append(motion_case('engineering:short-arc-and-discrete-rotation', synthetic,
                             dict(engine='flow-v1', overrides=[]), 4000000))
    for easing in ('linear', 'smoothstep', 'smootherstep'):
        for width in (0, 500000):
            config = dict(engine='flow-v1', overrides=[dict(
                id='boundary', mode='wave', start_us=1000000, end_us=3000000,
                transition_in_us=width, transition_out_us=width,
                interpolation=easing, params={}, note='engineering fixture')])
            motion.append(motion_case(f'engineering:{easing}:{width}', synthetic, config, 4000000))
    visual_file = works / 'p-51315e7a7fa2/plans/c7d7e0484f21ace930dd1b5bbbf40dc3724782eaaa0f0103202abcb2bdf73117/plan.json'
    saved = verified_cache(visual_file)
    inputs[str(visual_file.relative_to(ROOT))] = digest(visual_file)
    p = load_project(works / 'p-51315e7a7fa2', saved['source_system'])
    assert saved['raw_sha256'] == p['raw_sha256'] and saved['model_sha256'] == p['model_sha256']
    special = dict(times_us=[0, 4000000], states=[
        asdict(VisualState(hue_base=359, particle_count=200, trail_length=20)),
        asdict(VisualState(hue_base=1, particle_count=202, trail_length=22))])
    for name, plan in [('cached-visual', saved), ('engineering:visual-short-arc-rounding', special)]:
        times = queries(plan['times_us'])
        if name.startswith('engineering:'):
            times = sorted(set(times) | {999999, 1000000, 1000001, 2999999, 3000000, 3000001})
        states = [VisualState(**s) for s in plan['states']]
        expected = visual_at(states, np.array(plan['times_us']), np.array(times), []).tolist()
        visual.append(dict(name=name, plan=plan, times=times, edits=[], expected=expected))
    fixture = dict(schema=1, purpose='engineering verification, not training data',
                   tolerance=1e-9, motion=motion, visual=visual, input_sha256=inputs)
    for file, value in inputs.items():
        assert digest(ROOT / file) == value
    integration = renderer_checks()
    atomic_json(output / 'parity-fixture.json', fixture)
    atomic_json(output / 'python-checks.json', dict(status='passed',
                elapsed_seconds=time.perf_counter() - started, motion_cases=len(motion),
                motion_samples=sum(len(c['times']) for c in motion), visual_cases=len(visual),
                visual_samples=sum(len(c['times']) for c in visual), renderer=integration,
                input_sha256=inputs))
    print(f'PASS: {len(motion)} motion cases, {len(visual)} visual cases; deterministic bounded particle integration')


if __name__ == '__main__':
    main()
