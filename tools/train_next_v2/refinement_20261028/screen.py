"""Apply preregistered first-seed gates; never train, evaluate holdouts, or promote.

The utility reads existing reports and CPU checkpoints only. Training sources and
protocols remain immutable. Partial results cannot authorize repeat seeds.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tools.train_next.core import atomic_json, sha256
from tools.train_next_v2.common import ROOT, read_json, now, verify_hashes
from .observe import observe
from .run import WINDOW, PROTOCOL, locations

DOMAINS = ('deam', 'memo2496', 'pmemo')
AXES = ('valence', 'arousal')
GATED = ('ccc', 'mae', 'centered_rmse', 'derivative_mae', 'dynamics_beats_constant')
DISPLAY = GATED + ('rmse', 'mean_absolute_bias', 'std_ratio', 'constant_prediction_mae', 'prediction_span90')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_metrics(metrics):
    if set(metrics) != set(DOMAINS):
        raise ValueError('All three frozen development domains are required')
    all_ids = set()
    for domain in DOMAINS:
        group = metrics[domain]
        ids = [row['song_id'] for row in group['per_song']]
        if not ids or len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError('Empty or duplicate development song IDs')
        all_ids.update(ids)
        for axis in AXES:
            entries = [row[axis] for row in group['per_song'] if row[axis] is not None]
            macro = group['macro'][axis]
            if not entries or macro['songs'] != len(entries):
                raise ValueError('Axis support count changed')
            if macro['ccc_songs'] != sum(row['ccc'] is not None for row in entries):
                raise ValueError('CCC support count changed')
            for metric in entries[0]:
                if metric == 'values':
                    continue
                values = [row[metric] for row in entries if row[metric] is not None]
                if any(not finite(v) for v in values):
                    raise ValueError('Nonfinite per-song metric')
                expected = float(np.mean(values)) if values else None
                actual = macro.get(metric)
                if expected is None:
                    if actual is not None:
                        raise ValueError('Undefined metric became a number')
                elif not finite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                    raise ValueError('Macro metric differs from equal-song aggregation')
            if any(not finite(macro.get(m)) for m in GATED):
                raise ValueError('Missing or nonfinite gate metric')
    return all_ids


def compare(reference, candidate, selection):
    if validate_metrics(reference) != validate_metrics(candidate):
        raise ValueError('Different development song IDs')
    rows = []
    for domain in DOMAINS:
        ref_ids = {r['song_id']: r for r in reference[domain]['per_song']}
        cand_ids = {r['song_id']: r for r in candidate[domain]['per_song']}
        if ref_ids.keys() != cand_ids.keys():
            raise ValueError('Song moved between domains')
        for song_id, old in ref_ids.items():
            new = cand_ids[song_id]
            if old['group_id'] != new['group_id']:
                raise ValueError('Song grouping changed')
            for axis in AXES:
                a, b = old[axis], new[axis]
                if a is None or b is None:
                    if a != b:
                        raise ValueError('Independent axis mask support changed')
                    continue
                if any(a[k] != b[k] for k in ('values', 'truth_std', 'truth_span90')):
                    raise ValueError('Truth support changed')
                if any((a[k] is None) != (b[k] is None) for k in ('ccc', 'derivative_mae')):
                    raise ValueError('CCC or native-time derivative validity changed')
        for axis in AXES:
            a, b = reference[domain]['macro'][axis], candidate[domain]['macro'][axis]
            guard = selection['dynamic_guardrails']
            gates = dict(
                ccc=b['ccc'] - a['ccc'] >= -selection['maximum_axis_domain_point_ccc_drop'] - 1e-12,
                mae=b['mae'] - a['mae'] <= selection['maximum_axis_domain_mae_increase'] + 1e-12,
                centered_rmse=b['centered_rmse'] <= a['centered_rmse'] * (1 + guard['centered_rmse_max_relative_increase']) + 1e-12,
                derivative_mae=b['derivative_mae'] <= a['derivative_mae'] * (1 + guard['derivative_mae_max_relative_increase']) + 1e-12,
                dynamics_beats_constant=b['dynamics_beats_constant'] - a['dynamics_beats_constant'] >= -guard['dynamics_beats_constant_max_absolute_drop'] - 1e-12)
            details = {m: dict(reference=a.get(m), candidate=b.get(m),
                       delta=b[m]-a[m] if finite(a.get(m)) and finite(b.get(m)) else None) for m in DISPLAY}
            rows.append(dict(dataset=domain, axis=axis, metrics=details, gates=gates,
                             eligible=all(gates.values())))
    reference_mean = float(np.mean([r['metrics']['ccc']['reference'] for r in rows]))
    candidate_mean = float(np.mean([r['metrics']['ccc']['candidate'] for r in rows]))
    gain = candidate_mean - reference_mean
    positive_gain = gain > selection['screening_mean_delta_min_exclusive']
    return dict(reference_mean_ccc=reference_mean, candidate_mean_ccc=candidate_mean,
                mean_ccc_delta=gain, positive_mean_gain=positive_gain,
                eligible=positive_gain and all(r['eligible'] for r in rows), rows=rows,
                promotion_authorized=False,
                scope='First-seed internal development screening only; no confidence or release claim')


def verified_report(arm, seed, protocol):
    run_id, directory, model_dir = locations(arm, seed)
    state = observe(arm, seed)
    if state['active'] or state['indeterminate'] or state['run_lock_held']:
        raise ValueError('Worker has not verifiably exited')
    report, config, launch = (read_json(directory / name) for name in ('report.json', 'config.json', 'launch.json'))
    if report['status'] != 'completed' or report['completed_steps'] != protocol['budget']['steps_per_run']:
        raise ValueError('Incomplete fixed-budget report')
    if report['run_id'] != run_id or report['arm'] != arm or report['seed'] != seed:
        raise ValueError('Report run identity mismatch')
    if sha256(PROTOCOL) != report['protocol_sha256'] or report['protocol_sha256'] != config['protocol_sha256']:
        raise ValueError('Protocol identity mismatch')
    if sha256(directory / 'config.json') != launch['config_hash'] or config['code_hashes'] != launch['code_hashes']:
        raise ValueError('Registered configuration changed')
    verify_hashes(config['code_hashes'])
    verify_hashes(report['protected_hashes'])
    spec = next(item for item in protocol['arms'] if item['id'] == arm)
    if config['settings'] != {**protocol['defaults'], **spec['overrides'], 'seed': seed}:
        raise ValueError('Settings changed from the single-factor protocol')
    source_path = ROOT / protocol['source_protocol']
    if sha256(source_path) != protocol['source_protocol_sha256']:
        raise ValueError('Source protocol changed')
    source = read_json(source_path)
    expected_ids = {key for key, value in source['split']['assignments'].items() if value == 'development'}
    if validate_metrics(report['best_development']) != expected_ids:
        raise ValueError('Report did not cover the frozen development set')
    if (report['train_songs'] != protocol['fit_songs'] or report['development_songs'] != protocol['development_songs']
            or any(report[k] != 0 for k in ('calibration_songs_used', 'original_validation_songs_used', 'test_songs_used'))):
        raise ValueError('Data role/count contract failed')
    if not report['checkpoint_reload_verified'] or report['maximum_metric_difference'] > 1e-6:
        raise ValueError('Checkpoint reload was not verified')
    for kind in ('best', 'last'):
        path = model_dir / (kind + '.pth')
        if sha256(path) != report['checkpoint_hashes'][path.name]:
            raise ValueError('Checkpoint hash changed')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['provenance'] != config or checkpoint['settings'] != config['settings']:
            raise ValueError('Checkpoint provenance mismatch')
        metric_key = 'development' if kind == 'best' else 'best_development'
        if checkpoint[metric_key] != report['best_development']:
            raise ValueError('Checkpoint/report per-song metrics mismatch')
        expected_step = report['best_step'] if kind == 'best' else protocol['budget']['steps_per_run']
        if checkpoint['step'] != expected_step:
            raise ValueError('Checkpoint step mismatch')
        if kind == 'last':
            for key in ('optimizer', 'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'best_state_dict'):
                if key not in checkpoint:
                    raise ValueError('Incomplete recovery checkpoint')
    return report, dict(report_sha256=sha256(directory/'report.json'), config_sha256=sha256(directory/'config.json'),
                        checkpoint_sha256=report['checkpoint_hashes'], status=state['worker_process'])


def select_candidate(comparisons, protocol):
    expected = set(protocol['budget']['screening']) - {'control'}
    if set(comparisons) != expected:
        raise ValueError('Incomplete first-seed matrix cannot authorize repeats')
    candidates = [(arm, row) for arm, row in comparisons.items() if row['eligible']]
    candidates.sort(key=lambda pair: (-pair[1]['candidate_mean_ccc'], pair[0]))
    return dict(eligible=bool(candidates), selected_arm=candidates[0][0] if candidates else None,
                promotion_authorized=False,
                decision='repeat_best_candidate_and_control' if candidates else 'stop_model_selection_no_candidate_passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('mean_loss', 'tone_residual'))
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.arm is None) == (not args.finalize):
        parser.error('Use exactly one of --arm or --finalize')
    destination = args.output.resolve()
    if (not destination.is_relative_to(WINDOW.resolve()) or destination.exists()
            or destination == (WINDOW/'selection.json').resolve()):
        raise ValueError('Write a new analysis artifact inside this window; never overwrite evidence')
    protocol = read_json(PROTOCOL)
    if sha256(PROTOCOL) != read_json(WINDOW/'plan.json')['protocol_sha256']:
        raise ValueError('Window protocol differs from preregistration')
    replay = read_json(WINDOW/'rounds/002/control-replay.json')
    if not replay['passed'] or not all(replay['checks'].values()):
        raise ValueError('Historical control replay did not pass')
    seed = protocol['seeds'][0]
    control, reference_proof = verified_report('control', seed, protocol)
    if reference_proof['report_sha256'] != replay['candidate_report_sha256']:
        raise ValueError('Replayed control report changed')
    arms = [a for a in protocol['budget']['screening'] if a != 'control'] if args.finalize else [args.arm]
    comparisons, evidence = {}, {'control': reference_proof}
    for arm in arms:
        candidate, evidence[arm] = verified_report(arm, seed, protocol)
        comparisons[arm] = compare(control['best_development'], candidate['best_development'], protocol['selection'])
    report = dict(created_at=now(), protocol_sha256=sha256(PROTOCOL), seed=seed,
                  comparisons=comparisons, evidence=evidence, analysis_source_sha256=sha256(Path(__file__)))
    if args.finalize:
        selection_path = WINDOW/'selection.json'
        if selection_path.exists():
            raise ValueError('Selection already frozen; inspect instead of overwriting')
        report['selection'] = select_candidate(comparisons, protocol)
        atomic_json(destination, report)
        atomic_json(selection_path, dict(**report['selection'], created_at=now(), seed=seed,
            protocol_sha256=sha256(PROTOCOL), evidence=str(destination.relative_to(ROOT)),
            evidence_sha256=sha256(destination)))
    else:
        atomic_json(destination, report)
    print(json.dumps(dict(output=str(destination), summary={arm: {k: row[k] for k in
        ('reference_mean_ccc', 'candidate_mean_ccc', 'mean_ccc_delta', 'eligible')} for arm,row in comparisons.items()},
        selection=report.get('selection')), ensure_ascii=False))


if __name__ == '__main__':
    main()
