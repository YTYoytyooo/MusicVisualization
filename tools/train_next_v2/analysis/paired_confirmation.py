"""Paired three-seed, song-group bootstrap on internal development reports."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import traceback

import numpy as np

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import AXES, BASE, ROOT, now, read_json, verify_hashes
from tools.train_next_v2.status import snapshot

METRICS = (
    'ccc', 'mae', 'rmse', 'mean_absolute_bias', 'centered_rmse',
    'derivative_mae', 'truth_std', 'prediction_std', 'std_ratio',
    'truth_span90', 'prediction_span90', 'constant_prediction_mae',
    'dynamics_beats_constant',
)


def connected_groups(records):
    """Retain transitive song-identity relationships, including duplicate audio."""
    parent = {r['song_id']: r['song_id'] for r in records}
    if len(parent) != len(records) or any(r['split'] != 'train' for r in records):
        raise ValueError('Expected unique original-training song identities')

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for field in ('group_id', 'audio_sha256'):
        owners = {}
        for row in records:
            value = row[field]
            if not value:
                raise ValueError(f'Missing group identity: {field}')
            if value in owners:
                left, right = find(row['song_id']), find(owners[value])
                parent[max(left, right)] = min(left, right)
            else:
                owners[value] = row['song_id']
    result = defaultdict(list)
    for row in sorted(records, key=lambda r: r['song_id']):
        result[find(row['song_id'])].append(row)
    return dict(sorted(result.items()))


def weighted_macro(sums, counts, weights):
    """Resample whole clusters while preserving song-weighted macro metrics."""
    numerator = np.matmul(weights, sums)
    denominator = np.matmul(weights, counts)
    if np.any(denominator <= 0):
        raise ValueError('Bootstrap replicate contains no valid metric observations')
    result = numerator / denominator
    if not np.isfinite(result).all():
        raise ValueError('Non-finite resampled metric')
    return result


def group_totals(per_seed, groups, axis):
    sums = np.zeros((len(per_seed), len(groups), len(METRICS)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for seed_index, songs in enumerate(per_seed):
        for group_index, group in enumerate(groups):
            for row in group:
                values = songs[row['song_id']][axis]
                if values is None:
                    continue
                for metric_index, name in enumerate(METRICS):
                    value = values[name]
                    if value is not None:
                        if not np.isfinite(value):
                            raise ValueError('Non-finite source metric')
                        sums[seed_index, group_index, metric_index] += value
                        counts[seed_index, group_index, metric_index] += 1
    return sums, counts


def check_pairs(reference, candidate, records):
    expected = {r['song_id']: r for r in records}
    if reference.keys() != candidate.keys() or reference.keys() != expected.keys():
        raise ValueError('Song identities differ from the frozen development split')
    for song_id, original in reference.items():
        current = candidate[song_id]
        if original['group_id'] != current['group_id'] or original['group_id'] != expected[song_id]['group_id']:
            raise ValueError('Song group identity changed')
        for axis in AXES:
            left, right = original[axis], current[axis]
            if left is None or right is None:
                if left != right:
                    raise ValueError('Axis validity changed')
            elif any(left[k] != right[k] for k in ('values', 'truth_std', 'truth_span90')):
                raise ValueError('Target statistics changed')


def checked_reports(experiment, seeds, source_hashes):
    reports = []
    for seed in seeds:
        run_id = f'{experiment}-s{seed}'
        if snapshot(run_id)['outcome'] != 'completed_verified':
            raise ValueError(f'Run not completed and verified: {run_id}')
        report_path = BASE / 'runs' / run_id / 'report.json'
        verification_path = BASE / 'reports' / f'{run_id}-verification.json'
        report, verification = read_json(report_path), read_json(verification_path)
        if verification['report_sha256'] != sha256(report_path):
            raise ValueError('Independent verification report hash changed')
        if not verification.get('fresh_process_checkpoint_load_verified', verification.get('fresh_process_best_load_verified', False)):
            raise ValueError('Independent checkpoint reload missing')
        checkpoint = ROOT / 'models/next_va_v2' / run_id / 'best.pth'
        expected = verification.get('best_checkpoint_sha256', verification.get('checkpoint_sha256'))
        if sha256(checkpoint) != expected or expected != report['checkpoint_sha256']:
            raise ValueError('Checkpoint hash changed')
        if report['completed_steps'] != 20000 or report['development_songs'] != 383:
            raise ValueError('Budget or evaluation split changed')
        if any(report[k] != 0 for k in ('calibration_songs_used', 'original_validation_songs_used', 'test_songs_used')):
            raise ValueError('Held-out data was consumed')
        for path in (report_path, verification_path, checkpoint):
            source_hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
        reports.append(report)
    return reports


def analyze(protocol_path: Path, candidate: str):
    protocol = read_json(protocol_path)
    if candidate not in protocol['candidates']:
        raise ValueError('Candidate is not in the frozen shortlist')
    source_hashes = {
        protocol['source_protocol']: protocol['source_protocol_sha256'],
        protocol['shortlist_protocol']: protocol['shortlist_protocol_sha256'],
    }
    verify_hashes(source_hashes)
    source = read_json(ROOT / protocol['source_protocol'])
    shortlist = read_json(ROOT / protocol['shortlist_protocol'])
    if protocol['selection_policy'] != shortlist['selection_policy']:
        raise ValueError('Selection gates changed')
    if protocol['seeds'] != shortlist['seeds'] or protocol['reference'] != shortlist['reference']:
        raise ValueError('Paired seeds or reference changed')
    if protocol['replicates'] != 2000 or protocol['confidence'] != 0.95:
        raise ValueError('Frozen confidence calculation changed')
    manifest = Path(source['source_manifest'])
    if sha256(manifest) != protocol['manifest_sha256'] or protocol['manifest_sha256'] != source['manifest_sha256']:
        raise ValueError('Manifest changed')
    with manifest.open(encoding='utf-8') as stream:
        records = [r for r in map(json.loads, stream)
                   if source['split']['assignments'].get(r['song_id']) == 'development']
    groups = connected_groups(records)
    if len(records) != protocol['expected_development_songs'] or len(groups) != protocol['expected_development_groups']:
        raise ValueError('Frozen development group counts changed')
    by_domain = defaultdict(list)
    for group in groups.values():
        domains = {r['source_dataset'] for r in group}
        if len(domains) != 1:
            raise ValueError('Cross-dataset component requires an explicit joint resampling protocol')
        by_domain[next(iter(domains))].append(group)
    if {d: len(g) for d, g in by_domain.items()} != protocol['expected_dataset_groups']:
        raise ValueError('Dataset group counts changed')
    source_hashes[manifest.relative_to(ROOT).as_posix()] = sha256(manifest)
    source_hashes[protocol_path.relative_to(ROOT).as_posix()] = sha256(protocol_path)
    source_hashes[Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha256(Path(__file__))
    seeds = protocol['seeds']
    reference = checked_reports(protocol['reference'], seeds, source_hashes)
    candidates = checked_reports(candidate, seeds, source_hashes)
    if any(r['protocol_sha256'] != protocol['source_protocol_sha256'] for r in reference + candidates):
        raise ValueError('A completed report belongs to another data/training protocol')
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    alpha = (1 - protocol['confidence']) / 2
    rows, overall_draws = [], []
    for domain in sorted(by_domain):
        domain_groups = by_domain[domain]
        domain_records = [r for group in domain_groups for r in group]
        refs = [{r['song_id']: r for r in report['best_development'][domain]['per_song']} for report in reference]
        cands = [{r['song_id']: r for r in report['best_development'][domain]['per_song']} for report in candidates]
        for ref, cand in zip(refs, cands):
            check_pairs(ref, cand, domain_records)
        weights = rng.multinomial(len(domain_groups), np.full(len(domain_groups), 1 / len(domain_groups)), size=protocol['replicates'])
        ones = np.ones((1, len(domain_groups)))
        for axis in AXES:
            rsum, rcount = group_totals(refs, domain_groups, axis)
            csum, ccount = group_totals(cands, domain_groups, axis)
            rp = weighted_macro(rsum, rcount, ones)[:, 0, :]
            cp = weighted_macro(csum, ccount, ones)[:, 0, :]
            for si in range(len(seeds)):
                for mi, metric in enumerate(METRICS):
                    np.testing.assert_allclose(rp[si, mi], reference[si]['best_development'][domain]['macro'][axis][metric], rtol=1e-10, atol=1e-12)
                    np.testing.assert_allclose(cp[si, mi], candidates[si]['best_development'][domain]['macro'][axis][metric], rtol=1e-10, atol=1e-12)
            draws = (weighted_macro(csum, ccount, weights) - weighted_macro(rsum, rcount, weights)).mean(axis=0)
            ci = np.quantile(draws, [alpha, 1 - alpha], axis=0, method=protocol['quantile_method'])
            delta = cp - rp
            metrics = {m: {'reference_mean': float(rp[:, i].mean()), 'candidate_mean': float(cp[:, i].mean()),
                           'mean_delta': float(delta[:, i].mean()), 'delta_seed_sample_std': float(delta[:, i].std(ddof=1)),
                           'delta_ci95': ci[:, i].tolist(), 'per_seed_delta': delta[:, i].tolist()}
                       for i, m in enumerate(METRICS)}
            rows.append({'dataset': domain, 'axis': axis, 'songs': len(domain_records), 'groups': len(domain_groups), 'metrics': metrics})
            overall_draws.append(draws[:, 0])
    per_seed_gain = [c['best_development_score'] - r['best_development_score'] for c, r in zip(candidates, reference)]
    mean_gain = statistics.mean(per_seed_gain)
    np.testing.assert_allclose(mean_gain, statistics.mean(r['metrics']['ccc']['mean_delta'] for r in rows), atol=1e-12)
    selection = protocol['selection_policy']
    gates = {
        'mean_ccc_gain': mean_gain >= selection['minimum_mean_gain'],
        'all_axis_dataset_ccc_point_bounds': all(r['metrics']['ccc']['mean_delta'] >= -selection['maximum_axis_domain_point_ccc_drop'] for r in rows),
        'all_axis_dataset_mae_point_bounds': all(r['metrics']['mae']['mean_delta'] <= selection['maximum_axis_domain_mae_increase'] for r in rows),
        'all_axis_dataset_ccc_ci_lower_bounds': all(r['metrics']['ccc']['delta_ci95'][0] > -0.02 for r in rows),
    }
    return {'created_at': now(), 'candidate': candidate, 'reference': protocol['reference'], 'seeds': seeds,
            'status': 'completed_internal_development_confirmation', 'source_hashes': source_hashes,
            'bootstrap_protocol': protocol_path.relative_to(ROOT).as_posix(), 'bootstrap_replicates': protocol['replicates'],
            'development_songs': len(records), 'development_groups': len(groups), 'rows': rows,
            'per_seed': [{'seed': seed, 'candidate_score': c['best_development_score'], 'reference_score': r['best_development_score'], 'gain': gain}
                         for seed, c, r, gain in zip(seeds, candidates, reference, per_seed_gain)],
            'mean_candidate_ccc': statistics.mean(c['best_development_score'] for c in candidates),
            'mean_reference_ccc': statistics.mean(r['best_development_score'] for r in reference),
            'mean_paired_ccc_gain': mean_gain, 'paired_gain_seed_sample_std': statistics.stdev(per_seed_gain),
            'overall_ccc_gain_ci95': np.quantile(np.mean(overall_draws, axis=0), [alpha, 1 - alpha], method=protocol['quantile_method']).tolist(),
            'gates': gates, 'passes_internal_development_gates': all(gates.values()),
            'release_status': 'Not released; this analysis cannot replace original validation, calibration, final test or listening gates.',
            'scope_and_limitations': protocol['scope_and_limitations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol_path, output = args.protocol.resolve(), args.output.resolve()
    if not protocol_path.is_relative_to(BASE) or not output.is_relative_to(BASE / 'reports'):
        raise ValueError('Analysis inputs/outputs must stay within the optimization workspace')
    if output.exists():
        raise FileExistsError('Preserve existing analysis; choose a distinct version for a changed calculation')
    with run_lease(f'confirmation-{args.candidate}-v1'):
        if output.exists():
            raise FileExistsError('A completed analysis already exists')
        with output.with_suffix('.log').open('a', encoding='utf-8') as log:
            log.write(json.dumps({'started_at': now(), 'candidate': args.candidate}) + '\n')
            log.flush()
            try:
                result = analyze(protocol_path, args.candidate)
                atomic_json(output, result)
                summary = {k: result[k] for k in ('candidate', 'mean_paired_ccc_gain', 'overall_ccc_gain_ci95', 'gates', 'passes_internal_development_gates')}
                log.write(json.dumps({'completed_at': now(), **summary}) + '\n')
                print(json.dumps(summary))
            except Exception:
                log.write(traceback.format_exc())
                raise


if __name__ == '__main__':
    main()
