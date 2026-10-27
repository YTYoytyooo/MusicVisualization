"""Apply the frozen C pilot gates to all six variants and three paired seeds."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import statistics
import sys
import traceback

import numpy as np

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import AXES, BASE, ROOT, PLAN, now, paths, process_identity, read_json, record_event, verify_hashes
from tools.train_next_v2.run import Tee
from tools.train_next_v2.pilot_training.data import CONTRACT
from tools.train_next_v2.pilot_training.execution import MATRIX, PROTOCOL, QUEUE_DIR, QUEUE_ID, verification_matches
from .paired_confirmation import METRICS, check_pairs, group_totals, weighted_macro

JOB_ID = 'c-pilot-comparison-v1'
DIRECTORY = BASE / 'analyses' / JOB_ID
OUTPUT = BASE / 'reports/C-pilot-decision-v1.json'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def selection_gates(per_seed_gain, rows):
    """Point gates were fixed before extraction; bootstrap CIs are exploratory only."""
    if len(per_seed_gain) != 3 or len(rows) != 6:
        raise ValueError('Require all three seeds and all six dataset-axis pairs')
    return dict(mean_ccc_gain_at_least_002=statistics.mean(per_seed_gain) >= .02,
                positive_gain_in_every_seed=all(value > 0 for value in per_seed_gain),
                no_dataset_axis_mean_ccc_drop_over_001=all(row['metrics']['ccc']['mean_delta'] >= -.01 for row in rows),
                no_dataset_axis_mean_mae_increase_over_001=all(row['metrics']['mae']['mean_delta'] <= .01 for row in rows))


def paired_comparison(reference, candidate, records, seeds, bootstrap):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[row['source_dataset']][row['linked_component']].append(row)
    rng = np.random.default_rng(bootstrap['seed'])
    rows, overall_draws = [], []
    for domain, components in sorted(grouped.items()):
        groups = [components[key] for key in sorted(components)]
        domain_records = [row for group in groups for row in group]
        refs = [{row['song_id']: row for row in report['best_development'][domain]['per_song']} for report in reference]
        cands = [{row['song_id']: row for row in report['best_development'][domain]['per_song']} for report in candidate]
        for left, right in zip(refs, cands):
            check_pairs(left, right, domain_records)
        weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=bootstrap['replicates'])
        ones = np.ones((1, len(groups)))
        for axis in AXES:
            rs, rn = group_totals(refs, groups, axis)
            cs, cn = group_totals(cands, groups, axis)
            # Exactly the same songs, masks and metric eligibility must be paired.
            np.testing.assert_array_equal(rn, cn)
            rp, cp = weighted_macro(rs, rn, ones)[:, 0, :], weighted_macro(cs, cn, ones)[:, 0, :]
            for index in range(3):
                for metric_index, metric in enumerate(METRICS):
                    np.testing.assert_allclose(rp[index, metric_index], reference[index]['best_development'][domain]['macro'][axis][metric], rtol=1e-10, atol=1e-12)
                    np.testing.assert_allclose(cp[index, metric_index], candidate[index]['best_development'][domain]['macro'][axis][metric], rtol=1e-10, atol=1e-12)
            draws = (weighted_macro(cs, cn, weights)-weighted_macro(rs, rn, weights)).mean(axis=0)
            intervals = np.quantile(draws, [.025, .975], axis=0, method='linear')
            delta = cp-rp
            metrics = {metric: dict(reference_mean=float(rp[:, i].mean()), candidate_mean=float(cp[:, i].mean()),
                mean_delta=float(delta[:, i].mean()), per_seed_delta=delta[:, i].tolist(),
                delta_seed_sample_std=float(delta[:, i].std(ddof=1)), delta_ci95=intervals[:, i].tolist())
                for i, metric in enumerate(METRICS)}
            rows.append(dict(dataset=domain, axis=axis, songs=len(domain_records), groups=len(groups), metrics=metrics))
            overall_draws.append(draws[:, 0])
    gains = [right['best_development_score']-left['best_development_score'] for left, right in zip(reference, candidate)]
    np.testing.assert_allclose(statistics.mean(gains), statistics.mean(row['metrics']['ccc']['mean_delta'] for row in rows), atol=1e-12)
    gates = selection_gates(gains, rows)
    return dict(rows=rows, per_seed=[dict(seed=seed, reference_ccc=left['best_development_score'],
                candidate_ccc=right['best_development_score'], gain=gain)
                for seed, left, right, gain in zip(seeds, reference, candidate, gains)],
                mean_reference_ccc=statistics.mean(row['best_development_score'] for row in reference),
                mean_candidate_ccc=statistics.mean(row['best_development_score'] for row in candidate),
                candidate_seed_sample_std=statistics.stdev(row['best_development_score'] for row in candidate),
                mean_paired_ccc_gain=statistics.mean(gains), paired_gain_seed_sample_std=statistics.stdev(gains),
                overall_ccc_gain_ci95=np.quantile(np.mean(overall_draws, axis=0), [.025, .975], method='linear').tolist(),
                gates=gates, eligible_for_larger_internal_experiment=all(gates.values()))


def register():
    if (DIRECTORY / 'launch.json').exists() or OUTPUT.exists():
        raise ValueError('Preserve existing comparison registration/results')
    matrix = read_json(MATRIX)
    contract = read_json(CONTRACT)
    protected = dict(matrix['protected_hashes'])
    protected.update(contract['builder_code_hashes'])
    for path in (MATRIX, CONTRACT, PROTOCOL, Path(__file__), Path(__file__).with_name('paired_confirmation.py')):
        protected[relative(path)] = sha256(path)
    verify_hashes(protected)
    env = dict(read_json(QUEUE_DIR / 'launch.json')['environment'])
    env.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    argv = [str(Path(sys.executable).resolve()), '-u', '-m', 'tools.train_next_v2.analysis.c_pilot', '--launch', str(DIRECTORY / 'launch.json')]
    launch = dict(job_id=JOB_ID, registered_at=now(), original_argv=argv, cwd=str(ROOT), python=argv[0],
        environment=env, protected_hashes=protected, output=relative(OUTPUT), os_lock=str(lock_path(JOB_ID)),
        log=str(DIRECTORY / 'console.log'), gpu_used=False,
        bootstrap=dict(seed=20261004, replicates=2000, confidence=.95, quantile_method='linear',
            resampling='Paired connected song groups within each dataset; identical draws across axes, variants and the three fixed seeds. Average seeds, then equal-weight the six dataset-axis means.',
            interpretation='Exploratory intervals conditional on the chosen checkpoints and only 30 development songs; not a final generalization or seed-population confidence interval.'),
        gates_source=relative(PROTOCOL), recovery='Never analyze partial runs or overwrite prior results. Wait for 18 verified completions. No training or heldout labels are read.',
        tie_break='Largest mean candidate CCC among eligible variants; lexical variant ID only for an exact tie.')
    DIRECTORY.mkdir(parents=True)
    atomic_json(DIRECTORY / 'launch.json', launch)
    record_event(dict(event='C_pilot_paired_analysis_registered', job_id=JOB_ID, launch=relative(DIRECTORY / 'launch.json')),
                 updates=dict(pending_C_analysis=relative(DIRECTORY / 'launch.json')))
    print(json.dumps(dict(job_id=JOB_ID, registered=True)))


def analyze(launch):
    if is_running(QUEUE_ID):
        raise RuntimeError('C training queue still active; wait for complete evidence')
    queue_report = read_json(QUEUE_DIR / 'report.json')
    if queue_report['status'] != 'completed_awaiting_paired_analysis' or queue_report['count'] != 18:
        raise ValueError('C training matrix is not complete')
    protocol, contract, matrix = read_json(PROTOCOL), read_json(CONTRACT), read_json(MATRIX)
    if queue_report['matrix_sha256'] != sha256(MATRIX):
        raise ValueError('Queue completed another matrix')
    records = [row for row in contract['songs'] if row['role'] == 'development']
    if len(records) != 30 or len({row['linked_component'] for row in records}) != 30:
        raise ValueError('C development groups changed')
    hashes = dict(launch['protected_hashes'])
    hashes[relative(QUEUE_DIR / 'report.json')] = sha256(QUEUE_DIR / 'report.json')
    reports, checks = defaultdict(list), defaultdict(list)
    expected = {row['run_id'] for row in matrix['runs']}
    if expected != set(queue_report['completed_runs']) or len(expected) != 18:
        raise ValueError('Queue completion membership mismatch')
    for variant in protocol['experiments']:
        for seed in protocol['seeds']:
            run_id = f'{variant["id"]}-s{seed}'
            run_dir, model_dir = paths(run_id)
            if run_id not in expected or is_running(run_id) or not verification_matches(run_id):
                raise ValueError('C run is not independently completed: ' + run_id)
            report = read_json(run_dir / 'report.json')
            verification = read_json(run_dir / 'verification.json')
            config = read_json(run_dir / 'config.json')
            if report['protocol_sha256'] != sha256(PROTOCOL) or report['contract_sha256'] != sha256(CONTRACT):
                raise ValueError('Report belongs to another C dataset/protocol')
            if config['settings'] != {**protocol['settings'], 'seed': seed} or config['experiment'] != variant['id']:
                raise ValueError('Training budgets or variant settings differ')
            if report['completed_steps'] != 2000 or report['train_songs'] != 60 or report['development_songs'] != 30:
                raise ValueError('C data or optimizer budget differs')
            if any(report[key] != 0 for key in ('calibration_songs_used', 'original_validation_songs_used', 'test_songs_used')):
                raise ValueError('C comparison consumed heldout data')
            for path in [run_dir / name for name in ('config.json', 'report.json', 'verification.json')] + [model_dir / name for name in ('best.pth', 'last.pth')]:
                hashes[relative(path)] = sha256(path)
            reports[variant['id']].append(report)
            checks[variant['id']].append(verification)
    reference = reports[protocol['control']]
    results = []
    for variant in protocol['experiments']:
        name = variant['id']
        pair = paired_comparison(reference, reports[name], records, protocol['seeds'], launch['bootstrap'])
        pair.update(variant=name, is_control=name == protocol['control'],
            costs=dict(extraction=contract['extraction_costs'].get(name, dict(source='Existing cache reused; no new extraction timing collected')),
                training_seconds=[row['training_seconds'] for row in reports[name]],
                gpu_training_peak_bytes=[row['gpu_peak_bytes'] for row in reports[name]],
                parameter_count=[row['parameter_count'] for row in reports[name]],
                cached_development_inference_seconds=[row['cached_feature_inference_seconds'] for row in checks[name]],
                feature_dimensions=variant['feature_dimensions']),
            best_steps=[row['best_step'] for row in reports[name]],
            checkpoints=[row['checkpoint'] for row in reports[name]])
        results.append(pair)
    eligible = [row for row in results if not row['is_control'] and row['eligible_for_larger_internal_experiment']]
    selected = sorted(eligible, key=lambda row: (-row['mean_candidate_ccc'], row['variant']))[0]['variant'] if eligible else None
    verify_hashes(hashes)
    verify_hashes(contract['protected_hashes'])
    return dict(status='completed_verified', completed_at=now(), job_id=JOB_ID, reference=protocol['control'],
        seeds=protocol['seeds'], fit_songs=60, development_songs=30, development_groups=30,
        source_hashes=hashes, bootstrap=launch['bootstrap'], results=results, selected_for_expansion=selected,
        decision='expand_one_candidate_internally' if selected else 'no_candidate_passed_frozen_pilot_gates',
        legacy_model_unchanged=True, historical_baseline_unchanged=True,
        calibration_original_validation_test_songs_used=0,
        release_status='No model promoted; only a small internal feature screen.',
        limitations=protocol['limitations'] + ['Bootstrap intervals are exploratory and do not add or replace the frozen point gates.',
        'Reported embedding extraction seconds sum per-song extraction work; model loading and migration overhead are recorded separately in the feature-job reports.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--launch', type=Path)
    args = parser.parse_args()
    with run_lease(JOB_ID):
        if args.register:
            register()
            return
        if args.launch.resolve() != DIRECTORY / 'launch.json' or OUTPUT.exists():
            raise ValueError('Use the registered launch and preserve previous results')
        launch = read_json(args.launch)
        verify_hashes(launch['protected_hashes'])
        for key, value in launch['environment'].items():
            if os.environ.get(key) != value:
                raise ValueError('Comparison environment changed: ' + key)
        atomic_json(DIRECTORY / 'process.json', dict(started_at=now(), process_identity=process_identity()))
        with (DIRECTORY / 'console.log').open('a', encoding='utf-8', buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    result = analyze(launch)
                    atomic_json(OUTPUT, result)
                    stages = read_json(PLAN)['stages']
                    stages['C'].update(pilot_decision=relative(OUTPUT), selected_for_expansion=result['selected_for_expansion'],
                        phase='pilot_passed_ready_for_internal_expansion' if result['selected_for_expansion'] else 'completed_no_feature_selected')
                    if result['selected_for_expansion'] is None:
                        stages['C'].update(status='completed', completed_at=now(), conclusion=result['decision'])
                        stages['D'].update(status='in_progress', phase='source_license_time_origin_dedup_preparation')
                    record_event(dict(event='C_pilot_frozen_gate_decision', decision=result['decision'], selected=result['selected_for_expansion'], report=relative(OUTPUT)),
                        updates=dict(stages=stages, current_stage='C' if result['selected_for_expansion'] else 'D',
                        next_action='为唯一入围C特征登记完整内部划分确认实验，仍不接触原验证或测试集。' if result['selected_for_expansion'] else
                        'C阶段没有候选通过冻结门槛，不扩大特征提取；推进D阶段MERGE小包许可、标签来源、音频原点与排重审计，再登记静态辅助对照。'))
                    print(json.dumps(dict(decision=result['decision'], selected=result['selected_for_expansion'],
                        variants=[dict(variant=row['variant'], mean_ccc=row['mean_candidate_ccc'], gain=row['mean_paired_ccc_gain'], gates=row['gates']) for row in result['results']]), ensure_ascii=False))
                except BaseException as exc:
                    traceback.print_exc()
                    atomic_json(DIRECTORY / 'failure.json', dict(failed_at=now(), error=f'{type(exc).__name__}: {exc}'))
                    record_event(dict(event='C_pilot_analysis_failed', error=f'{type(exc).__name__}: {exc}'))
                    raise


if __name__ == '__main__':
    main()
