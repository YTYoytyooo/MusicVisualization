"""Apply the predeclared D gates only after all paired runs pass fresh reload."""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np
import torch

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, lock_path, is_running
from tools.train_next_v2.common import AXES, BASE, ROOT, PLAN, now, paths, process_identity, read_json, record_event, verify_hashes
from tools.train_next_v2.run import Tee
from tools.train_next_v2.jamendo_training.data import CONTRACT, PROTOCOL, relative
from tools.train_next_v2.jamendo_training.execution import MATRIX, QUEUE_DIR, QUEUE_ID, verification_matches, settings_for
from .paired_confirmation import METRICS, connected_groups, check_pairs, group_totals, weighted_macro

JOB = 'd-jamendo-paired-comparison-v1'
DIRECTORY = BASE/'analyses'/JOB
OUTPUT = BASE/'reports/D-Jamendo-paired-decision-v1.json'
TEST_REPORT = BASE/'analyses/d-jamendo-analysis-tests-v1/report.json'


def analysis_code():
    files = [Path(__file__), Path(__file__).with_name('paired_confirmation.py'),
             ROOT/'tools/train_next_v2/tests/test_d_jamendo_analysis.py']
    return {relative(p): sha256(p) for p in files}


def development_groups(protocol, contract):
    b = read_json(ROOT/protocol['source_B_protocol'])
    manifest = Path(b['source_manifest'])
    verify_hashes({relative(manifest): b['manifest_sha256'],
                   protocol['source_B_protocol']: protocol['source_B_protocol_sha256']})
    # Read identity metadata only. Per-song metrics come from verified development reports.
    rows = [r for r in map(json.loads, manifest.read_text(encoding='utf-8').splitlines())
            if b['split']['assignments'].get(r['song_id']) == 'development']
    admitted = {s['song_id']: s for s in contract['main_songs'] if s['role'] == 'development'}
    if len(rows) != 383 or set(admitted) != {r['song_id'] for r in rows}:
        raise ValueError('Frozen development membership changed')
    for row in rows:
        if any(row[k] != admitted[row['song_id']][k] for k in ('group_id', 'audio_sha256', 'source_dataset')):
            raise ValueError('Development recording identity changed')
    groups = connected_groups(rows)
    by_domain = defaultdict(list)
    for group in groups.values():
        domains = {r['source_dataset'] for r in group}
        if len(domains) != 1:
            raise ValueError('Cross-domain cluster requires a separately frozen joint resampling protocol')
        by_domain[next(iter(domains))].append(group)
    counts = dict(songs=len(rows), groups=len(groups),
        domain_songs=dict(Counter(r['source_dataset'] for r in rows)),
        domain_groups={d:len(g) for d,g in by_domain.items()})
    return dict(by_domain), counts


def selection_gates(gains, rows, selection):
    expected = {(domain, axis) for domain in ('deam', 'memo2496', 'pmemo') for axis in AXES}
    if len(gains) != 3 or len(rows) != 6 or {(r['dataset'],r['axis']) for r in rows} != expected:
        raise ValueError('Require exactly three seeds and the six unique domain-axis pairs')
    numbers = list(gains)
    for row in rows:
        numbers += [row['metrics']['ccc']['mean_delta'], row['metrics']['mae']['mean_delta'],
                    *row['metrics']['ccc']['delta_ci95']]
    if not np.isfinite(numbers).all():
        raise ValueError('Non-finite selection evidence')
    return dict(mean_ccc_gain=statistics.mean(gains) >= selection['minimum_mean_CCC_gain'],
        positive_gain_in_every_seed=(all(g > 0 for g in gains) if selection['require_positive_gain_in_every_seed'] else True),
        all_axis_dataset_ccc_point_bounds=all(r['metrics']['ccc']['mean_delta'] >= -selection['maximum_dataset_axis_CCC_drop'] for r in rows),
        all_axis_dataset_mae_point_bounds=all(r['metrics']['mae']['mean_delta'] <= selection['maximum_dataset_axis_MAE_increase'] for r in rows),
        all_axis_dataset_ccc_ci_lower_bounds=all(r['metrics']['ccc']['delta_ci95'][0] > selection['minimum_dataset_axis_CCC_CI_lower'] for r in rows))


def check_pair_fairness(reference, candidate, left_config, right_config):
    left, right = dict(left_config['settings']), dict(right_config['settings'])
    if left.pop('tag_loss_weight') != 0. or right.pop('tag_loss_weight') != .1 or left != right:
        raise ValueError('Paired configurations differ beyond the frozen auxiliary coefficient')
    for field in ('contract_sha256', 'protocol_sha256', 'feature_dimensions', 'code', 'runtime', 'arithmetic'):
        if left_config[field] != right_config[field]:
            raise ValueError('Paired data/code/runtime differs: '+field)
    for field in ('initial_state_sha256', 'exposure_chain', 'samplers', 'completed_steps',
                  'train_songs', 'development_songs', 'auxiliary_songs'):
        if reference[field] != candidate[field]:
            raise ValueError('Paired initialization or exposure mismatch: '+field)


def compare(reference, candidate, by_domain, seeds, bootstrap, selection):
    if len(reference) != 3 or len(candidate) != 3 or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('All three distinct paired seeds are required')
    rng = np.random.default_rng(bootstrap['seed'])
    rows, overall_draws = [], []
    alpha = (1-bootstrap['confidence'])/2
    for domain, groups in sorted(by_domain.items()):
        records = [r for g in groups for r in g]
        refs = [{r['song_id']:r for r in report['best_development'][domain]['per_song']} for report in reference]
        cands = [{r['song_id']:r for r in report['best_development'][domain]['per_song']} for report in candidate]
        for ref, cand in zip(refs, cands):
            check_pairs(ref, cand, records)
        # Resample connected groups, not independent rows; reuse draws across axes and seeds.
        weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=bootstrap['replicates'])
        ones = np.ones((1,len(groups)))
        for axis in AXES:
            rs, rn = group_totals(refs, groups, axis)
            cs, cn = group_totals(cands, groups, axis)
            np.testing.assert_array_equal(rn, cn)
            rp = weighted_macro(rs, rn, ones)[:,0,:]
            cp = weighted_macro(cs, cn, ones)[:,0,:]
            for index in range(3):
                for mi, metric in enumerate(METRICS):
                    np.testing.assert_allclose(rp[index,mi],reference[index]['best_development'][domain]['macro'][axis][metric],rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(cp[index,mi],candidate[index]['best_development'][domain]['macro'][axis][metric],rtol=1e-10,atol=1e-12)
            draws = (weighted_macro(cs,cn,weights)-weighted_macro(rs,rn,weights)).mean(axis=0)
            intervals = np.quantile(draws,[alpha,1-alpha],axis=0,method=bootstrap['quantile_method'])
            delta = cp-rp
            metrics = {metric:dict(reference_mean=float(rp[:,i].mean()),candidate_mean=float(cp[:,i].mean()),
                mean_delta=float(delta[:,i].mean()),per_seed_delta=delta[:,i].tolist(),
                delta_seed_sample_std=float(delta[:,i].std(ddof=1)),delta_ci95=intervals[:,i].tolist())
                for i,metric in enumerate(METRICS)}
            rows.append(dict(dataset=domain,axis=axis,songs=len(records),groups=len(groups),metrics=metrics))
            overall_draws.append(draws[:,0])
    gains = [c['best_development_score']-r['best_development_score'] for r,c in zip(reference,candidate)]
    np.testing.assert_allclose(statistics.mean(gains),statistics.mean(row['metrics']['ccc']['mean_delta'] for row in rows),atol=1e-12)
    gates = selection_gates(gains,rows,selection)
    return dict(rows=rows,per_seed=[dict(seed=s,reference_ccc=r['best_development_score'],candidate_ccc=c['best_development_score'],gain=g)
        for s,r,c,g in zip(seeds,reference,candidate,gains)],
        mean_reference_ccc=statistics.mean(r['best_development_score'] for r in reference),
        mean_candidate_ccc=statistics.mean(r['best_development_score'] for r in candidate),
        mean_paired_ccc_gain=statistics.mean(gains),paired_gain_seed_sample_std=statistics.stdev(gains),
        candidate_seed_sample_std=statistics.stdev(r['best_development_score'] for r in candidate),
        overall_ccc_gain_ci95=np.quantile(np.mean(overall_draws,axis=0),[alpha,1-alpha],method=bootstrap['quantile_method']).tolist(),
        gates=gates,passes_internal_development_gates=all(gates.values()))


def register():
    if DIRECTORY.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve previous D analysis registrations/results')
    tests = read_json(TEST_REPORT)
    if tests['status'] != 'passed' or tests['code_hashes'] != analysis_code():
        raise ValueError('Fresh analysis tests required before registration')
    protocol, contract, matrix = read_json(PROTOCOL), read_json(CONTRACT), read_json(MATRIX)
    protected = dict(matrix['protected_hashes'])
    protected.update(contract['builder_code_hashes']); protected.update(analysis_code())
    for path in (MATRIX,CONTRACT,PROTOCOL,TEST_REPORT):
        protected[relative(path)] = sha256(path)
    verify_hashes(protected)
    _,counts = development_groups(protocol,contract)
    env = dict(read_json(QUEUE_DIR/'launch.json')['environment'])
    env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    argv = [str(Path(sys.executable).resolve()),'-u','-m','tools.train_next_v2.analysis.d_jamendo','--launch',str(DIRECTORY/'launch.json')]
    selection = protocol['selection']
    launch = dict(job_id=JOB,registered_at=now(),original_argv=argv,cwd=str(ROOT),python=argv[0],environment=env,
        protected_hashes=protected,output=relative(OUTPUT),os_lock=str(lock_path(JOB)),log=relative(DIRECTORY/'console.log'),
        gpu_used=False,expected_development=counts,
        bootstrap=dict(seed=20261004,replicates=selection['paired_song_group_bootstrap_replicates'],
            confidence=selection['confidence'],quantile_method='linear',
            resampling='Connected song groups within each dataset, paired across models, axes and all three fixed seeds. Preserve song-weighted domain macros; equal-weight six domain-axis means.',
            interpretation='Conditional on the three fixed seeds and selected checkpoints; internal development has already been used for selection. Not a seed-population or final generalization confidence interval.'),
        gates_source=relative(PROTOCOL),required_completed_verified_runs=6,
        recovery='Run only after all6 workers and queue have exited and fresh verification matches. No partial inference or score-based retry. Preserve existing results; changed analysis requires a new version.')
    DIRECTORY.mkdir(parents=True);atomic_json(DIRECTORY/'launch.json',launch)
    record_event(dict(event='D_Jamendo_auxiliary_paired_analysis_registered',job_id=JOB,launch=relative(DIRECTORY/'launch.json')),
        updates=dict(pending_D_analysis=relative(DIRECTORY/'launch.json')))
    print(json.dumps(dict(job_id=JOB,status='registered_waiting_for_all6_verified',development=counts)))


def analyze(launch):
    if is_running(QUEUE_ID):
        raise RuntimeError('D training queue is still active; partial comparisons are forbidden')
    queue_report = read_json(QUEUE_DIR/'report.json')
    matrix,protocol,contract = read_json(MATRIX),read_json(PROTOCOL),read_json(CONTRACT)
    expected = {r['run_id'] for r in matrix['runs']}
    if (queue_report['status'] != 'completed_awaiting_paired_analysis' or queue_report['count'] != 6
            or len(expected) != 6 or len(queue_report['completed_runs']) != 6
            or expected != set(queue_report['completed_runs']) or queue_report['matrix_sha256'] != sha256(MATRIX)):
        raise ValueError('All six registered runs must be complete')
    by_domain,counts = development_groups(protocol,contract)
    if counts != launch['expected_development']:
        raise ValueError('Development bootstrap clusters changed')
    if (launch['bootstrap']['replicates'] != protocol['selection']['paired_song_group_bootstrap_replicates']
            or launch['bootstrap']['confidence'] != protocol['selection']['confidence']):
        raise ValueError('Bootstrap differs from frozen D selection specification')
    hashes = dict(launch['protected_hashes'])
    hashes[relative(QUEUE_DIR/'report.json')] = sha256(QUEUE_DIR/'report.json')
    reports,configs,checks,lasts = [defaultdict(list) for _ in range(4)]
    for variant in protocol['variants']:
        for seed in protocol['optimization']['seeds']:
            rid = variant['id']+'-v1-s'+str(seed)
            run_dir,model_dir = paths(rid)
            if rid not in expected or is_running(rid) or is_running('verify-'+rid) or not verification_matches(rid):
                raise ValueError('Run is not independently completed: '+rid)
            report,config,check = [read_json(run_dir/name) for name in ('report.json','config.json','verification.json')]
            if (report['protocol_sha256'] != sha256(PROTOCOL) or report['contract_sha256'] != sha256(CONTRACT)
                    or config['settings'] != settings_for(protocol,variant,seed) or config['experiment'] != variant['id']
                    or report['completed_steps'] != 20000 or report['train_songs'] != 3070
                    or report['development_songs'] != 383 or report['auxiliary_songs'] != 255):
                raise ValueError('Report data/config/budget mismatch: '+rid)
            if any(report[k] != 0 for k in ('calibration_songs_used','original_validation_songs_used','test_songs_used')):
                raise ValueError('Heldout data was consumed')
            if report['continuous_auxiliary_VA_used'] or report['pseudo_labels_used']:
                raise ValueError('Auxiliary label semantics violated')
            last = torch.load(model_dir/'last.pth',map_location='cpu',weights_only=True)
            for path in [run_dir/name for name in ('config.json','report.json','verification.json')] + [model_dir/name for name in ('best.pth','last.pth')]:
                hashes[relative(path)] = sha256(path)
            key = variant['id'];reports[key].append(report);configs[key].append(config);checks[key].append(check)
            lasts[key].append({k:last[k] for k in ('torch_rng','cuda_rng','samplers','initial_state_sha256','exposure_chain','mean','std')})
    ref_key,cand_key = [v['id'] for v in protocol['variants']]
    if ref_key != 'dj00-control' or cand_key != 'dj01-observed-tags-010':
        raise ValueError('Frozen control/candidate identity changed')
    parity=[]
    for i,seed in enumerate(protocol['optimization']['seeds']):
        check_pair_fairness(reports[ref_key][i],reports[cand_key][i],configs[ref_key][i],configs[cand_key][i])
        r,c = lasts[ref_key][i],lasts[cand_key][i]
        for field in ('torch_rng','mean','std'):
            torch.testing.assert_close(r[field],c[field],rtol=0,atol=0)
        if len(r['cuda_rng']) != len(c['cuda_rng']) or not r['cuda_rng']:
            raise ValueError('Missing paired CUDA state')
        for left,right in zip(r['cuda_rng'],c['cuda_rng']):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        parity.append(dict(seed=seed,initial_state_sha256=r['initial_state_sha256'],exposure_chain=r['exposure_chain'],
                           main_exposures=r['samplers']['main_exposures'],auxiliary_forward_exposures=r['samplers']['auxiliary_forward_exposures'],
                           identical_scaler_and_final_torch_cuda_rng=True))
    result = compare(reports[ref_key],reports[cand_key],by_domain,protocol['optimization']['seeds'],launch['bootstrap'],protocol['selection'])
    costs = {key:dict(training_seconds=[r['training_seconds'] for r in reports[key]],
        gpu_training_peak_bytes=[r['gpu_peak_bytes'] for r in reports[key]],
        cached_development_inference_seconds=[r['cached_feature_inference_seconds'] for r in checks[key]],
        best_steps=[r['best_step'] for r in reports[key]],checkpoints=[r['checkpoint'] for r in reports[key]]) for key in (ref_key,cand_key)}
    verify_hashes(hashes);verify_hashes(contract['protected_hashes'])
    return dict(status='completed_verified',job_id=JOB,completed_at=now(),reference=ref_key,candidate=cand_key,
        source_hashes=hashes,bootstrap=launch['bootstrap'],development=counts,paired_fairness=parity,
        result=result,costs=costs,selection_thresholds=protocol['selection'],
        decision='eligible_for_original_validation_confirmation' if result['passes_internal_development_gates'] else 'Jamendo_auxiliary_not_selected',
        calibration_original_validation_test_targets_used=0,legacy_teacher_mixed_unchanged=True,
        release_status='Research decision only. Original validation confirmation, calibration, final test and human listening gates are still required.',
        limitations=contract['limitations']+[launch['bootstrap']['interpretation']])


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true');mode.add_argument('--launch',type=Path);args=p.parse_args()
    with run_lease(JOB):
        if args.register:
            register();return
        if args.launch.resolve() != DIRECTORY/'launch.json' or OUTPUT.exists():
            raise ValueError('Use the registered launch; preserve existing results')
        launch=read_json(args.launch);verify_hashes(launch['protected_hashes'])
        for name,value in launch['environment'].items():
            if os.environ.get(name)!=value:raise ValueError('Analysis environment changed: '+name)
        atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=process_identity()))
        with (DIRECTORY/'console.log').open('a',encoding='utf-8',buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                try:
                    started=time.monotonic();result=analyze(launch);result['analysis_seconds']=time.monotonic()-started
                    atomic_json(OUTPUT,result)
                    stages=read_json(PLAN)['stages']
                    stages['D'].update(jamendo_decision=relative(OUTPUT),phase='Jamendo_analysis_complete_D_summary_pending')
                    record_event(dict(event='D_Jamendo_auxiliary_paired_decision',decision=result['decision'],report=relative(OUTPUT)),
                        updates=dict(stages=stages,current_stage='D',pending_D_analysis=None,next_action=
                        'Record Jamendo and MERGE paired conclusions with all frozen gates; then close D with resource availability limits and prepare E calibration using the retained non-leaking B00 reference ensemble unless a candidate passed all gates. Keep heldout calibration/validation/test targets unopened until their protocols are frozen.'))
                    print(json.dumps(dict(decision=result['decision'],gain=result['result']['mean_paired_ccc_gain'],gates=result['result']['gates'])))
                except BaseException as exc:
                    traceback.print_exc();atomic_json(DIRECTORY/'failure.json',dict(failed_at=now(),error=f'{type(exc).__name__}: {exc}'))
                    record_event(dict(event='D_Jamendo_auxiliary_analysis_failed',error=f'{type(exc).__name__}: {exc}'));raise


if __name__=='__main__':main()
