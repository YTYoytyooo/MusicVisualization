"""Run one frozen stage-B experiment with cache-only data and exact step resume."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from tools.train_next.core import atomic_json, collate_songs, fit_scaler, sha256, write_status
from tools.train_next.run import save_checkpoint
from tools.train_next.runtime import run_lease, lock_path
from .common import (AXES, ROOT, BASE, PLAN, GPU_LEASE, assert_gpu_available, code_hashes,
                     now, paths, process_identity, read_json, record_event, verify_hashes)
from .data import load_internal, StepSampler
from .evaluation import evaluate, score, assert_metrics_match
from .model import SequenceModel, loss


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_loop(songs, settings, provenance, run_dir, model_dir, device, resume=False,
               progress=None, after_checkpoint=None):
    fit, development = songs['fit'], songs['development']
    mean, std = fit_scaler(fit)
    model = SequenceModel(len(mean), settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    sampler = StepSampler(fit, settings['sampling_alpha'], settings['seed'])
    history, start_step, best_step, best_score = [], 1, 0, -float('inf')
    best_state, best_metrics = None, None
    running_loss, running_count, total_seconds = 0., 0, 0.
    if resume and (model_dir / 'last.pth').is_file():
        saved = torch.load(model_dir / 'last.pth', map_location='cpu', weights_only=True)
        if saved['provenance'] != provenance or saved['settings'] != settings:
            raise ValueError('Resume code, configuration, data or feature version mismatch')
        np.testing.assert_array_equal(saved['mean'].numpy(), mean)
        np.testing.assert_array_equal(saved['std'].numpy(), std)
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        ns = saved['numpy_rng']
        np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        start_step = saved['step'] + 1
        best_step, best_score, best_state = saved['best_step'], saved['best_score'], saved['best_state_dict']
        best_metrics, history = saved['best_development'], saved['history']
        running_loss, running_count = saved['running_loss'], saved['running_count']
        total_seconds = saved['training_seconds']
    else:
        initial = evaluate(model, development, mean, std, device, settings['batch_size'])
        best_score, best_metrics, best_state = score(initial), initial, cpu_state(model)
        history.append(dict(step=0, development_score=best_score, training_loss=None,
                            development={name: m['macro'] for name, m in initial.items()}, best=True))
    started = time.monotonic()

    def best_payload():
        return dict(schema_version=1, architecture='whole-song-va-experiment-v2', label_order=list(AXES),
            state_dict=best_state, settings=settings, provenance=provenance,
            mean=torch.from_numpy(mean), std=torch.from_numpy(std), step=best_step,
            development=best_metrics, development_score=best_score)

    # Reconstruct best from the authoritative full last checkpoint after a crash.
    save_checkpoint(model_dir / 'best.pth', best_payload())
    atomic_json(run_dir / 'metrics.json', history)
    last_step = start_step - 1
    for step in range(start_step, settings['max_steps'] + 1):
        model.train()
        indices = sampler.indices(step, settings['batch_size'])
        batch = [fit[i] for i in indices]
        x, target, mask, lengths = collate_songs(batch, mean, std)
        prediction = model(x.to(device), lengths)
        value = loss(prediction, target.to(device), mask.to(device), [s['native_seconds'] for s in batch], settings)
        if not torch.isfinite(value):
            raise ValueError('Non-finite objective')
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_gradient'], error_if_nonfinite=True)
        optimizer.step()
        running_loss += float(value.detach().cpu())
        running_count += 1
        last_step = step
        row = None
        if step % settings['evaluate_every'] == 0 or step == settings['max_steps']:
            metrics = evaluate(model, development, mean, std, device, settings['batch_size'])
            current = score(metrics)
            improved = current > best_score + 1e-4
            if improved:
                best_score, best_step, best_state, best_metrics = current, step, cpu_state(model), metrics
            row = dict(step=step, training_loss=running_loss / running_count, development_score=current,
                       best=improved, best_step=best_step, best_score=best_score,
                       development={name: m['macro'] for name, m in metrics.items()},
                       gradient_norm=float(gradient), learning_rate=settings['lr'],
                       training_seconds=total_seconds + time.monotonic() - started)
            history.append(row)
            running_loss, running_count = 0., 0
        checkpoint_due = row is not None or step % settings['checkpoint_every'] == 0
        if checkpoint_due:
            ns = np.random.get_state()
            checkpoint = dict(schema_version=1, step=step, settings=settings, provenance=provenance,
                state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                best_step=best_step, best_score=best_score, best_state_dict=best_state,
                best_development=best_metrics, history=history,
                running_loss=running_loss, running_count=running_count,
                training_seconds=total_seconds + time.monotonic() - started,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                python_rng=random.getstate(), numpy_rng=(ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]))
            save_checkpoint(model_dir / 'last.pth', checkpoint)
            save_checkpoint(model_dir / 'best.pth', best_payload())
            if row is not None:
                atomic_json(run_dir / 'metrics.json', history)
            if after_checkpoint:
                after_checkpoint(step)
        if progress and (step % 50 == 0 or step == 1 or row is not None):
            status = dict(stage='training', step=step, max_steps=settings['max_steps'],
                best_step=best_step, best_development_score=best_score,
                gradient_norm=float(gradient), learning_rate=settings['lr'],
                elapsed_training_seconds=round(total_seconds + time.monotonic() - started, 1),
                valid_labels=int(mask.sum()), gpu_peak_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
            if row:
                status.update(training_loss=row['training_loss'], development_score=row['development_score'])
            progress(status)
    best = torch.load(model_dir / 'best.pth', map_location='cpu', weights_only=True)
    verified = SequenceModel(len(mean), settings).to(device)
    verified.load_state_dict(best['state_dict'], strict=True)
    actual = evaluate(verified, development, mean, std, device, settings['batch_size'])
    difference = assert_metrics_match(best['development'], actual)
    return dict(status='completed', best_step=best_step, completed_steps=last_step,
        best_development=actual, best_development_score=score(actual),
        checkpoint_reload_verified=True, maximum_metric_difference=difference,
        train_songs=len(fit), development_songs=len(development),
        calibration_songs_used=0, original_validation_songs_used=0, test_songs_used=0,
        initialization='random; no previous trained weights',
        training_seconds=round(total_seconds + time.monotonic() - started, 2),
        gpu_peak_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)


def configuration(args):
    protocol_path = args.protocol.resolve()
    if not protocol_path.is_relative_to(BASE):
        raise ValueError('Protocol must be inside the optimization directory')
    protocol = read_json(protocol_path)
    experiment = next((item for item in protocol['experiments'] if item['id'] == args.experiment), None)
    if experiment is None:
        raise ValueError('Experiment is not in the frozen matrix')
    settings = {**protocol['defaults'], **experiment['overrides']}
    if args.seed is not None:
        if args.seed not in protocol['selection']['repeat_seeds']:
            raise ValueError('Seed outside frozen repeat set')
        settings['seed'] = args.seed
    return protocol, dict(run_id=args.run_id, experiment=args.experiment, stage='B', settings=settings,
        device=args.device, protocol=str(protocol_path), protocol_sha256=sha256(protocol_path),
        code=code_hashes(), feature_key=protocol['feature_key'], manifest_sha256=protocol['manifest_sha256'],
        initialization=protocol['initialization'], python=str(Path(sys.executable).resolve()))


def job_record(args, config, run_dir, model_dir, status):
    argv = [str(Path(sys.executable).resolve()), '-u', '-m', 'tools.train_next_v2.run',
            '--run-id', args.run_id, '--experiment', args.experiment,
            '--protocol', str(args.protocol.resolve()), '--device', args.device,
            '--seed', str(config['settings']['seed']), '--resume']
    return dict(run_id=args.run_id, stage='B', status=status, original_argv=argv, cwd=str(ROOT),
        python=config['python'], environment={name: os.environ.get(name) for name in
            ('PYTHONUTF8', 'HF_HUB_OFFLINE', 'HF_HUB_DISABLE_PROGRESS_BARS', 'TEMP', 'TMP',
             'CUBLAS_WORKSPACE_CONFIG', 'CUDA_CACHE_PATH', 'HF_HOME', 'TORCH_HOME', 'NUMBA_CACHE_DIR')},
        process_identity=process_identity() if status == 'running' else None,
        os_lock=str(lock_path(args.run_id)), gpu_lock=str(lock_path(GPU_LEASE)),
        log_paths=[str(run_dir / 'console.log')], config_hash=sha256(run_dir / 'config.json'),
        code_hashes=config['code'], data_hashes=dict(protocol=config['protocol_sha256'], manifest=config['manifest_sha256']),
        feature_spec=config['feature_key'], step_budget=config['settings']['max_steps'],
        checkpoint_paths=dict(best=str(model_dir / 'best.pth'), last=str(model_dir / 'last.pth')),
        started_at=now() if status == 'running' else None, registered_at=now(),
        recovery='Use original_argv with --resume; last.pth is authoritative, and exact code/data/config checks are mandatory.')


def main(args):
    protocol, config = configuration(args)
    run_dir, model_dir = paths(args.run_id)
    if run_dir.exists():
        if not (args.resume or args.register_only):
            raise ValueError('Run exists; use --resume')
        if not (run_dir / 'config.json').is_file() or read_json(run_dir / 'config.json') != config:
            raise ValueError('Existing configuration/code differs; do not overwrite or weaken checks')
    else:
        if args.resume:
            raise ValueError('Cannot resume an unregistered run')
        if model_dir.exists():
            raise ValueError('Model directory already exists')
        run_dir.mkdir(parents=True)
        model_dir.mkdir(parents=True)
        atomic_json(run_dir / 'config.json', config)
    if args.register_only:
        if (run_dir / 'report.json').exists() or (model_dir / 'last.pth').exists():
            raise ValueError('Do not re-register a started/completed run')
        row = job_record(args, config, run_dir, model_dir, 'registered')
        atomic_json(run_dir / 'launch.json', row)
        record_event(dict(event='experiment_registered', run_id=args.run_id), run=row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return
    if (run_dir / 'report.json').is_file():
        raise ValueError('Run already completed; inspect the report instead of restarting')
    assert_gpu_available()
    if args.device == 'cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching the registered command')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(config['settings']['threads'])
    torch.manual_seed(config['settings']['seed'])
    np.random.seed(config['settings']['seed'])
    random.seed(config['settings']['seed'])
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    row = job_record(args, config, run_dir, model_dir, 'running')
    atomic_json(run_dir / 'process.json', row)
    record_event(dict(event='experiment_started', run_id=args.run_id), run=row,
                 updates=dict(next_action=f'监测{args.run_id}的真实进度，完成后核验报告并启动下一组冻结实验。'))

    def progress(data):
        write_status(run_dir / 'status.json', dict(**data, process_identity=row['process_identity']))
        print(json.dumps(data, ensure_ascii=False), flush=True)

    started = time.monotonic()
    try:
        verify_hashes(config['code'])
        if sha256(ROOT / 'models/emotion_model.pth') != protocol['legacy_model_sha256']:
            raise ValueError('Legacy model changed')
        if sha256(ROOT / 'models/next_va/mixed-20261003-recovery/best.pth') != protocol['baseline_checkpoint_sha256']:
            raise ValueError('Baseline model changed')
        progress(dict(stage='verifying_and_loading_cache', protocol_sha256=config['protocol_sha256']))
        songs = load_internal(protocol, device, progress)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        result = train_loop(songs, config['settings'], config, run_dir, model_dir, device, args.resume, progress)
        verify_hashes(config['code'])
        legacy_ok = sha256(ROOT / 'models/emotion_model.pth') == protocol['legacy_model_sha256']
        baseline_ok = sha256(ROOT / 'models/next_va/mixed-20261003-recovery/best.pth') == protocol['baseline_checkpoint_sha256']
        if not legacy_ok or not baseline_ok:
            raise ValueError('Protected models changed during the experiment')
        result.update(run_id=args.run_id, experiment=args.experiment, stage='B', completed_at=now(),
            elapsed_seconds=round(time.monotonic() - started, 2), legacy_model_unchanged=legacy_ok,
            baseline_model_unchanged=baseline_ok, checkpoint=str(model_dir / 'best.pth'),
            checkpoint_sha256=sha256(model_dir / 'best.pth'), protocol_sha256=config['protocol_sha256'])
        atomic_json(run_dir / 'report.json', result)
        progress(dict(stage='completed', report=str(run_dir / 'report.json'), best_step=result['best_step'],
                      best_development_score=result['best_development_score']))
        record_event(dict(event='experiment_completed', run_id=args.run_id, best_step=result['best_step'],
                          score=result['best_development_score']),
                     run=dict(run_id=args.run_id, status='completed', result=str(run_dir / 'report.json'),
                              last_progress=dict(step=result['completed_steps'], best_score=result['best_development_score'])),
                     updates=dict(next_action='检查已完成的B阶段报告；按冻结矩阵顺序登记并运行下一个未完成对照，8组后进行筛选与三种子复验。'))
    except BaseException as exc:
        failure = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        progress(dict(stage=failure, error=f'{type(exc).__name__}: {exc}'))
        record_event(dict(event='experiment_' + failure, run_id=args.run_id, error=str(exc)),
                     run=dict(run_id=args.run_id, status=failure, error=str(exc)),
                     updates=dict(next_action=f'定位{args.run_id}的{failure}原因；确认OS锁释放后，严格按原配置及last.pth恢复。'))
        raise


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--protocol', type=Path, default=BASE / 'protocols/b-v1.json')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--register-only', action='store_true')
    args = parser.parse_args()
    if args.register_only:
        with run_lease(args.run_id):
            main(args)
    else:
        # The global GPU lease prevents overlap even if another run ID is used.
        with run_lease(GPU_LEASE), run_lease(args.run_id):
            run_dir, _ = paths(args.run_id)
            if not run_dir.is_dir():
                raise ValueError('Register --register-only before launch so commands and versions are durable')
            with (run_dir / 'console.log').open('a', encoding='utf-8', buffering=1) as log:
                with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                    main(args)
