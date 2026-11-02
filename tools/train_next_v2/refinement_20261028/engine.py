"""Frozen-loop equivalent with explicit model seam and checkpointed time limit.

Derived from train_next_v2.run.train_loop; the historical file is untouched.
Control equivalence and interrupted/resumed equivalence are integration tested.
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch

from tools.train_next.core import atomic_json, collate_songs, fit_scaler
from tools.train_next.run import save_checkpoint
from tools.train_next_v2.common import AXES
from tools.train_next_v2.data import StepSampler
from tools.train_next_v2.evaluation import evaluate, score, assert_metrics_match
from tools.train_next_v2.run import cpu_state
from .model import build_model, objective


def train_loop(songs, settings, provenance, run_dir, model_dir, device, resume=False,
               progress=None, should_stop=None, after_checkpoint=None):
    fit, development = songs['fit'], songs['development']
    mean, std = fit_scaler(fit)
    model = build_model(len(mean), settings).to(device)
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
        # Caller rejects an expired window before loading data; this check also
        # permits a unit/integration caller to fail closed before evaluation.
        if should_stop and should_stop():
            return dict(status='paused_deadline', completed_steps=0, checkpoint_saved=False)
        initial = evaluate(model, development, mean, std, device, settings['batch_size'])
        best_score, best_metrics, best_state = score(initial), initial, cpu_state(model)
        history.append(dict(step=0, development_score=best_score, training_loss=None,
                            development={name: m['macro'] for name, m in initial.items()}, best=True))
    started = time.monotonic()

    def best_payload():
        return dict(schema_version=1, architecture='v3-refinement-20261028', label_order=list(AXES),
                    state_dict=best_state, settings=settings, provenance=provenance,
                    mean=torch.from_numpy(mean), std=torch.from_numpy(std), step=best_step,
                    development=best_metrics, development_score=best_score)

    def checkpoint(step):
        ns = np.random.get_state()
        payload = dict(schema_version=1, step=step, settings=settings, provenance=provenance,
            state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            mean=torch.from_numpy(mean), std=torch.from_numpy(std),
            best_step=best_step, best_score=best_score, best_state_dict=best_state,
            best_development=best_metrics, history=history,
            running_loss=running_loss, running_count=running_count,
            training_seconds=total_seconds + time.monotonic() - started,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            python_rng=random.getstate(), numpy_rng=(ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]))
        save_checkpoint(model_dir / 'last.pth', payload)
        save_checkpoint(model_dir / 'best.pth', best_payload())
        atomic_json(run_dir / 'metrics.json', history)
        if after_checkpoint:
            after_checkpoint(step)

    save_checkpoint(model_dir / 'best.pth', best_payload())
    atomic_json(run_dir / 'metrics.json', history)
    last_step = start_step - 1
    for step in range(start_step, settings['max_steps'] + 1):
        if should_stop and should_stop():
            checkpoint(last_step)
            return dict(status='paused_deadline', completed_steps=last_step, best_step=best_step,
                        checkpoint_saved=True, best_development_score=best_score)
        model.train()
        batch = [fit[i] for i in sampler.indices(step, settings['batch_size'])]
        x, target, mask, lengths = collate_songs(batch, mean, std)
        prediction = model(x.to(device), lengths)
        value = objective(prediction, target.to(device), mask.to(device), [s['native_seconds'] for s in batch], settings)
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
        if row is not None or step % settings['checkpoint_every'] == 0:
            checkpoint(step)
        if progress and (step % 50 == 0 or step == 1 or row is not None):
            progress(dict(stage='training', step=step, max_steps=settings['max_steps'],
                best_step=best_step, best_development_score=best_score,
                gradient_norm=float(gradient), learning_rate=settings['lr'],
                elapsed_training_seconds=round(total_seconds + time.monotonic() - started, 1),
                valid_labels=int(mask.sum()), gpu_peak_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0))
    if should_stop and should_stop():
        return dict(status='paused_deadline', completed_steps=last_step, best_step=best_step,
                    checkpoint_saved=True, best_development_score=best_score)
    best = torch.load(model_dir / 'best.pth', map_location='cpu', weights_only=True)
    verified = build_model(len(mean), settings).to(device)
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
