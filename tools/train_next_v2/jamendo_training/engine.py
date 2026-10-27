"""Equal-exposure paired training with strict, complete step-level recovery."""
from __future__ import annotations

import hashlib
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from tools.train_next.core import atomic_json, collate_songs, fit_scaler
from tools.train_next.run import save_checkpoint
from tools.train_next_v2.common import AXES
from tools.train_next_v2.evaluation import evaluate, score, assert_metrics_match
from tools.train_next_v2.model import loss
from tools.train_next_v2.run import cpu_state
from .model import AuxiliaryModel, tag_loss
from tools.train_next_v2.jamendo_data.contract import TAG_ORDER
from .sampling import PairedSampler, collate_auxiliary, digest

ARCHITECTURE = 'whole-song-va-with-observed-tags-v1'


def tensor_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def check_resume(saved, settings, provenance, mean, std, sampler):
    required = {'architecture', 'state_dict', 'optimizer', 'mean', 'std', 'samplers', 'torch_rng',
        'cuda_rng', 'python_rng', 'numpy_rng', 'step', 'best_step', 'best_score', 'best_state_dict',
        'best_development', 'history', 'running_losses', 'running_count', 'training_seconds',
        'initial_state_sha256', 'exposure_chain', 'amp_scaler', 'settings', 'provenance'}
    if not required.issubset(saved):
        raise ValueError('Incomplete full resume checkpoint')
    if (saved['architecture'] != ARCHITECTURE or saved['provenance'] != provenance
            or saved['settings'] != settings or saved['amp_scaler'] is not None):
        raise ValueError('Resume configuration/data/code/arithmetic mismatch')
    if not 0 < saved['step'] <= settings['max_steps'] or not 0 <= saved['best_step'] <= saved['step']:
        raise ValueError('Invalid checkpoint step')
    sampler.verify(saved['samplers'], saved['step'])
    np.testing.assert_array_equal(saved['mean'].numpy(), mean)
    np.testing.assert_array_equal(saved['std'].numpy(), std)
    if not np.isfinite(saved['best_score']):
        raise ValueError('Non-finite saved score')


def train_loop(songs, auxiliary, settings, provenance, run_dir, model_dir, device,
               resume=False, progress=None, after_checkpoint=None):
    fit, development = songs['fit'], songs['development']
    # Deliberately no auxiliary/development contribution to these statistics.
    mean, std = fit_scaler(fit)
    model = AuxiliaryModel(len(mean), settings).to(device)
    initial_hash = tensor_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    sampler = PairedSampler(fit, auxiliary, settings)
    history, start_step, best_step, total_seconds = [], 1, 0, 0.
    running_losses, running_count = [0., 0., 0.], 0
    exposure_chain = digest(sampler.definition)
    if resume and (model_dir / 'last.pth').is_file():
        saved = torch.load(model_dir / 'last.pth', weights_only=True, map_location='cpu')
        check_resume(saved, settings, provenance, mean, std, sampler)
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            if not saved['cuda_rng']:
                raise ValueError('Missing CUDA RNG state')
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        random.setstate(saved['python_rng'])
        ns = saved['numpy_rng']
        np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        start_step, best_step = saved['step'] + 1, saved['best_step']
        best_score, best_state = saved['best_score'], saved['best_state_dict']
        best_metrics, history = saved['best_development'], saved['history']
        running_losses, running_count = saved['running_losses'], saved['running_count']
        total_seconds, exposure_chain = saved['training_seconds'], saved['exposure_chain']
        initial_hash = saved['initial_state_sha256']
    else:
        initial = evaluate(model, development, mean, std, device, settings['batch_size'])
        best_score, best_metrics, best_state = score(initial), initial, cpu_state(model)
        history.append(dict(step=0, development_score=best_score, main_loss=None, tag_loss=None,
            total_loss=None, development={d: m['macro'] for d, m in initial.items()}, best=True))
    started = time.monotonic()

    def best_payload():
        return dict(schema_version=1, architecture=ARCHITECTURE, label_order=list(AXES),
            tag_order=list(TAG_ORDER), state_dict=best_state,
            settings=settings, provenance=provenance, mean=torch.from_numpy(mean), std=torch.from_numpy(std),
            step=best_step, development=best_metrics, development_score=best_score,
            initial_state_sha256=initial_hash)

    # A fully saved last checkpoint is authoritative if an exit interrupted the best write.
    save_checkpoint(model_dir / 'best.pth', best_payload())
    atomic_json(run_dir / 'metrics.json', history)
    last_step = start_step - 1
    for step in range(start_step, settings['max_steps'] + 1):
        model.train()
        main_ids, aux_ids = sampler.indices(step)
        exposure_chain = digest([exposure_chain, step, main_ids, aux_ids])
        batch = [fit[i] for i in main_ids]
        clips = [auxiliary[i] for i in aux_ids]
        x, target, mask, lengths = collate_songs(batch, mean, std)
        ax, classes, alengths = collate_auxiliary(clips, mean, std)
        prediction = model(x.to(device), lengths)
        main_value = loss(prediction, target.to(device), mask.to(device), [s['native_seconds'] for s in batch], settings)
        # BOTH arms execute the same auxiliary forward/backward graph and draws.
        logits = model.classify(ax.to(device), alengths)
        aux_value = tag_loss(logits, classes.to(device))
        value = main_value + settings['tag_loss_weight'] * aux_value
        if not torch.isfinite(value) or not torch.isfinite(aux_value):
            raise ValueError('Non-finite main/auxiliary objective')
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_gradient'], error_if_nonfinite=True)
        optimizer.step()
        numbers = [float(v.detach().cpu()) for v in (main_value, aux_value, value)]
        running_losses = [a + b for a, b in zip(running_losses, numbers)]
        running_count += 1
        last_step, row = step, None
        if step % settings['evaluate_every'] == 0 or step == settings['max_steps']:
            metrics = evaluate(model, development, mean, std, device, settings['batch_size'])
            current = score(metrics)
            improved = current > best_score + 1e-4
            if improved:
                best_score, best_step, best_state, best_metrics = current, step, cpu_state(model), metrics
            row = dict(step=step, main_loss=running_losses[0]/running_count,
                tag_loss=running_losses[1]/running_count, total_loss=running_losses[2]/running_count,
                development_score=current, best=improved, best_step=best_step, best_score=best_score,
                development={d: m['macro'] for d, m in metrics.items()}, gradient_norm=float(gradient),
                learning_rate=settings['lr'], training_seconds=total_seconds + time.monotonic()-started)
            history.append(row)
            running_losses, running_count = [0., 0., 0.], 0
        if row is not None or step % settings['checkpoint_every'] == 0:
            ns = np.random.get_state()
            checkpoint = dict(schema_version=1, architecture=ARCHITECTURE, step=step,
                settings=settings, provenance=provenance, state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                mean=torch.from_numpy(mean), std=torch.from_numpy(std), amp_scaler=None,
                samplers=sampler.state(step), initial_state_sha256=initial_hash, exposure_chain=exposure_chain,
                best_step=best_step, best_score=best_score, best_state_dict=best_state,
                best_development=best_metrics, history=history, running_losses=running_losses,
                running_count=running_count, training_seconds=total_seconds + time.monotonic()-started,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                python_rng=random.getstate(), numpy_rng=(ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]))
            save_checkpoint(model_dir / 'last.pth', checkpoint)
            save_checkpoint(model_dir / 'best.pth', best_payload())
            if row is not None:
                atomic_json(run_dir / 'metrics.json', history)
            if after_checkpoint:
                after_checkpoint(step)
        if progress and (step % 50 == 0 or step == 1 or row is not None):
            status = dict(stage='training', step=step, max_steps=settings['max_steps'], best_step=best_step,
                best_development_score=best_score, main_loss=numbers[0], tag_loss=numbers[1],
                auxiliary_weight=settings['tag_loss_weight'], gradient_norm=float(gradient),
                learning_rate=settings['lr'], valid_main_labels=int(mask.sum()),
                main_exposures=step*settings['batch_size'], auxiliary_forward_exposures=step*settings['aux_batch_size'],
                elapsed_training_seconds=round(total_seconds+time.monotonic()-started, 1),
                gpu_peak_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
            if row:
                status['development_score'] = row['development_score']
            progress(status)
    best = torch.load(model_dir / 'best.pth', weights_only=True, map_location='cpu')
    verified = AuxiliaryModel(len(mean), settings).to(device)
    verified.load_state_dict(best['state_dict'], strict=True)
    actual = evaluate(verified, development, mean, std, device, settings['batch_size'])
    difference = assert_metrics_match(best['development'], actual)
    return dict(status='completed', best_step=best_step, completed_steps=last_step,
        best_development=actual, best_development_score=score(actual),
        checkpoint_reload_verified=True, maximum_metric_difference=difference,
        train_songs=len(fit), development_songs=len(development), auxiliary_songs=len(auxiliary),
        calibration_songs_used=0, original_validation_songs_used=0, test_songs_used=0,
        continuous_auxiliary_VA_used=False, pseudo_labels_used=False, initialization='fresh paired random weights',
        initial_state_sha256=initial_hash, exposure_chain=exposure_chain, samplers=sampler.state(last_step),
        training_seconds=round(total_seconds+time.monotonic()-started, 2),
        gpu_peak_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0)
