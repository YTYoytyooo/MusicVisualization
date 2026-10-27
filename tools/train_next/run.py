"""Train a new DEAM-supervised V/A teacher; never overwrite the legacy model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

from tools.datasets.deam_dataset import DEFAULT_PREPARED, DeamSequences
from .core import (AXES, ROOT, SequenceVA, atomic_json, attach_targets, collate_songs,
                   evaluate, fit_scaler, output_paths, regression_metrics, select_records,
                   sha256, supervised_loss, write_status, replace_with_retry)
from .features import FeatureStore
from .runtime import run_lease


def save_checkpoint(path, value):
    temporary = path.with_suffix('.pth.tmp')
    torch.save(value, temporary)
    replace_with_retry(temporary, path)


def score(metrics):
    values = [metrics['macro'][axis]['ccc'] for axis in AXES]
    if any(v is None or not np.isfinite(v) for v in values):
        raise ValueError('Validation must contain varying labels for both axes')
    return float(np.mean(values))


def train(songs, config, run_dir, model_dir, device, feature_config, resume=False):
    training, validation = songs['train'], songs['validation']
    mean, std = fit_scaler(training)
    model = SequenceVA(len(mean), config['hidden']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    start_epoch, best_score, bad_epochs = 1, -float('inf'), 0
    if resume and (model_dir / 'last.pth').is_file():
        saved = torch.load(model_dir / 'last.pth', weights_only=True, map_location=device)
        if saved['config'] != config or saved['feature_spec'] != feature_config:
            raise ValueError('Checkpoint configuration mismatch')
        np.testing.assert_array_equal(saved['mean'].cpu().numpy(), mean)
        np.testing.assert_array_equal(saved['std'].cpu().numpy(), std)
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch, best_score, bad_epochs = saved['epoch'] + 1, saved['best_score'], saved['bad_epochs']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    target_mean = np.array([np.concatenate([s['y'][s['mask'][:, i], i] for s in training]).mean() for i in range(2)])
    baseline = regression_metrics([np.broadcast_to(target_mean, s['y'].shape) for s in validation], validation)
    atomic_json(run_dir / 'mean-baseline.json', baseline)
    epoch_history = []
    if resume and (run_dir / 'metrics.json').is_file():
        epoch_history = [r for r in json.loads((run_dir / 'metrics.json').read_text()) if r['epoch'] < start_epoch]
    for epoch in range(start_epoch, config['epochs'] + 1):
        if bad_epochs >= config['patience']:
            break
        started = time.monotonic()
        model.train()
        order = list(range(len(training)))
        random.Random(config['seed'] + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), config['song_batch']):
            batch = [training[i] for i in order[start:start + config['song_batch']]]
            x, target, mask, lengths = collate_songs(batch, mean, std)
            prediction = model(x.to(device), lengths)
            loss = supervised_loss(prediction, target.to(device), mask.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, validation, mean, std, device, config['song_batch'])
        current_score = score(metrics)
        improved = current_score > best_score + 1e-4
        if improved:
            best_score, bad_epochs = current_score, 0
        else:
            bad_epochs += 1
        checkpoint = dict(schema_version=1, architecture='whole-song-bilstm-va', label_order=AXES,
                          state_dict=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                          config=config, feature_spec=feature_config, mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                          best_score=best_score, bad_epochs=bad_epochs, validation=metrics,
                          torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        if improved:
            save_checkpoint(model_dir / 'best.pth', checkpoint)
        save_checkpoint(model_dir / 'last.pth', checkpoint)
        row = dict(epoch=epoch, training_loss=float(np.mean(losses)), validation=metrics['macro'],
                   mean_ccc=current_score, best=improved, seconds=round(time.monotonic() - started, 2))
        epoch_history.append(row)
        atomic_json(run_dir / 'metrics.json', epoch_history)
        write_status(run_dir / 'status.json', dict(stage='training', **row))
        print(json.dumps(row), flush=True)
    best = torch.load(model_dir / 'best.pth', map_location='cpu', weights_only=True)
    return dict(status='completed', stage=config['stage'], train_songs=len(training), validation_songs=len(validation),
                test_songs_used=0, best_epoch=best['epoch'], best_validation=best['validation'],
                mean_baseline=baseline['macro'], checkpoint=str(model_dir / 'best.pth'),
                feature_spec=feature_config, note='Pilot establishes pipeline viability, not accuracy or pseudo-label reliability.'
                if config['stage'] == 'pilot' else 'Supervised teacher candidate; test evaluation and pseudo-label calibration are separate steps.')


def main(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use the isolated training environment after setup_gpu.ps1, or explicitly select --device cpu.')
    dataset = DeamSequences(args.prepared, split=None)
    limits = {'train': 20, 'validation': 8} if args.stage == 'pilot' else {'train': 0, 'validation': 0}
    selected = {split: select_records(dataset.records, split, limit, args.seed) for split, limit in limits.items()}
    if any(len(records) < 2 for records in selected.values()):
        raise ValueError('At least two songs per train/validation split are required')
    for key in ('group_id', 'audio_sha256'):
        if {r[key] for r in selected['train']} & {r[key] for r in selected['validation']}:
            raise ValueError(f'Train/validation overlap in {key}')
    run_dir, model_dir = output_paths(args.run_id)
    if not args.resume and (run_dir.exists() or model_dir.exists()):
        raise ValueError('Run already exists; choose a new run-id or use --resume')
    if args.resume and not (run_dir / 'config.json').is_file():
        raise ValueError('Cannot resume a run without its configuration')
    labels = {r['song_id']: sha256(dataset.root / r['labels_path']) for rows in selected.values() for r in rows}
    config = dict(stage=args.stage, seed=args.seed, epochs=args.epochs or (5 if args.stage == 'pilot' else 50),
                  patience=8, hidden=128, lr=3e-4, song_batch=args.song_batch, feature_batch=args.feature_batch,
                  threads=args.threads, device=str(device), prepared=str(Path(args.prepared).resolve()),
                  manifest_sha256=sha256(Path(args.prepared) / 'manifest.jsonl'),
                  labels_sha256={str(k): v for k, v in labels.items()},
                  selected_ids={k: [r['song_id'] for r in v] for k, v in selected.items()},
                  code={name: sha256(Path(__file__).with_name(name)) for name in ('core.py', 'features.py', 'run.py', 'runtime.py')})
    if args.resume and json.loads((run_dir / 'config.json').read_text()) != config:
        raise ValueError('Run configuration or source data/code changed; use a new run-id')
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    model_dir.mkdir(parents=True, exist_ok=args.resume)
    atomic_json(run_dir / 'config.json', config)
    legacy = ROOT / 'models/emotion_model.pth'
    legacy_hash = sha256(legacy) if legacy.is_file() else None
    atomic_json(run_dir / 'environment.json', dict(python=sys.version, torch=torch.__version__, platform=platform.platform(),
                gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None, legacy_model_sha256=legacy_hash,
                packages={d.metadata['Name']: d.version for d in distributions()}))
    started = time.monotonic()
    try:
        write_status(run_dir / 'status.json', dict(stage='loading_feature_extractor'))
        store = FeatureStore(device, args.feature_batch,
                             progress=lambda update: write_status(run_dir / 'status.json', dict(stage='extracting_features', **update)))
        songs = {split: [] for split in selected}
        mapping = {r['song_id']: i for i, r in enumerate(dataset.records)}
        total = sum(map(len, selected.values()))
        finished = 0
        for split, records in selected.items():
            for record in records:
                times, features, cached = store.get(record, dataset.root)
                item = dataset[mapping[record['song_id']]]
                target, mask = attach_targets(times, item['times_seconds'], item['targets'], item['mask'])
                songs[split].append(dict(song_id=record['song_id'], x=features, y=target, mask=mask))
                finished += 1
                write_status(run_dir / 'status.json', dict(stage='extracting_features', done=finished, total=total, song_id=record['song_id'], cached=cached))
                print(json.dumps(dict(stage='features', done=finished, total=total, song_id=record['song_id'], cached=cached,
                                      frames=len(times), elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
        store.unload()
        report = train(songs, config, run_dir, model_dir, device, store.spec, args.resume)
        if legacy_hash is not None and sha256(legacy) != legacy_hash:
            raise RuntimeError('Legacy model changed during this run; investigate before using results')
        report.update(elapsed_seconds=round(time.monotonic() - started, 1), legacy_model_unchanged=True,
                      completed_at=datetime.now(timezone.utc).isoformat())
        atomic_json(run_dir / 'report.json', report)
        write_status(run_dir / 'status.json', dict(stage='completed', report=str(run_dir / 'report.json')))
        print(json.dumps(dict(status='completed', report=str(run_dir / 'report.json'), checkpoint=report['checkpoint'])), flush=True)
    except BaseException as exc:
        write_status(run_dir / 'status.json', dict(stage='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--prepared', type=Path, default=DEFAULT_PREPARED)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--feature-batch', type=int, default=8)
    parser.add_argument('--song-batch', type=int, default=4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if any(v is not None and v < 1 for v in (args.epochs, args.feature_batch, args.song_batch, args.threads)):
        parser.error('Epochs, batch sizes and thread count must be positive')
    with run_lease(args.run_id):
        main(args)
