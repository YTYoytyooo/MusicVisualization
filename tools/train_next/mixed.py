"""Continue a frozen-feature DEAM teacher with audited supplemental labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from tools.datasets.deam_dataset import DeamSequences
from .core import (ROOT, AXES, SequenceVA, atomic_json, attach_targets, collate_songs,
                   evaluate, output_paths, sha256, supervised_loss, write_status)
from .features import FeatureStore
from .run import save_checkpoint
from .runtime import run_lease


def domain_metrics(model, songs, mean, std, device, batch_size):
    groups = defaultdict(list)
    for song in songs:
        groups[song['source_dataset']].append(song)
    return {name: evaluate(model, group, mean, std, device, batch_size)
            for name, group in sorted(groups.items())}


def metric_score(metrics):
    values = [result['macro'][axis]['ccc'] for result in metrics.values() for axis in AXES]
    if not values or any(v is None or not np.isfinite(v) for v in values):
        raise ValueError('Each validation dataset needs varying labels for both axes')
    return float(np.mean(values))


def balanced_order(songs, epoch, seed):
    groups = defaultdict(list)
    for i, song in enumerate(songs):
        groups[song['source_dataset']].append(i)
    rng = random.Random(seed + epoch)
    size = max(map(len, groups.values()))
    order = []
    for group in groups.values():
        items = []
        while len(items) < size:
            shuffled = list(group)
            rng.shuffle(shuffled)
            items.extend(shuffled)
        order.extend(items[:size])
    rng.shuffle(order)
    return order


def main(args):
    torch.set_num_threads(8)
    torch.manual_seed(20261002)
    np.random.seed(20261002)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    run_dir, model_dir = output_paths(args.run_id)
    if not args.resume and (run_dir.exists() or model_dir.exists()):
        raise ValueError('Run exists; choose a new run-id or use --resume')
    initial = torch.load(args.initialize, weights_only=True, map_location='cpu')
    dataset = DeamSequences(args.prepared, split=None)
    selected = [r for r in dataset.records if r['split'] in ('train', 'validation')]
    for key in ('group_id', 'audio_sha256'):
        groups = [{r[key] for r in selected if r['split'] == split} for split in ('train', 'validation')]
        if groups[0] & groups[1]:
            raise ValueError(f'Validation leakage in {key}')
    config = dict(stage='mixed_supervised', epochs=args.epochs, patience=8, hidden=128,
                  lr=1e-4, seed=20261002, song_batch=4, feature_batch=8, device=str(device),
                  prepared=str(args.prepared.resolve()), manifest_sha256=sha256(args.prepared / 'manifest.jsonl'),
                  initialization_sha256=sha256(args.initialize),
                  initialization=str(args.initialize.resolve()),
                  labels_sha256={str(r['song_id']): sha256(dataset.root / r['labels_path']) for r in selected},
                  code={name: sha256(Path(__file__).with_name(name)) for name in ('core.py', 'features.py', 'mixed.py', 'run.py', 'runtime.py')},
                  normalization='frozen DEAM-training scaler from initializer',
                  sampling='equal examples per dataset per epoch, cycle shuffled smaller datasets',
                  checkpoint_guard='DEAM mean V/A CCC cannot fall >0.02 below initializer',
                  pseudo_labels_used=False)
    if args.resume:
        if not (run_dir / 'config.json').exists() or json.loads((run_dir / 'config.json').read_text()) != config:
            raise ValueError('Resume configuration, source labels, or code changed')
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    model_dir.mkdir(parents=True, exist_ok=args.resume)
    atomic_json(run_dir / 'config.json', config)
    legacy_hash = sha256(ROOT / 'models/emotion_model.pth')
    started = time.monotonic()
    try:
        store = FeatureStore(device, 8, progress=lambda row: write_status(run_dir / 'status.json', dict(stage='extracting_features', **row)))
        if store.spec != initial['feature_spec']:
            raise ValueError('Initializer and current feature extraction specifications differ')
        songs = {'train': [], 'validation': []}
        indices = {str(r['song_id']): i for i, r in enumerate(dataset.records)}
        for n, row in enumerate(selected, 1):
            times, values, cached = store.get(row, dataset.root)
            item = dataset[indices[str(row['song_id'])]]
            observed = item['mask'].any(axis=1)
            target, mask = attach_targets(times, item['times_seconds'][observed], item['targets'][observed], item['mask'][observed])
            if not mask.any():
                raise ValueError(f'No valid labels for {row["song_id"]}')
            songs[row['split']].append(dict(song_id=row['song_id'], source_dataset=row['source_dataset'],
                                           x=values, y=target, mask=mask))
            status = dict(stage='extracting_features', done=n, total=len(selected), song_id=row['song_id'], cached=cached)
            write_status(run_dir / 'status.json', status)
            print(json.dumps(status), flush=True)
        store.unload()
        mean, std = initial['mean'].numpy(), initial['std'].numpy()
        model = SequenceVA(len(mean), 128).to(device)
        model.load_state_dict(initial['state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        initial_metrics = domain_metrics(model, songs['validation'], mean, std, device, 4)
        atomic_json(run_dir / 'initial-validation.json', initial_metrics)
        deam_floor = metric_score({'deam': initial_metrics['deam']}) - .02
        best_score, bad_epochs, start_epoch, history = metric_score(initial_metrics), 0, 1, []

        def checkpoint(epoch, metrics):
            return dict(schema_version=1, architecture='whole-song-bilstm-va', label_order=AXES,
                        state_dict=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                        config=config, feature_spec=store.spec, mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                        best_score=best_score, bad_epochs=bad_epochs, validation_by_dataset=metrics,
                        epoch_history=history,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])

        if args.resume and (model_dir / 'last.pth').exists():
            saved = torch.load(model_dir / 'last.pth', weights_only=True, map_location=device)
            if saved['config'] != config:
                raise ValueError('Checkpoint configuration changed')
            model.load_state_dict(saved['state_dict'])
            optimizer.load_state_dict(saved['optimizer'])
            torch.set_rng_state(saved['torch_rng'].cpu())
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
            start_epoch, best_score, bad_epochs = saved['epoch'] + 1, saved['best_score'], saved['bad_epochs']
            saved_history = saved.get('epoch_history')
            if saved_history is None:
                saved_history = json.loads((run_dir / 'metrics.json').read_text()) if (run_dir / 'metrics.json').exists() else []
            history = [r for r in saved_history if r['epoch'] < start_epoch]
            atomic_json(run_dir / 'metrics.json', history)
        else:
            save_checkpoint(model_dir / 'best.pth', checkpoint(0, initial_metrics))
        for epoch in range(start_epoch, args.epochs + 1):
            if bad_epochs >= config['patience']:
                break
            model.train()
            order = balanced_order(songs['train'], epoch, config['seed'])
            losses = []
            for pos in range(0, len(order), 4):
                batch = [songs['train'][i] for i in order[pos:pos + 4]]
                x, y, mask, lengths = collate_songs(batch, mean, std)
                prediction = model(x.to(device), lengths)
                loss = supervised_loss(prediction, y.to(device), mask.to(device))
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            metrics = domain_metrics(model, songs['validation'], mean, std, device, 4)
            score = metric_score(metrics)
            guard = metric_score({'deam': metrics['deam']}) >= deam_floor
            improved = guard and score > best_score + 1e-4
            if improved:
                best_score, bad_epochs = score, 0
            else:
                bad_epochs += 1
            row = dict(epoch=epoch, training_loss=float(np.mean(losses)), mean_dataset_ccc=score,
                       deam_guard_passed=guard, best=improved,
                       validation={name: result['macro'] for name, result in metrics.items()})
            history.append(row)
            state = checkpoint(epoch, metrics)
            if improved:
                save_checkpoint(model_dir / 'best.pth', state)
            save_checkpoint(model_dir / 'last.pth', state)
            atomic_json(run_dir / 'metrics.json', history)
            write_status(run_dir / 'status.json', dict(stage='training', **row))
            print(json.dumps(row), flush=True)
        best = torch.load(model_dir / 'best.pth', weights_only=True, map_location='cpu')
        assert sha256(ROOT / 'models/emotion_model.pth') == legacy_hash
        report = dict(status='completed', best_epoch=best['epoch'], improved_over_initializer=best['epoch'] > 0,
                      best_validation=best['validation_by_dataset'], initializer_validation=initial_metrics,
                      counts={split: dict(Counter(s['source_dataset'] for s in rows)) for split, rows in songs.items()},
                      test_songs_used=0, pseudo_labels_used=False, legacy_model_unchanged=True,
                      elapsed_seconds=round(time.monotonic() - started, 1), checkpoint=str(model_dir / 'best.pth'))
        atomic_json(run_dir / 'report.json', report)
        write_status(run_dir / 'status.json', dict(stage='completed', report=str(run_dir / 'report.json')))
        print(json.dumps(dict(status='completed', checkpoint=report['checkpoint'])), flush=True)
    except BaseException as exc:
        write_status(run_dir / 'status.json', dict(stage='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, default=ROOT / 'data/datasets/combined/prepared/v1')
    p.add_argument('--initialize', type=Path, default=ROOT / 'models/next_va/teacher-20261002/best.pth')
    p.add_argument('--run-id', default='mixed-20261002')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.epochs < 1:
        p.error('epochs must be positive')
    with run_lease(args.run_id):
        main(args)
