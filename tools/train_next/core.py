"""Song-level sequence training, with independent missing-label masks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT = Path(__file__).resolve().parents[2]
AXES = ['valence', 'arousal']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def replace_with_retry(source, destination):
    """Readers without FILE_SHARE_DELETE can briefly block Windows replacement."""
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.05 * 2**attempt, .5))


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_status(path, data):
    """Progress reporting is best effort; checkpoints and configuration are not."""
    try:
        atomic_json(path, {**data, 'updated_at': datetime.now(timezone.utc).isoformat()})
        return True
    except OSError as exc:
        print(json.dumps(dict(event='status_write_failed', path=str(path), error=str(exc))), file=sys.stderr, flush=True)
        return False


def output_paths(run_id):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', run_id):
        raise ValueError('run-id must contain only letters, digits, underscores or hyphens')
    return ROOT / 'data/training/next_va/runs' / run_id, ROOT / 'models/next_va' / run_id


def select_records(records, split, limit, seed):
    if split not in ('train', 'validation'):
        raise ValueError('Training cannot consume the test split')
    selected = sorted((r for r in records if r['split'] == split), key=lambda r: r['song_id'])
    if limit:
        longest = max(selected, key=lambda r: r.get('duration_seconds', 0)) if selected else None
        others = [r for r in selected if r is not longest]
        random.Random(seed).shuffle(others)
        selected = ([longest] + others[:limit - 1]) if longest else []
    return sorted(selected, key=lambda r: r['song_id'])


def attach_targets(times, label_times, labels, label_mask):
    """Require exact timestamp joins, not row truncation or gap interpolation."""
    targets = np.zeros((len(times), 2), dtype=np.float32)
    mask = np.zeros_like(targets, dtype=bool)
    indices = np.searchsorted(times, label_times)
    if np.any(indices >= len(times)) or not np.allclose(times[indices], label_times, atol=1e-7, rtol=0):
        raise ValueError('Feature timestamps do not contain every label timestamp')
    targets[indices] = labels
    mask[indices] = label_mask
    return targets, mask


def fit_scaler(train_songs):
    if not train_songs:
        raise ValueError('Empty training set')
    count = 0
    total = np.zeros(train_songs[0]['x'].shape[1], dtype=np.float64)
    squares = np.zeros_like(total)
    for song in train_songs:
        values = song['x'].astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite features')
        total += values.sum(axis=0)
        squares += (values * values).sum(axis=0)
        count += len(values)
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(0, squares / count - mean * mean)), 1e-5)
    return mean.astype(np.float32), std.astype(np.float32)


def collate_songs(songs, mean, std):
    lengths = torch.tensor([len(s['x']) for s in songs], dtype=torch.int64)
    longest = int(lengths.max())
    x = torch.zeros((len(songs), longest, len(mean)), dtype=torch.float32)
    y = torch.zeros((len(songs), longest, 2), dtype=torch.float32)
    mask = torch.zeros_like(y, dtype=torch.bool)
    for i, song in enumerate(songs):
        n = len(song['x'])
        x[i, :n] = torch.from_numpy((song['x'] - mean) / std)
        y[i, :n] = torch.from_numpy(song['y'])
        mask[i, :n] = torch.from_numpy(song['mask'])
    return x, y, mask, lengths


class SequenceVA(nn.Module):
    """Offline bidirectional sequence model; each forward item is one song."""
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(.15), nn.Linear(hidden * 2, 2), nn.Tanh())

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.norm(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        return self.head(output)


def supervised_loss(prediction, target, mask, ccc_weight=.2):
    terms = []
    for song in range(len(prediction)):
        for axis in range(2):
            good = mask[song, :, axis]
            if not torch.any(good):
                continue
            p, y = prediction[song, good, axis], target[song, good, axis]
            term = nn.functional.smooth_l1_loss(p, y, beta=.2)
            if len(y) > 1 and torch.var(y, unbiased=False) > 1e-6:
                pm, ym = p.mean(), y.mean()
                covariance = ((p - pm) * (y - ym)).mean()
                denominator = p.var(unbiased=False) + y.var(unbiased=False) + (pm - ym).square()
                ccc = 2 * covariance / (denominator + 1e-8)
                term = term + ccc_weight * (1 - ccc)
            terms.append(term)
    if not terms:
        raise ValueError('Batch has no supervised values')
    return torch.stack(terms).mean()


def regression_metrics(predictions, songs):
    per_song = []
    for prediction, song in zip(predictions, songs):
        row = {'song_id': song['song_id']}
        for index, axis in enumerate(AXES):
            valid = song['mask'][:, index]
            p, y = prediction[valid, index].astype(np.float64), song['y'][valid, index].astype(np.float64)
            if not len(y):
                row[axis] = None
                continue
            if not np.isfinite(p).all():
                raise ValueError('Non-finite prediction')
            denominator = p.var() + y.var() + (p.mean() - y.mean()) ** 2
            ccc = float(2 * ((p - p.mean()) * (y - y.mean())).mean() / max(denominator, 1e-12)) if y.var() > 1e-6 else None
            row[axis] = dict(mae=float(np.abs(p - y).mean()), rmse=float(np.sqrt(((p - y) ** 2).mean())), ccc=ccc, values=len(y))
        per_song.append(row)
    macro = {}
    for axis in AXES:
        rows = [r[axis] for r in per_song if r[axis] is not None]
        macro[axis] = {}
        for metric in ('mae', 'rmse', 'ccc'):
            values = [r[metric] for r in rows if r[metric] is not None]
            macro[axis][metric] = float(np.mean(values)) if values else None
        macro[axis]['ccc_songs'] = sum(r['ccc'] is not None for r in rows)
    return dict(macro=macro, per_song=per_song)


@torch.inference_mode()
def evaluate(model, songs, mean, std, device, batch_size=4):
    model.eval()
    predictions = []
    for start in range(0, len(songs), batch_size):
        group = songs[start:start + batch_size]
        x, _, _, lengths = collate_songs(group, mean, std)
        output = model(x.to(device), lengths).cpu().numpy()
        predictions.extend(output[i, :len(song['x'])] for i, song in enumerate(group))
    return regression_metrics(predictions, songs)
