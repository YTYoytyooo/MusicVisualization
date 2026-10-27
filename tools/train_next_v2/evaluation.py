from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from tools.train_next.core import collate_songs
from .common import AXES


def metrics_for_axis(prediction, song, axis):
    valid = song['mask'][:, axis]
    p, y = prediction[valid, axis].astype(float), song['y'][valid, axis].astype(float)
    if not len(y):
        return None
    if not np.isfinite(p).all():
        raise ValueError('Non-finite prediction')
    pm, ym, ps, ys = p.mean(), y.mean(), p.std(), y.std()
    denominator = ps ** 2 + ys ** 2 + (pm - ym) ** 2
    elapsed = np.diff(song['times'][valid])
    adjacent = np.isclose(elapsed, song['native_seconds'], atol=1e-7, rtol=0)
    slope_error = np.abs((np.diff(p)[adjacent] - np.diff(y)[adjacent]) / elapsed[adjacent])
    return dict(values=len(y), mae=float(np.abs(p - y).mean()), rmse=float(np.sqrt(((p - y) ** 2).mean())),
        ccc=float(2 * ((p - pm) * (y - ym)).mean() / max(denominator, 1e-12)) if ys ** 2 > 1e-6 else None,
        mean_bias=float(pm - ym), mean_absolute_bias=float(abs(pm - ym)),
        truth_std=float(ys), prediction_std=float(ps), std_ratio=float(ps / ys) if ys > 1e-3 else None,
        truth_span90=float(np.percentile(y, 95) - np.percentile(y, 5)),
        prediction_span90=float(np.percentile(p, 95) - np.percentile(p, 5)),
        constant_prediction_mae=float(np.abs(pm - y).mean()),
        dynamics_beats_constant=float(np.abs(p - y).mean() < np.abs(pm - y).mean()),
        centered_rmse=float(np.sqrt(((p - pm - y + ym) ** 2).mean())),
        derivative_mae=float(slope_error.mean()) if len(slope_error) else None)


@torch.inference_mode()
def evaluate(model, songs, mean, std, device, batch_size=4):
    model.eval()
    by_domain = defaultdict(list)
    for song in songs:
        by_domain[song['source_dataset']].append(song)
    result = {}
    for domain, group in sorted(by_domain.items()):
        per_song = []
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            x, _, _, lengths = collate_songs(batch, mean, std)
            output = model(x.to(device), lengths).cpu().numpy()
            for i, song in enumerate(batch):
                prediction = output[i, :len(song['x'])]
                per_song.append(dict(song_id=song['song_id'], group_id=song['group_id'],
                    **{axis: metrics_for_axis(prediction, song, a) for a, axis in enumerate(AXES)}))
        macro, variation_groups = {}, {}
        for axis in AXES:
            rows = [r[axis] for r in per_song if r[axis] is not None]
            macro[axis] = {}
            if not rows:
                macro[axis] = dict(mae=None, rmse=None, ccc=None, ccc_songs=0, songs=0)
            for metric in rows[0] if rows else ():
                if metric == 'values':
                    continue
                values = [r[metric] for r in rows if r[metric] is not None]
                macro[axis][metric] = float(np.mean(values)) if values else None
            macro[axis]['ccc_songs'] = sum(r['ccc'] is not None for r in rows)
            macro[axis]['songs'] = len(rows)
            variation_groups[axis] = {}
            for label, predicate in [('low', lambda r: r['truth_std'] < .05), ('high', lambda r: r['truth_std'] >= .05)]:
                selected = [r for r in rows if predicate(r)]
                ccc = [r['ccc'] for r in selected if r['ccc'] is not None]
                variation_groups[axis][label] = dict(songs=len(selected), ccc_songs=len(ccc),
                    ccc=float(np.mean(ccc)) if ccc else None,
                    mae=float(np.mean([r['mae'] for r in selected])) if selected else None)
        result[domain] = dict(macro=macro, variation_groups=variation_groups, per_song=per_song)
    return result


def score(metrics):
    values = [group['macro'][axis]['ccc'] for group in metrics.values() for axis in AXES]
    if not values or any(v is None or not np.isfinite(v) for v in values):
        raise ValueError('Every development domain/axis needs valid varying targets')
    return float(np.mean(values))


def assert_metrics_match(expected, actual):
    if expected.keys() != actual.keys():
        raise ValueError('Development domains changed')
    largest = 0.
    for domain in expected:
        original = {r['song_id']: r for r in expected[domain]['per_song']}
        current = {r['song_id']: r for r in actual[domain]['per_song']}
        if original.keys() != current.keys():
            raise ValueError('Development IDs changed')
        for key in original:
            for axis in AXES:
                if original[key][axis] is None or current[key][axis] is None:
                    if original[key][axis] != current[key][axis]:
                        raise ValueError('Metric axis mask changed')
                    continue
                for metric in ('mae', 'rmse', 'ccc', 'mean_bias', 'centered_rmse', 'derivative_mae'):
                    old, new = original[key][axis][metric], current[key][axis][metric]
                    if old is None or new is None:
                        if old != new:
                            raise ValueError('Metric validity changed')
                    else:
                        np.testing.assert_allclose(old, new, rtol=1e-5, atol=1e-6)
                        largest = max(largest, abs(old - new))
    return largest
