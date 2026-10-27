"""Read-only validation diagnostics from a verified checkpoint and existing caches."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


AXES = ('valence', 'arousal')
SELECTIONS = (
    ('typical', '典型歌曲', .50, .50, 1., 1.),
    ('steady', '较平稳', .15, .50, 1., .25),
    ('dynamic', '变化明显', .85, .50, 1., .25),
    ('better', '拟合较好', .50, .85, .25, 1.),
    ('worse', '拟合较差', .50, .15, .25, 1.),
)


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else None


def lag_profile(times, truth, prediction, valid, maximum=8.):
    """Positive lag compares y(t) with p(t+lag): prediction follows the annotation.

    All shifts use the same native observed target points, excluding both edges.
    No target interpolation, circular shift, alignment correction, or test tuning.
    """
    times = np.asarray(times, float)
    observed = np.flatnonzero(valid)
    result = dict(best_seconds=None, best_correlation=None, zero_correlation=None,
                  improvement=None, points=0, search_seconds=[], correlations=[],
                  status='有效标注不足', credible=False)
    if len(observed) < 12:
        return result
    hop = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), hop, rtol=0, atol=1e-7):
        raise ValueError('Prediction timeline must be regularly sampled')
    native_steps = max(1, round(float(np.median(np.diff(times[observed]))) / hop))
    radius = int(min(maximum / hop, (observed[-1] - observed[0]) / 4))
    radius = radius // native_steps * native_steps
    core = observed[(observed >= observed[0] + radius) & (observed <= observed[-1] - radius)]
    if len(core) < 12 or radius == 0:
        return result
    shifts = np.arange(-radius, radius + 1, native_steps, dtype=int)
    correlations = [correlation(truth[core], prediction[core + shift]) for shift in shifts]
    result.update(points=len(core), search_seconds=(shifts * hop).tolist(), correlations=correlations)
    finite = [i for i, value in enumerate(correlations) if value is not None]
    if not finite:
        result['status'] = '曲线为常量，时延不可识别'
        return result
    best = max(finite, key=lambda i: (correlations[i], -abs(shifts[i])))
    zero = correlations[list(shifts).index(0)]
    gain = correlations[best] - zero if zero is not None else None
    if np.std(truth[core]) < .02 or np.std(prediction[core]) < .005:
        status = '曲线波动较小，时延不稳定'
    elif correlations[best] < .3:
        status = '相关性较弱，时延不可靠'
    elif abs(shifts[best]) == radius:
        status = '峰值在搜索边缘，时延不可靠'
    elif gain is None or gain < .05:
        status = '移位改善很小，无明显时延证据'
    else:
        status = '候选错位，需结合曲线核查'
    result.update(best_seconds=float(shifts[best] * hop), best_correlation=correlations[best],
                  zero_correlation=zero, improvement=gain, status=status,
                  credible=status == '候选错位，需结合曲线核查')
    return result


def axis_diagnostics(times, truth, prediction, valid):
    p, y = prediction[valid].astype(float), truth[valid].astype(float)
    if not len(y):
        return None
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite values in diagnostic input')
    pm, ym = float(p.mean()), float(y.mean())
    ps, ys = float(p.std()), float(y.std())
    span_p = float(np.percentile(p, 95) - np.percentile(p, 5))
    span_y = float(np.percentile(y, 95) - np.percentile(y, 5))
    denominator = ps * ps + ys * ys + (pm - ym) ** 2
    ccc = float(2 * ((p - pm) * (y - ym)).mean() / max(denominator, 1e-12)) if ys * ys > 1e-6 else None
    return dict(values=len(y), truth_mean=ym, prediction_mean=pm, mean_bias=pm - ym,
                truth_std=ys, prediction_std=ps, std_ratio=ps / ys if ys > 1e-3 else None,
                truth_span90=span_y, prediction_span90=span_p,
                span90_ratio=span_p / span_y if span_y > 1e-3 else None,
                mae=float(np.abs(p - y).mean()), rmse=float(np.sqrt(((p - y) ** 2).mean())),
                ccc=ccc, pearson=correlation(y, p),
                constant_prediction_mae=float(np.abs(pm - y).mean()),
                centered_rmse=float(np.sqrt(((p - pm - y + ym) ** 2).mean())),
                lag=lag_profile(times, truth, prediction, valid))


def percentile_rank(values):
    values = np.asarray(values, float)
    return np.array([np.mean(values < x) + .5 * np.mean(values == x) for x in values])


def select_representatives(rows):
    """Six deterministic examples, selected by disclosed validation-only strata."""
    if len(rows) < 6:
        raise ValueError('At least six validation songs per dataset are required')
    rows = sorted(rows, key=lambda r: r['song_id'])
    variation = percentile_rank([np.mean([r[a]['truth_std'] for a in AXES]) for r in rows])
    quality = percentile_rank([np.mean([r[a]['ccc'] if r[a]['ccc'] is not None else 0 for a in AXES]) for r in rows])
    selected, used = [], set()
    for key, label, v, q, wv, wq in SELECTIONS:
        index = min((i for i in range(len(rows)) if i not in used),
                    key=lambda i: (wv * abs(variation[i] - v) + wq * abs(quality[i] - q), rows[i]['song_id']))
        used.add(index)
        selected.append(dict(song_id=rows[index]['song_id'], category=key, reason=label,
                             variation_percentile=float(variation[index]), quality_percentile=float(quality[index])))
    index = max((i for i in range(len(rows)) if i not in used), key=lambda i: (rows[i]['duration_seconds'], rows[i]['song_id']))
    selected.append(dict(song_id=rows[index]['song_id'], category='long', reason='较长歌曲',
                         variation_percentile=float(variation[index]), quality_percentile=float(quality[index])))
    return selected


def collect(args):
    import torch
    from tools.datasets.deam_dataset import DeamSequences, contained_path
    from .core import ROOT, SequenceVA, attach_targets, collate_songs, sha256, atomic_json
    from .features import FeatureStore

    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Output must be a new directory inside Vm')
    torch.set_num_threads(8)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if checkpoint['architecture'] != 'whole-song-bilstm-va' or checkpoint['label_order'] != list(AXES):
        raise ValueError('Unsupported checkpoint architecture')
    for name, expected in config['code'].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Training code changed: {name}')
    prepared = Path(config['prepared'])
    if sha256(prepared / 'manifest.jsonl') != config['manifest_sha256']:
        raise ValueError('Prepared manifest changed')
    completion = json.loads((ROOT / 'data/training/next_va/completion-20261003.json').read_text(encoding='utf-8'))
    if completion['runs']['mixed']['checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Checkpoint does not match the verified mixed model')
    legacy_path = ROOT / 'models/emotion_model.pth'
    legacy_hash = sha256(legacy_path)
    if legacy_hash != completion['legacy_model_sha256']:
        raise ValueError('Legacy model changed since training completion')
    dataset = DeamSequences(prepared, split='validation')
    device = torch.device(args.device)
    store = FeatureStore(device)
    if store.spec != checkpoint['feature_spec']:
        raise ValueError('Feature specification or dependencies changed')
    for row in dataset.records:
        if not (store.directory / (row['audio_sha256'] + '.npz')).is_file():
            raise FileNotFoundError(f'Existing cache required: {row["song_id"]}; extraction is disabled')
    mean, std = checkpoint['mean'].numpy(), checkpoint['std'].numpy()
    model = SequenceVA(len(mean), config['hidden']).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    groups = defaultdict(list)
    for i, row in enumerate(dataset.records):
        groups[row['source_dataset']].append(i)
    rows, curves = [], {}
    max_metric_difference = 0.
    with torch.inference_mode():
        for domain, indices in sorted(groups.items()):
            expected_rows = {r['song_id']: r for r in checkpoint['validation_by_dataset'][domain]['per_song']}
            for start in range(0, len(indices), 4):
                batch, metadata, clocks = [], [], []
                for index in indices[start:start + 4]:
                    row = dataset.records[index]
                    if sha256(contained_path(dataset.root, row['labels_path'])) != config['labels_sha256'][row['song_id']]:
                        raise ValueError(f'Labels changed: {row["song_id"]}')
                    times, values, cached = store.get(row, dataset.root)
                    if not cached:
                        raise RuntimeError('Unexpected feature extraction')
                    item = dataset[index]
                    observed = item['mask'].any(axis=1)
                    target, mask = attach_targets(times, item['times_seconds'][observed], item['targets'][observed], item['mask'][observed])
                    batch.append(dict(song_id=row['song_id'], x=values, y=target, mask=mask))
                    metadata.append(row)
                    clocks.append(times)
                x, _, _, lengths = collate_songs(batch, mean, std)
                predictions = model(x.to(device), lengths).cpu().numpy()
                for song, meta, times, prediction in zip(batch, metadata, clocks, predictions):
                    prediction = prediction[:len(times)]
                    summary = {key: meta.get(key, '') for key in ('song_id', 'source_dataset', 'title', 'artist', 'duration_seconds', 'audio_path', 'group_id')}
                    summary['split'] = 'validation'
                    for a, axis in enumerate(AXES):
                        summary[axis] = axis_diagnostics(times, song['y'][:, a], prediction[:, a], song['mask'][:, a])
                        for metric in ('mae', 'rmse', 'ccc'):
                            actual, expected = summary[axis][metric], expected_rows[meta['song_id']][axis][metric]
                            if actual is None or expected is None:
                                if actual != expected:
                                    raise ValueError('Metric applicability changed')
                            else:
                                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                                max_metric_difference = max(max_metric_difference, abs(actual - expected))
                    rows.append(summary)
                    curves[meta['song_id']] = dict(times=times.tolist(), prediction=prediction.tolist(),
                                                   truth=[[float(v) if good else None for v, good in zip(pair, valid)]
                                                          for pair, valid in zip(song['y'], song['mask'])])
                if start % 80 == 0 or start + 4 >= len(indices):
                    print(json.dumps(dict(dataset=domain, done=min(start + 4, len(indices)), total=len(indices))), flush=True)
    representatives, domain_summaries = [], {}
    for domain in sorted(groups):
        group = [r for r in rows if r['source_dataset'] == domain]
        representatives.extend(select_representatives(group))
        domain_summaries[domain] = dict(songs=len(group))
        for axis in AXES:
            metrics = [r[axis] for r in group if r[axis] is not None]
            average = lambda key: float(np.mean([m[key] for m in metrics if m[key] is not None]))
            ratio = [m['std_ratio'] for m in metrics if m['std_ratio'] is not None]
            domain_summaries[domain][axis] = dict(mae=average('mae'), ccc=average('ccc'),
                mean_bias=average('mean_bias'), mean_absolute_bias=float(np.mean([abs(m['mean_bias']) for m in metrics])),
                median_std_ratio=float(np.median(ratio)),
                dynamics_beats_constant=int(sum(m['mae'] < m['constant_prediction_mae'] for m in metrics)),
                lag_candidates=int(sum(m['lag']['credible'] for m in metrics)))
    if sha256(legacy_path) != legacy_hash or sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError('Model changed during diagnostics')
    output.mkdir(parents=True)
    selected_ids = {r['song_id'] for r in representatives}
    report = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                  epoch=checkpoint['epoch'], feature_key=store.key, manifest_sha256=config['manifest_sha256'],
                  diagnostic_code_sha256=sha256(__file__), device=str(device),
                  validation_songs=len(rows), test_songs_used=0, training_performed=False,
                  checkpoint_metrics_reproduced=True, max_metric_difference=max_metric_difference,
                  legacy_model_unchanged=True, legacy_model_sha256=legacy_hash,
                  selection_policy='Six per dataset: near median, variability P15/P85, CCC P85/P15, longest remaining. Deterministic unique IDs; ranks use both axes.',
                  lag_policy='Pearson y(t) versus p(t+lag), same central observed points at every shift; native label cadence; max +/-8s capped at quarter observed duration. Positive means prediction follows label. Heuristic quality flags; no correction applied.',
                  domain_summaries=domain_summaries, songs=rows, representatives=representatives,
                  curves={key: value for key, value in curves.items() if key in selected_ids})
    atomic_json(output / 'diagnostics.json', report)
    print(json.dumps(dict(status='completed', output=str(output), selected=len(representatives),
                          validation_songs=len(rows), max_metric_difference=max_metric_difference)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/next_va/mixed-20261003-recovery/best.pth'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    collect(parser.parse_args())
