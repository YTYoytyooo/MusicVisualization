"""Freeze group-isolated internal splits and eight bounded cache experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json

import numpy as np

from .common import ROOT, BASE, PLAN, read_json, record_event
from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease


def split_groups(records, seed=20261003):
    if any(r['split'] != 'train' for r in records):
        raise ValueError('Internal splits may only consume original training records')
    if len({r['song_id'] for r in records}) != len(records):
        raise ValueError('Duplicate song IDs')
    parent = {r['song_id']: r['song_id'] for r in records}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for field in ('group_id', 'audio_sha256'):
        owners = {}
        for row in records:
            key = row[field]
            if key in owners:
                parent[find(row['song_id'])] = find(owners[key])
            owners[key] = row['song_id']
    groups = defaultdict(list)
    for row in records:
        groups[find(row['song_id'])].append(row)
    domains = sorted({r['source_dataset'] for r in records})
    counts = np.array([sum(r['source_dataset'] == d for r in records) for d in domains])
    fractions = np.array([.8, .1, .1])
    targets = fractions[:, None] * counts[None, :]
    actual = np.zeros_like(targets)
    roles = ('fit', 'development', 'calibration')
    assignments = {}
    ordered = sorted(groups.values(), key=lambda g: (-len(g), hashlib.sha256(
        (str(seed) + ':' + min(r['song_id'] for r in g)).encode()).hexdigest()))
    for group in ordered:
        vector = np.array([sum(r['source_dataset'] == d for r in group) for d in domains])
        # Greedy reduction of per-domain, size-normalized squared target error.
        costs = [float(np.sum(((actual[i] + vector - targets[i]) ** 2 -
                               (actual[i] - targets[i]) ** 2) / np.maximum(targets[i], 1))) for i in range(3)]
        chosen = min(range(3), key=lambda i: (costs[i], i))
        actual[chosen] += vector
        for row in group:
            assignments[row['song_id']] = roles[chosen]
    for field in ('group_id', 'audio_sha256'):
        memberships = defaultdict(set)
        for row in records:
            memberships[row[field]].add(assignments[row['song_id']])
        if any(len(v) > 1 for v in memberships.values()):
            raise ValueError(f'Internal leakage: {field}')
    return dict(assignments=dict(sorted(assignments.items())), counts={
        role: {d: int(actual[i, j]) for j, d in enumerate(domains)} for i, role in enumerate(roles)},
        seed=seed, method='original train only; union group_id/audio_sha256; domain-stratified 80/10/10 greedy assignment',
        grouping_limitation='Memo2496 artist metadata incomplete; preserved audited source groups are not a guarantee of complete artist separation')


def experiment_matrix():
    return [
        dict(id='b00-reference', name='原结构从头训练对照', overrides={}),
        dict(id='b01-branch-fusion', name='CLAP/声学独立投影归一化', overrides={'fusion': 'branches'}),
        dict(id='b02-axis-heads', name='V/A独立预测头', overrides={'separate_heads': True}),
        dict(id='b03-ccc-half', name='CCC权重0.5', overrides={'ccc_weight': .5}),
        dict(id='b04-ccc-one', name='CCC权重1.0', overrides={'ccc_weight': 1.}),
        dict(id='b05-derivative', name='原生时间变化率损失', overrides={'derivative_weight': .05}),
        dict(id='b06-sampling-sqrt', name='按数据集样本量平方根采样', overrides={'sampling_alpha': .5}),
        dict(id='b07-sampling-size', name='按数据集样本量采样', overrides={'sampling_alpha': 1.}),
    ]


def prepare():
    import torch
    from tools.train_next.features import FeatureStore
    plan = read_json(PLAN)
    prepared = ROOT / plan['paths']['prepared_dataset']
    if sha256(prepared / 'manifest.jsonl') != plan['baseline']['manifest_sha256']:
        raise ValueError('Baseline manifest changed')
    records = [json.loads(line) for line in (prepared / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    # Test/validation metadata only establishes disjoint group identities; no labels are opened.
    for field in ('group_id', 'audio_sha256'):
        values = [{r[field] for r in records if r['split'] == split} for split in ('train', 'validation', 'test')]
        if any(values[i] & values[j] for i in range(3) for j in range(i)):
            raise ValueError(f'External split leakage: {field}')
    train = [r for r in records if r['split'] == 'train']
    split = split_groups(train)
    baseline = ROOT / plan['baseline']['checkpoint']
    if sha256(baseline) != plan['baseline']['checkpoint_sha256']:
        raise ValueError('Baseline checkpoint changed')
    metadata = torch.load(baseline, map_location='cpu', weights_only=True)
    store = FeatureStore(torch.device('cpu'))
    if store.spec != metadata['feature_spec']:
        raise ValueError('Frozen extractor specification changed')
    feature_hashes = {}
    consumed = [r for r in train if split['assignments'][r['song_id']] in ('fit', 'development')]
    for i, row in enumerate(consumed, 1):
        path = store.directory / (row['audio_sha256'] + '.npz')
        if not path.is_file():
            raise FileNotFoundError(f'No existing features: {row["song_id"]}')
        feature_hashes[row['audio_sha256']] = sha256(path)
        if i % 500 == 0 or i == len(consumed):
            print(json.dumps(dict(stage='freeze_feature_hashes', done=i, total=len(consumed))), flush=True)
    protocol = dict(schema_version=1, id='b-v1', source_manifest=str(prepared / 'manifest.jsonl'),
                    manifest_sha256=plan['baseline']['manifest_sha256'],
                    prepared_config_sha256=sha256(prepared / 'config.json'),
                    baseline_checkpoint_sha256=plan['baseline']['checkpoint_sha256'],
                    legacy_model_sha256=plan['baseline']['legacy_model_sha256'],
                    feature_key=store.key, feature_spec=store.spec, feature_hashes=feature_hashes,
                    labels_sha256={r['song_id']: metadata['config']['labels_sha256'][r['song_id']] for r in consumed},
                    split=split, experiments=experiment_matrix(),
                    defaults=dict(hidden=128, layers=1, dropout=.15, fusion='concatenate', separate_heads=False,
                                  lr=3e-4, weight_decay=1e-4, ccc_weight=.2, derivative_weight=0.,
                                  sampling_alpha=0., batch_size=4, max_steps=20000, evaluate_every=1000,
                                  checkpoint_every=250, seed=20261003, threads=8, clip_gradient=1.),
                    initialization='random from fixed seed; no legacy/teacher/mixed trained weights',
                    normalization='fit on internal fit songs only, shared across all variants',
                    budget='20000 optimizer updates x 4 songs = 80000 exposures for every variant; no score-based early stopping',
                    sampling='per-step deterministic sampling with replacement; domain probability proportional to N**alpha, then uniform song within domain',
                    low_variation_truth_std_threshold=.05,
                    selection=dict(primary='equal mean of dataset x axis macro per-song CCC on internal development only',
                                   checkpoint='largest primary score; require improvement >0.0001 to replace best',
                                   shortlist_count=2, repeat_seeds=[20261003, 20261004, 20261005],
                                   shortlist='rank candidate variants by internal development score; retain two and also repeat the reference for paired three-seed comparison',
                                   minimum_mean_gain=0.02, maximum_axis_domain_point_ccc_drop=.01,
                                   maximum_axis_domain_mae_increase=.01,
                                   confidence='paired song-group bootstrap; 2000 replicates; 95% interval; require every axis/domain CCC lower bound above -0.02 for promotion',
                                   gate_scope='Promotion after three-seed confirmation, not early stopping. Failed variants retain complete records.'),
                    sealed_roles=['calibration', 'original_validation', 'test'],
                    valid_label_time='exact native timestamps; independent V/A mask; no interpolation or cross-gap differences')
    target = BASE / 'protocols/b-v1.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if read_json(target) != protocol:
            raise ValueError('Frozen protocol differs; do not overwrite it')
    else:
        atomic_json(target, protocol)
    stages = plan['stages']
    stages['B'].update(status='in_progress', protocol=str(target.relative_to(ROOT)), protocol_sha256=sha256(target),
                       split_counts=split['counts'], experiment_count=8)
    record_event(dict(event='stage_B_protocol_frozen', protocol=str(target), counts=split['counts']),
                 updates=dict(status='in_progress', stages=stages,
                              next_action='验证独立训练实现和恢复，然后运行b00-reference随机初始化基线；保留内部校准、原验证和测试集合。'))
    print(json.dumps(dict(protocol=str(target), sha256=sha256(target), split_counts=split['counts'],
                          feature_files=len(feature_hashes), experiments=8), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    with run_lease('next-va-v2-protocol'):
        prepare()
