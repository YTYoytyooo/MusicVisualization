from __future__ import annotations

from collections import defaultdict

import numpy as np

from tools.datasets.deam_dataset import DeamSequences, contained_path
from tools.train_next.core import ROOT, attach_targets, sha256
from tools.train_next.features import FeatureStore


class CachedFeatureStore(FeatureStore):
    def _load(self):
        raise RuntimeError('Stage B is cache-only; extraction is forbidden even if a cache disappears')


def load_internal(protocol, device, progress=None):
    from pathlib import Path
    manifest = Path(protocol['source_manifest'])
    if sha256(manifest) != protocol['manifest_sha256']:
        raise ValueError('Protocol manifest changed')
    if sha256(manifest.parent / 'config.json') != protocol['prepared_config_sha256']:
        raise ValueError('Prepared dataset configuration changed')
    dataset = DeamSequences(manifest.parent, split='train')
    assignment = protocol['split']['assignments']
    if {r['song_id'] for r in dataset.records} != set(assignment):
        raise ValueError('Original training IDs changed')
    chosen = sorted([(i, row) for i, row in enumerate(dataset.records)
                     if assignment[row['song_id']] in ('fit', 'development')], key=lambda pair: pair[1]['song_id'])
    store = CachedFeatureStore(device)
    if store.spec != protocol['feature_spec'] or store.key != protocol['feature_key']:
        raise ValueError('Frozen feature specification changed')
    songs = {'fit': [], 'development': []}
    # Verify every cache before get(): this path must never fall back to extraction.
    for _, row in chosen:
        path = store.directory / (row['audio_sha256'] + '.npz')
        if not path.is_file() or sha256(path) != protocol['feature_hashes'][row['audio_sha256']]:
            raise ValueError(f'Frozen cache missing/changed: {row["song_id"]}')
    for done, (index, row) in enumerate(chosen, 1):
        if sha256(contained_path(dataset.root, row['labels_path'])) != protocol['labels_sha256'][row['song_id']]:
            raise ValueError(f'Frozen labels changed: {row["song_id"]}')
        times, features, cached = store.get(row, dataset.root)
        if not cached:
            raise RuntimeError('New feature extraction is disabled in stage B')
        item = dataset[index]
        observed = item['mask'].any(axis=1)
        target, mask = attach_targets(times, item['times_seconds'][observed], item['targets'][observed], item['mask'][observed])
        if not mask.any():
            raise ValueError(f'Empty supervision: {row["song_id"]}')
        # Native clock is read before joining to the 0.5-second feature clock.
        native = float(np.median(np.diff(item['times_seconds'])))
        if native not in (.5, 1.):
            raise ValueError(f'Unexpected native annotation cadence: {native}')
        songs[assignment[row['song_id']]].append(dict(song_id=row['song_id'], source_dataset=row['source_dataset'],
            group_id=row['group_id'], x=features, y=target, mask=mask, times=times,
            native_seconds=native, audio_sha256=row['audio_sha256']))
        if progress and (done % 100 == 0 or done == len(chosen)):
            progress(dict(stage='loading_caches', done=done, total=len(chosen), song_id=row['song_id']))
    store.unload()
    return songs


class StepSampler:
    """Independent seeded draws per optimizer step allow exact crash recovery."""
    def __init__(self, songs, alpha, seed):
        groups = defaultdict(list)
        for i, song in enumerate(songs):
            groups[song['source_dataset']].append(i)
        self.names = sorted(groups)
        self.groups = [groups[name] for name in self.names]
        sizes = np.array([len(group) for group in self.groups], float)
        self.probabilities = sizes ** alpha / np.sum(sizes ** alpha)
        self.seed = seed

    def indices(self, step, count):
        rng = np.random.default_rng([self.seed, step])
        domains = rng.choice(len(self.groups), size=count, p=self.probabilities)
        return [self.groups[domain][rng.integers(len(self.groups[domain]))] for domain in domains]
