"""Strict C feature/target loading through the unchanged B timestamp join."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from tools.datasets.deam_dataset import DeamSequences, contained_path
from tools.train_next.core import ROOT, attach_targets, sha256
from tools.train_next_v2.common import BASE, read_json, verify_hashes

CONTRACT = BASE / 'protocols/c-pilot-training-inputs-v1.json'


def array_fingerprint(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_dataset(b_protocol):
    manifest = Path(b_protocol['source_manifest'])
    if sha256(manifest) != b_protocol['manifest_sha256'] or sha256(manifest.parent/'config.json') != b_protocol['prepared_config_sha256']:
        raise ValueError('Frozen source dataset manifest/configuration changed')
    dataset = DeamSequences(manifest.parent, split='train')
    return dataset, {row['song_id']: index for index, row in enumerate(dataset.records)}


def validate_membership(songs, assignments):
    if len(songs)!=90 or len({row['song_id'] for row in songs})!=90:
        raise ValueError('Pilot must contain exactly 90 distinct songs')
    if len({row['linked_component'] for row in songs})!=90 or len({row['audio_sha256'] for row in songs})!=90:
        raise ValueError('Pilot contains linked or duplicated recordings')
    counts = Counter((row['role'],row['source_dataset']) for row in songs)
    expected = {(role,domain):n for role,n in [('fit',20),('development',10)] for domain in ('deam','memo2496','pmemo')}
    if dict(counts)!=expected or any(assignments.get(row['song_id'])!=row['role'] for row in songs):
        raise ValueError('Pilot would change frozen internal roles or use calibration/heldout songs')


def read_feature(entry, times, audio_sha256):
    path = contained_path(ROOT, entry['path'])
    if sha256(path)!=entry['sha256']:
        raise ValueError('C training feature changed: '+entry['path'])
    with np.load(path,allow_pickle=False) as cached:
        np.testing.assert_array_equal(cached['times'],times)
        if str(cached['feature_key'].item())!=entry['feature_key']:
            raise ValueError('C feature key mismatch')
        if 'audio_sha256' in cached and str(cached['audio_sha256'].item())!=audio_sha256:
            raise ValueError('C feature audio identity mismatch')
        values=cached['features'].copy()
    if values.shape!=(len(times),entry['dimensions']) or not np.isfinite(values).all():
        raise ValueError('C feature shape/finite check failed')
    return values


def read_targets(dataset, index, row, times):
    source=dataset.records[index]
    for name in ('song_id','source_dataset','audio_sha256','group_id'):
        if source[name]!=row[name]:
            raise ValueError('Source recording identity changed: '+name)
    if source['split']!='train':
        raise ValueError('Original heldout data cannot enter C training')
    if sha256(contained_path(dataset.root,source['labels_path']))!=row['expected_labels_sha256']:
        raise ValueError('C pilot labels changed')
    item=dataset[index]
    observed=item['mask'].any(axis=1)
    target,mask=attach_targets(times,item['times_seconds'][observed],item['targets'][observed],item['mask'][observed])
    native=float(np.median(np.diff(item['times_seconds'])))
    if native not in (.5,1.) or not mask.any():
        raise ValueError('Invalid native cadence or empty pilot supervision')
    return target,mask,native


def load_variant(contract, variant):
    verify_hashes(contract['protected_hashes'])
    b=read_json(ROOT/contract['source_B_protocol'])
    validate_membership(contract['songs'],b['split']['assignments'])
    dataset,indices=source_dataset(b)
    result={'fit':[],'development':[]}
    for row in sorted(contract['songs'],key=lambda r:r['song_id']):
        times=np.arange(int(np.floor(row['duration_seconds']/.5))+1)*.5
        target,mask,native=read_targets(dataset,indices[row['song_id']],row,times)
        if array_fingerprint(times,target,mask)!=row['target_fingerprint']:
            raise ValueError('Pilot target or V/A mask changed')
        values=read_feature(row['features'][variant],times,row['audio_sha256'])
        result[row['role']].append(dict(song_id=row['song_id'],source_dataset=row['source_dataset'],
            group_id=row['group_id'],audio_sha256=row['audio_sha256'],x=values,y=target,mask=mask,
            times=times,native_seconds=native))
    return result
