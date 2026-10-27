"""Versioned, hash-checked C feature shards with crash-safe completion."""
import uuid

import numpy as np

from tools.train_next.core import atomic_json, sha256
from tools.train_next_v2.common import read_json


def validate(path, times, key, audio_sha256, dimensions):
    with np.load(path, allow_pickle=False) as cached:
        if str(cached['feature_key'].item()) != key or str(cached['audio_sha256'].item()) != audio_sha256:
            raise ValueError('C feature provenance mismatch')
        np.testing.assert_array_equal(cached['times'], times)
        values = cached['features'].copy()
        if values.shape != (len(times), dimensions) or not np.isfinite(values).all():
            raise ValueError('C feature values are invalid')
    return values


def read_shard(path, times, key, audio_sha256, dimensions):
    metadata_path = path.with_suffix('.json')
    if not metadata_path.exists():
        return None  # Missing/orphan shard requires recomputation, never blind reuse.
    metadata = read_json(metadata_path)
    if not path.is_file() or sha256(path) != metadata['output_sha256']:
        raise ValueError('Completed C feature shard hash changed')
    if metadata['feature_key'] != key or metadata['audio_sha256'] != audio_sha256:
        raise ValueError('C feature metadata provenance mismatch')
    validate(path, times, key, audio_sha256, dimensions)
    return metadata


def write_shard(path, times, values, key, audio_sha256, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix('.json').exists():
        raise FileExistsError('Completed feature shard is immutable')
    if not np.isfinite(values).all() or values.shape[0] != len(times):
        raise ValueError('Refusing invalid C features')
    if path.exists():
        previous = validate(path, times, key, audio_sha256, values.shape[1])
        # If a crash occurred between npz and sidecar replacement, certify the
        # orphan only after exact deterministic recomputation from its source.
        np.testing.assert_array_equal(previous, values)
    else:
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        with temporary.open('xb') as stream:
            np.savez_compressed(stream, times=times, features=values,
                                feature_key=key, audio_sha256=audio_sha256)
        temporary.replace(path)
    result = dict(metadata, feature_key=key, audio_sha256=audio_sha256,
                  output_sha256=sha256(path), bytes=path.stat().st_size,
                  frames=len(times), dimensions=values.shape[1])
    atomic_json(path.with_suffix('.json'), result)
    return result
