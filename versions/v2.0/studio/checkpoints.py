"""Local renderer checkpoints: JSON metadata + hashed NPZ arrays, never pickle.

A configuration key must identify the audio/revision/plan, seeds, simulation
dimensions/fps, renderer version and HUD timeline. A mismatched key is refused.
Snapshots describe the state BEFORE metadata.frame_idx, not a completed frame.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid
import zipfile

import numpy as np

from .store import atomic_json, digest, object_digest

MAX_ARRAY_BYTES = 512 * 1024 * 1024


def _key(value):
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError('Checkpoint config_key must be a nonempty string')
    return value


def _manifest(path, expected_key):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Checkpoint manifest is too large')
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError('Checkpoint manifest must be an object')
    if data.get('schema_version') != 1 or data.get('config_key') != _key(expected_key):
        raise ValueError('Checkpoint configuration key mismatch')
    expected_hash = object_digest({k: v for k, v in data.items() if k != 'sha256'})
    if data.get('sha256') != expected_hash:
        raise ValueError('Checkpoint manifest checksum mismatch')
    filename = data.get('arrays_file')
    if (not isinstance(filename, str) or Path(filename).name != filename
            or '/' in filename or '\\' in filename or not filename.endswith('.npz')):
        raise ValueError('Invalid checkpoint array filename')
    return data


def save_checkpoint(directory, config_key, renderer):
    """Publish a new immutable checkpoint manifest and return its absolute path."""
    config_key = _key(config_key)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = renderer.export_state()
    arrays = snapshot['arrays']
    schema = {}
    total = 0
    for key, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError('Checkpoint arrays must be non-object numpy arrays')
        if not key.replace('_', '').isalnum() or not key.isascii():
            raise ValueError('Invalid checkpoint array key')
        total += value.nbytes
        schema[key] = {'shape': list(value.shape), 'dtype': value.dtype.str}
    if total > MAX_ARRAY_BYTES or len(arrays) > 256:
        raise ValueError('Checkpoint exceeds safe array limits')
    prefix = hashlib.sha256(config_key.encode()).hexdigest()[:16]
    stem = f"{prefix}-{snapshot['metadata']['frame_idx']:012d}-{uuid.uuid4().hex[:8]}"
    arrays_path = directory / (stem + '.npz')
    manifest_path = directory / (stem + '.json')
    fd, pending = tempfile.mkstemp(prefix='.checkpoint-pending-', suffix='.npz', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, arrays_path)
        metadata = {'schema_version': 1, 'config_key': config_key,
                    'state': snapshot['metadata'], 'arrays': schema,
                    'arrays_file': arrays_path.name, 'arrays_sha256': digest(arrays_path)}
        metadata['sha256'] = object_digest(metadata)
        atomic_json(manifest_path, metadata)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)
    return manifest_path


def _validated_arrays(path, schema):
    if not isinstance(schema, dict) or not 1 <= len(schema) <= 256:
        raise ValueError('Checkpoint array schema is invalid')
    total = 0
    for key, entry in schema.items():
        if not isinstance(key, str) or not key.replace('_', '').isalnum() or not key.isascii():
            raise ValueError('Checkpoint array name is invalid')
        shape = entry['shape']
        dtype = np.dtype(entry['dtype'])
        if (dtype.hasobject or dtype.kind not in 'buif' or not isinstance(shape, list)
                or not 1 <= len(shape) <= 4 or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape)):
            raise ValueError('Checkpoint array shape/dtype is invalid')
        total += math.prod(shape) * dtype.itemsize
    if total > MAX_ARRAY_BYTES:
        raise ValueError('Checkpoint exceeds safe expanded size')
    # Inspect NPY headers BEFORE numpy allocates according to their shapes. This
    # also rejects duplicate ZIP members and malicious object/pickle payloads.
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(set(names)) != len(names) or set(names) != {key + '.npy' for key in schema}:
            raise ValueError('Checkpoint archive members do not match its schema')
        if sum(entry.file_size for entry in entries) > MAX_ARRAY_BYTES + 1024 * 1024:
            raise ValueError('Checkpoint archive exceeds safe expanded size')
        for entry in entries:
            specification = schema[entry.filename[:-4]]
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported checkpoint NPY format')
                if (list(shape) != specification['shape'] or dtype.str != specification['dtype']
                        or dtype.hasobject or entry.file_size != stream.tell() + math.prod(shape) * dtype.itemsize):
                    raise ValueError('Checkpoint array header does not match its schema')
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in schema}


def load_checkpoint(manifest_path, config_key, renderer):
    """Validate hashes/config/array schema, restore renderer and return next frame."""
    manifest_path = Path(manifest_path).resolve()
    data = _manifest(manifest_path, config_key)
    arrays_path = (manifest_path.parent / data['arrays_file']).resolve()
    if arrays_path.parent != manifest_path.parent:
        raise ValueError('Checkpoint array path escapes its directory')
    if digest(arrays_path) != data['arrays_sha256']:
        raise ValueError('Checkpoint array checksum mismatch')
    try:
        arrays = _validated_arrays(arrays_path, data['arrays'])
        renderer.restore_state({'metadata': data['state'], 'arrays': arrays})
    except (zipfile.BadZipFile, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError('Malformed renderer checkpoint') from exc
    return renderer.frame_idx


def find_checkpoint(directory, config_key, max_frame):
    """Find the latest matching manifest at/before a requested next-frame index.

    Final NPZ validation still happens in load_checkpoint. Invalid manifests are
    ignored so an interrupted publication cannot masquerade as a valid state.
    """
    _key(config_key)
    if isinstance(max_frame, bool) or not isinstance(max_frame, int) or max_frame < 0:
        raise ValueError('max_frame must be a nonnegative integer')
    prefix = hashlib.sha256(config_key.encode()).hexdigest()[:16]
    matches = []
    for path in Path(directory).glob(prefix + '-*.json'):
        try:
            data = _manifest(path, config_key)
            frame = data['state']['frame_idx']
            if (not isinstance(frame, bool) and isinstance(frame, int) and 0 <= frame <= max_frame
                    and (path.parent / data['arrays_file']).is_file()):
                matches.append((frame, path.name, path.resolve()))
        except (ValueError, TypeError, KeyError, OSError):
            continue
    return max(matches, default=(None, None, None))[2]
