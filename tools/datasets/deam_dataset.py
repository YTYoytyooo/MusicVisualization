"""Read prepared DEAM sequences without resampling or dividing songs into clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_PREPARED = Path(__file__).resolve().parents[2] / 'data/datasets/deam/prepared/v1'


def contained_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Dataset path escapes root: {relative}')
    return path


class DeamSequences:
    """One item per source recording; targets are ordered [valence, arousal].

    Set include_audio=True to decode the entire source recording as float32
    [audio_samples, channels] at its native sample rate. Labels have their own
    explicit timestamps in seconds; they are not indexed by audio sample.
    """

    def __init__(self, prepared=DEFAULT_PREPARED, split='train', include_audio=False):
        if split not in ('train', 'validation', 'test', None):
            raise ValueError('split must be train, validation, test or None')
        prepared = Path(prepared).resolve()
        config = json.loads((prepared / 'config.json').read_text(encoding='utf-8'))
        if config['schema_version'] != 1 or config['label_order'] != ['valence', 'arousal']:
            raise ValueError('Unsupported prepared dataset schema')
        self.root = (prepared / config['dataset_root']).resolve()
        self.include_audio = include_audio
        self.records = []
        ids = set()
        with (prepared / 'manifest.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                record = json.loads(line)
                if record['song_id'] in ids:
                    raise ValueError('Duplicate song in manifest')
                ids.add(record['song_id'])
                if split is None or record['split'] == split:
                    self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = contained_path(self.root, record['labels_path'])
        with np.load(path, allow_pickle=False) as labels:
            times = labels['times_seconds'].copy()
            targets = labels['targets'].copy()
            mask = labels['mask'].copy()
        if times.ndim != 1 or targets.shape != (len(times), 2) or mask.shape != targets.shape:
            raise ValueError(f'Invalid sequence shape: {record["song_id"]}')
        if mask.dtype != np.bool_ or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError(f'Invalid timestamps/mask: {record["song_id"]}')
        if not np.all(np.isfinite(targets)) or np.any(np.abs(targets[mask]) > 1):
            raise ValueError(f'Invalid targets: {record["song_id"]}')
        valid_times = times[mask.any(axis=1)]
        if np.any(valid_times < 0) or np.any(valid_times > record['duration_seconds']):
            raise ValueError(f'Valid labels outside audio: {record["song_id"]}')
        item = dict(song_id=record['song_id'], times_seconds=times, targets=targets,
                    mask=mask, sample_rate=record['sample_rate'], metadata=record)
        if self.include_audio:
            audio, sample_rate = sf.read(contained_path(self.root, record['audio_path']),
                                         dtype='float32', always_2d=True)
            if sample_rate != record['sample_rate'] or audio.shape != (record['decoded_frames'], record['channels']):
                raise ValueError(f'Audio differs from audited recording: {record["song_id"]}')
            item['audio'] = audio
        return item


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=DEFAULT_PREPARED)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='train')
    parser.add_argument('--audio', action='store_true', help='Also decode the first song')
    args = parser.parse_args()
    dataset = DeamSequences(args.prepared, args.split, args.audio)
    first = dataset[0]
    print(json.dumps(dict(songs=len(dataset), first_song_id=first['song_id'],
                          target_shape=first['targets'].shape,
                          valid_values=first['mask'].sum(axis=0).tolist(),
                          audio_shape=first['audio'].shape if args.audio else None)))
