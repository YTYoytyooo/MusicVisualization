"""Cross-check prepared DEAM against raw CSV labels and split membership."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

try:
    from .deam_dataset import DEFAULT_PREPARED, DeamSequences
except ImportError:
    from deam_dataset import DEFAULT_PREPARED, DeamSequences


def verify(prepared):
    prepared = Path(prepared).resolve()
    dataset = DeamSequences(prepared, split=None)
    root = dataset.root
    source = {}
    for axis in ('valence', 'arousal'):
        path = root / f'raw/annotations/annotations averaged per song/dynamic (per second annotations)/{axis}.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream)
            header = next(reader)
            stamps = [int(name.strip().removeprefix('sample_').removesuffix('ms')) / 1000 for name in header[1:]]
            source[axis] = {int(row[0]): dict(zip(stamps, [float(v) if v.strip() else None for v in row[1:]]))
                            for row in reader if row}
    seen_artists, seen_hashes = {}, {}
    split_ids = {name: set() for name in ('train', 'validation', 'test')}
    counts = np.zeros(2, dtype=np.int64)
    missing = 0
    for index, record in enumerate(dataset.records):
        song = dataset[index]
        sid, split = record['song_id'], record['split']
        split_ids[split].add(sid)
        artist = re.sub(r'[\W_]+', ' ', unicodedata.normalize('NFKC', record['artist']).casefold()).strip()
        for key, table in ((artist, seen_artists), (record['audio_sha256'], seen_hashes)):
            assert key not in table or table[key] == split, f'Split overlap: {sid}'
            table[key] = split
        assert record['label_order'] == ['valence', 'arousal']
        assert record['features']['training_enabled'] is False
        assert (root / record['audio_path']).is_file()
        assert (root / record['provided_features_path']).is_file()
        expected_times = sorted(set(source['valence'][sid]) | set(source['arousal'][sid]))
        np.testing.assert_array_equal(song['times_seconds'], expected_times)
        assert song['times_seconds'][0] == 15
        for axis_index, axis in enumerate(('valence', 'arousal')):
            for point, timestamp in enumerate(song['times_seconds']):
                value = source[axis][sid].get(timestamp)
                valid = value is not None and np.isfinite(value) and 0 <= timestamp <= record['duration_seconds']
                assert song['mask'][point, axis_index] == valid, f'Wrong mask: {sid}'
                expected = np.float32(value) if valid else np.float32(0)
                assert song['targets'][point, axis_index] == expected, f'Wrong label: {sid}'
        np.testing.assert_array_equal(song['mask'].sum(axis=0), record['valid_values'])
        assert len(song['times_seconds']) == record['label_count']
        counts += song['mask'].sum(axis=0)
        missing += int((~song['mask']).sum())
    for split, ids in split_ids.items():
        listed = [int(x) for x in (prepared / f'splits/{split}.txt').read_text().splitlines()]
        assert len(listed) == len(set(listed)) and set(listed) == ids
    excluded = {r['song_id'] for r in json.loads((prepared / 'excluded.json').read_text(encoding='utf-8'))}
    included = {r['song_id'] for r in dataset.records}
    assert not included & excluded
    assert set(source['valence']) == set(source['arousal']) == included | excluded
    smoke = []
    for split in split_ids:
        reader = DeamSequences(prepared, split=split, include_audio=True)
        choices = {0, max(range(len(reader)), key=lambda i: reader.records[i]['duration_seconds'])}
        for index in sorted(choices):
            song = reader[index]
            assert np.isfinite(song['audio']).all()
            smoke.append(dict(split=split, song_id=song['song_id'], audio_shape=list(song['audio'].shape)))
    result = dict(status='passed', songs_verified=len(dataset), raw_label_values_exactly_matched=True,
                  valid_values=counts.tolist(), missing_axis_values=missing,
                  splits={k: len(v) for k, v in split_ids.items()},
                  artist_and_exact_hash_split_disjoint=True, audio_loader_smoke=smoke)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=DEFAULT_PREPARED)
    parser.add_argument('--report', type=Path, help='Optional new JSON report file; refuses overwrite')
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error('Report already exists; choose a new path')
    result = verify(args.prepared)
    if args.report:
        with args.report.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
