"""Audit recording overlap and publish a fixed DEAM + supplemental split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import unicodedata
import uuid

import numpy as np

from tools.datasets.deam_dataset import DeamSequences
from tools.datasets.fetch_supplemental import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
FPCALC = ROOT / 'data/datasets/.tools/chromaprint/chromaprint-fpcalc-1.6.1-windows-x86_64/fpcalc.exe'


def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', value or '').casefold() if c.isalnum())


class Fingerprints:
    def __init__(self):
        self.directory = ROOT / 'data/datasets/.fingerprints' / digest(FPCALC)[:16]
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, row):
        path = self.directory / (row['audio_sha256'] + '.json')
        if path.exists():
            result = json.loads(path.read_text())
        else:
            audio = ROOT / row['audio_path']
            if digest(audio) != row['audio_sha256']:
                raise ValueError(f'Changed audio: {row["song_id"]}')
            process = subprocess.run([str(FPCALC), '-raw', '-json', '-length', '0', str(audio)],
                                     capture_output=True, text=True, timeout=30)
            if process.returncode:
                raise ValueError(f'Fingerprint failed: {row["song_id"]}: {process.stderr[:400]}')
            result = json.loads(process.stdout)
            write_json(path, result)
        return np.asarray(result['fingerprint'], dtype=np.uint32)

    def checked(self, row):
        try:
            return self.get(row), None
        except Exception as exc:
            return np.empty(0, dtype=np.uint32), str(exc)


class MatchIndex:
    """Offset-aware Chromaprint screening; conservative candidates are excluded."""
    def __init__(self):
        self.postings = defaultdict(list)
        self.arrays = []

    def add(self, values):
        index = len(self.arrays)
        self.arrays.append(values)
        for pos in range(0, len(values), 8):
            key = int(values[pos]) & 0xfffff000
            # Common fingerprints (silence, sustained tones) are not informative.
            if len(self.postings[key]) <= 200:
                self.postings[key].append((index, pos))

    def matches(self, values):
        votes = Counter()
        for pos, word in enumerate(values):
            matches = self.postings.get(int(word) & 0xfffff000, ())
            if len(matches) <= 200:
                for index, other_pos in matches:
                    votes[index, other_pos - pos] += 1
        found = {}
        for (index, offset), count in votes.most_common(40):
            if count < 5:
                break
            other = self.arrays[index]
            start_a, start_b = max(0, -offset), max(0, offset)
            length = min(len(values) - start_a, len(other) - start_b)
            minimum = max(80, min(160, int(.75 * min(len(values), len(other)))))
            if length < minimum:
                continue
            a, b = values[start_a:start_a + length], other[start_b:start_b + length]
            if len(np.unique(a)) < 30 or len(np.unique(b)) < 30:
                continue
            errors = np.bitwise_count(a ^ b)
            mean_error = float(errors.mean() / 32)
            if mean_error <= .12 and float((errors <= 8).mean()) >= .85:
                if index not in found or mean_error < found[index]['bit_error']:
                    found[index] = dict(offset_frames=offset, compared_frames=length, bit_error=mean_error)
        return found


def load_records(name):
    dataset = DeamSequences(ROOT / f'data/datasets/{name}/prepared/v1', split=None)
    rows = []
    for r in dataset.records:
        r = dict(r)
        for key in ('audio_path', 'labels_path'):
            path = (dataset.root / r[key]).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('Dataset path outside workspace')
            r[key] = path.relative_to(ROOT).as_posix()
        r['song_id'] = f'deam:{r["song_id"]}' if name == 'deam' else str(r['song_id'])
        r['source_dataset'] = name
        rows.append(r)
    return rows


def build(output, supplements):
    if output.exists():
        raise ValueError('Output already exists; use a new output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + '.pending-' + uuid.uuid4().hex[:8])
    pending.mkdir()
    deam = load_records('deam')
    candidates = [r for name in supplements for r in load_records(name) if r.get('eligible_dynamic', True)]
    all_rows = deam + candidates
    store = Fingerprints()
    fingerprints = []
    fingerprint_errors = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, (values, error) in enumerate(pool.map(store.checked, all_rows)):
            if error:
                if i < len(deam):
                    raise ValueError(f'Cannot verify fixed DEAM reference {all_rows[i]["song_id"]}: {error}')
                fingerprint_errors[i - len(deam)] = error
            fingerprints.append(values)
            if (i + 1) % 100 == 0:
                print(json.dumps(dict(stage='fingerprints', done=i + 1, total=len(all_rows))), flush=True)
    index = MatchIndex()
    for f in fingerprints[:len(deam)]:
        index.add(f)
    deam_hashes = {r['audio_sha256'] for r in deam}
    deam_artists = {normalized(r.get('artist')) for r in deam} - {''}
    deam_titles = {normalized(r.get('title')) for r in deam if len(normalized(r.get('title'))) >= 8}
    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(max(a, b))] = find(min(a, b))

    groups = {}
    exclusions, duplicate_links = [], []
    excluded_indices = set()
    for i, row in enumerate(candidates):
        reasons = []
        if i in fingerprint_errors:
            reasons.append('fingerprint_unavailable')
        if row['audio_sha256'] in deam_hashes:
            reasons.append('identical_DEAM_audio')
        artist, title = normalized(row.get('artist')), normalized(row.get('title'))
        if artist and artist in deam_artists:
            reasons.append('DEAM_artist_overlap')
        if len(title) >= 8 and title in deam_titles:
            reasons.append('DEAM_title_overlap_candidate')
        values = fingerprints[len(deam) + i]
        matched = index.matches(values)
        for other, evidence in matched.items():
            duplicate_links.append(dict(a=row['song_id'], b=all_rows[other]['song_id'], **evidence))
            if other < len(deam):
                reasons.append('DEAM_acoustic_overlap')
            else:
                union(i, other - len(deam))
        for key in (('artist', artist), ('title', title if len(title) >= 8 else ''), ('hash', row['audio_sha256'])):
            if not key[1]:
                continue
            if key in groups:
                union(i, groups[key])
            else:
                groups[key] = i
        index.add(values)
        if reasons:
            excluded_indices.add(i)
            exclusions.append(dict(song_id=row['song_id'], reasons=sorted(set(reasons))))
        if (i + 1) % 100 == 0:
            print(json.dumps(dict(stage='overlap', done=i + 1, total=len(candidates))), flush=True)
    # If one component overlaps DEAM, exclude every connected supplemental item.
    excluded_groups = {find(i) for i in excluded_indices}
    members = defaultdict(list)
    for i, r in enumerate(candidates):
        members[find(i)].append(r['song_id'])
    kept = []
    for i, row in enumerate(candidates):
        group = find(i)
        if group in excluded_groups:
            if i not in excluded_indices:
                exclusions.append(dict(song_id=row['song_id'], reasons=['group_connected_to_DEAM_overlap']))
            continue
        token = '|'.join(sorted(members[group]))
        key = hashlib.sha256(('20261002|' + token).encode()).hexdigest()
        fraction = int(key[:8], 16) / 2**32
        row.update(group_id='supplemental-' + key[:16], split='train' if fraction < .8 else 'validation' if fraction < .9 else 'test',
                   training_enabled=True, dedup_method='metadata_sha256_chromaprint_offset_v1')
        kept.append(row)
    rows = deam + kept
    with (pending / 'manifest.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts = {name: dict(Counter(r['split'] for r in rows if r['source_dataset'] == name)) for name in ['deam'] + supplements}
    write_json(pending / 'config.json', dict(schema_version=1, label_order=['valence', 'arousal'],
               dataset_root=os.path.relpath(ROOT, output), supplements=supplements, seed=20261002,
               label_resampling='none; exact original timestamps with per-axis masks',
               provenance={name: digest(ROOT / f'data/datasets/{name}/prepared/v1/manifest.jsonl') for name in ['deam'] + supplements}))
    write_json(pending / 'exclusions.json', exclusions)
    write_json(pending / 'duplicates.json', duplicate_links)
    write_json(pending / 'fingerprint-errors.json', {candidates[i]['song_id']: error for i, error in fingerprint_errors.items()})
    report = dict(status='ready', counts=counts, excluded=len(exclusions), fingerprint_links=len(duplicate_links),
                  test_use='held out; fingerprints used only to prevent recording overlap, never model inputs',
                  limitations=['Chromaprint screening cannot guarantee detection of covers, heavy edits or all short overlaps',
                               'Memo2496 artist metadata is incomplete; grouping uses available metadata and recording fingerprints'])
    write_json(pending / 'report.json', report)
    pending.replace(output)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'data/datasets/combined/prepared/v1')
    p.add_argument('--supplements', nargs='+', choices=['pmemo', 'memo2496'], default=['pmemo', 'memo2496'])
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'data/datasets'):
        p.error('Output must stay inside workspace data/datasets')
    build(output, args.supplements)
