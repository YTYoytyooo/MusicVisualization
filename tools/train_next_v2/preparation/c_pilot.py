"""Freeze 90 stage-C songs using source metadata only, without fitting a model."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import ROOT, BASE, read_json


def select(rows, protocol):
    assignments = protocol['split']['assignments']
    train = [row for row in rows if row['split'] == 'train']
    if {row['song_id'] for row in train} != set(assignments):
        raise ValueError('Original training identities changed')
    parent = {row['song_id']: row['song_id'] for row in train}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for field in ('group_id', 'audio_sha256'):
        owners = {}
        for row in train:
            if row[field] in owners:
                parent[find(row['song_id'])] = find(owners[row[field]])
            owners[row[field]] = row['song_id']
    components = defaultdict(list)
    for row in train:
        components[find(row['song_id'])].append(row['song_id'])
    component = {sid: min(ids) for ids in components.values() for sid in ids}
    representatives, used = [], set()
    for role, count in [('fit', 20), ('development', 10)]:
        for domain in ('deam', 'pmemo', 'memo2496'):
            eligible = defaultdict(list)
            for row in train:
                if assignments[row['song_id']] == role and row['source_dataset'] == domain and component[row['song_id']] not in used:
                    eligible[component[row['song_id']]].append(row)
            # One recording per linked group; pick within each group without labels.
            pool = [min(group, key=lambda row: hashlib.sha256(('C-pilot-20261003:' + row['song_id']).encode()).hexdigest())
                    for group in eligible.values()]
            pool.sort(key=lambda row: (row['duration_seconds'], row['song_id']))
            if len(pool) < count:
                raise ValueError(f'Too few independent groups in {role}/{domain}')
            indices = [round(i * (len(pool) - 1) / (count - 1)) for i in range(count)]
            for index in indices:
                row = pool[index]
                linked = component[row['song_id']]
                assert linked not in used
                used.add(linked)
                representatives.append(dict(
                    **{key: row[key] for key in ('song_id', 'source_dataset', 'group_id', 'audio_sha256',
                        'audio_path', 'labels_path', 'duration_seconds', 'sample_rate', 'decoded_frames', 'channels')},
                    role=role, linked_component=linked,
                    expected_labels_sha256=protocol['labels_sha256'][row['song_id']],
                    original_feature_sha256=protocol['feature_hashes'][row['audio_sha256']]))
    assert len(representatives) == 90
    assert len({r['song_id'] for r in representatives}) == 90
    assert len({r['audio_sha256'] for r in representatives}) == 90
    assert len({r['linked_component'] for r in representatives}) == 90
    assert all(assignments[r['song_id']] in ('fit', 'development') for r in representatives)
    return representatives


def main():
    protocol_path = BASE / 'protocols/b-v1.json'
    protocol = read_json(protocol_path)
    manifest = Path(protocol['source_manifest'])
    if sha256(manifest) != protocol['manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    selected = select(rows, protocol)
    # A fresh reversed-input selection must produce the same frozen identities.
    assert selected == select(list(reversed(rows)), protocol)
    counts = Counter((row['role'], row['source_dataset']) for row in selected)
    payload = dict(schema_version=1, id='c-pilot-songs-v1', status='sample_preparation_only',
        source_B_protocol_sha256=sha256(protocol_path), source_manifest_sha256=sha256(manifest),
        selector_sha256=sha256(__file__), selection_uses_labels=False,
        selection='One recording per transitive group/audio component; seeded within-group choice; spread across duration ranks including endpoints in each role/domain',
        counts={role: {domain: counts[(role, domain)] for domain in ('deam', 'pmemo', 'memo2496')}
                for role in ('fit', 'development')},
        original_validation_used=0, calibration_used=0, test_used=0,
        unique_audio_and_group_components=90,
        total_audio_seconds=sum(row['duration_seconds'] for row in selected),
        remaining='Freeze encoder revisions, feature timing, training budget and selection rule before extracting or fitting. No C encoder has been downloaded or run by this tool.',
        limitations='Small pilot cannot establish final generalization; source grouping cannot guarantee all unknown recording/artist overlaps.',
        songs=selected)
    target = BASE / 'protocols/c-pilot-songs-v1.json'
    if target.exists():
        if read_json(target) != payload:
            raise ValueError('Existing pilot differs; refusing to overwrite')
    else:
        atomic_json(target, payload)
    print(json.dumps(dict(path=str(target.relative_to(ROOT)), sha256=sha256(target),
        counts=payload['counts'], unique_groups=90, audio_minutes=round(payload['total_audio_seconds'] / 60, 2),
        labels_read=0, selection_repeat_verified=True), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    with run_lease('next-va-v2-c-pilot-preparation'):
        main()
