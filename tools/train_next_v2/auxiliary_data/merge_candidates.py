"""Freeze a conservative MERGE auxiliary candidate pool using metadata only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, process_identity, read_json, record_event, verify_hashes
from .merge_inspect import csv_rows

JOB = 'd-merge-candidates-v1'
FOLDER = BASE / 'analyses' / JOB
OUTPUT = ROOT / 'data/datasets/merge/prepared/auxiliary-candidates-v1'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def normalized(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', value or '').casefold() if c.isalnum())


def prefix(sid):
    return 'MT' if sid.startswith('MT') else 'A' if sid.startswith('A') else 'L' if sid.startswith('L') else 'other'


def provisional_va(row, quadrant):
    # Deliberate named lookup: release order is A,V but this project is V,A.
    raw = [float(row['Valence']), float(row['Arousal'])]
    if not all(math.isfinite(x) and 0 <= x <= 1 and x != .5 for x in raw):
        raise ValueError('Invalid or ambiguous training-only static score')
    v, a = raw
    derived = 'Q1' if v > .5 and a > .5 else 'Q2' if v < .5 and a > .5 else 'Q3' if v < .5 and a < .5 else 'Q4'
    if quadrant != derived:
        raise ValueError('Training-only score disagrees with recorded quadrant')
    return raw, [2 * value - 1 for value in raw]


def connected_groups(identities):
    """Connect artist/recording identities, propagating held-out exclusions transitively."""
    parent = {sid: sid for sid in identities}
    def root(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid
    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)
    first = {}
    for sid, row in sorted(identities.items()):
        for key in row['identity_keys']:
            if key in first:
                union(sid, first[key])
            else:
                first[key] = sid
    groups = defaultdict(list)
    for sid in identities:
        groups[root(sid)].append(sid)
    return {sid: sorted(groups[root(sid)]) for sid in identities}


def register():
    if FOLDER.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve existing candidates and registration')
    source = read_json(BASE / 'analyses/d-merge-schema-v1/launch.json')
    hashes = dict(source['protected_hashes'])
    for path in [Path(__file__), ROOT/'tools/train_next_v2/common.py', ROOT/'tools/train_next/runtime.py',
                 ROOT/'tools/train_next/core.py', BASE/'protocols/b-v1.json', BASE/'analyses/d-merge-schema-v1/report.json']:
        hashes[relative(path)] = sha256(path)
    verify_hashes(hashes)
    argv = [str(Path(sys.executable).resolve()), '-m', 'tools.train_next_v2.auxiliary_data.merge_candidates', '--run']
    launch = dict(job_id=JOB, registered_at=now(), original_argv=argv, cwd=str(ROOT), python=argv[0],
        environment=source['environment'], protected_hashes=hashes, files=source['files'],
        source_manifest=source['source_manifest'], os_lock=str(lock_path(JOB)),
        output=relative(OUTPUT), report=relative(FOLDER/'report.json'), gpu_used=False,
        label_scope='Static score diagnostics use official 70/15/15 training IDs only; existing main corpus consulted for identity metadata only.',
        recovery='No automatic retry; preserve output, version changed inputs/logic independently.',
        proposed_first_task='whole-clip quadrant classification; numeric static regression disabled pending scale/source clarification')
    FOLDER.mkdir(parents=True)
    atomic_json(FOLDER/'launch.json', launch)
    record_event(dict(event='D_MERGE_candidate_screen_registered', launch=relative(FOLDER/'launch.json')))


def run():
    launch = read_json(FOLDER/'launch.json')
    verify_hashes(launch['protected_hashes'])
    if OUTPUT.exists() or (FOLDER/'report.json').exists():
        raise FileExistsError('Metadata screen already has output; preserve it')
    atomic_json(FOLDER/'process.json', dict(started_at=now(), process_identity=process_identity()))
    files = launch['files']
    metadata = csv_rows(ROOT/files['MERGE_Audio_Complete/merge_audio_complete_metadata.csv'])
    values = csv_rows(ROOT/files['MERGE_Audio_Complete/merge_audio_complete_av_values.csv'])
    roles = {}
    for role in ('train', 'validate', 'test'):
        key = f'MERGE_Audio_Complete/tvt_dataframes/tvt_70_15_15/tvt_70_15_15_{role}_audio_complete.csv'
        for sid in csv_rows(ROOT/files[key]):
            if sid in roles:
                raise ValueError('Official split IDs overlap')
            roles[sid] = role
    with (ROOT/launch['source_manifest']).open(encoding='utf-8') as stream:
        main = [json.loads(line) for line in stream]
    assignments = read_json(BASE/'protocols/b-v1.json')['split']['assignments']
    identities, main_artist_title, main_titles = {}, defaultdict(list), defaultdict(list)
    for row in main:
        sid = row['song_id']
        artist, title = normalized(row.get('artist')), normalized(row.get('title'))
        role = assignments[sid] if row['split'] == 'train' else row['split']
        keys = [('main_group', row['group_id']), ('audio_sha256', row['audio_sha256'])]
        if artist:
            keys.append(('artist', artist))
        if artist and title:
            keys.append(('artist_title', artist, title))
            main_artist_title[artist, title].append(sid)
        if title:
            main_titles[title].append(sid)
        identities[sid] = dict(role=role, corpus='main', identity_keys=keys)
    for sid, row in metadata.items():
        artist, title = normalized(row['Artist']), normalized(row['Title'])
        keys = []
        if artist:
            keys.append(('artist', artist))
        if artist and title:
            keys.append(('artist_title', artist, title))
        for track in sorted(set(re.findall(r'MT\d+', row['AllMusic Id']+' '+row['AppearancesTrackIDs']))):
            keys.append(('allmusic_track', track))
        identities['merge:'+sid] = dict(role=roles[sid], corpus='merge', identity_keys=keys)
    groups = connected_groups(identities)
    rows, source_audit = [], defaultdict(list)
    for sid in sorted(s for s in metadata if roles[s] == 'train'):
        meta = metadata[sid]
        raw, converted = provisional_va(values[sid], meta['Quadrant'])
        source_audit[prefix(sid)].append(raw)
        artist, title = normalized(meta['Artist']), normalized(meta['Title'])
        members = groups['merge:'+sid]
        heldout = [s for s in members if identities[s]['role'] not in ('train', 'fit')]
        exact_main = main_artist_title.get((artist, title), [])
        possible_titles = main_titles.get(title, []) if title else []
        reasons = []
        if not artist or not title:
            reasons.append('missing_artist_or_title')
        if any(identities[s]['corpus'] == 'main' for s in heldout):
            reasons.append('linked_main_heldout_artist_or_recording_group')
        if any(identities[s]['corpus'] == 'merge' for s in heldout):
            reasons.append('linked_MERGE_official_heldout_artist_or_recording_group')
        if exact_main:
            reasons.append('same_artist_title_in_main_corpus')
        if possible_titles and not exact_main:
            reasons.append('title_only_possible_main_overlap_pending_audio')
        rows.append(dict(song_id='merge:'+sid, release_id=sid, official_role='train',
            group_id='merge-metadata-'+hashlib.sha256('\n'.join(members).encode()).hexdigest()[:20],
            artist=meta['Artist'], title=meta['Title'], source_id_prefix=prefix(sid),
            prefix_provenance='Observed release ID prefix; legacy source lineage not independently resolved per row.',
            quadrant=meta['Quadrant'], raw_static_va=raw, raw_axis_order=['valence','arousal'],
            provisional_static_va_minus1_plus1=converted, numeric_regression_enabled=False,
            source_duration_metadata=meta['Duration'], clip_duration_seconds=None,
            source_song_offset_seconds=None, local_audio_time_origin_seconds=0,
            label_scope='whole_released_clip_static', time_varying_labels=False,
            metadata_candidate=not reasons, exclusion_reasons=reasons,
            linked_heldout_ids=heldout, exact_main_identity_ids=exact_main,
            possible_title_overlap_ids=possible_titles,
            fingerprint_audit='pending', training_admitted=False))
    candidates = [row for row in rows if row['metadata_candidate']]
    contract = dict(id='merge-auxiliary-label-contract-v1', created_at=now(),
        source_release='https://zenodo.org/records/13939205', source_release_version='1.1.0',
        declared_record_license='cc-by-nc-4.0', usage='noncommercial local research; no audio redistribution',
        chosen_split='official audio complete 70/15/15; further remove groups linked to reserved data',
        primary_paper='https://arxiv.org/html/2407.06060v3', source_sections=['III-B','III-C'],
        numeric_scores=dict(source='Automatic emotion-word mapping; not direct continuous human ratings.',
            release_column_order=['Song','Arousal','Valence'], stored_axis_order=['valence','arousal'],
            paper_scale=[-1,1], inferred_release_scale=[0,1], conversion='2*x-1',
            scale_status='Provisional inference from all 2490 official training scores and exact quadrant agreement at center 0.5; author confirmation absent.',
            training_enabled=False, raw_values_preserved=True),
        quadrant_task=dict(target='Published Q1/Q2/Q3/Q4 labels', granularity='whole released audio clip',
            supervision_source='Published quadrant validation; this project has not conducted new human review.',
            intended_head='Separate clip-level classification head with temporal pooling; never broadcast to timepoint VA truth.',
            training_enabled=False, remaining_gates=['verified audio decode/duration','fingerprint and recording exclusion','frozen equal-step experiment protocol']),
        timeline=dict(native_clip_start_seconds=0, source_full_song_offset='unknown; never infer from Duration or normalize a song timeline',
            dynamic_ground_truth=False),
        original_heldout_targets_used=False, metadata_identity_screen_only=True,
        limitations=['Normalized artist names do not resolve all aliases or featured-artist combinations.',
                    'Main Memo2496 artist metadata is incomplete; metadata separation is not proof of recording separation.',
                    'Some title-only matches may be false positives; quarantine until audio evidence is reviewed.',
                    'A/MT/L prefixes are recorded without inventing a per-song label provenance mapping.'])
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT/'label-contract.json', contract)
    atomic_json(OUTPUT/'training-screen.json', rows)
    counts = {name: dict(training_songs=len(items), quadrant_center=.5,
                         raw_ranges={axis:[min(row[i] for row in items), max(row[i] for row in items)]
                                     for i,axis in enumerate(('valence','arousal'))})
              for name,items in source_audit.items()}
    report = dict(job_id=JOB, status='completed_metadata_screen_audio_gate_pending', completed_at=now(),
        official_train_songs=len(rows), metadata_candidates=len(candidates),
        candidate_groups=len({r['group_id'] for r in candidates}), excluded_or_quarantined=len(rows)-len(candidates),
        exclusion_reason_counts=dict(Counter(reason for row in rows for reason in row['exclusion_reasons'])),
        candidate_quadrants=dict(Counter(row['quadrant'] for row in candidates)), source_prefix_audit=counts,
        numeric_regression_enabled=False, training_admitted=False, original_heldout_targets_used=False,
        protected_exact_matches=[row for row in rows if row['exact_main_identity_ids']],
        source_hashes=launch['protected_hashes'],
        outputs={relative(path):sha256(path) for path in OUTPUT.glob('*.json')})
    atomic_json(FOLDER/'report.json', report)
    stages = read_json(PLAN)['stages']
    stages['D'].update(phase='MERGE_audio_acquisition_and_identity_screen',
        candidate_screen=dict(report=relative(FOLDER/'report.json'), candidates=len(candidates),
                              groups=report['candidate_groups'], training_admitted=False),
        label_contract=relative(OUTPUT/'label-contract.json'))
    record_event(dict(event='D_MERGE_metadata_candidates_frozen', candidates=len(candidates),
                      excluded_or_quarantined=report['excluded_or_quarantined']), updates=dict(stages=stages))
    print(json.dumps({key: report[key] for key in ('status','official_train_songs','metadata_candidates',
          'candidate_groups','excluded_or_quarantined','exclusion_reason_counts','candidate_quadrants','source_prefix_audit')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    with run_lease(JOB):
        register() if args.register else run()
