"""Read-only metadata admission diagnostics; no audio, VA targets, or training."""
import csv
from collections import Counter
import json
import re
from pathlib import Path
import sys

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes, process_identity

JOB = 'd-jamendo-schema-v1'
DIRECTORY = BASE / 'analyses' / JOB
SOURCE = ROOT / 'data/datasets/mtg_jamendo/raw/metadata-v2/source'


def literal_tsv(path):
    # The release is literal TSV: quotation marks inside song names are text.
    # CSV's default quote handling silently joins unrelated physical records.
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE))


def tags(path):
    rows = literal_tsv(path)
    if rows.pop(0) != ['TRACK_ID','ARTIST_ID','ALBUM_ID','PATH','DURATION','TAGS']:
        raise ValueError('Unexpected tagging schema')
    ids = set()
    for row in rows:
        if (len(row) < 6 or row[0] in ids or not re.fullmatch(r'track_\d+', row[0])
                or not re.fullmatch(r'\d+/\d+\.mp3', row[3]) or not 0 < float(row[4]) < 86400
                or any(not tag.startswith('mood/theme---') for tag in row[5:])):
            raise ValueError('Invalid tagging row: ' + str(row[:1]))
        ids.add(row[0])
    return rows


def audit():
    mood = tags(SOURCE / 'data/autotagging_moodtheme.tsv')
    splits = {s: tags(SOURCE / f'data/splits/split-0/autotagging_moodtheme-{s}.tsv')
              for s in ('train','validation','test')}
    meta = literal_tsv(SOURCE / 'data/raw.meta.tsv')
    header = meta.pop(0)
    if header != ['TRACK_ID','ARTIST_ID','ALBUM_ID','TRACK_NAME','ARTIST_NAME','ALBUM_NAME','RELEASEDATE','URL']:
        raise ValueError('Unexpected identity schema')
    counts = Counter(len(row) for row in meta)
    invalid = [dict(line=i+2, track_id=row[0], columns=len(row), reason='extra_unescaped_tab_no_guessing')
               for i,row in enumerate(meta) if len(row) != len(header)]
    valid = {row[0]: dict(zip(header,row)) for row in meta if len(row) == len(header)}
    if len(valid) != len(meta)-len(invalid):
        raise ValueError('Duplicate identity IDs')
    license_lines = (SOURCE / 'audio_licenses.txt').read_text(encoding='utf-8').splitlines()
    license_paths = {line for line in license_lines if re.fullmatch(r'\d+/\d+\.mp3', line)}
    overlaps = {}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        overlaps[a+'_'+b] = dict(tracks=len({r[0] for r in splits[a]} & {r[0] for r in splits[b]}),
                                 artists=len({r[1] for r in splits[a]} & {r[1] for r in splits[b]}))
    if any(any(row.values()) for row in overlaps.values()):
        raise ValueError('Official metadata train/evaluation identities overlap')
    return dict(
        mood_tracks=len(mood), mood_vocabulary=sorted({tag for row in mood for tag in row[5:]}),
        identity_physical_rows=len(meta), identity_valid_rows=len(valid), identity_column_counts=dict(counts),
        ambiguous_identity_rows=invalid,
        published_readme_identity_count=56693,
        identity_count_note='Pinned file has fewer physical rows than README; use measured records. No rows repaired or guessed.',
        mood_tracks_without_valid_identity=sorted({r[0] for r in mood}-valid.keys()),
        mood_paths_without_license_block=sorted({r[3] for r in mood}-license_paths),
        official_splits={s:dict(tracks=len(rows), artists=len({r[1] for r in rows}),
            vocabulary=sorted({tag for r in rows for tag in r[5:]})) for s,rows in splits.items()},
        official_split_overlaps=overlaps,
        license_contract=dict(code='Apache-2.0', metadata='CC-BY-NC-SA-4.0',
            audio='Individual Creative Commons declarations in audio_licenses.txt; per-selected-file audit still required',
            dataset_scope='Official README limits dataset to non-commercial research and academic use; commercial permission not obtained',
            source='https://github.com/MTG/mtg-jamendo-dataset/blob/cafd8e20c265ed84f1e61f1c875327971f43a62f/README.md'),
        label_contract=dict(scope='Whole-track uploader mood/theme tags', kind='weak_multilabel_categories',
            continuous_va_targets=False, absence_is_verified_negative=False,
            preprocessing='Keep positive/missing distinction; any closed-world classification loss is an explicit experimental assumption',
            time_origin='Full-track dataset metadata; decode and duration verification required before use'),
        training_admitted=False, audio_downloaded=False, main_corpus_labels_read=False,
        pending=['Quarantine ambiguous/missing identity metadata',
                 'Exclude matching artists/recordings with main development/calibration/validation/test and auxiliary heldouts',
                 'Check MusAV/ManyMusic/Jamendo identities before assigning roles',
                 'Select bounded tag/artist-diverse train-only subset, audit per-track licenses and available audio URLs',
                 'Decode/fingerprint against protected corpora and freeze auxiliary comparison before training'])


def main():
    with run_lease(JOB):
        launch=read_json(DIRECTORY/'launch.json');verify_hashes(launch['protected_hashes'])
        if (DIRECTORY/'report.json').exists(): raise FileExistsError('Preserve completed schema audit')
        if Path(sys.executable).resolve()!=Path(launch['python']):raise ValueError('Incorrect Python')
        atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=process_identity()))
        result=dict(status='completed_verified',job_id=JOB,completed_at=now(),**audit())
        result['source_hashes']=launch['protected_hashes']
        verify_hashes(launch['protected_hashes']);atomic_json(DIRECTORY/'report.json',result)
        with run_lease('next-va-v2-d-ledger'):
            stages=read_json(PLAN)['stages']
            stages['D']['jamendo_schema_audit']=dict(report=str((DIRECTORY/'report.json').relative_to(ROOT)),
                mood_tracks=result['mood_tracks'],official_train_tracks=result['official_splits']['train']['tracks'],
                ambiguous_identity_rows=len(result['ambiguous_identity_rows']),training_admitted=False)
            record_event(dict(event='D_Jamendo_schema_audited',job_id=JOB),updates=dict(stages=stages))
        print(json.dumps({key:result[key] for key in ('status','mood_tracks','identity_physical_rows','identity_valid_rows',
            'ambiguous_identity_rows','mood_tracks_without_valid_identity','official_split_overlaps','training_admitted')}))


if __name__=='__main__':main()
