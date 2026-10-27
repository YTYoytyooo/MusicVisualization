"""Audit MERGE identifiers, official splits and training-only static-label conventions."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes

JOB = 'd-merge-schema-v1'
FOLDER = BASE / 'analyses' / JOB
PACKAGE = ROOT / 'data/datasets/merge/raw/v1.1.0/package-metadata'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def normalized(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text or '').casefold()))


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len({row['Song'] for row in rows}) != len(rows):
        raise ValueError('Duplicate Song identifiers: ' + str(path))
    return {row['Song']: row for row in rows}


def register():
    if FOLDER.exists():
        raise FileExistsError('Preserve previous schema audit')
    members = read_json(PACKAGE / 'metadata-members.json')
    files = {row['archive_name']: row['path'] for row in members}
    hashes = {row['path']: row['sha256'] for row in members}
    source = BASE / 'protocols/b-v1.json'
    protocol = read_json(source)
    manifest = Path(protocol['source_manifest'])
    hashes.update({relative(manifest): protocol['manifest_sha256'], relative(Path(__file__)): sha256(Path(__file__)),
                   relative(PACKAGE / 'archive-index.json'): sha256(PACKAGE / 'archive-index.json')})
    verify_hashes(hashes)
    argv = [str(Path(sys.executable).resolve()), '-m', 'tools.train_next_v2.auxiliary_data.merge_inspect', '--run']
    environment = read_json(BASE / 'analyses/d-merge-package-metadata-v1/launch.json')['environment']
    launch = dict(job_id=JOB, registered_at=now(), original_argv=argv, cwd=str(ROOT), python=argv[0],
        environment=environment, files=files, protected_hashes=hashes, source_manifest=relative(manifest),
        os_lock=str(lock_path(JOB)), output=relative(FOLDER / 'report.json'), gpu_used=False,
        label_scope='Numerical label diagnostics use MERGE official 70/15/15 training IDs only; no model predictions or original heldout labels.',
        recovery='Finite read-only audit; preserve report, reject changed hashes, no automatic retry.')
    FOLDER.mkdir(parents=True)
    atomic_json(FOLDER / 'launch.json', launch)
    record_event(dict(event='D_MERGE_schema_inspection_registered', launch=relative(FOLDER / 'launch.json')))


def run():
    if (FOLDER / 'report.json').exists():
        raise FileExistsError('Schema audit already completed')
    launch = read_json(FOLDER / 'launch.json')
    verify_hashes(launch['protected_hashes'])
    files = launch['files']
    metadata = csv_rows(ROOT / files['MERGE_Audio_Complete/merge_audio_complete_metadata.csv'])
    values = csv_rows(ROOT / files['MERGE_Audio_Complete/merge_audio_complete_av_values.csv'])
    index = read_json(PACKAGE / 'archive-index.json')
    audio = {Path(row['name']).stem: row for row in index if row['name'].lower().endswith('.mp3')}
    if set(audio) != set(metadata) or set(values) != set(metadata) or len(metadata) != 3554:
        raise ValueError('Audio/metadata/static-value IDs do not match all 3554 songs')
    split_summaries, schemes = {}, {}
    for scheme in ('70_15_15', '40_30_30'):
        splits = {}
        for role in ('train', 'validate', 'test'):
            name = f'MERGE_Audio_Complete/tvt_dataframes/tvt_{scheme}/tvt_{scheme}_{role}_audio_complete.csv'
            splits[role] = csv_rows(ROOT / files[name])
            if any(row['Quadrant'] != metadata[sid]['Quadrant'] for sid, row in splits[role].items()):
                raise ValueError('Official split quadrant differs from metadata')
        roles = list(splits)
        if set().union(*(set(rows) for rows in splits.values())) != set(metadata):
            raise ValueError('Official split missing IDs')
        if sum(len(rows) for rows in splits.values()) != len(metadata):
            raise ValueError('Official split IDs overlap')
        artists = {role: {normalized(metadata[sid]['Artist']) for sid in rows if metadata[sid]['Artist'].strip()} for role, rows in splits.items()}
        overlaps = {f'{left}/{right}': len(artists[left] & artists[right])
                    for i, left in enumerate(roles) for right in roles[i+1:]}
        split_summaries[scheme] = dict(counts={role: len(rows) for role, rows in splits.items()},
                                      artist_overlap_counts=overlaps, song_ids_disjoint=True)
        schemes[scheme] = splits
    train_ids = sorted(schemes['70_15_15']['train'])
    floats, missing = [], []
    for sid in train_ids:
        try:
            v, a = float(values[sid]['Valence']), float(values[sid]['Arousal'])
            if not (math.isfinite(v) and math.isfinite(a)):
                raise ValueError('nonfinite')
        except (ValueError, TypeError):
            missing.append(sid)
            continue
        floats.append((sid, v, a))
    matches = {}
    for center in (0., .5):
        matching, ambiguous = 0, 0
        for sid, v, a in floats:
            if v == center or a == center:
                ambiguous += 1
                continue
            quadrant = 'Q1' if v > center and a > center else 'Q2' if v < center and a > center else 'Q3' if v < center and a < center else 'Q4'
            matching += quadrant == metadata[sid]['Quadrant']
        matches[str(center)] = dict(matching=matching, ambiguous=ambiguous, total=len(floats))
    # Original corpus is consulted for recording identity fields only, never target values.
    with (ROOT / launch['source_manifest']).open(encoding='utf-8') as stream:
        main = [json.loads(line) for line in stream]
    main_identity = {}
    for row in main:
        key = (normalized(row.get('artist')), normalized(row.get('title')))
        if all(key):
            main_identity.setdefault(key, []).append(dict(song_id=row['song_id'], split=row['split'], group_id=row['group_id']))
    cross = []
    for sid, row in metadata.items():
        key = (normalized(row['Artist']), normalized(row['Title']))
        if key in main_identity:
            cross.append(dict(merge_song_id=sid, main_recordings=main_identity[key]))
    license_names = [row['name'] for row in index if any(word in Path(row['name']).name.lower() for word in ('license', 'licence', 'readme', 'copying'))]
    report = dict(job_id=JOB, status='completed_metadata_inspection', completed_at=now(),
        source_hashes=launch['protected_hashes'], songs=len(metadata), archive_audio_members=len(audio),
        official_splits=split_summaries, package_license_or_readme_files=license_names,
        declared_record_license='cc-by-nc-4.0', whole_archive_md5_verified=False,
        static_value_column_order=['Song', 'Arousal', 'Valence'],
        training_label_audit=dict(training_ids=len(train_ids), valid_numeric=len(floats), missing_ids=missing,
            ranges={axis: dict(min=min(row[col] for row in floats), max=max(row[col] for row in floats))
                    for axis, col in [('valence', 1), ('arousal', 2)]},
            quadrant_agreement_by_center=matches,
            interpretation='Compare file-scale conventions to published documentation before defining any transformation. No normalization has been applied.'),
        exact_artist_title_cross_corpus_matches=cross,
        identity_limitations='Metadata matching is only a first screen; audio fingerprints and recording/segment identity still required.',
        training_admitted=False, original_heldout_labels_used=0, audio_downloaded=False,
        primary_sources=[dict(url='https://arxiv.org/html/2407.06060v3', sections='III and dataset description',
            findings='Static V/A scores are derived from mapped emotion words, with published scale [-1,1]. Quadrant validation does not manually validate the numeric V/A scores.'),
            dict(url='https://zenodo.org/records/13939205', findings='Version 1.1.0 release supplies audio excerpts, metadata, static scores and official splits.')],
        remaining=['Resolve numeric scale and per-source provenance, including private-collection A identifiers.',
            'Record package/license evidence, acquire checksummed audio, decode native duration and audit clip offsets.',
            'Exclude duplicate recordings and linked groups across training, development, calibration and final-test partitions.',
            'Freeze independent static/quadrant auxiliary tasks and equal-step controls before fitting.'])
    atomic_json(FOLDER / 'report.json', report)
    stages = read_json(PLAN)['stages']
    stages['D'].update(phase='MERGE_metadata_inspected_scale_identity_audio_pending', schema_audit=relative(FOLDER / 'report.json'))
    record_event(dict(event='D_MERGE_metadata_schema_inspected', report=relative(FOLDER / 'report.json')),
        updates=dict(stages=stages, next_action='继续核实MERGE包内静态V/A量纲与分来源标签映射，处理艺人/录音跨划分重叠，再获取校验音频；不得直接把静态数值复制成时间曲线。'))
    print(json.dumps({key: report[key] for key in ('songs', 'official_splits', 'training_label_audit', 'package_license_or_readme_files')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    with run_lease(JOB):
        register() if args.register else run()
