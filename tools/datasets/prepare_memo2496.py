"""Prepare Memo2496 at its native 1 Hz clock; candidates require dataset deduplication."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid
import zipfile

import numpy as np
import soundfile as sf

WORKSPACE = Path(__file__).resolve().parents[2]
AXES = ['valence', 'arousal']
RAW_KEYS = {'valence': 'V', 'arousal': 'A'}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def dynamic_csv(path, standard_deviation=False):
    """Keep the full header clock and pad absent cells with NaN, never interpolate."""
    songs = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if not header or header[0].strip() != 'song_id':
            raise ValueError(f'Invalid label header: {path}')
        milliseconds = []
        for name in header[1:]:
            match = re.fullmatch(r'sample_(\d+)ms', name.strip())
            if not match:
                raise ValueError(f'Invalid 1 Hz timestamp header: {name}')
            milliseconds.append(int(match[1]))
        if not milliseconds or milliseconds != list(range(0, len(milliseconds) * 1000, 1000)):
            raise ValueError(f'Expected increasing 1 Hz timestamps starting at 0: {path}')
        times = np.asarray(milliseconds, dtype=np.float64) / 1000
        for line, row in enumerate(reader, 2):
            if not row:
                continue
            sid = int(row[0])
            if sid <= 0 or sid in songs or len(row) > len(header):
                raise ValueError(f'Duplicate ID or invalid row at {path}:{line}')
            values = np.full(len(times), np.nan, dtype=np.float64)
            values[:len(row)-1] = [float(v) if v.strip() else np.nan for v in row[1:]]
            finite = values[np.isfinite(values)]
            lower = 0 if standard_deviation else -1
            if np.any(np.isinf(values)) or np.any(finite < lower) or np.any(finite > 1):
                kind = 'standard deviation' if standard_deviation else 'dynamic label'
                raise ValueError(f'Out-of-range {kind} for song {sid} in {path}; expected [{lower}, 1]')
            songs[sid] = (times.copy(), values)
    if not songs:
        raise ValueError(f'Empty label table: {path}')
    return songs


def read_metadata(path):
    result, filenames = {}, set()
    required = {'song_id', 'file_name', 'Song title', 'duration (min.sec)', 'emotion', 'distance', 'platform'}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Unexpected songs_info_all.csv schema')
        for row in reader:
            sid = int(row['song_id'])
            filename = row['file_name'].strip()
            if sid <= 0 or sid in result:
                raise ValueError(f'Duplicate or invalid metadata song_id: {sid}')
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.mp3', filename, re.IGNORECASE):
                raise ValueError(f'Unsafe metadata audio filename: {filename}')
            if filename.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1, 10)], *[f'LPT{i}' for i in range(1, 10)]}:
                raise ValueError(f'Unsafe Windows metadata audio filename: {filename}')
            if filename.casefold() in filenames:
                raise ValueError(f'Ambiguous metadata audio filename: {filename}')
            filenames.add(filename.casefold())
            # The min.sec field and categorical emotion are provenance only.
            result[sid] = dict(file_name=filename, title=row['Song title'], artist='',
                               metadata_duration_min_sec=row['duration (min.sec)'],
                               source_emotion=row['emotion'], source_distance=row['distance'],
                               platform=row['platform'])
    return result


def index_audio(archive, metadata):
    """Resolve exact basenames; reject unsafe paths, links and duplicate members."""
    wanted = {entry['file_name'].casefold(): sid for sid, entry in metadata.items()}
    if len(wanted) != len(metadata):
        raise ValueError('Ambiguous metadata audio filenames')
    members, paths = {}, set()
    for info in archive.infolist():
        name = info.orig_filename
        stripped = name[:-1] if name.endswith('/') else name
        parts = stripped.split('/')
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if (not stripped or '\\' in name or '\x00' in name or ':' in name
                or any(p in ('', '.', '..') or p.endswith((' ', '.')) for p in parts)
                or kind not in (0, stat.S_IFREG, stat.S_IFDIR)):
            raise ValueError(f'Unsafe ZIP member: {name!r}')
        key = stripped.casefold()
        if key in paths:
            raise ValueError(f'Duplicate ZIP path: {name}')
        paths.add(key)
        if info.is_dir():
            continue
        sid = wanted.get(parts[-1].casefold())
        if sid is None:
            continue
        if sid in members:
            raise ValueError(f'Ambiguous ZIP audio basename: {parts[-1]}')
        if parts[-1] != metadata[sid]['file_name']:
            raise ValueError(f'ZIP audio filename case differs from metadata: {parts[-1]}')
        if info.flag_bits & 1 or info.file_size <= 0:
            raise ValueError(f'Encrypted or empty ZIP audio: {name}')
        members[sid] = info
    return members


def verify_raw_ratings(path, means, stds):
    """Independently reproduce the supplied mean and population standard deviation."""
    entries = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(entries, list) or not entries:
        raise ValueError('Raw-rating JSON must contain a nonempty list')
    groups = defaultdict(list)
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get('MID')) is not int or entry['MID'] <= 0:
            raise ValueError('Invalid raw-rating MID')
        for key in RAW_KEYS.values():
            values = entry.get(key)
            if (not isinstance(values, list) or not values
                    or any(type(value) not in (int, float) for value in values)):
                raise ValueError(f'Invalid raw-rating {key} sequence for MID {entry["MID"]}')
            values = np.asarray(values, dtype=np.float64)
            if np.any(~np.isfinite(values)) or np.any(np.abs(values) > 1):
                raise ValueError(f'Out-of-range raw-rating {key} for MID {entry["MID"]}')
        groups[entry['MID']].append(entry)
    compared = {'mean': 0, 'std': 0}
    max_error = {'mean': 0., 'std': 0.}
    missing = []
    for axis, key in RAW_KEYS.items():
        for sid in sorted(set(means[axis]) | set(stds[axis])):
            if sid not in groups:
                raise ValueError(f'Raw-rating records missing for supplied song {sid}')
            rows = groups[sid]
            length = max(len(row[key]) for row in rows)
            matrix = np.full((len(rows), length), np.nan, dtype=np.float64)
            for index, row in enumerate(rows):
                matrix[index, :len(row[key])] = row[key]
            expected_mean = np.nanmean(matrix, axis=0)
            expected_std = np.nanstd(matrix, axis=0, ddof=0)
            for kind, table, expected in [('mean', means[axis], expected_mean), ('std', stds[axis], expected_std)]:
                if sid not in table:
                    missing.append(dict(source_song_id=sid, axis=axis, statistic=kind, missing_row=True))
                    continue
                times, supplied = table[sid]
                valid = np.isfinite(supplied)
                indices = times[valid].astype(np.int64)
                if np.any(indices >= length):
                    raise ValueError(f'Raw-rating {kind} support mismatch: {axis}, song {sid}')
                errors = np.abs(supplied[valid] - expected[indices])
                maximum = float(np.max(errors)) if len(errors) else 0.
                if maximum > 1e-7:
                    raise ValueError(f'Raw-rating {kind} mismatch: {axis}, song {sid}, max error {maximum}')
                compared[kind] += int(valid.sum())
                max_error[kind] = max(max_error[kind], maximum)
                missing_count = length - int(valid.sum())
                if missing_count:
                    missing.append(dict(source_song_id=sid, axis=axis, statistic=kind,
                                        missing_values=missing_count))
    return dict(status='verified', ratings=len(entries), songs=len(groups),
                mean_values_compared=compared['mean'], std_values_compared=compared['std'],
                maximum_absolute_error=max_error, std_definition='population (ddof=0)',
                missing_supplied_values=missing)


def audit_audio(path, audio_sha256=None):
    before = path.stat()
    sha = audio_sha256 or digest(path)
    decoded = 0
    with sf.SoundFile(path) as stream:
        sample_rate, channels, header_frames = stream.samplerate, stream.channels, len(stream)
        if sample_rate <= 0 or channels <= 0:
            raise ValueError('Invalid audio sample rate or channels')
        while True:
            position = stream.tell()
            block = stream.read(65536, dtype='float32', always_2d=True)
            actual = stream.tell() - position
            # Some MP3 headers overstate length; decoder cursor progress is authoritative.
            if actual < 0 or actual > len(block):
                raise ValueError('Invalid decoder frame progress')
            if not actual:
                break
            if not np.all(np.isfinite(block[:actual])):
                raise ValueError('Non-finite decoded audio')
            decoded += actual
    if not decoded:
        raise ValueError('Empty audio decode')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Audio changed during audit')
    return dict(audio_sha256=sha, duration_seconds=decoded / sample_rate,
                decoded_frames=decoded, reported_header_frames=header_frames,
                sample_rate=sample_rate, channels=channels, audio_bytes=before.st_size,
                audio_decode_ok=True)


def align_labels(valence, arousal, valence_std, arousal_std, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Expected a positive decoded audio duration')
    times = np.unique(np.concatenate([pair[0] for pair in (valence, arousal, valence_std, arousal_std)]))
    targets = np.zeros((len(times), 2), dtype=np.float32)
    mask = np.zeros_like(targets, dtype=bool)
    std = np.zeros_like(targets)
    std_mask = np.zeros_like(mask)
    outside = []
    for axis, (axis_times, values) in enumerate([valence, arousal]):
        indices = np.searchsorted(times, axis_times)
        finite = np.isfinite(values)
        good = finite & (axis_times >= 0) & (axis_times <= duration)
        targets[indices[good], axis] = values[good]
        mask[indices[good], axis] = True
        outside.append(int(np.sum(finite & ((axis_times < 0) | (axis_times > duration)))))
    for axis, (axis_times, values) in enumerate([valence_std, arousal_std]):
        indices = np.searchsorted(times, axis_times)
        good = np.isfinite(values) & mask[indices, axis]
        std[indices[good], axis] = values[good]
        std_mask[indices[good], axis] = True
    return times, targets, mask, std, std_mask, outside


def prepare(root, output, workers=4, expected_songs=2496):
    started = time.monotonic()
    root, output = Path(root).resolve(), Path(output).resolve()
    if (output == root / 'prepared' or not output.is_relative_to(root / 'prepared') or output.exists()):
        raise ValueError('Use a new output directory under dataset_root/prepared')
    if not root.is_relative_to(WORKSPACE) or not output.is_relative_to(WORKSPACE):
        raise ValueError('All dataset files must remain inside the Vm workspace')
    if not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    raw = root / 'raw'
    if not raw.resolve().is_relative_to(WORKSPACE):
        raise ValueError('Raw dataset directory escapes the Vm workspace')
    staging = output.with_name(output.name + '.pending-' + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    report = dict(dataset='Memo2496', created_at=datetime.now(timezone.utc).isoformat(),
                  status='preparing', training_enabled=False, range_issues=[],
                  missing_audio=[], missing_labels={}, audio_decode_failures=[])
    try:
        inputs = [raw / f'{axis}_all_{kind}.csv' for axis in AXES for kind in ('average', 'std')]
        inputs += [raw / 'songs_info_all.csv', raw / 'Label_pseudonymous.json', raw / 'MusicRawData.zip']
        for path in inputs:
            if not path.resolve().is_relative_to(WORKSPACE):
                raise ValueError(f'Input path escapes the Vm workspace: {path}')
        snapshots = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in inputs}
        means = {axis: dynamic_csv(raw / f'{axis}_all_average.csv') for axis in AXES}
        stds = {axis: dynamic_csv(raw / f'{axis}_all_std.csv', True) for axis in AXES}
        metadata = read_metadata(raw / 'songs_info_all.csv')
        if len(metadata) != expected_songs:
            raise ValueError(f'Expected {expected_songs} metadata songs, found {len(metadata)}')
        report['source_songs'] = len(metadata)
        for axis in AXES:
            for kind, table in [('mean', means[axis]), ('std', stds[axis])]:
                extra = sorted(set(table) - set(metadata))
                if extra:
                    raise ValueError(f'{axis} {kind} has unknown metadata IDs: {extra}')
                report['missing_labels'][f'{axis}_{kind}'] = sorted(set(metadata) - set(table))
        report['raw_rating_verification'] = verify_raw_ratings(raw / 'Label_pseudonymous.json', means, stds)
        report['label_validation'] = {}
        for axis in AXES:
            report['label_validation'][axis] = {}
            for kind, table in [('mean', means[axis]), ('std', stds[axis])]:
                values = np.concatenate([pair[1] for pair in table.values()])
                finite = values[np.isfinite(values)]
                report['label_validation'][axis][kind] = dict(
                    rows=len(table), finite_values=len(finite), missing_values=int(np.isnan(values).sum()),
                    expected_scale=[0 if kind == 'std' else -1, 1],
                    observed_min=float(finite.min()) if len(finite) else None,
                    observed_max=float(finite.max()) if len(finite) else None)
        (staging / 'audio').mkdir()
        (staging / 'dynamic').mkdir()
        extracted = {}
        with zipfile.ZipFile(raw / 'MusicRawData.zip') as archive:
            members = index_audio(archive, metadata)
            report['missing_audio'] = sorted(set(metadata) - set(members))
            report['archive_members'] = len(archive.infolist())
            for index, sid in enumerate(sorted(members), 1):
                info = members[sid]
                destination = staging / 'audio' / metadata[sid]['file_name']
                hasher = hashlib.sha256()
                size = 0
                with archive.open(info, 'r') as source, destination.open('xb') as target:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        target.write(block)
                        hasher.update(block)
                        size += len(block)
                if size != info.file_size:
                    raise ValueError(f'Extracted audio size mismatch: {sid}')
                extracted[sid] = dict(path=destination, sha256=hasher.hexdigest(),
                                      archive_member=info.filename, archive_crc32=f'{info.CRC:08x}')
                if index % 100 == 0 or index == len(members):
                    print(f'Extracted audio {index}/{len(members)}', flush=True)
        audited = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(audit_audio, item['path'], item['sha256']): sid for sid, item in extracted.items()}
            for index, future in enumerate(as_completed(futures), 1):
                sid = futures[future]
                try:
                    audited[sid] = future.result()
                except Exception as exc:
                    report['audio_decode_failures'].append(dict(source_song_id=sid, error=str(exc)))
                if index % 100 == 0 or index == len(futures):
                    print(f'Decoded audio {index}/{len(futures)}; failures={len(report["audio_decode_failures"])}', flush=True)
        records, outside_audio, missing_values = [], [], []
        # Missing whole rows remain missing on the supplied source clock.
        clocks = [pair[0] for table in list(means.values()) + list(stds.values()) for pair in table.values()]
        clock = np.unique(np.concatenate(clocks))
        empty = (clock, np.full(len(clock), np.nan))
        for sid in sorted(audited):
            pairs = [means[axis].get(sid, empty) for axis in AXES] + [stds[axis].get(sid, empty) for axis in AXES]
            times, targets, mask, std, std_mask, outside = align_labels(*pairs, audited[sid]['duration_seconds'])
            np.savez_compressed(staging / 'dynamic' / f'{sid}.npz', times_seconds=times, targets=targets,
                                mask=mask, label_std=std, label_std_mask=std_mask)
            if any(outside):
                outside_audio.append(dict(source_song_id=sid, masked_values=outside))
            missing_values.append(dict(source_song_id=sid,
                                       mean_values=[int(np.isnan(pair[1]).sum()) for pair in pairs[:2]],
                                       std_values=[int(np.isnan(pair[1]).sum()) for pair in pairs[2:]]))
            valid = mask.sum(axis=0).tolist()
            records.append(dict(dataset='Memo2496', song_id=f'memo2496:{sid}', source_song_id=sid,
                                **metadata[sid], **audited[sid],
                                audio_path=(output / 'audio' / metadata[sid]['file_name']).relative_to(root).as_posix(),
                                labels_path=(output / 'dynamic' / f'{sid}.npz').relative_to(root).as_posix(),
                                archive_member=extracted[sid]['archive_member'], archive_crc32=extracted[sid]['archive_crc32'],
                                group_id=f'memo2496:pending:{sid}', grouping_verified=False, split='candidate',
                                training_enabled=False, eligible_dynamic=all(count > 0 for count in valid),
                                label_order=AXES, sample_hz=1, label_count=len(times), valid_values=valid,
                                first_label_seconds=float(times[0]), last_label_seconds=float(times[-1]),
                                static_labels=None))
        if not records or not any(record['eligible_dynamic'] for record in records):
            raise ValueError('No audio recordings with valid dynamic labels on both axes')
        report.update(status='candidate_pending_cross_dataset_deduplication', songs=len(records),
                      eligible_dynamic_songs=sum(record['eligible_dynamic'] for record in records),
                      outside_audio_masked=outside_audio, missing_label_values=missing_values,
                      audio_full_decode=True, soundfile_version=sf.__version__,
                      libsndfile_version=sf.__libsndfile_version__,
                      audio_hours=sum(record['duration_seconds'] for record in records) / 3600,
                      audio_header_frame_mismatches=[dict(source_song_id=r['source_song_id'], reported=r['reported_header_frames'], decoded=r['decoded_frames'])
                                                    for r in records if r['reported_header_frames'] != r['decoded_frames']],
                      static_labels_provided=False, models_modified=False,
                      limitations=['Artist is absent from supplied metadata; no artist is inferred from titles.',
                                   'Split and grouping require cross-dataset duplicate checks before training.',
                                   'Categorical emotion and min.sec metadata do not create static VA labels or audio frames.'],
                      elapsed_seconds=round(time.monotonic() - started, 2))
        provenance = [dict(path=path.relative_to(root).as_posix(), bytes=snapshots[path][0], sha256=digest(path)) for path in inputs[:-1]]
        # Do not read the 12 GB ZIP twice; each selected member is CRC-checked and SHA256-hashed during extraction.
        provenance.append(dict(path='raw/MusicRawData.zip', bytes=snapshots[inputs[-1]][0],
                               member_integrity='ZIP CRC32 checked during extraction; extracted audio has SHA256'))
        for path, snapshot in snapshots.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != snapshot:
                raise ValueError(f'Source changed during preparation: {path}')
        with (staging / 'manifest.jsonl').open('w', encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        write_json(staging / 'annotation-provenance.json', provenance)
        write_json(staging / 'config.json', dict(schema_version=1, dataset='Memo2496',
                   dataset_root=Path(os.path.relpath(root, output)).as_posix(), label_order=AXES,
                   sample_hz=1, target_scale=[-1, 1], missing_target_storage=0,
                   missing_target_rule='mask=false; exclude from loss and metrics',
                   label_std_rule='population standard deviation; label_std_mask marks supplied valid values',
                   static_label_conversion=None, training_enabled=False,
                   split_policy='candidate only; requires cross-dataset deduplication'))
        write_json(staging / 'report.json', report)
        staging.rename(output)
        print(f'Prepared candidates: {output}; songs={len(records)}; training_enabled=false', flush=True)
        return report
    except Exception as exc:
        report.update(status='failed', error=str(exc), elapsed_seconds=round(time.monotonic() - started, 2))
        if 'Out-of-range' in str(exc):
            report['range_issues'].append(str(exc))
        write_json(staging / 'report.json', report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=WORKSPACE / 'data/datasets/memo2496')
    parser.add_argument('--output', type=Path, help='New ROOT/prepared directory; default: ROOT/prepared/v1')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    prepare(args.root, args.output or args.root / 'prepared/v1', args.workers)
