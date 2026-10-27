"""Prepare PMEmo chorus labels as audited candidates; never enable training."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import unicodedata
import zipfile

import numpy as np
import soundfile as sf

WORKSPACE = Path(__file__).resolve().parents[2]
AXES = ['valence', 'arousal']
ANNOTATIONS = ('dynamic_annotations.csv', 'dynamic_annotations_std.csv',
               'static_annotations.csv', 'static_annotations_std.csv')
EXPECTED_SHA256 = 'acfcd0e5aefd3905458e73a583aa17e667fe96f58defe1908196b11b1034b198'
EXPECTED_BYTES = 679668214


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_rows(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'Missing or duplicate CSV columns: {path}')
        if not set(required).issubset(reader.fieldnames):
            raise ValueError(f'Unexpected CSV schema: {path}')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Malformed CSV row: {path}, line {reader.line_num}')
            yield {key: value.strip() for key, value in row.items()}


def source_id(value):
    if not re.fullmatch(r'[1-9][0-9]*', value):
        raise ValueError(f'Invalid musicId: {value}')
    return int(value)


def annotation_value(value, statistic):
    if statistic not in ('mean', 'std'):
        raise ValueError('Statistic must be mean or std')
    number = float(value) if value else np.nan
    if np.isnan(number):
        return number
    if not np.isfinite(number) or number < 0 or (statistic == 'mean' and number > 1):
        raise ValueError(f'Invalid {statistic} annotation: {value}')
    return number


def dynamic_csv(path, statistic='mean'):
    """Read the supplied chorus-relative frameTime, without resampling/offsets."""
    columns = [f'{axis.title()}({statistic})' for axis in AXES]
    result = {}
    for row in csv_rows(path, ['musicId', 'frameTime', *columns]):
        sid = source_id(row['musicId'])
        timestamp = float(row['frameTime'])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f'Invalid frameTime: {sid}')
        times, values = result.setdefault(sid, ([], []))
        if times and timestamp <= times[-1]:
            raise ValueError(f'Non-increasing or duplicate frameTime: {sid}')
        times.append(timestamp)
        values.append([annotation_value(row[column], statistic) for column in columns])
    return {sid: (np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64))
            for sid, (times, values) in result.items()}


def static_csv(path, statistic='mean'):
    columns = [f'{axis.title()}({statistic})' for axis in AXES]
    result = {}
    for row in csv_rows(path, ['musicId', *columns]):
        sid = source_id(row['musicId'])
        if sid in result:
            raise ValueError(f'Duplicate static annotation: {sid}')
        result[sid] = np.asarray([annotation_value(row[column], statistic) for column in columns])
    return result


def align_dynamic(means, deviations, duration):
    """Align observed rows only; masked target placeholders are finite zeroes."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Audio duration must be finite and positive')
    empty = (np.empty(0, dtype=np.float64), np.empty((0, 2), dtype=np.float64))
    means = empty if means is None else means
    deviations = empty if deviations is None else deviations
    for times, values in (means, deviations):
        if times.ndim != 1 or values.shape != (len(times), 2):
            raise ValueError('Invalid dynamic annotation shape')
        if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError('Invalid dynamic annotation timestamps')
        if np.any(np.isinf(values)):
            raise ValueError('Infinite annotation values')
    if np.any((means[1] < 0) | (means[1] > 1)) or np.any(deviations[1] < 0):
        raise ValueError('Out-of-range annotations')
    times = np.union1d(means[0], deviations[0])
    targets = np.zeros((len(times), 2), dtype=np.float32)
    mask = np.zeros(targets.shape, dtype=bool)
    label_std = np.zeros(targets.shape, dtype=np.float32)
    std_mask = np.zeros(targets.shape, dtype=bool)
    indices = np.searchsorted(times, means[0])
    finite = np.isfinite(means[1])
    valid = finite & (means[0][:, None] <= duration)
    converted = np.where(valid, 2 * means[1] - 1, 0)
    targets[indices] = converted
    mask[indices] = valid
    outside = int(np.sum(finite & (means[0][:, None] > duration)))
    indices = np.searchsorted(times, deviations[0])
    available = np.isfinite(deviations[1])
    label_std[indices] = np.where(available, 2 * deviations[1], 0)
    std_mask[indices] = available
    if not np.all(np.isfinite(label_std)):
        raise ValueError('Standard deviation cannot be represented as float32')
    return times, targets, mask, label_std, std_mask, outside


def read_metadata(path):
    fields = ['musicId', 'fileName', 'title', 'artist', 'album', 'duration',
              'chorus_start_time', 'chorus_end_time']
    result = {}
    for row in csv_rows(path, fields):
        sid = source_id(row['musicId'])
        if sid in result or row['fileName'] != f'{sid}.mp3':
            raise ValueError(f'Duplicate ID or unexpected metadata audio filename: {sid}')
        duration = float(row['duration'])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'Invalid metadata duration: {sid}')
        result[sid] = dict(artist=row['artist'], title=row['title'], album=row['album'],
                           metadata_duration_seconds=duration,
                           chorus_start_time=row['chorus_start_time'], chorus_end_time=row['chorus_end_time'])
    return result


def extract_selected(archive, destination):
    """Preflight every ZIP path, then exclusively copy the narrow allowlist."""
    destination = Path(destination).resolve()
    fixed = {'PMEmo2019/metadata.csv'} | {'PMEmo2019/annotations/' + name for name in ANNOTATIONS}
    with zipfile.ZipFile(archive) as source:
        selected, seen = [], set()
        for info in source.infolist():
            name = info.filename
            parts = name.rstrip('/').split('/')
            if '\\' in name or ':' in name or name.startswith('/') or any(p in ('', '.', '..') for p in parts):
                raise ValueError(f'Unsafe ZIP path: {name}')
            if name not in fixed and not re.fullmatch(r'PMEmo2019/chorus/[1-9][0-9]*\.mp3', name):
                continue
            if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir():
                raise ValueError(f'Unsupported ZIP member: {name}')
            if name in seen:
                raise ValueError(f'Duplicate ZIP member: {name}')
            seen.add(name)
            target = (destination / PurePosixPath(name).relative_to('PMEmo2019')).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'ZIP member escapes destination: {name}')
            selected.append((info, target))
        if not fixed.issubset(seen) or not any('/chorus/' in name for name in seen):
            raise ValueError('Archive is missing required PMEmo files')
        for info, target in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as input_stream, target.open('xb') as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    return [target for _, target in selected]


def audit_audio(path):
    before = path.stat()
    sha = digest(path)
    with sf.SoundFile(path) as stream:
        sample_rate, channels, reported_frames = stream.samplerate, stream.channels, len(stream)
        if sample_rate <= 0 or channels <= 0:
            raise ValueError(f'Invalid audio format: {path}')
        decoded = 0
        position = stream.tell()
        # blocks() may expose an oversized buffer after a short MP3 read.
        # Only the decoder's actual frame-position advance is valid audio.
        for block in stream.blocks(blocksize=65536, dtype='float32', always_2d=True):
            next_position = stream.tell()
            actual = next_position - position
            if actual < 0 or actual > len(block):
                raise ValueError(f'Invalid decoder frame count: {path}')
            if not actual:
                break
            if not np.all(np.isfinite(block[:actual])):
                raise ValueError(f'Non-finite decoded audio: {path}')
            decoded += actual
            position = next_position
        if not decoded:
            raise ValueError(f'Empty audio decode: {path}')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Audio changed during audit: {path}')
    return dict(audio_sha256=sha, audio_bytes=before.st_size, audio_decode_ok=True,
                sample_rate=sample_rate, channels=channels, decoded_frames=decoded,
                reported_header_frames=reported_frames, duration_seconds=decoded / sample_rate)


def artist_key(name):
    return re.sub(r'[\W_]+', ' ', unicodedata.normalize('NFKC', name).casefold()).strip()


def static_record(sid, means, deviations):
    values = means.get(sid, np.full(2, np.nan))
    std = deviations.get(sid, np.full(2, np.nan))
    as_list = lambda array: [float(value) if np.isfinite(value) else None for value in array]
    return dict(song_id=f'pmemo:{sid}', source_song_id=sid, label_order=AXES,
                mean_0_1=as_list(values), mean_minus1_1=as_list(2 * values - 1),
                std_0_1=as_list(std), std_minus1_1=as_list(2 * std),
                mask=np.isfinite(values).tolist(), std_mask=np.isfinite(std).tolist())


def prepare(root, output=None, *, expected_sha256=EXPECTED_SHA256,
            expected_bytes=EXPECTED_BYTES, workers=4):
    started = time.monotonic()
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root / 'prepared/v1'
    prepared_root = (root / 'prepared').resolve()
    if (not prepared_root.is_relative_to(root) or not output.is_relative_to(root)
            or output == prepared_root or not output.is_relative_to(prepared_root) or output.exists()):
        raise ValueError('Use a new output directory under dataset_root/prepared')
    if not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    archive = root / 'raw/PMEmo2019.zip'
    before = archive.stat()
    if expected_bytes is not None and before.st_size != expected_bytes:
        raise ValueError('PMEmo archive byte size does not match expected source')
    archive_sha = digest(archive)
    if expected_sha256 is not None and archive_sha != expected_sha256.lower():
        raise ValueError('PMEmo archive SHA256 does not match expected source')
    staging = output.with_name(output.name + '.pending')
    staging.mkdir(parents=True, exist_ok=False)
    # A failed run retains its exclusive .pending directory for inspection.
    # Re-running requires a fresh output name; neither source nor outputs are deleted.
    source = staging / 'source'
    extract_selected(archive, source)
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Archive changed during preparation')
    annotations = source / 'annotations'
    means = dynamic_csv(annotations / 'dynamic_annotations.csv')
    deviations = dynamic_csv(annotations / 'dynamic_annotations_std.csv', 'std')
    static_means = static_csv(annotations / 'static_annotations.csv')
    static_deviations = static_csv(annotations / 'static_annotations_std.csv', 'std')
    metadata = read_metadata(source / 'metadata.csv')
    audio = {int(path.stem): path for path in (source / 'chorus').glob('*.mp3')}
    ids = set(metadata)
    if set(audio) != ids:
        raise ValueError(f'Audio/metadata IDs mismatch: missing={sorted(ids-set(audio))}, extra={sorted(set(audio)-ids)}')
    for name, rows in [('dynamic', means), ('dynamic_std', deviations), ('static', static_means), ('static_std', static_deviations)]:
        if set(rows) - ids:
            raise ValueError(f'Unknown {name} annotation IDs: {sorted(set(rows)-ids)}')
    print(f'Auditing {len(ids)} PMEmo chorus files with {workers} workers', flush=True)
    audited, errors = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(audit_audio, audio[sid]): sid for sid in sorted(ids)}
        for count, future in enumerate(as_completed(futures), 1):
            sid = futures[future]
            try:
                audited[sid] = future.result()
            except Exception as exc:
                errors.append(dict(source_song_id=sid, error=str(exc)))
            if count % 100 == 0 or count == len(ids):
                print(f'Checked {count}/{len(ids)}; errors={len(errors)}', flush=True)
    if errors:
        write_json(staging / 'errors.json', errors)
        raise ValueError(f'Audio audit failed; see {staging / "errors.json"}; no release published')
    (staging / 'dynamic').mkdir()
    records, outside_audio = [], []
    for sid in sorted(ids):
        info = audited[sid]
        times, targets, mask, label_std, std_mask, outside = align_dynamic(means.get(sid), deviations.get(sid), info['duration_seconds'])
        np.savez_compressed(staging / 'dynamic' / f'{sid}.npz', times_seconds=times,
                            targets=targets, mask=mask, label_std=label_std, label_std_mask=std_mask)
        if outside:
            outside_audio.append(dict(source_song_id=sid, masked_values=outside))
        artist = artist_key(metadata[sid]['artist'])
        records.append(dict(dataset='PMEmo', song_id=f'pmemo:{sid}', source_song_id=sid,
                            **metadata[sid], **info, group_id='artist:' + artist if artist else f'pmemo:unresolved:{sid}',
                            group_id_tentative=True, split='candidate', training_enabled=False,
                            eligible_dynamic=bool(mask.any()),
                            audio_path=(output / 'source/chorus' / f'{sid}.mp3').relative_to(root).as_posix(),
                            labels_path=(output / 'dynamic' / f'{sid}.npz').relative_to(root).as_posix(),
                            label_order=AXES, label_count=len(times), valid_values=mask.sum(axis=0).tolist(),
                            first_label_seconds=float(times[0]) if len(times) else None,
                            last_label_seconds=float(times[-1]) if len(times) else None,
                            audio_time_origin='chorus_start; frameTime retained unchanged',
                            metadata_duration_difference_seconds=info['duration_seconds']-metadata[sid]['metadata_duration_seconds']))
    with (staging / 'manifest.jsonl').open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    with (staging / 'static.jsonl').open('x', encoding='utf-8') as stream:
        for sid in sorted(ids):
            stream.write(json.dumps(static_record(sid, static_means, static_deviations), ensure_ascii=False, allow_nan=False) + '\n')
    write_json(staging / 'config.json', dict(schema_version=1, dataset='PMEmo',
               dataset_root=Path(os.path.relpath(root, output)).as_posix(), label_order=AXES,
               target_scale=[-1, 1], target_conversion='2 * original_mean_0_1 - 1',
               std_conversion='2 * original_std_0_1', missing_target_storage=0,
               missing_target_rule='mask=false; exclude from loss and metrics',
               timestamp_source='dynamic CSV frameTime in seconds relative to supplied chorus audio',
               chorus_start_time_applied=False, static_labels='static.jsonl; never tiled over dynamic time',
               training_enabled=False, split_policy='candidate only; cross-dataset split and deduplication pending'))
    provenance = [dict(path='raw/PMEmo2019.zip', bytes=before.st_size, sha256=archive_sha)]
    for path in [source / 'metadata.csv', *sorted(annotations.glob('*.csv'))]:
        provenance.append(dict(path=(output / path.relative_to(staging)).relative_to(root).as_posix(),
                               bytes=path.stat().st_size, sha256=digest(path)))
    write_json(staging / 'annotation-provenance.json', provenance)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), status='candidate_pending_cross_dataset_dedup_and_split',
                  songs=len(records), dynamic_songs=sum(r['eligible_dynamic'] for r in records),
                  no_eligible_dynamic_song_ids=[r['song_id'] for r in records if not r['eligible_dynamic']],
                  training_enabled=False, full_audio_decode=True, audio_decoder=f'soundfile {sf.__version__}',
                  outside_audio_masked=outside_audio,
                  audio_hours=sum(r['duration_seconds'] for r in records) / 3600,
                  audio_header_frame_mismatches=[dict(song_id=r['song_id'], reported=r['reported_header_frames'], decoded=r['decoded_frames'])
                                                for r in records if r['reported_header_frames'] != r['decoded_frames']],
                  metadata_duration_differences=[dict(song_id=r['song_id'], difference_seconds=r['metadata_duration_difference_seconds'])
                                                 for r in records if abs(r['metadata_duration_difference_seconds']) > .1],
                  source_archive_preserved=True, extracted_content=['chorus audio', 'four annotation CSVs', 'metadata CSV'],
                  limitations=['Artist names are normalized tentative groups; aliases and audio overlaps are not yet resolved.',
                               'SoundFile full decoding validates finite samples and duration; it is not a separate strict FFmpeg corruption check.'],
                  elapsed_seconds=round(time.monotonic() - started, 2))
    write_json(staging / 'report.json', report)
    if output.exists():
        raise ValueError('Output appeared during preparation; refusing to overwrite')
    staging.rename(output)
    print(json.dumps(dict(songs=report['songs'], dynamic_songs=report['dynamic_songs'], output=str(output)), ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=WORKSPACE / 'data/datasets/pmemo')
    parser.add_argument('--output', type=Path, help='New directory under ROOT/prepared; default ROOT/prepared/v1')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--expected-sha256', default=EXPECTED_SHA256)
    parser.add_argument('--expected-bytes', type=int, default=EXPECTED_BYTES)
    args = parser.parse_args()
    prepare(args.root, args.output, workers=args.workers,
            expected_sha256=args.expected_sha256, expected_bytes=args.expected_bytes)
