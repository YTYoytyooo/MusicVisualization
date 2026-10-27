"""Prepare DEAM song sequences; never train a model or rewrite source data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time
import unicodedata
import uuid
import zipfile

import numpy as np
import soundfile as sf

WORKSPACE = Path(__file__).resolve().parents[2]
AXES = ['valence', 'arousal']
AVERAGED = Path('annotations/annotations averaged per song')
DEFAULT_EXCLUSIONS = Path(__file__).with_name('deam_exclusions.json')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def dynamic_csv(path):
    """Ragged rows use the corresponding header prefix, never a guessed offset."""
    songs = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        times = []
        for name in header[1:]:
            match = re.fullmatch(r'sample_(\d+)ms', name.strip())
            if not match: raise ValueError(f'Invalid timestamp header: {name}')
            times.append(int(match[1]) / 1000)
        times = np.asarray(times, dtype=np.float64)
        if not len(times) or np.any(np.diff(times) != .5): raise ValueError('Expected increasing 2 Hz timestamps')
        for row in reader:
            if not row: continue
            sid = int(row[0])
            if sid in songs or len(row) > len(header): raise ValueError(f'Duplicate ID or extra values: {sid}')
            values = np.asarray([float(v) if v.strip() else np.nan for v in row[1:]], dtype=np.float64)
            if np.any(np.isinf(values)) or np.any(np.abs(values[np.isfinite(values)]) > 1):
                raise ValueError(f'Out-of-range dynamic label: {sid}')
            songs[sid] = (times[:len(values)], values)
    return songs


def align_labels(valence, arousal, duration):
    times = np.union1d(valence[0], arousal[0])
    targets = np.zeros((len(times), 2), dtype=np.float32)
    mask = np.zeros_like(targets, dtype=bool)
    outside = 0
    for axis, (axis_times, values) in enumerate([valence, arousal]):
        indices = np.searchsorted(times, axis_times)
        good = np.isfinite(values) & (axis_times >= 0) & (axis_times <= duration)
        outside += int(np.sum(np.isfinite(values) & (axis_times > duration)))
        targets[indices[good], axis] = values[good]
        mask[indices[good], axis] = True
    return times, targets, mask, outside


def static_csvs(folder):
    result = {}
    for path in sorted(folder.glob('*.csv')):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            for raw in reader:
                row = {k.strip(): v.strip() for k, v in raw.items()}
                sid = int(row['song_id'])
                if sid in result: raise ValueError(f'Duplicate static label: {sid}')
                means = [float(row[f'{axis}_mean']) for axis in AXES]
                stds = [float(row[f'{axis}_std']) for axis in AXES]
                if any(not math.isfinite(v) or not 1 <= v <= 9 for v in means) or any(not math.isfinite(v) or v < 0 for v in stds):
                    raise ValueError(f'Invalid static labels: {sid}')
                result[sid] = dict(mean_1_9=means, mean_minus1_1=[(v-5)/4 for v in means], std_1_9=stds)
    return result


def read_metadata(archive):
    result = {}
    with zipfile.ZipFile(archive) as source:
        for year in (2013, 2014, 2015):
            name = f'metadata/metadata_{year}.csv'
            data = source.read(name).decode('utf-8-sig')
            reader = csv.reader(io.StringIO(data)); next(reader)
            for row in reader:
                if not row: continue
                sid = int(row[0].strip())
                if sid in result: raise ValueError(f'Duplicate metadata ID: {sid}')
                row = [v.strip() for v in row]
                if year == 2013:
                    artist, title, genre = row[2], row[3], row[6]
                    # Preserve the original min.sec strings; do not interpret as decimal minutes.
                    offsets = row[4:6]
                elif year == 2014:
                    artist, title, genre, offsets = row[1], row[3], row[4], row[5:7]
                else:
                    artist, title, genre, offsets = row[3], row[2], row[5], []
                result[sid] = dict(artist=artist, title=title, genre=genre, metadata_year=year,
                                   original_segment_fields=offsets)
    return result


def artist_key(name):
    return re.sub(r'[\W_]+', ' ', unicodedata.normalize('NFKC', name).casefold()).strip()


def split_by_artist(records, seed):
    parent = {r['song_id']: r['song_id'] for r in records}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]; key = parent[key]
        return key
    def union(a, b): parent[find(b)] = find(a)
    seen_artist, seen_audio = {}, {}
    for r in records:
        sid = r['song_id']; artist = artist_key(r['artist'])
        if not artist: raise ValueError(f'Missing artist metadata: {sid}')
        for seen, key in [(seen_artist, artist), (seen_audio, r['audio_sha256'])]:
            if key in seen: union(sid, seen[key])
            else: seen[key] = sid
    groups = defaultdict(list)
    for r in records: groups[find(r['song_id'])].append(r)
    ordered = list(groups.values()); random.Random(seed).shuffle(ordered); ordered.sort(key=len, reverse=True)
    targets = dict(train=len(records)*.7, validation=len(records)*.15, test=len(records)*.15)
    counts = dict.fromkeys(targets, 0)
    for group in ordered:
        split = max(targets, key=lambda k: targets[k]-counts[k])
        group_id = 'artist-group-' + str(min(r['song_id'] for r in group))
        for r in group: r.update(split=split, group_id=group_id)
        counts[split] += len(group)
    for field in ('group_id', 'audio_sha256'):
        seen = {}
        for r in records:
            if r[field] in seen and seen[r[field]] != r['split']: raise AssertionError('Split leakage')
            seen[r[field]] = r['split']
    return counts, len(groups)


def audit_audio(path, ffmpeg, known_exclusion=None):
    before = path.stat()
    sha = digest(path)
    with sf.SoundFile(path) as stream:
        duration, sample_rate, channels, frames = len(stream)/stream.samplerate, stream.samplerate, stream.channels, len(stream)
        strict = subprocess.run([str(ffmpeg), '-nostdin', '-v', 'error', '-xerror', '-err_detect', 'explode',
                                 '-i', str(path), '-map', '0:a:0', '-f', 'null', '-'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
        codec_error = strict.stderr.decode('utf-8', errors='replace').strip()
        if known_exclusion:
            if sha != known_exclusion['audio_sha256']:
                raise ValueError('Previously excluded audio changed; review the exclusion before preparing it again')
            return dict(audio_sha256=sha, duration_seconds=duration, sample_rate=sample_rate, channels=channels,
                        decoded_frames=0, audio_bytes=before.st_size, audio_decode_ok=False,
                        audio_decode_error=known_exclusion['reason'])
        if strict.returncode or codec_error:
            return dict(audio_sha256=sha, duration_seconds=duration, sample_rate=sample_rate, channels=channels,
                        decoded_frames=0, audio_bytes=before.st_size, audio_decode_ok=False,
                        audio_decode_error=f'FFmpeg exit {strict.returncode}: {codec_error[:4000]}')
        decoded = 0
        # Count actual returned frames: blocks() may expose a full-sized buffer
        # after a short MP3 read when the container's length estimate is too large.
        while True:
            block = stream.read(65536, dtype='float32', always_2d=True)
            if not len(block): break
            if not np.all(np.isfinite(block)): raise ValueError('Non-finite decoded audio')
            decoded += len(block)
        if decoded == 0: raise ValueError('Empty audio decode')
    if (before.st_size, before.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns): raise ValueError('Audio changed during audit')
    return dict(audio_sha256=sha, duration_seconds=decoded/sample_rate, sample_rate=sample_rate, channels=channels,
                decoded_frames=decoded, reported_header_frames=frames,
                audio_bytes=before.st_size, audio_decode_ok=True)


def audit_features(path, duration):
    before = path.stat(); sha = digest(path)
    with path.open(encoding='utf-8-sig') as stream:
        header = stream.readline().strip().split(';')
        if len(header) != 261 or header[0] != 'frameTime': raise ValueError('Unexpected feature schema')
        first = last = None; rows = 0; non_half_second_steps = 0
        for line in stream:
            if not line.strip(): continue
            values = np.fromstring(line, sep=';')
            if len(values) != len(header) or not np.all(np.isfinite(values)): raise ValueError(f'Invalid feature row {rows+1}')
            t = float(values[0])
            if t < 0 or (last is not None and t <= last): raise ValueError('Feature timestamps not increasing')
            if last is not None and abs(t-last-.5) > 1e-5: non_half_second_steps += 1
            if first is None: first = t
            last = t; rows += 1
    if not rows: raise ValueError('Empty feature file')
    if (before.st_size, before.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns): raise ValueError('Features changed during audit')
    return dict(sha256=sha, bytes=before.st_size, rows=rows, columns=260, first_seconds=first, last_seconds=last,
                extends_beyond_audio=last > duration+.5, non_half_second_steps=non_half_second_steps,
                header_sha256=hashlib.sha256(';'.join(header).encode()).hexdigest(),
                training_enabled=False, reason='Temporal origin and extraction provenance are not independently verified')


def prepare(root, output, workers=4, seed=20261002, expected_songs=1802, ffmpeg=None, exclusions_path=DEFAULT_EXCLUSIONS):
    started = time.monotonic(); root = root.resolve(); output = output.resolve(); raw = root/'raw'
    if output == root/'prepared' or not output.is_relative_to(root/'prepared') or output.exists(): raise ValueError('Use a new output directory under dataset_root/prepared')
    if ffmpeg is None:
        bundled = WORKSPACE/'versions/v2.0/.runtime/bin/ffmpeg.exe'
        ffmpeg = bundled if bundled.is_file() else shutil.which('ffmpeg')
    if not ffmpeg: raise ValueError('FFmpeg is required for strict audio decoding; use --ffmpeg PATH')
    version = subprocess.run([str(ffmpeg), '-version'], capture_output=True, check=True, timeout=10).stdout.decode('utf-8', errors='replace').splitlines()[0]
    staging = output.with_name(output.name+'.pending-'+uuid.uuid4().hex[:8]); staging.mkdir(parents=True)
    (staging/'dynamic').mkdir(); (staging/'splits').mkdir()
    averaged = raw/AVERAGED
    axes = {axis: dynamic_csv(averaged/'dynamic (per second annotations)'/(axis+'.csv')) for axis in AXES}
    static = static_csvs(averaged/'song_level')
    metadata = read_metadata(raw/'metadata.zip')
    exclusions = {r['song_id']: r for r in json.loads(Path(exclusions_path).read_text(encoding='utf-8'))}
    audio = {int(p.stem): p for p in (raw/'audio').glob('*.mp3')}
    features = {int(p.stem): p for p in (raw/'features').glob('*.csv')}
    ids = set(audio)
    for label, table in [('valence',axes['valence']),('arousal',axes['arousal']),('static',static),('metadata',metadata),('features',features)]:
        if set(table) != ids: raise ValueError(f'{label} IDs mismatch: missing={sorted(ids-set(table))}, extra={sorted(set(table)-ids)}')
    if len(ids) != expected_songs: raise ValueError(f'Expected {expected_songs} songs, found {len(ids)}')
    print(f'Auditing {len(ids)} songs with {workers} workers: full audio decode + hashes + all feature rows', flush=True)
    def inspect(sid):
        result = audit_audio(audio[sid], ffmpeg, exclusions.get(sid))
        result['features'] = audit_features(features[sid], result['duration_seconds'])
        return result
    audited, errors = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(inspect, sid):sid for sid in sorted(ids)}
        for n, future in enumerate(as_completed(futures), 1):
            sid = futures[future]
            try: audited[sid] = future.result()
            except Exception as exc: errors.append(dict(song_id=sid, error=str(exc)))
            if n % 100 == 0 or n == len(ids): print(f'Checked {n}/{len(ids)}; errors={len(errors)}', flush=True)
    if errors:
        write_json(staging/'errors.json', errors)
        raise ValueError(f'Audit failed; see {staging}/errors.json. No prepared release published.')
    excluded = [dict(song_id=sid, **audited[sid]) for sid in sorted(ids) if not audited[sid]['audio_decode_ok']]
    write_json(staging/'excluded.json', excluded)
    usable_ids = [sid for sid in sorted(ids) if audited[sid]['audio_decode_ok']]
    if not usable_ids: raise ValueError('No recordings passed strict decoding')
    mismatches, outside_times, records = [], [], []
    for sid in usable_ids:
        info = audited[sid]
        v, a = axes['valence'][sid], axes['arousal'][sid]
        times, targets, mask, outside = align_labels(v, a, info['duration_seconds'])
        if len(v[0]) != len(a[0]): mismatches.append(dict(song_id=sid,valence=len(v[0]),arousal=len(a[0])))
        if outside: outside_times.append(dict(song_id=sid,masked_values=outside))
        if not np.any(mask): raise ValueError(f'No trainable dynamic labels: {sid}')
        np.savez_compressed(staging/'dynamic'/f'{sid}.npz', times_seconds=times, targets=targets, mask=mask)
        records.append(dict(dataset='DEAM',song_id=sid,**metadata[sid],**info,
                            audio_path=audio[sid].relative_to(root).as_posix(),
                            labels_path=(output/'dynamic'/f'{sid}.npz').relative_to(root).as_posix(),
                            provided_features_path=features[sid].relative_to(root).as_posix(),
                            label_order=AXES, label_count=len(times), valid_values=mask.sum(axis=0).tolist(),
                            first_label_seconds=float(times[0]),last_label_seconds=float(times[-1]),
                            static_labels=static[sid]))
    counts, group_count = split_by_artist(records, seed)
    with (staging/'manifest.jsonl').open('w', encoding='utf-8') as stream:
        for record in records: stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    for split in counts:
        (staging/'splits'/f'{split}.txt').write_text(''.join(f"{r['song_id']}\n" for r in records if r['split']==split),encoding='utf-8')
    # Statistics may be used for diagnostics; there is no all-data input scaler.
    statistics = {}
    for split in counts:
        totals = np.zeros(2); squares = np.zeros(2); valid_counts = np.zeros(2,dtype=int)
        for r in records:
            if r['split'] != split: continue
            with np.load(staging/'dynamic'/f"{r['song_id']}.npz",allow_pickle=False) as z:
                valid_counts += z['mask'].sum(axis=0); totals += (z['targets']*z['mask']).sum(axis=0,dtype=np.float64)
                squares += (z['targets'].astype(np.float64)**2*z['mask']).sum(axis=0)
        means = totals/valid_counts
        statistics[split] = dict(songs=counts[split],valid_values=valid_counts.tolist(),mean=means.tolist(),std=np.sqrt(np.maximum(0,squares/valid_counts-means**2)).tolist())
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),source_songs=len(ids),songs=len(records),status='ready_for_audio_dynamic_va_training',
                  audio_full_decode=True,strict_audio_decoder=version,excluded_song_ids=[r['song_id'] for r in excluded],
                  feature_files_fully_parsed=True,models_modified=False,
                  split_policy='local artist groups + exact audio hash unions; NOT official DEAM benchmark split',seed=seed,
                  split_counts=counts,artist_groups=group_count,metadata_year_counts=dict(Counter(r['metadata_year'] for r in records)),
                  axis_length_mismatches=mismatches,outside_audio_masked=outside_times,
                  provided_features_training_enabled=False,
                  features_beyond_audio=[sid for sid in sorted(ids) if audited[sid]['features']['extends_beyond_audio']],
                  features_non_half_second_steps=[sid for sid in sorted(ids) if audited[sid]['features']['non_half_second_steps']],
                  feature_header_variants=len({r['features']['header_sha256'] for r in audited.values()}),
                  audio_header_frame_mismatches=[dict(song_id=r['song_id'],reported=r['reported_header_frames'],decoded=r['decoded_frames'])
                                                for r in records if r['reported_header_frames'] != r['decoded_frames']],
                  audio_hours=sum(r['duration_seconds'] for r in records)/3600,
                  exact_audio_duplicate_groups=[group for group in duplicate_ids(records) if len(group)>1],
                  limitations=['Supplied openSMILE features are retained but excluded from training pending temporal-origin verification.',
                               'Artist grouping uses normalized supplied names; aliases, covers and near-duplicate recordings are not fingerprint-verified.',
                               'No embeddings, fitted feature scaler, trained weights or measured prediction metrics are produced.'],
                  statistics=statistics,elapsed_seconds=round(time.monotonic()-started,2))
    write_json(staging/'report.json',report)
    provenance = []
    for path in sorted((raw/'annotations').rglob('*')):
        if path.is_file(): provenance.append(dict(path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=digest(path)))
    provenance.append(dict(path='raw/metadata.zip',sha256=digest(raw/'metadata.zip'),url='https://cvml.unige.ch/databases/DEAM/metadata.zip'))
    write_json(staging/'applied-exclusions.json',list(exclusions.values()))
    write_json(staging/'annotation-provenance.json',provenance)
    write_json(staging/'config.json',dict(schema_version=1,dataset_root=Path(os.path.relpath(root,output)).as_posix(),label_order=AXES,sample_hz=2,
                 target_scale=[-1,1],missing_target_storage=0,missing_target_rule='mask=false; exclude from loss and metrics',
                 static_label_conversion='(mean_1_9 - 5) / 4; static supervision only',provided_features_training_enabled=False,
                 intended_model='next-version VA only; do not overwrite models/emotion_model.pth'))
    staging.rename(output)
    print(json.dumps({k:report[k] for k in ('songs','split_counts','artist_groups','audio_hours','elapsed_seconds')},ensure_ascii=False),flush=True)
    print(f'Prepared: {output}',flush=True)


def duplicate_ids(records):
    groups = defaultdict(list)
    for r in records: groups[r['audio_sha256']].append(r['song_id'])
    return list(groups.values())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=WORKSPACE/'data/datasets/deam')
    parser.add_argument('--output',type=Path,help='New directory under ROOT/prepared; defaults to ROOT/prepared/v1')
    parser.add_argument('--workers',type=int,choices=range(1,9),default=4)
    parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--ffmpeg',type=Path,help='FFmpeg executable; defaults to bundled runtime or PATH')
    parser.add_argument('--exclusions',type=Path,default=DEFAULT_EXCLUSIONS,help='Reviewed source exclusions with matching SHA256')
    args=parser.parse_args();prepare(args.root,args.output or args.root/'prepared/v1',args.workers,args.seed,ffmpeg=args.ffmpeg,exclusions_path=args.exclusions)
