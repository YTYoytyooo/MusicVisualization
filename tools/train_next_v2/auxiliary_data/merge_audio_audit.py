"""Decode MERGE clips and screen recording overlap before any auxiliary fitting."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
import zipfile

import numpy as np
import soundfile as sf

from tools.datasets.index_supplemental import FPCALC, MatchIndex
from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, process_identity, read_json, record_event, verify_hashes
from .merge_candidates import connected_groups

JOB = 'd-merge-audio-audit-v1'
FOLDER = BASE / 'analyses' / JOB
RAW = ROOT / 'data/datasets/merge/raw/v1.1.0'
OUTPUT = ROOT / 'data/datasets/merge/prepared/audio-audit-v1'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def safe_audio_id(name):
    parts = name.split('/')
    if len(parts) != 3 or parts[0] != 'MERGE_Audio_Complete' or parts[1] not in ('Q1','Q2','Q3','Q4'):
        raise ValueError('Unexpected audio member path')
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.mp3', parts[2]):
        raise ValueError('Unsafe audio member name')
    return Path(parts[2]).stem


def decode(path):
    frames, squared, peak = 0, 0., 0.
    with sf.SoundFile(path) as stream:
        rate, channels, declared_frames = stream.samplerate, stream.channels, stream.frames
        if not 8000 <= rate <= 192000 or not 1 <= channels <= 8:
            raise ValueError('Unexpected native sample rate/channel count')
        for block in stream.blocks(blocksize=65536, dtype='float32', always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError('Non-finite decoded samples')
            frames += len(block)
            squared += float(np.square(block.astype(np.float64)).sum())
            peak = max(peak, float(np.abs(block).max(initial=0)))
    if frames != declared_frames or not 1 <= frames / rate <= 600:
        raise ValueError('Incomplete decode or unexpected clip duration')
    rms = (squared / max(1, frames * channels)) ** .5
    if rms <= 1e-7:
        raise ValueError('Silent clip')
    return dict(frames=frames, sample_rate=rate, channels=channels, duration_seconds=frames/rate,
                rms=rms, peak=peak, decoded_all_samples=True, local_time_origin_seconds=0,
                source_song_offset_seconds=None)


def fingerprint(path, audio_sha, cache):
    saved = cache / (audio_sha + '.json')
    if saved.exists():
        row = read_json(saved)
    else:
        proc = subprocess.run([str(FPCALC), '-raw', '-json', '-length', '0', str(path)],
            capture_output=True, text=True, timeout=45, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if proc.returncode:
            raise ValueError('Fingerprint failed: ' + proc.stderr[:400])
        row = json.loads(proc.stdout)
        atomic_json(saved, row)
    values = row.get('fingerprint')
    if not isinstance(values, list) or len(values) < 80 or not all(isinstance(x,int) and 0 <= x < 2**32 for x in values):
        raise ValueError('Missing/too-short/invalid raw Chromaprint fingerprint')
    return dict(path=relative(saved), sha256=sha256(saved), frames=len(values)), np.asarray(values,dtype=np.uint32)


def register():
    if FOLDER.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve previous audio audit')
    downloaded = read_json(BASE/'analyses/d-merge-audio-download-v1/report.json')
    if not downloaded['official_md5_verified']:
        raise ValueError('Whole archive checksum required')
    source = read_json(BASE/'analyses/d-merge-candidates-v1/launch.json')
    hashes = dict(source['protected_hashes'])
    paths = [Path(__file__), FPCALC, ROOT/'tools/datasets/index_supplemental.py',
             ROOT/'data/datasets/merge/prepared/auxiliary-candidates-v1/training-screen.json',
             ROOT/'data/datasets/merge/prepared/auxiliary-candidates-v1/label-contract.json',
             BASE/'analyses/d-merge-audio-download-v1/report.json']
    hashes.update({relative(path):sha256(path) for path in paths})
    hashes[downloaded['archive']['path']] = downloaded['archive']['sha256']
    verify_hashes(hashes)
    argv = [str(Path(sys.executable).resolve()), '-u', '-m', 'tools.train_next_v2.auxiliary_data.merge_audio_audit', '--run']
    launch = dict(job_id=JOB, stage='D', registered_at=now(), original_argv=argv, recovery_argv=argv+['--resume'],
        cwd=str(ROOT), python=argv[0], environment=source['environment'], protected_hashes=hashes,
        source_manifest=source['source_manifest'], archive=downloaded['archive']['path'],
        candidate_screen='data/datasets/merge/prepared/auxiliary-candidates-v1/training-screen.json',
        fingerprint_cache=relative(ROOT/'data/datasets/.fingerprints'/sha256(FPCALC)[:16]),
        os_lock=str(lock_path(JOB)), log=relative(FOLDER/'console.log'), status_path=relative(FOLDER/'status.json'),
        report=relative(FOLDER/'report.json'), worker_count=4, gpu_used=False, targets_used=False,
        recovery='No automatic retry. On interruption verify inactive OS lease and unchanged inputs/code, then --resume. Preserve per-clip checks; a final report cannot be rerun.',
        scope='Full archive audio integrity; main/auxiliary reserved audio only for identity exclusion, never fitting or metrics.')
    FOLDER.mkdir(parents=True)
    atomic_json(FOLDER/'launch.json', launch)
    stages = read_json(PLAN)['stages']
    stages['D']['audio_audit'] = dict(job_id=JOB, status='registered', launch=relative(FOLDER/'launch.json'))
    record_event(dict(event='D_MERGE_audio_audit_registered', launch=relative(FOLDER/'launch.json')), updates=dict(stages=stages))


def run(resume):
    launch = read_json(FOLDER/'launch.json')
    verify_hashes(launch['protected_hashes'])
    if (FOLDER/'report.json').exists():
        raise FileExistsError('Final audio audit already exists; do not restart')
    if (FOLDER/'process.json').exists() and not resume:
        raise ValueError('Prior audit requires explicit --resume')
    identity = process_identity()
    atomic_json(FOLDER/'process.json', dict(started_at=now(), process_identity=identity,
        actual_argv=launch['recovery_argv'] if resume else launch['original_argv']))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audio_root, records = RAW/'audio', OUTPUT/'records'
    audio_root.mkdir(exist_ok=True); records.mkdir(exist_ok=True)
    cache = ROOT / launch['fingerprint_cache']; cache.mkdir(parents=True, exist_ok=True)
    with (FOLDER/'console.log').open('a', encoding='utf-8') as log:
        def publish(value):
            value = dict(**value, updated_at=now(), process_identity=identity)
            atomic_json(FOLDER/'status.json', value)
            text = json.dumps(value); log.write(text+'\n'); log.flush(); print(text,flush=True)
        try:
            audio_rows = []
            with zipfile.ZipFile(ROOT/launch['archive']) as archive:
                audio_members = [info for info in archive.infolist() if info.filename.lower().endswith('.mp3')]
                ids = [safe_audio_id(info.filename) for info in audio_members]
                if len(ids) != 3554 or len(set(ids)) != len(ids):
                    raise ValueError('Audio IDs are not unique/complete')
                for i,info in enumerate(audio_members):
                    sid = safe_audio_id(info.filename)
                    if info.file_size > 20*1024*1024 or info.flag_bits & 1:
                        raise ValueError('Oversized/encrypted audio member')
                    data = archive.read(info)  # zipfile verifies each member CRC.
                    digest = hashlib.sha256(data).hexdigest()
                    path = audio_root / (sid+'.mp3')
                    if path.exists():
                        if sha256(path) != digest:
                            raise ValueError('Existing extracted audio changed; preserved')
                    else:
                        partial = path.with_suffix('.mp3.part')
                        if partial.exists():
                            if sha256(partial) != digest:
                                raise ValueError('Interrupted partial differs; preserve and diagnose')
                        else:
                            with partial.open('xb') as stream:
                                stream.write(data); stream.flush(); os.fsync(stream.fileno())
                        partial.rename(path)
                    audio_rows.append(dict(song_id='merge:'+sid, release_id=sid, audio_path=relative(path),
                        audio_sha256=digest, archive_member=info.filename, archive_member_crc32=f'{info.CRC:08x}'))
                    if (i+1) % 250 == 0 or i+1 == len(ids):
                        publish(dict(stage='extract_crc_sha_verified', completed=i+1, total=len(ids)))
            def inspect(row):
                path, saved = ROOT/row['audio_path'], records/(row['release_id']+'.json')
                if saved.exists():
                    previous = read_json(saved)
                    if previous['audio_sha256'] != row['audio_sha256']:
                        raise ValueError('Saved audio audit belongs to another audio version')
                    if previous.get('fingerprint'):
                        verify_hashes({previous['fingerprint']['path']: previous['fingerprint']['sha256']})
                    return previous
                try:
                    native = decode(path)
                    fp,_ = fingerprint(path,row['audio_sha256'],cache)
                    result = dict(**row,status='verified_audio_and_fingerprint',native_audio=native,fingerprint=fp)
                except Exception as exc:
                    result = dict(**row,status='quarantined',error=f'{type(exc).__name__}: {exc}')
                atomic_json(saved,result)
                return result
            inspected = []
            with ThreadPoolExecutor(max_workers=launch['worker_count']) as pool:
                for row in pool.map(inspect,audio_rows):
                    inspected.append(row)
                    if len(inspected) % 100 == 0 or len(inspected) == len(audio_rows):
                        publish(dict(stage='decode_and_fingerprint',completed=len(inspected),total=len(audio_rows)))
            with (ROOT/launch['source_manifest']).open(encoding='utf-8') as stream:
                main = [json.loads(line) for line in stream]
            def main_fingerprint(row):
                path = ROOT/row['audio_path']
                if sha256(path) != row['audio_sha256']:
                    raise ValueError('Main corpus audio changed: '+row['song_id'])
                receipt, values = fingerprint(path,row['audio_sha256'],cache)
                return row,receipt,values
            index, index_ids, main_receipts = MatchIndex(),[],[]
            graph = {}
            with ThreadPoolExecutor(max_workers=launch['worker_count']) as pool:
                for row,receipt,values in pool.map(main_fingerprint,main):
                    index.add(values); index_ids.append(row['song_id'])
                    graph[row['song_id']] = dict(identity_keys=[('audio_sha256',row['audio_sha256']),('fp_link',row['song_id'])])
                    main_receipts.append(dict(song_id=row['song_id'],audio_sha256=row['audio_sha256'],fingerprint=receipt))
                    if len(index_ids) % 500 == 0 or len(index_ids) == len(main):
                        publish(dict(stage='protected_main_audio_identity_index',completed=len(index_ids),total=len(main)))
            links = []
            for i,row in enumerate(sorted(inspected,key=lambda r:r['song_id'])):
                keys = [('audio_sha256',row['audio_sha256']),('fp_link',row['song_id'])]
                if row['status'] == 'verified_audio_and_fingerprint':
                    values = np.asarray(read_json(ROOT/row['fingerprint']['path'])['fingerprint'],dtype=np.uint32)
                    for matched,details in index.matches(values).items():
                        keys.append(('fp_link',index_ids[matched]))
                        links.append(dict(song_id=row['song_id'],matched_song_id=index_ids[matched],**details))
                    index.add(values); index_ids.append(row['song_id'])
                graph[row['song_id']] = dict(identity_keys=keys)
                if (i+1) % 100 == 0 or i+1 == len(inspected):
                    publish(dict(stage='offset_fingerprint_overlap',completed=i+1,total=len(inspected),links=len(links)))
            groups = connected_groups(graph)
            screened = read_json(ROOT/launch['candidate_screen'])
            candidates = {row['song_id']:row for row in screened if row['metadata_candidate']}
            verified = {row['song_id']:row for row in inspected if row['status'] == 'verified_audio_and_fingerprint'}
            accepted, rejected = [],[]
            for sid,row in sorted(candidates.items()):
                group = groups[sid]
                blocked = [other for other in group if other not in candidates or other not in verified]
                reason = 'audio_integrity_or_fingerprint_failed' if sid not in verified else 'recording_link_to_main_or_reserved_auxiliary' if blocked else 'duplicate_candidate_recording' if sid != min(group) else None
                if reason:
                    rejected.append(dict(song_id=sid,reason=reason,recording_group=group))
                    continue
                accepted.append(dict(**row,audio=verified[sid],recording_group=group,
                                     audio_gate_passed=True,auxiliary_task='clip_quadrant',
                                     fitting_enabled=False))
            # No labels from main/aux held-out songs are used by this job.
            atomic_json(OUTPUT/'audio-inventory.json',inspected)
            atomic_json(OUTPUT/'main-fingerprint-receipts.json',main_receipts)
            atomic_json(OUTPUT/'recording-links.json',links)
            atomic_json(OUTPUT/'eligible-quadrant-candidates.json',accepted)
            atomic_json(OUTPUT/'recording-exclusions.json',rejected)
            durations = [row['native_audio']['duration_seconds'] for row in verified.values()]
            report = dict(job_id=JOB,status='completed_audio_screen_protocol_pending',completed_at=now(),
                process_identity=identity,archive_sha256=launch['protected_hashes'][launch['archive']],
                decoded_verified=len(verified),audio_total=len(inspected),quarantined=len(inspected)-len(verified),
                native_duration_seconds=dict(min=min(durations),max=max(durations),median=float(np.median(durations))),
                main_identity_records=len(main),fingerprint_links=len(links),metadata_candidates=len(candidates),
                audio_screened_candidates=len(accepted),additional_recording_exclusions=len(rejected),
                exclusion_reasons=dict(Counter(row['reason'] for row in rejected)),
                candidate_quadrants=dict(Counter(row['quadrant'] for row in accepted)),
                fitting_enabled=False,numeric_regression_enabled=False,original_heldout_targets_used=False,
                source_full_song_offsets_known=False,local_clip_time_origin_seconds=0,
                limitation='Metadata plus offset-aware Chromaprint is conservative screening, not proof against every cover, remix, alias or very short overlap. Whole-clip static targets need no fabricated full-song offset.',
                remaining=['Freeze equal-step quadrant-auxiliary/control experiment protocol','Prepare eligible clips feature cache with separate static-task masks'],
                source_hashes=launch['protected_hashes'],outputs={relative(p):sha256(p) for p in OUTPUT.glob('*.json')})
            atomic_json(FOLDER/'report.json',report)
            stages = read_json(PLAN)['stages']
            stages['D'].update(phase='MERGE_audio_screen_complete_auxiliary_protocol_pending')
            stages['D']['audio_audit'].update(status=report['status'],report=relative(FOLDER/'report.json'),
                                             screened_candidates=len(accepted))
            record_event(dict(event='D_MERGE_audio_screen_complete',candidates=len(accepted),report=relative(FOLDER/'report.json')),
                         updates=dict(stages=stages))
            publish(dict(stage=report['status'],audio_screened_candidates=len(accepted),excluded=len(rejected)))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            log.write(traceback.format_exc()+'\n');log.flush()
            publish(dict(stage='failed',error=error))
            atomic_json(FOLDER/('failure-'+now().replace(':','-')+'.json'),dict(error=error,process_identity=identity,time=now()))
            record_event(dict(event='D_MERGE_audio_audit_failed',error=error))
            raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    with run_lease(JOB):
        register() if args.register else run(args.resume)
