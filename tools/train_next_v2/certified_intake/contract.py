"""Bind raw annotation takes to certified native EOF, without header stretching."""
import copy
import hashlib
import math

import numpy as np
import soundfile as sf

from tools.train_next.core import sha256
from tools.train_next_v2.jamendo_audit_v2.native import read_unseeked_block

VERSION = 'native-eof-raw-takes-v1'
AXES = ('valence', 'arousal')


def validate_certificate(certificate):
    c = certificate
    if not c['bulk_stream_exact'] or not c['ffmpeg_stderr_empty'] or c['ffmpeg_returncode'] != 0:
        raise ValueError('Missing successful independent decoder evidence')
    if c['actual_frames'] <= 0 or c['actual_sample_rate'] <= 0 or c['actual_channels'] <= 0:
        raise ValueError('Invalid native geometry')
    if not c['actual_frames'] == c['stream_frames'] == c['ffmpeg_frames']:
        raise ValueError('Certified frame counts disagree')
    if c['bulk_float32_sha256'] != c['stream_float32_sha256']:
        raise ValueError('Certified native PCM hashes disagree')


def verify_native(path, certificate):
    """Read to actual EOF, requiring the original bytes and certified float32 PCM."""
    validate_certificate(certificate)
    c = certificate
    if sha256(path) != c['audio_sha256']:
        raise ValueError('Audio source bytes changed')
    digest = hashlib.sha256()
    frames = 0
    with sf.SoundFile(path) as stream:
        if (stream.samplerate, stream.channels) != (c['actual_sample_rate'], c['actual_channels']):
            raise ValueError('Native rate/channels changed')
        header_frames = stream.frames
        while True:
            block = read_unseeked_block(stream, 65536)
            if not len(block):
                break
            if not np.isfinite(block).all():
                raise ValueError('Non-finite native PCM')
            frames += len(block)
            if frames > c['actual_frames']:
                raise ValueError('Unexpected extra PCM frames')
            digest.update(block.tobytes(order='C'))
    if frames != c['actual_frames'] or digest.hexdigest() != c['bulk_float32_sha256']:
        raise ValueError('Native EOF frame count or PCM hash mismatch')
    if sha256(path) != c['audio_sha256']:
        raise ValueError('Audio changed while decoding')
    return dict(frames=frames, sample_rate=c['actual_sample_rate'], channels=c['actual_channels'],
                float32_sha256=digest.hexdigest(), header_frames=header_frames,
                header_matches_eof=header_frames == frames,
                duration_seconds=frames / c['actual_sample_rate'])


def adapt_raw_takes(song, certificate, max_rate_gap_seconds):
    """Expose original rows and per-axis masks; rate edges never cross takes/gaps.

    The explicit gap limit is an eligibility bound, not a new sampling clock.
    No overlap resolution, time interpolation, delay correction or label filling.
    """
    validate_certificate(certificate)
    if not math.isfinite(max_rate_gap_seconds) or max_rate_gap_seconds <= 0:
        raise ValueError('A positive rate-gap limit must be recorded')
    if song['source']['song_id'] != certificate['annotation_song_id'] or song['source']['audio_sha256'] != certificate['audio_sha256']:
        raise ValueError('Annotation/audio identity mismatch')
    duration = certificate['actual_frames'] / certificate['actual_sample_rate']
    takes = []
    for take in song['takes']:
        times = [p['time'] for p in take['points']]
        if any(isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t) or t < 0 or t > duration for t in times):
            raise ValueError('Raw timestamp outside native EOF clock')
        if any(b < a for a, b in zip(times, times[1:])):
            raise ValueError('Raw timestamps run backwards within a take')
        dt = [b - a for a, b in zip(times, times[1:])]
        masks = {}
        edges = {}
        for axis in AXES:
            mask = []
            for point in take['points']:
                value = point.get(axis)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1):
                    raise ValueError('Invalid raw V/A value')
                mask.append(axis in take['axes'] and point.get('status') == 'annotated' and value is not None)
            masks[axis] = mask
            edges[axis] = [mask[i] and mask[i+1] and 0 < gap <= max_rate_gap_seconds for i, gap in enumerate(dt)]
        takes.append(dict(raw=copy.deepcopy(take), valid_masks=masks, delta_time_seconds=dt,
                          rate_edge_masks=edges))
    return dict(contract_version=VERSION, source=copy.deepcopy(song['source']),
                reviewer=song['reviewer'], declared_completed=copy.deepcopy(song['completed']),
                raw_song=copy.deepcopy(song), takes=takes, max_rate_gap_seconds=max_rate_gap_seconds,
                timeline_scaled=False, labels_shifted=False, gaps_imputed=False,
                overlap_resolution='not_applied_keep_takes_separate',
                browser_priming_verified=False, psychological_quality_verified=False,
                training_admitted=False, formal_evaluation_admitted=False)
