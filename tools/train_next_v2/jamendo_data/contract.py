"""Keep observed whole-track tags separate from continuous VA supervision."""
from collections import Counter
import math

import numpy as np


TAG_ORDER = tuple('mood/theme---'+name for name in (
    'calm', 'dark', 'emotional', 'energetic', 'happy', 'hopeful',
    'melancholic', 'positive', 'relaxing', 'romantic', 'sad', 'upbeat'))


def observed_tag_distribution(tags):
    """Distribution over recorded tags, NOT binary tag-presence ground truth.

    Zero mass means the tag was not observed in this annotation. The auxiliary
    softmax predicts an observed tag drawn from the track's recorded tag set;
    it does not estimate twelve calibrated emotion-presence probabilities.
    """
    if not tags or len(set(tags)) != len(tags) or not set(tags).issubset(TAG_ORDER):
        raise ValueError('Missing, duplicate or unknown observed tag')
    target = np.zeros(len(TAG_ORDER), dtype=np.float32)
    target[[TAG_ORDER.index(tag) for tag in tags]] = 1./len(tags)
    return target


def validate_candidates(rows):
    if not rows or len({r['song_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate auxiliary song IDs')
    digests = [r['audio']['audio_sha256'] for r in rows]
    if len(set(digests)) != len(digests):
        raise ValueError('Duplicate auxiliary recording')
    for row in rows:
        if (row['official_role'] != 'train' or not row['metadata_eligible']
                or not row['audio_gate_passed'] or row['auxiliary_task'] != 'whole_track_weak_category'
                or row['training_admitted'] or row['fitting_enabled']):
            raise ValueError('Wrong source role or admission state')
        if any(row[name] for name in ('heldout_identity_ids', 'possible_overlap_ids', 'exclusion_reasons')):
            raise ValueError('Unresolved metadata/heldout exclusion')
        if any(name in row for name in ('y', 'mask', 'timepoint_labels', 'continuous_va', 'pseudo_labels')):
            raise ValueError('Whole-track weak records cannot contain invented continuous targets')
        if row['recording_group'] != [row['song_id']]:
            raise ValueError('This pilot requires independently screened single recording groups')
        codes = row['audio_license']['codes']
        if not codes or not set(codes).issubset({'by', 'by-sa', 'by-nc', 'by-nc-sa'}):
            raise ValueError('Audio license outside the frozen research pilot')
        tags = row['selected_positive_tags']
        observed_tag_distribution(tags)
        if not set(tags).issubset(row['positive_tags']):
            raise ValueError('Selected tag is not an observed source annotation')
        audio, native = row['audio'], row['audio']['native_audio']
        if (not audio['passed'] or audio['audio_sha256'] != row['expected_audio_sha256']
                or audio['song_id'] != row['song_id'] or audio['track_id'] != row['track_id']):
            raise ValueError('Native receipt/audio identity mismatch')
        if (not native['bulk_stream_sample_parity'] or not native['native_stderr_empty']
                or native['strict_ffmpeg_returncode'] != 0 or native['timeline_stretched']
                or native['local_time_origin_seconds'] != 0 or native['source_song_offset_seconds'] != 0
                or row['source_song_offset_seconds'] != 0):
            raise ValueError('Native decoding/time-origin gate failed')
        duration = native['duration_seconds']
        if (not math.isfinite(duration) or duration <= 0 or native['frames'] <= 0
                or not 8000 <= native['sample_rate'] <= 192000 or not 1 <= native['channels'] <= 8
                or abs(duration-native['frames']/native['sample_rate']) > 1e-9
                or abs(duration-row['metadata_duration_seconds']) > .25):
            raise ValueError('Native sample clock or full-track duration mismatch')
    return dict(songs=len(rows), groups=len({r['group_id'] for r in rows}),
        tag_counts=dict(Counter(t for r in rows for t in r['selected_positive_tags'])),
        hours=sum(r['audio']['native_audio']['duration_seconds'] for r in rows)/3600)


def feature_record(row):
    validate_candidates([row])
    audio, native = row['audio'], row['audio']['native_audio']
    return dict(song_id=row['song_id'], audio_path=audio['audio_path'], audio_sha256=audio['audio_sha256'],
        duration_seconds=native['duration_seconds'], decoded_frames=native['frames'],
        sample_rate=native['sample_rate'], channels=native['channels'])


def validate_feature_arrays(times, values, record):
    expected = np.arange(int(np.floor(record['duration_seconds']/.5))+1, dtype=np.float64)*.5
    if times.dtype != np.float64 or not np.array_equal(times, expected):
        raise ValueError('Whole-track native .5second clock changed')
    if values.dtype != np.float32 or values.shape != (len(expected), 531) or not np.isfinite(values).all():
        raise ValueError('Baseline feature dimensions or finite values changed')
