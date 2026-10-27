"""Exact length contract of the pinned librosa 0.11 resampler."""
import math


def resampled_length(source_samples, source_rate, target_rate):
    if source_samples < 1 or source_rate <= 0 or target_rate <= 0:
        raise ValueError('Invalid physical sample clock')
    # librosa.core.audio.resample computes ratio first, then multiplies. The
    # order is observable at integer boundaries due to IEEE-754 roundoff.
    # Keep this exact check rather than accepting an arbitrary +/-1 tolerance.
    return math.ceil(source_samples * (float(target_rate) / source_rate))


def sample_clock_metadata(source_samples, source_rate, target_rate, actual_length):
    expected = resampled_length(source_samples, source_rate, target_rate)
    if actual_length != expected:
        raise ValueError('Resampled length differs from the pinned implementation')
    rational = (source_samples * target_rate + source_rate - 1) // source_rate
    return dict(source_samples=source_samples, source_rate=source_rate, target_rate=target_rate,
        implementation_expected_samples=expected, rational_ceil_samples=rational,
        implementation_rounding_samples=expected-rational,
        source_duration_seconds=source_samples/source_rate,
        resampled_duration_seconds=actual_length/target_rate,
        timestamp_mapping='Resampled sample i is at i/target_rate; target timestamps are unchanged.')
