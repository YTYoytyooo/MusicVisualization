"""Nine acoustic descriptors in c-feature-pilot-v1; never emotion labels."""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    'log_harmonic_spectral_energy_2s', 'log_percussive_spectral_energy_2s',
    'percussive_spectral_energy_fraction_2s', 'onset_acf_peak_60_120_bpm_8s',
    'onset_acf_peak_120_240_bpm_8s', 'dominant_onset_pulse_bpm_30_240_8s',
    'chroma_entropy_2s', 'chroma_cosine_change_prev_next_1s', 'chroma_change_valid',
]

IMPLEMENTATION = dict(sample_rate=24000, n_fft=2048, hop_length=240, stft_center=True,
    stft_pad_mode='constant', hpss_kernel_size=31, hpss_power=2, hpss_margin=1,
    chroma_stft_norm='inf (librosa default)', chroma_stft_tuning=None,
    chroma_window_normalization='L1 after averaging source frames',
    time_windows='Half-open intervals; exclude frame centers >= original decoded source duration.',
    onset='Positive magnitude spectral differences summed across frequency; initial flux zero.',
    autocorrelation='Demean the actual 8s source window; normalize each lag by both lagged energies; clip floating-point roundoff to [-1,1].',
    pulse_tie_break='Smallest lag (highest BPM) in the declared range.',
    feature_names=FEATURE_NAMES)


def pulse_features(onset_window, frame_rate=100.):
    x = np.asarray(onset_window, dtype=np.float64)
    if len(x) < 2:
        return np.zeros(3)
    x = x - x.mean()
    if np.dot(x, x) == 0:
        return np.zeros(3)
    minimum_lag = int(np.ceil(60 * frame_rate / 240))
    maximum_lag = min(len(x)-1, int(np.floor(60 * frame_rate / 30)))
    if maximum_lag < minimum_lag:
        return np.zeros(3)
    lags = np.arange(minimum_lag, maximum_lag+1)
    bpm = 60 * frame_rate / lags
    scores = []
    for lag in lags:
        left, right = x[:-lag], x[lag:]
        denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
        scores.append(float(np.dot(left, right) / denominator) if denominator > 0 else 0.)
    scores = np.clip(scores, -1., 1.)
    peaks = []
    for low, high in ((60, 120), (120, 240)):
        selected = (bpm >= low) & (bpm <= high)
        peaks.append(float(np.max(scores[selected])) if selected.any() else 0.)
    return np.array([*peaks, float(bpm[int(np.argmax(scores))])])


def pool_acoustics(times, frame_times, harmonic_energy, percussive_energy, chroma, onset):
    clock = np.asarray(frame_times, dtype=np.float64)
    harmonic_energy = np.asarray(harmonic_energy, dtype=np.float64)
    percussive_energy = np.asarray(percussive_energy, dtype=np.float64)
    chroma = np.asarray(chroma, dtype=np.float64)
    onset = np.asarray(onset, dtype=np.float64)
    if not len(clock) or chroma.shape != (12, len(clock)) or not np.all(np.diff(clock)>0):
        raise ValueError('Invalid acoustic frame clock')
    if any(len(x) != len(clock) for x in (harmonic_energy, percussive_energy, onset)):
        raise ValueError('Acoustic arrays have different physical clocks')
    if any(not np.isfinite(x).all() for x in (harmonic_energy, percussive_energy, chroma, onset)):
        raise ValueError('Nonfinite acoustic source')
    def region(left, right):
        return slice(np.searchsorted(clock, left, side='left'), np.searchsorted(clock, right, side='left'))
    def valid_chroma(section):
        h = harmonic_energy[section]
        c = chroma[:, section]
        if not len(h) or np.sum(h) <= 0:
            return None
        vector = c.mean(1)
        return vector if np.linalg.norm(vector)>0 else None
    result = []
    for time in times:
        section = region(time-1, time+1)
        h, p = harmonic_energy[section], percussive_energy[section]
        if not len(h):
            raise ValueError('No physical source frames for acoustic pooling')
        log_h, log_p = np.log(max(h.mean(), 1e-10)), np.log(max(p.mean(), 1e-10))
        fraction = p.sum() / max(h.sum()+p.sum(), 1e-10)
        pulse = pulse_features(onset[region(time-4, time+4)])
        chroma_mean = chroma[:, section].mean(1)
        probability = chroma_mean / max(chroma_mean.sum(), 1e-10)
        positive = probability > 0
        entropy = -np.sum(probability[positive] * np.log(probability[positive])) / np.log(12)
        previous, following = valid_chroma(region(time-1, time)), valid_chroma(region(time, time+1))
        valid = previous is not None and following is not None
        change = (1 - np.clip(np.dot(previous, following) / (np.linalg.norm(previous)*np.linalg.norm(following)), -1, 1)) if valid else 0.
        result.append([log_h, log_p, fraction, *pulse, entropy, change, float(valid)])
    values = np.asarray(result, dtype=np.float32)
    if values.shape != (len(times), 9) or not np.isfinite(values).all():
        raise ValueError('Invalid rhythm/harmony output')
    return values


def extract_rhythm_harmony(audio24, source_duration, times):
    import librosa
    magnitude = np.abs(librosa.stft(audio24, n_fft=2048, hop_length=240,
                                   center=True, pad_mode='constant'))
    harmonic, percussive = librosa.decompose.hpss(magnitude, kernel_size=31, power=2, margin=1)
    harmonic_power = harmonic ** 2
    harmonic_energy = np.mean(harmonic_power, axis=0)
    percussive_energy = np.mean(percussive ** 2, axis=0)
    chroma = librosa.feature.chroma_stft(S=harmonic_power, sr=24000, n_fft=2048, hop_length=240)
    onset = np.concatenate(([0.], np.maximum(np.diff(magnitude, axis=1), 0).sum(0)))
    clock = np.arange(magnitude.shape[1], dtype=np.float64) * .01
    valid = clock < source_duration
    values = pool_acoustics(times, clock[valid], harmonic_energy[valid], percussive_energy[valid], chroma[:, valid], onset[valid])
    metadata = dict(source_duration_seconds=source_duration, resampled_samples=len(audio24),
        stft_frames_total=len(clock), valid_source_frame_count=int(valid.sum()),
        first_source_frame_seconds=float(clock[valid][0]), last_source_frame_seconds=float(clock[valid][-1]),
        source_frame_step_seconds=.01, implementation=IMPLEMENTATION)
    return values, metadata
