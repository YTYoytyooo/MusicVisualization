"""Keep full-track native samples unchanged and prove bulk/stream parity."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

from tools.train_next_v2.common import ROOT

FFMPEG = ROOT/'versions/v2.0/.runtime/bin/ffmpeg.exe'


def decode_exact(path, metadata_duration, duration_tolerance=0.25):
    samples, rate = sf.read(path, dtype='float32', always_2d=True)
    if not 8000 <= rate <= 192000 or not 1 <= samples.shape[1] <= 8:
        raise ValueError('Unexpected native sample rate or channels')
    if not np.isfinite(samples).all() or not len(samples):
        raise ValueError('Empty or non-finite native samples')
    bulk_digest = hashlib.sha256(samples.tobytes(order='C')).hexdigest()
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    peak = float(np.max(np.abs(samples)))
    if rms <= 1e-7:
        raise ValueError('Silent track')
    native_duration = len(samples)/rate
    if abs(native_duration-float(metadata_duration)) > duration_tolerance:
        raise ValueError(f'Native duration differs from metadata: {native_duration:.6f} vs {metadata_duration}')
    frames = 0
    streamed = hashlib.sha256()
    with sf.SoundFile(path) as stream:
        if stream.samplerate != rate or stream.channels != samples.shape[1] or stream.frames != len(samples):
            raise ValueError('Bulk/native header frame or channel mismatch')
        for block in stream.blocks(blocksize=65536, dtype='float32', always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError('Non-finite streamed samples')
            streamed.update(block.tobytes(order='C'))
            frames += len(block)
    if frames != len(samples) or streamed.hexdigest() != bulk_digest:
        raise ValueError('Bulk and streamed native samples differ; no repair or rescaling')
    return dict(frames=frames, channels=samples.shape[1], sample_rate=rate, duration_seconds=native_duration,
        metadata_duration_seconds=float(metadata_duration), duration_error_seconds=native_duration-float(metadata_duration),
        float32_sample_sha256=bulk_digest, bulk_stream_sample_parity=True, rms=rms, peak=peak,
        local_time_origin_seconds=0, source_song_offset_seconds=0, timeline_stretched=False,
        timing_scope='Official full-track release with low-bitrate mono encoding; local native samples retained. No continuous targets exist for these tracks.')


def inspect(path, metadata_duration, log_directory):
    """Run sequentially: fd2 capture is process-wide, never use threads here."""
    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    command = [str(FFMPEG), '-nostdin', '-hide_banner', '-v', 'error', '-xerror',
        '-err_detect', 'explode', '-threads', '1', '-i', str(path), '-map', '0:a:0', '-f', 'null', '-']
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (log_directory/'ffmpeg.log').write_text(completed.stderr, encoding='utf-8')
    if completed.returncode or completed.stderr.strip():
        raise ValueError(f'Strict ffmpeg decoder rejected track: code={completed.returncode}; {completed.stderr[-500:]}')
    warning_path = log_directory/'soundfile.log'
    original = os.dup(2)
    try:
        with warning_path.open('wb') as warnings:
            sys.stderr.flush()
            os.dup2(warnings.fileno(), 2)
            try:
                details = decode_exact(path, metadata_duration)
            finally:
                sys.stderr.flush()
    finally:
        os.dup2(original, 2)
        os.close(original)
    warning = warning_path.read_text(encoding='utf-8', errors='replace')
    if warning.strip():
        raise ValueError('Native decoder emitted diagnostics: '+warning[-500:])
    return dict(**details, strict_ffmpeg_returncode=completed.returncode, native_stderr_empty=True)
