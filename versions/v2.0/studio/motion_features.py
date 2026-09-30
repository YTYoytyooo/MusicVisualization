"""Small, audio-derived motion controls; never reads or changes model predictions."""
import os
from contextlib import contextmanager
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import uuid
import zipfile
import zlib

import numpy as np
import soundfile as sf

from .store import atomic_json, digest, load_project, object_digest, project_lock, read_json


KIND = 'audio-rms-spectral-flux-v1'
STEP_US = 100000
FFT_SIZE = 2048
RMS_FLOOR = .08
FLUX_FLOOR = .04
FLUX_NOISE = .002
FIELDS = ('times_us', 'rms', 'onset', 'activity', 'pulse_strength')
_LOADED_SOURCE_HASH = digest(Path(__file__))


class MotionFeatureCancelled(Exception):
    """Cooperative cancellation, distinct from decoding/cache errors."""


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise MotionFeatureCancelled('音频运动特征提取已取消')


def _fingerprint():
    source = digest(Path(__file__))
    if source != _LOADED_SOURCE_HASH:
        raise ValueError('运动特征代码已变化，请重启服务后重试')
    return object_digest({
        'source': source, 'kind': KIND, 'step_us': STEP_US, 'fft_size': FFT_SIZE,
        'rms_floor': RMS_FLOOR, 'flux_floor': FLUX_FLOOR, 'flux_noise': FLUX_NOISE,
        'numpy': np.__version__, 'soundfile': sf.__version__,
        'libsndfile': sf.__libsndfile_version__,
    })


def _report(progress, done, total):
    if progress is not None:
        args = ('audio-motion-features', done / max(1, total))
        if getattr(progress, 'accepts_details', False) is True:
            progress(*args, {'done': done, 'total': total, 'unit': '音频片段'})
        else:
            progress(*args)


@contextmanager
def _motion_audio(audio, folder, cancel):
    """Decode MP3 once to temporary PCM; do not seek in inaccurate VBR streams.

    Chunked libsndfile reads may seek using estimated MP3 frame positions.
    FFmpeg decodes sequentially into a private scratch file, preserving bounded
    memory during subsequent 100 ms feature reads and leaving the source alone.
    """
    if sf.info(str(audio)).format != 'MP3':
        yield audio, None
        return
    from .pipeline import ffmpeg_path
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise ValueError('MP3 运动特征需要本地 FFmpeg 解码器')
    decoder = {'name': 'ffmpeg-pcm-f64le', 'sha256': digest(ffmpeg)}
    _check_cancel(cancel)
    with tempfile.TemporaryDirectory(prefix='.motion-audio-', dir=folder) as temporary:
        decoded = Path(temporary) / 'decoded.wav'
        log_path = Path(temporary) / 'decoder.log'
        with log_path.open('wb') as log:
            proc = subprocess.Popen([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                '-i', str(audio), '-map', '0:a:0', '-vn', '-c:a', 'pcm_f64le',
                '-rf64', 'auto', str(decoded)], stdout=subprocess.DEVNULL, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                while proc.poll() is None:
                    _check_cancel(cancel)
                    time.sleep(.05)
            except BaseException:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise
        _check_cancel(cancel)
        if proc.returncode != 0 or not decoded.is_file():
            with log_path.open('rb') as log:
                detail = log.read(4096).decode('utf-8', errors='replace')
            raise ValueError('MP3 解码失败：' + detail)
        yield decoded, decoder


def _extract(audio, count, rate, frames, progress, cancel):
    """Read at most one 100 ms block; FFT temporary storage is fixed per channel."""
    rms = np.empty(count, dtype=np.float64)
    flux = np.zeros(count, dtype=np.float64)
    window_size = min(FFT_SIZE, max(3, rate * STEP_US // 1000000))
    window = np.hanning(window_size)
    normalization = np.sqrt(FFT_SIZE * np.sum(window * window) / 2)
    previous = None
    tail = None
    with sf.SoundFile(str(audio)) as stream:
        if stream.samplerate != rate or len(stream) != frames:
            raise ValueError('音频在特征提取过程中变化')
        for index in range(count):
            _check_cancel(cancel)
            # Rational boundaries avoid accumulating drift at unusual sample rates.
            left = index * rate * STEP_US // 1000000
            right = min(frames, (index + 1) * rate * STEP_US // 1000000)
            block = stream.read(right - left, dtype='float64', always_2d=True)
            if len(block) != right - left or not len(block) or not np.isfinite(block).all():
                raise ValueError('音频片段为空、损坏或包含非有限数值')
            # Channel energy, not the mono sum: anti-phase stereo must not disappear.
            rms[index] = np.sqrt(np.mean(block * block))
            available = block if tail is None else np.concatenate((tail, block), axis=0)
            padded = np.zeros((window_size, block.shape[1]), dtype=np.float64)
            length = min(len(available), window_size)
            padded[-length:] = available[-length:]
            tail = available[-window_size:].copy()
            spectrum = np.sqrt(np.mean(np.abs(np.fft.rfft(
                padded * window[:, None], n=FFT_SIZE, axis=0)) ** 2, axis=1)) / normalization
            if previous is not None:
                flux[index] = np.linalg.norm(np.maximum(spectrum - previous, 0))
            previous = spectrum
            if index % 20 == 0 or index + 1 == count:
                _report(progress, index + 1, count)
        if len(stream.read(1, dtype='float64', always_2d=True)):
            raise ValueError('音频在特征提取过程中变化')
    activity = np.clip(rms / max(RMS_FLOOR, float(np.percentile(rms, 95))), 0, 1)
    # Reject stationary numerical/phase variation and constant noise floors.
    excess = np.maximum(flux - float(np.median(flux)) - FLUX_NOISE, 0)
    pulse = np.clip(excess / max(FLUX_FLOOR, float(np.percentile(excess, 95))), 0, 1)
    pulse *= np.clip(rms / RMS_FLOOR, 0, 1)
    return {'times_us': np.arange(count, dtype=np.int64) * STEP_US,
            'rms': rms, 'onset': flux, 'activity': activity, 'pulse_strength': pulse}


def _validate(arrays, count):
    if set(arrays) != set(FIELDS):
        raise ValueError('运动特征缓存字段不匹配')
    for field in FIELDS:
        value = arrays[field]
        dtype = np.dtype('int64' if field == 'times_us' else 'float64')
        if value.dtype != dtype or value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError('运动特征缓存形状或数值无效')
        if np.any(value < 0) or (field in ('activity', 'pulse_strength') and np.any(value > 1)):
            raise ValueError('运动特征缓存数值越界')
    if not np.array_equal(arrays['times_us'], np.arange(count, dtype=np.int64) * STEP_US):
        raise ValueError('运动特征缓存时间轴不匹配')


def _read_cache(data_path, meta_path, expected, count):
    meta = read_json(meta_path)
    if not isinstance(meta, dict) or any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError('运动特征缓存来源不匹配')
    if digest(data_path) != meta.get('sha256'):
        raise ValueError('运动特征缓存校验失败')
    # Inspect headers before numpy allocates; never allow pickle/object arrays.
    with zipfile.ZipFile(data_path) as archive:
        if sorted(archive.namelist()) != sorted(field + '.npy' for field in FIELDS):
            raise ValueError('运动特征缓存成员无效')
        for field in FIELDS:
            entry = archive.getinfo(field + '.npy')
            if entry.file_size > count * 8 + 4096:
                raise ValueError('运动特征缓存成员过大')
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('运动特征缓存数组版本不支持')
                if shape != (count,) or order or dtype != np.dtype('int64' if field == 'times_us' else 'float64'):
                    raise ValueError('运动特征缓存数组头无效')
    with np.load(data_path, allow_pickle=False) as stored:
        arrays = {field: stored[field].copy() for field in FIELDS}
    _validate(arrays, count)
    return arrays


def _quarantine(folder, paths):
    existing = [path for path in paths if path.exists()]
    if existing:
        invalid = folder / ('invalid-' + uuid.uuid4().hex)
        invalid.mkdir()
        for path in existing:
            path.rename(invalid / path.name)


def load_motion_features(project_path, progress=None, cancel=None):
    """Return real audio features on a left-edge 0.1-second clock, caching locally.

    Each RMS summarizes [time, time + .1s), truncated at EOF. Onset is the L2
    positive difference between consecutive 2048-point Hann magnitude spectra
    (the Hann support is capped at one full hop; partial EOF uses past samples);
    it is an audio novelty signal, not a model prediction or beat annotation.
    No prediction, revision, project metadata or source audio is modified.
    """
    path = Path(project_path).resolve()
    project = load_project(path)
    audio = (path / project['audio_file']).resolve()
    if audio.parent != path or not audio.is_file():
        raise ValueError('项目音频路径无效或文件不存在')
    _check_cancel(cancel)
    audio_hash = digest(audio)
    if audio_hash != project['audio_sha256']:
        raise ValueError('项目音频校验失败；不会为已修改音频生成运动特征')
    fingerprint = _fingerprint()
    with _motion_audio(audio, path, cancel) as (decoded, decoder):
        return _load_decoded_features(path, audio, decoded, decoder, audio_hash,
                                      fingerprint, progress, cancel)


def _load_decoded_features(path, audio, decoded, decoder, audio_hash, fingerprint, progress, cancel):
    info = sf.info(str(decoded))
    if info.frames <= 0 or info.samplerate < 10 or info.channels < 1:
        raise ValueError('音频时长、采样率或声道无效')
    frames = info.frames
    count = (frames * 1000000 + info.samplerate * STEP_US - 1) // (info.samplerate * STEP_US)
    identity = {'kind': KIND, 'audio_sha256': audio_hash, 'fingerprint': fingerprint,
                'step_us': STEP_US, 'sample_rate': info.samplerate, 'frames': frames,
                'channels': info.channels, 'count': count}
    if decoder is not None:
        identity['decoder'] = decoder
    key = object_digest(identity)
    expected = {**identity, 'schema_version': 1, 'key': key}
    folder = path / 'motion_features'
    folder.mkdir(exist_ok=True)
    data_path, meta_path = folder / (key + '.npz'), folder / (key + '.json')

    def unchanged():
        _check_cancel(cancel)
        if digest(audio) != audio_hash:
            raise ValueError('音频在特征提取过程中变化；拒绝发布缓存')
        if _fingerprint() != fingerprint:
            raise ValueError('运动特征代码在提取过程中变化；拒绝发布缓存')

    with project_lock(folder):
        _check_cancel(cancel)
        arrays = None
        if data_path.exists() or meta_path.exists():
            try:
                arrays = _read_cache(data_path, meta_path, expected, count)
            except (OSError, ValueError, KeyError, TypeError, EOFError,
                    zipfile.BadZipFile, struct.error, zlib.error):
                _quarantine(folder, (data_path, meta_path))
        if arrays is None:
            arrays = _extract(decoded, count, info.samplerate, frames, progress, cancel)
            _validate(arrays, count)
            fd, temporary = tempfile.mkstemp(prefix='.pending-', suffix='.npz', dir=folder)
            try:
                with os.fdopen(fd, 'wb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                unchanged()
                checksum = digest(temporary)
                os.replace(temporary, data_path)
                atomic_json(meta_path, {**expected, 'sha256': checksum})
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        else:
            unchanged()
        _report(progress, count, count)
        return {**arrays, 'step_us': STEP_US, 'audio_sha256': audio_hash,
                'key': key, 'kind': KIND}
