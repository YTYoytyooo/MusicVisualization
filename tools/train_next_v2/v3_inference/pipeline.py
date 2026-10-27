"""Native-clock offline inference with immutable raw outputs and commit receipts."""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from tools.train_next.core import atomic_json, sha256
from tools.train_next_v2.common import ROOT, now, read_json, verify_hashes
from .manifest import (MANIFEST, COMPARISON, B_PROTOCOL, array_digest, contained,
                       relative, validate_manifest)


class AnalysisCancelled(InterruptedError):
    """Cooperative cancellation; committed feature receipts remain reusable."""


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise AnalysisCancelled('Analysis cancelled; completed raw artifacts are preserved')


def _progress(callback, stage, cancelled=None, **values):
    _check(cancelled)
    if callback is not None:
        callback(dict(stage=stage, **values))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def validate_output_system(system, duration, legacy=False):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid native decoded duration')
    times = np.asarray(system['times_seconds'], dtype=np.float64)
    raw = np.asarray(system['raw_va'], dtype=np.float64)
    hop = .1 if legacy else .5
    if (times.ndim != 1 or not len(times) or raw.shape != (len(times), 2)
            or not np.isfinite(times).all() or not np.isfinite(raw).all()
            or np.any(np.abs(raw) > 1)
            or not np.allclose(times, np.arange(len(times)) * hop, rtol=0, atol=1e-7)
            or times[-1] > duration + 1e-7):
        raise ValueError('Invalid raw model values or native time axis')
    if system.get('seed_disagreement_is_confidence', False) is not False:
        raise ValueError('Seed disagreement is not calibrated confidence')
    if legacy:
        # librosa fixes resampled length to ceil(native_duration * 22050).
        # Record the actual resampled count for real outputs; fixtures and old
        # comparison arrays can also be checked against native duration alone.
        expected_resampled = math.ceil(duration * 22050 - 1e-8)
        decoded_frames = system.get('decoded_frames_22050', expected_resampled)
        if (type(decoded_frames) is not int or decoded_frames <= 0
                or abs(decoded_frames - expected_resampled) > 1
                or len(times) != max(1, decoded_frames // 2205)):
            raise ValueError('Legacy output is truncated or its decoded geometry changed')
    if not legacy:
        expected = math.floor(duration / .5) + 1
        members = np.asarray(system['member_raw_va'], dtype=np.float64)
        disagreement = np.asarray(system['seed_disagreement_std'], dtype=np.float64)
        if (len(times) != expected or members.shape != (3, expected, 2)
                or not np.isfinite(members).all() or np.any(np.abs(members) > 1)
                or disagreement.shape != raw.shape or not np.isfinite(disagreement).all()
                or np.any(disagreement < 0)
                or not np.array_equal(raw, members.mean(axis=0))
                or not np.array_equal(disagreement, members.std(axis=0, ddof=1))):
            raise ValueError('Three-seed aggregation changed')


def verify_result(project_dir, manifest_path=MANIFEST, expected_audio_sha256=None, include_v2=None):
    """Read-only verification of the project and committed raw result."""
    directory = contained(project_dir)
    result_path = directory / 'analysis/result.json'
    receipt_path = directory / 'analysis/receipt.json'
    project_path = directory / 'project.json'
    receipt, project, result = read_json(receipt_path), read_json(project_path), read_json(result_path)
    mh = sha256(contained(manifest_path))
    if (receipt.get('schema_version') != 3 or receipt.get('result_sha256') != sha256(result_path)
            or receipt.get('project_sha256') != sha256(project_path)
            or receipt.get('manifest_sha256') != mh or result.get('manifest_sha256') != mh
            or project.get('manifest_sha256') != mh or project.get('schema_version') != 3
            or result.get('schema_version') != 3 or result.get('project_id') != directory.name
            or project.get('id') != directory.name or result.get('audio') != project.get('audio')
            or receipt.get('audio_sha256') != result['audio']['sha256']):
        raise ValueError('Raw result receipt or project binding mismatch')
    if expected_audio_sha256 is not None and result['audio']['sha256'] != expected_audio_sha256:
        raise ValueError('Project belongs to different audio')
    if sha256(contained(result['audio']['path'])) != result['audio']['sha256']:
        raise ValueError('Source audio changed since analysis')
    if include_v2 is not None and ('v2' in result['systems']) != include_v2:
        raise ValueError('Completed project has a different V2 comparison option; create a new project')
    if (not {'b00', 'b01'}.issubset(result['systems'])
            or not set(result['systems']).issubset({'b00', 'b01', 'v2'})
            or result['calibration']['status'] != 'unavailable'
            or result['tension']['status'] != 'unavailable'):
        raise ValueError('Research output contract changed')
    for field in ('calibration', 'tension'):
        if any(result[field].get(key) is not None for key in ('values', 'intervals')):
            raise ValueError('Unavailable outputs cannot carry fabricated values or intervals')
    for name, system in result['systems'].items():
        validate_output_system(system, result['audio']['duration_seconds'], name == 'v2')
    acoustic = result['acoustic']
    n = len(result['systems']['b01']['times_seconds'])
    if acoustic['times_seconds'] != result['systems']['b01']['times_seconds']:
        raise ValueError('Acoustic clock differs from sequence clock')
    for name in ('rms_dbfs', 'rms_amplitude', 'spectral_centroid_hz'):
        values = np.asarray(acoustic[name])
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError('Invalid acoustic output')
        if name != 'rms_dbfs' and np.any(values < 0):
            raise ValueError('RMS amplitude and spectral centroid cannot be negative')
    if acoustic.get('dbfs_floor_amplitude', 1e-8) != 1e-8:
        raise ValueError('Frozen acoustic dBFS floor changed')
    expected_dbfs = 20 * np.log10(np.maximum(np.asarray(acoustic['rms_amplitude']), 1e-8))
    if not np.allclose(acoustic['rms_dbfs'], expected_dbfs, rtol=0, atol=1e-10):
        raise ValueError('dBFS no longer corresponds to the exported RMS amplitude')
    return result


def _native_audio(path, digest):
    import soundfile as sf
    samples, sample_rate = sf.read(path, dtype='float32', always_2d=True)
    if (samples.ndim != 2 or not len(samples) or not samples.shape[1]
            or sample_rate <= 0 or not np.isfinite(samples).all()):
        raise ValueError('Audio must decode to nonempty finite native PCM')
    duration = len(samples) / sample_rate
    if duration < .1:
        raise ValueError('Audio must contain at least 0.1 seconds')
    if sha256(path) != digest:
        raise ValueError('Source audio changed during native decoding')
    audio = dict(path=relative(path), sha256=digest, name=path.name,
                 duration_seconds=duration, sample_rate=sample_rate,
                 channels=samples.shape[1], frames=len(samples))
    pcm_hash = hashlib.sha256(samples.tobytes(order='C')).hexdigest()
    del samples
    return audio, pcm_hash


def _features(audio, manifest, device, progress, cancelled):
    from tools.train_next_v2.research_inference.adapter import (
        FrozenFeatureStore, validate_clock, validate_features)
    # Reuse only feature files whose exact bytes were pinned in the old audited
    # comparison or the frozen internal-development protocol.
    comparison, b = read_json(COMPARISON), read_json(B_PROTOCOL)
    historical = next((r for r in comparison['validation_songs']
                       if r['audio_sha256'] == audio['sha256']), None)
    cache = None
    expected = None
    if historical:
        cache = contained(historical['sequence_cache'])
        expected = comparison['pinned_hashes'][historical['sequence_cache']]
    elif audio['sha256'] in b['feature_hashes']:
        cache = ROOT / 'data/training/next_va/features' / manifest['feature_key'] / (audio['sha256'] + '.npz')
        expected = b['feature_hashes'][audio['sha256']]
    if cache is not None:
        if sha256(cache) != expected:
            raise ValueError('Historical feature cache no longer matches its frozen receipt')
        with np.load(cache, allow_pickle=False) as saved:
            times, x, key = saved['times'].copy(), saved['features'].copy(), str(saved['feature_key'].item())
        if key != manifest['feature_key']:
            raise ValueError('Historical feature version mismatch')
        validate_features(x)
        validate_clock(times, len(x), audio['duration_seconds'])
        _check(cancelled)
        return times, x, dict(source='verified_existing_frozen_cache', path=relative(cache),
                             sha256=expected, reused=True, feature_key=key)
    directory = ROOT / 'data/v3.0/cache/sequence' / manifest['feature_key'] / device.replace(':', '-')
    def notify(values):
        _progress(progress, 'features', cancelled, **values)
    store = FrozenFeatureStore(manifest, device, directory, notify)
    record = dict(song_id=audio['sha256'][:24], audio_path=audio['path'], audio_sha256=audio['sha256'],
                  duration_seconds=audio['duration_seconds'], decoded_frames=audio['frames'],
                  channels=audio['channels'], sample_rate=audio['sample_rate'])
    try:
        times, x, reused = store.get(record)
        _check(cancelled)
        cache = directory / (audio['sha256'] + '.npz')
        return times, x, dict(source='v3_frozen_extractor_cache', path=relative(cache), sha256=sha256(cache),
                             receipt_sha256=sha256(cache.with_suffix('.receipt.json')),
                             reused=reused, feature_key=store.key, feature_device=device)
    finally:
        store.unload()


def _sequence(kind, manifest, times, features, duration, device, progress, cancelled):
    import torch
    from tools.train_next_v2.calibration.job_v2 import load_models
    from tools.train_next_v2.research_inference.adapter import standardize, validate_clock
    selected = manifest['systems'][kind]
    models, scaler = load_models(selected, torch.device(device))
    try:
        for member in selected['members']:
            if (array_digest(scaler[0]) != member['scaler_mean_sha256']
                    or array_digest(scaler[1]) != member['scaler_std_sha256']):
                raise ValueError('Loaded scaler differs from frozen V3 manifest')
        x = standardize(features, *scaler)
        validate_clock(times, len(x), duration)
        tensor = torch.from_numpy(x).unsqueeze(0).to(device)
        lengths = torch.tensor([len(x)], dtype=torch.int64)
        members = []
        with torch.inference_mode():
            for i, model in enumerate(models, 1):
                _progress(progress, 'inference', cancelled, system=kind, members_done=i-1, members_total=3)
                members.append(model(tensor, lengths).cpu().numpy()[0])
                _check(cancelled)
        members = np.asarray(members, dtype=np.float64)
        system = dict(times_seconds=times, raw_va=members.mean(axis=0), member_raw_va=members,
            seed_disagreement_std=members.std(axis=0, ddof=1), seed_disagreement_is_confidence=False,
            description=f'{kind.upper()} fixed three-seed arithmetic mean; raw offline whole-song V/A',
            member_run_ids=[m['run_id'] for m in selected['members']],
            temporal_scope='bidirectional whole-song context; not real-time', output_hop_seconds=.5)
        validate_output_system(system, duration)
        return system
    finally:
        del models
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


def _legacy(audio, manifest, progress, cancelled):
    import torch
    from tools.train_next_v2.cross_version.adapters import load_legacy_source
    legacy = manifest['legacy']
    folder = ROOT / 'data/v3.0/cache/legacy_cpu'
    folder.mkdir(parents=True, exist_ok=True)
    contract = hashlib.sha256(json.dumps(dict(legacy=legacy,
        checkpoint=manifest['pinned_hashes'][legacy['checkpoint']],
        source={k:v for k,v in manifest['pinned_hashes'].items() if k.startswith('versions/v2.0/') and k.endswith('.py')},
        libraries=manifest['feature_spec']['libraries']), sort_keys=True).encode()).hexdigest()
    cached = folder / (audio['sha256'] + '-' + contract[:16] + '.npz')
    receipt = cached.with_suffix('.receipt.json')
    if cached.exists() != receipt.exists():
        raise ValueError('Unreceipted legacy artifact requires inspection')
    if cached.exists():
        row = read_json(receipt)
        if row['audio_sha256'] != audio['sha256'] or row['contract'] != contract or row['cache_sha256'] != sha256(cached):
            raise ValueError('Legacy cache receipt mismatch')
        with np.load(cached, allow_pickle=False) as data:
            times, values = data['times'].copy(), data['raw_va'].copy()
            decoded_frames = int(data['decoded_frames_22050'].item())
        reused = True
    else:
        features, source = load_legacy_source()
        _progress(progress, 'legacy_cpu_decode', cancelled)
        y, rate = features.load_audio(str(contained(audio['path'])))
        if rate != 22050:
            raise ValueError('Original V2 decoder contract changed')
        decoded_frames = len(y)
        torch.manual_seed(legacy['seed'])
        np.random.seed(legacy['seed'])
        embeddings = source.extract_clap_embeddings(y, rate, batch_size=16,
            clap_source=contained(legacy['snapshot']), cancel=cancelled,
            progress=lambda done,total: _progress(progress, 'legacy_cpu_features', cancelled,
                                                   frames_done=done, frames_total=total))
        interface = source.EmotionInterface.load(str(contained(legacy['checkpoint'])))
        predicted = interface.predict_sequence(embeddings, cancel=cancelled,
            progress=lambda done,total: _progress(progress, 'legacy_cpu_head', cancelled,
                                                   frames_done=done, frames_total=total))
        values = np.asarray([r['emotion'][:2] for r in predicted], dtype=np.float64)
        times = np.arange(len(values), dtype=np.float64) * .1
        _check(cancelled)
        validate_output_system(dict(times_seconds=times, raw_va=values,
                                    decoded_frames_22050=decoded_frames), audio['duration_seconds'], True)
        tmp = cached.with_suffix('.tmp')
        if tmp.exists():
            raise ValueError('Partial legacy cache requires inspection')
        with tmp.open('xb') as stream:
            np.savez_compressed(stream, times=times, raw_va=values,
                                decoded_frames_22050=np.array(decoded_frames, dtype=np.int64))
        tmp.replace(cached)
        atomic_json(receipt, dict(audio_sha256=audio['sha256'], contract=contract,
                                 cache_sha256=sha256(cached), created_at=now(), device='cpu'))
        reused = False
    system = dict(times_seconds=times, raw_va=values, seed_disagreement_std=None,
        decoded_frames_22050=decoded_frames,
        description='Original V2 CPU decoder, CPU CLAP and unchanged CPU head; raw V/A only',
        output_hop_seconds=.1, execution_device='cpu', calibrated=False,
        seed_disagreement_is_confidence=False)
    validate_output_system(system, audio['duration_seconds'], True)
    return system, dict(path=relative(cached), sha256=sha256(cached), reused=reused,
                        source='original_v2_cpu', gpu_port_accepted=False)


def analyze(audio_path, project_dir, device='cuda', progress=None, cancelled=None,
            include_v2=False, manifest_path=MANIFEST):
    """Analyze one existing local file. Caller owns process/GPU/project locks."""
    import torch
    from tools.train_next_v2.research_inference.adapter import acoustic_properties
    if type(include_v2) is not bool or device not in ('cpu', 'cuda', 'cuda:0'):
        raise ValueError('Use device cpu/cuda/cuda:0 and an explicit boolean include_v2')
    started = time.perf_counter()
    _progress(progress, 'verify_manifest', cancelled)
    manifest = validate_manifest(manifest_path)
    mh = sha256(contained(manifest_path))
    path, directory = contained(audio_path), contained(project_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in ('.mp3', '.flac', '.wav', '.ogg'):
        raise ValueError('Supported preview audio: MP3, FLAC, WAV, OGG')
    digest = sha256(path)
    if digest in manifest['excluded_final_test_audio_sha256']:
        raise ValueError('Final-test audio/recording group remains sealed')
    if 'fabio' in {part.casefold() for part in path.parts} or path.is_relative_to(ROOT / 'data/annotations'):
        raise ValueError('Paused new annotation audio is excluded from this V3 preview')
    result_path, receipt_path = directory/'analysis/result.json', directory/'analysis/receipt.json'
    if result_path.exists() or receipt_path.exists():
        if not result_path.exists() or not receipt_path.exists():
            raise ValueError('Unreceipted raw result requires inspection; it will not be overwritten')
        verify_result(directory, manifest_path, digest, include_v2)
        _progress(progress, 'completed', cancelled, reused=True, result_path=relative(result_path))
        return result_path
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the pinned environment; explicitly choose CPU')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    measurements = dict(device=device, include_v2=include_v2, clock='time.perf_counter',
                        sleep_or_suspension_not_excluded=True)
    phase = time.perf_counter()
    _progress(progress, 'native_decode', cancelled)
    audio, pcm_hash = _native_audio(path, digest)
    measurements['decode_seconds'] = time.perf_counter()-phase
    project_path = directory / 'project.json'
    project = dict(schema_version=3, id=directory.name, audio=audio,
                   manifest_sha256=mh, created_at=now())
    if project_path.exists():
        old = read_json(project_path)
        if any(old.get(k) != project[k] for k in ('schema_version','id','audio','manifest_sha256')):
            raise ValueError('Existing project metadata differs; preserve it and create another project')
        project = old
    else:
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(project_path, project)
    _progress(progress, 'features', cancelled, duration_seconds=audio['duration_seconds'])
    phase = time.perf_counter()
    times, x, feature_receipt = _features(audio, manifest, device, progress, cancelled)
    measurements['features_seconds'] = time.perf_counter()-phase
    measurements['feature_cache_reused'] = feature_receipt['reused']
    systems = {}
    for kind in ('b01', 'b00'):
        phase = time.perf_counter()
        systems[kind] = _sequence(kind, manifest, times, x, audio['duration_seconds'], device, progress, cancelled)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        measurements[kind+'_load_and_inference_seconds'] = time.perf_counter()-phase
    legacy_receipt = None
    if include_v2:
        phase = time.perf_counter()
        systems['v2'], legacy_receipt = _legacy(audio, manifest, progress, cancelled)
        measurements['v2_cpu_seconds'] = time.perf_counter()-phase
    acoustic = dict(times_seconds=times, **acoustic_properties(x))
    measurements['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0
    measurements['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0
    measurements['total_before_serialization_seconds'] = time.perf_counter()-started
    measurements['audio_seconds_per_elapsed_second'] = audio['duration_seconds']/max(measurements['total_before_serialization_seconds'],1e-9)
    if sha256(path) != digest or sha256(contained(manifest_path)) != mh:
        raise ValueError('Input or manifest changed during analysis')
    result = dict(schema_version=3, project_id=directory.name, audio=audio, manifest_sha256=mh,
        systems=systems, acoustic=acoustic, calibration=dict(status='unavailable', values=None,
            reason='No validated calibration for B01; B00 failed audit; calibrators are not transferable'),
        tension=dict(status='unavailable', values=None), display=dict(status='not_applied_to_raw'),
        performance=measurements, provenance=dict(created_at=now(),
            manifest_path=relative(manifest_path), native_float32_pcm_sha256=pcm_hash,
            sequence_features=feature_receipt, legacy=legacy_receipt,
            label_values_loaded=False, training_performed=False, promoted_model=False,
            time_origin='native decoded sample zero; no label shift, rescale, or time stretching'))
    _progress(progress, 'save_raw_result', cancelled)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    # The commit receipt is written last. Interrupted orphan outputs fail closed.
    atomic_json(result_path, _jsonable(result))
    atomic_json(receipt_path, dict(schema_version=3, result_sha256=sha256(result_path),
        project_sha256=sha256(project_path), audio_sha256=digest, manifest_sha256=mh,
        completed_at=now(), models={kind:[m['checkpoint_sha256'] for m in system['members']]
                                   for kind,system in manifest['systems'].items()},
        feature_cache_sha256=feature_receipt['sha256'], include_v2=include_v2))
    verify_result(directory, manifest_path, digest, include_v2)
    _progress(progress, 'completed', None, reused=False, result_path=relative(result_path))
    return result_path
