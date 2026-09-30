"""Analysis, deterministic planning and revision-locked rendering."""
from dataclasses import asdict
import csv
import json
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
import numpy as np
from .store import (ROOT, atomic_json, create_project, digest, load_project,
                    load_raw, load_revision, now, object_digest, read_json, project_lock)
from .timeline import (EMOTIONS, VISUAL_BOUNDS, INTEGER_FIELDS, apply_edits,
                       emotion_at, event_times, frame_times, raw_at,
                       validation_times, validate_edits, number, integer)

VISUAL_FIELDS = tuple(VISUAL_BOUNDS)
from .smoothing import validate_smoothing, emotion_values, baseline_at, check_times


class Cancelled(Exception):
    pass


def engine_hash():
    files = ['emotion_model.py', 'feature_extraction.py', 'mcts.py', 'renderer.py',
             'prediction_overlay.py', 'studio/timeline.py', 'studio/pipeline.py']
    return object_digest({f: digest(ROOT / f) for f in files})


def packages():
    result = {}
    for name in ('numpy', 'torch', 'opencv-python', 'librosa', 'transformers', 'soundfile'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def ffmpeg_path():
    configured = os.environ.get('STUDIO_FFMPEG')
    candidates = [configured, shutil.which('ffmpeg'),
                  str(ROOT / '.runtime/bin/ffmpeg.exe'),
                  r'C:\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files\ffmpeg\bin\ffmpeg.exe']
    candidates += [str(p) for p in (ROOT / '.runtime/imageio_ffmpeg/binaries').glob('ffmpeg*.exe')]
    # WinGet installs may be visible to the user's terminal but not a desktop
    # worker's inherited PATH. Discover this known per-user location directly.
    local = os.environ.get('LOCALAPPDATA')
    if local:
        packages_dir = Path(local) / 'Microsoft/WinGet/Packages'
        candidates += [str(p) for p in sorted(
            packages_dir.glob('Gyan.FFmpeg_*/ffmpeg*/bin/ffmpeg.exe'), reverse=True)]
    for candidate in candidates:
        try:
            if candidate and Path(candidate).is_file():
                return str(Path(candidate).resolve())
        except OSError:
            continue  # An inaccessible system install must not hide a usable local one.
    return None


def doctor():
    import sys
    ff = ffmpeg_path()
    encoders = subprocess.run([ff, '-hide_banner', '-encoders'], capture_output=True,
                              text=True, timeout=10).stdout if ff else ''
    return {'studio': str(ROOT), 'python': sys.executable, 'version': sys.version,
            'packages': packages(), 'ffmpeg': ff, 'h264': 'libx264' in encoders,
            'aac': ' aac ' in encoders,
            'note': '正式 MP4 需要 H.264/AAC；不会自动安装依赖或训练模型'}


def analyze(audio_path, model_path, root, name=None, seed=42, progress=None, cancel=None):
    audio, model = Path(audio_path).resolve(), Path(model_path).resolve()
    if not audio.is_file() or not model.is_file():
        raise ValueError('音频或模型不存在；请提供完整路径。不会自动训练。')
    integer(seed, 'seed')
    def report(stage, ratio, details=None):
        if cancel and cancel():
            raise Cancelled('用户取消分析')
        if progress:
            if getattr(progress, 'accepts_details', False) is True:
                progress(stage, ratio, details)
            else:
                progress(stage, ratio)
    report('loading', 0.)
    from feature_extraction import load_audio, extract_global_info
    from emotion_model import extract_clap_embeddings, EmotionInterface, AnalysisCancelled
    from .fingerprints import clap_snapshot, feature_key, prediction_key, stage_fingerprint
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    # Avoid excessive worker contention on a local desktop.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    audio_hash, model_hash = digest(audio), digest(model)
    frozen_stages = {stage:stage_fingerprint(stage) for stage in ('features','prediction')}
    snapshot = clap_snapshot()
    key = feature_key(audio_hash, snapshot, frozen_stages['features'])
    prediction = prediction_key(key, model_hash, frozen_stages['prediction'])
    cache = Path(root) / '.cache' / key
    cache.mkdir(parents=True, exist_ok=True)
    y, sr = load_audio(str(audio))
    if len(y) < sr * .1:
        raise ValueError('音频至少需要 0.1 秒')
    info = extract_global_info(y, sr)
    report('embedding', .15)
    embedding_file, cache_meta = cache / 'clap.npy', cache / 'metadata.json'
    with project_lock(cache):
        valid = False
        if embedding_file.exists() and cache_meta.exists():
            try:
                meta = read_json(cache_meta)
                valid = meta.get('key') == key and meta.get('sha256') == digest(embedding_file)
            except (OSError, ValueError):
                pass
        if valid:
            embeddings = np.load(embedding_file, allow_pickle=False)
        else:
            # Preserve incomplete/corrupt artifacts and rebuild with a fresh name.
            tag = uuid.uuid4().hex
            for orphan in (embedding_file, cache_meta):
                if orphan.exists():
                    os.replace(orphan, cache / f'quarantine-{tag}-{orphan.name}')
            pending = cache / f'pending-{tag}.npy'
            try:
                embeddings = extract_clap_embeddings(y, sr, cache_path=str(pending),
                    clap_source=snapshot['source'], cancel=cancel,
                    progress=lambda done,total: report('embedding', done/max(1,total),
                        {'done':done,'total':total,'unit':'音频片段'}))
            except AnalysisCancelled as exc:
                raise Cancelled(str(exc)) from exc
            report('embedding', 1.)
            if embeddings.ndim != 2 or embeddings.shape[1] != 512 or not len(embeddings) or not np.isfinite(embeddings).all():
                raise ValueError('CLAP 嵌入格式非法')
            if stage_fingerprint('features') != frozen_stages['features']:
                raise ValueError('特征代码在分析过程中变化，未发布缓存；请重试')
            os.replace(pending, embedding_file)
            atomic_json(cache_meta, {'key': key, 'sha256': digest(embedding_file), 'audio_sha256': audio_hash,
                                    'clap_snapshot':snapshot, 'stage_fingerprint':frozen_stages['features']})
        if embeddings.ndim != 2 or embeddings.shape[1] != 512 or not len(embeddings) or not np.isfinite(embeddings).all():
            raise ValueError('CLAP 嵌入格式非法')
    report('inference', .75)
    prediction_folder = Path(root) / '.cache/predictions' / prediction
    prediction_folder.mkdir(parents=True, exist_ok=True)
    prediction_file, prediction_meta = prediction_folder / 'values.npy', prediction_folder / 'metadata.json'
    with project_lock(prediction_folder):
        valid_prediction = False
        if prediction_file.is_file() and prediction_meta.is_file():
            try:
                meta = read_json(prediction_meta)
                valid_prediction = meta['key'] == prediction and meta['sha256'] == digest(prediction_file)
            except (OSError, ValueError, KeyError):
                pass
        if valid_prediction:
            raw = np.load(prediction_file, allow_pickle=False)
        else:
            try:
                states = EmotionInterface.load(str(model)).predict_sequence(embeddings, cancel=cancel,
                    progress=lambda done,total: report('inference', done/max(1,total),
                        {'done':done,'total':total,'unit':'情绪采样'}))
            except AnalysisCancelled as exc:
                raise Cancelled(str(exc)) from exc
            raw = np.array([[state[k] for k in EMOTIONS] for state in states])
            report('inference', 1.)
            if raw.shape != (len(embeddings), 5) or not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1:
                raise ValueError('情绪预测格式非法')
            if stage_fingerprint('prediction') != frozen_stages['prediction']:
                raise ValueError('预测代码在分析过程中变化，未发布缓存；请重试')
            for old in (prediction_file, prediction_meta):
                if old.exists():
                    os.replace(old, prediction_folder / ('quarantine-' + uuid.uuid4().hex + '-' + old.name))
            pending_prediction = prediction_folder / ('pending-' + uuid.uuid4().hex + '.npy')
            np.save(pending_prediction, raw)
            os.replace(pending_prediction, prediction_file)
            atomic_json(prediction_meta, {'key':prediction,'sha256':digest(prediction_file),
                        'model_sha256':model_hash,'stage_fingerprint':frozen_stages['prediction']})
    if digest(audio) != audio_hash or digest(model) != model_hash:
        raise ValueError('分析过程中源音频或模型被修改，未发布项目')
    if any(stage_fingerprint(stage) != fingerprint for stage,fingerprint in frozen_stages.items()):
        raise ValueError('分析代码在任务期间变化，未发布项目；请重试')
    report('saving', .95)
    path = create_project(root, name, audio, model, raw, info, seed,
                          {'kind': 'model', 'engine_hash': engine_hash(), 'packages': packages(),
                           'stage_fingerprints':frozen_stages,
                           'clap_cache_key': key, 'prediction_cache_key':prediction,
                           'clap_snapshot':snapshot, 'clap_window_seconds': 2.,
                           'clap_timestamp': 'window-start; offline, includes future audio'})
    report('analyzed', 1.)
    return path


def timeline_payload(path, edits=None, revision=None, smoothing=None, motion=None):
    from .timeline import preview_times
    p = load_project(path)
    raw = load_raw(path)
    if edits is None:
        saved = load_revision(path, revision)
        edits = saved['edits']
        smoothing = saved.get('smoothing') if smoothing is None else smoothing
        motion = saved.get('motion') if motion is None else motion
    elif motion is None:
        motion = load_revision(path,revision).get('motion')
    smoothing = validate_smoothing(smoothing)
    edits = validate_edits(edits, p['duration_us'])
    # Validate on the complete source clock, independently of display sampling.
    emotion_values(raw, check_times(raw, p['duration_us'], edits, p['step_us'], smoothing), edits, p['step_us'], smoothing)
    ts = preview_times(len(raw), p['duration_us'], edits, p['step_us'])
    effective = emotion_values(raw, ts, edits, p['step_us'], smoothing)
    # Charts may decimate on the client. Server returns exact sample/event values.
    waveform = Path(path) / 'analysis/waveform.json'
    from .motion import build_motion_plan
    motion_plan = build_motion_plan(path, raw, edits, smoothing, motion)
    return {'times': [int(t) / 1e6 for t in ts], 'times_us': [int(t) for t in ts],
            'source_raw': raw.tolist(), 'step_us': p['step_us'], 'smoothing':smoothing,
            'baseline':baseline_at(raw, ts, p['step_us'], smoothing).tolist(),
            'raw': raw_at(raw, ts, p['step_us']).tolist(),
            'effective': effective.tolist(), 'waveform': read_json(waveform) if waveform.exists() else None,
            'motion': motion_plan}


def planning_key(path, edits, iterations=200, smoothing=None):
    from .fingerprints import stage_fingerprint
    p = load_project(path)
    edits = validate_edits(edits, p['duration_us'])
    # Ignore note/identity-only changes, disabled operations and visual edits.
    keys = ('layer', 'field', 'operation', 'time_us', 'start_us', 'end_us', 'value',
            'transition_in_us', 'transition_out_us', 'interpolation')
    active = [{k: e[k] for k in keys if k in e} for e in edits
              if e['layer'] == 'emotion' and e.get('enabled', True)]
    active.sort(key=lambda e: (e['field'], e.get('time_us', e.get('start_us', 0))))
    return object_digest({'raw': p['raw_sha256'], 'edits': active,
                          'seed': p['seed'], 'stage': stage_fingerprint('planning'), 'iterations': iterations,
                          'smoothing':validate_smoothing(smoothing)})


def cached_visual_plan(path, edits, iterations=200, smoothing=None):
    p = load_project(path)
    edits = validate_edits(edits, p['duration_us'])
    key = planning_key(path, edits, iterations, smoothing)
    cache = Path(path) / 'plans' / key / 'plan.json'
    if not cache.is_file():
        return None
    data = read_json(cache)
    if data.get('key') != key or data.get('sha256') != object_digest({k: v for k, v in data.items() if k != 'sha256'}):
        raise ValueError('视觉规划缓存校验失败')
    return {'key': key, 'emotion_key': key, 'times_us': data['times_us'],
            'states': data['states'], 'fields': list(VISUAL_FIELDS)}


def prepare_visual_plan(path, base_revision, edits, progress=None, cancel=None, smoothing=None):
    load_revision(path, base_revision)
    p = load_project(path)
    edits = validate_edits(edits, p['duration_us'])
    raw = load_raw(path)
    emotion_values(raw, check_times(raw, p['duration_us'], edits, p['step_us'], smoothing), edits, p['step_us'], smoothing)
    _, _, key = plan(path, {'edits': edits, 'smoothing':smoothing}, progress=progress, cancel=cancel)
    return {'project_id': Path(path).name, 'emotion_key': key, 'plan_key': key}


def validate_preview(path, edits, smoothing=None, motion=None):
    """Authoritative draft validation; never synthesize a fake visual baseline."""
    p = load_project(path)
    edits = validate_edits(edits, p['duration_us'])
    payload = timeline_payload(path, edits=edits, smoothing=smoothing, motion=motion)
    if any(e['layer'] == 'visual' and e['enabled'] for e in edits):
        cached = cached_visual_plan(path, edits, smoothing=smoothing)
        if cached is None:
            raise ValueError('视觉规划尚未准备好，请等待真实规划完成后再确认修改')
        from mcts import VisualState
        from .timeline import visual_validation_times
        states = [VisualState(**s) for s in cached['states']]
        ts = sorted(set(validation_times(len(payload['source_raw']), p['duration_us'], edits, p['step_us']).tolist())
                    | set(frame_times(p['duration_us'], 60).tolist()) | set(cached['times_us'])
                    | set(visual_validation_times(cached, p['duration_us'], edits).tolist()))
        visual_at(states, np.array(cached['times_us']), ts, edits)
    return payload


def plan(path, revision, iterations=200, progress=None, cancel=None):
    from mcts import MCTS, VisualState
    p, raw = load_project(path), load_raw(path)
    edits = validate_edits(revision['edits'], p['duration_us'])
    # Visual-only revisions share this cache; random source is isolated from particles.
    smoothing = validate_smoothing(revision.get('smoothing'))
    key = planning_key(path, edits, iterations, smoothing)
    directory = Path(path) / 'plans' / key
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / 'plan.json'
    if cache.exists():
        data = read_json(cache)
        if data.get('sha256') != object_digest({k: v for k, v in data.items() if k != 'sha256'}):
            raise ValueError('视觉规划缓存校验失败')
        return np.array(data['times_us'], dtype=np.int64), [VisualState(**v) for v in data['states']], key
    sampling = set(range(0, p['duration_us'], 500000)) | event_times(edits, 'emotion') | {p['duration_us']}
    # Short ramps must affect actual planning, not only the overlay. Use the
    # same interior nodes for every interpolation mode so comparisons are fair.
    for e in edits:
        if not e['enabled'] or e['layer'] != 'emotion':
            continue
        start = e.get('time_us', e.get('start_us'))
        end = e.get('time_us', e.get('end_us'))
        for a, b in ((start - e['transition_in_us'], start), (end, end + e['transition_out_us'])):
            sampling.update(a + (b - a) * i // 8 for i in range(1, 8))
    ts = sorted(sampling)
    values = emotion_values(raw, ts, edits, p['step_us'], smoothing)
    mcts = MCTS(n_iter=iterations, branching=5, rng_seed=p['seed'])
    states, previous = [], None
    for index, row in enumerate(values):
        if cancel and cancel():
            raise Cancelled('用户取消视觉规划')
        if progress and index % 10 == 0:
            if getattr(progress, 'accepts_details', False) is True:
                progress('planning', index / max(1,len(values)),{'done':index,'total':len(values),'unit':'节点'})
            else:
                progress('planning', index / max(1, len(values)))
        previous = mcts.search(float(row[0]), float(row[1]), init_state=previous)
        states.append(previous)
    data = {'times_us': ts, 'states': [asdict(s) for s in states], 'key': key}
    data['sha256'] = object_digest(data)
    atomic_json(cache, data)
    return np.array(ts, dtype=np.int64), states, key


def visual_at(states, state_times, times_us, edits):
    from legacy_pipeline import _interpolate_visual_state
    base = np.array([[getattr(s, k) for k in VISUAL_FIELDS] for s in
                     (_interpolate_visual_state(states, state_times / 1e6, t / 1e6) for t in times_us)])
    values = apply_edits(base, times_us, edits, 'visual', VISUAL_FIELDS)
    for k in INTEGER_FIELDS:
        values[:, VISUAL_FIELDS.index(k)] = np.rint(values[:, VISUAL_FIELDS.index(k)])
    return values


def validate_render_options(p, options):
    result = {'mode': options.get('mode', 'analysis'), 'fps': options.get('fps', 30),
              'width': options.get('width', 1280), 'height': options.get('height', 720),
              'start': options.get('start', 0), 'end': options.get('end')}
    if result['mode'] not in ('analysis', 'presentation'):
        raise ValueError('未知导出模式')
    for k, lower, upper in (('fps', 1, 60), ('width', 320, 1920), ('height', 180, 1080)):
        integer(result[k], k, lower)
        if result[k] > upper:
            raise ValueError(f'{k} 超过限制 {upper}')
    if result['width'] % 2 or result['height'] % 2:
        raise ValueError('视频宽高必须为偶数')
    result['start'] = number(result['start'], 'start')
    result['end'] = p['duration'] if result['end'] is None else number(result['end'], 'end')
    if not 0 <= result['start'] < result['end'] <= p['duration'] + 1e-6:
        raise ValueError('预览范围必须位于歌曲内且结束晚于开始')
    return result


def create_preview_snapshot(path, base_revision, edits, smoothing=None, edit_id=None,
                            motion=None, motion_edit_id=None):
    """Freeze a validated draft independently of the formal revision pointer."""
    path = Path(path)
    previous = load_revision(path, base_revision)
    p = load_project(path)
    smoothing = validate_smoothing(smoothing)
    from motion_schema import validate_config
    motion = validate_config(previous.get('motion') if motion is None else motion, p['duration_us'])
    edits = validate_edits(edits, p['duration_us'])
    if any(e['layer'] == 'visual' and e['enabled'] for e in edits) and cached_visual_plan(path,edits,smoothing=smoothing) is None:
        raise ValueError('请先等待当前修改的真实规划校验完成，再创建预览')
    validate_preview(path, edits, smoothing, motion)
    selected = next((e for e in edits if e['id'] == edit_id), None)
    if edit_id and selected is None:
        raise ValueError('预览选择的修改不存在')
    selected_motion = next((e for e in motion['overrides'] if e['id']==motion_edit_id), None)
    if motion_edit_id and selected_motion is None:
        raise ValueError('预览选择的运动覆盖不存在')
    if selected_motion:
        left = selected_motion['start_us']-selected_motion['transition_in_us']
        right = selected_motion['end_us']+selected_motion['transition_out_us']
        start, end = max(0.,left/1e6-2), min(p['duration'],right/1e6+2)
    elif selected:
        from .timeline import support
        left, right = support(selected)
        start, end = max(0., left/1e6-2), min(p['duration'],right/1e6+2)
    else:
        start, end = 0., min(p['duration'],10.)
    data = {'id':'s-' + uuid.uuid4().hex[:12], 'base_revision':base_revision,
             'analysis_sha256':p['raw_sha256'], 'created_at':now(), 'edits':edits,'smoothing':smoothing,
             'motion':motion,
            'options':{'start':start,'end':end,'width':640,'height':360,'fps':30,'mode':'analysis'}}
    data['sha256'] = object_digest(data)
    atomic_json(path / 'previews' / data['id'] / 'snapshot.json', data)
    return data


def render_preview(path, snapshot_id, options=None, progress=None, cancel=None):
    import re
    if not isinstance(snapshot_id,str) or not re.fullmatch(r's-[a-f0-9]{12}',snapshot_id):
        raise ValueError('非法预览快照')
    path = Path(path)
    saved = read_json(path / 'previews' / snapshot_id / 'snapshot.json')
    if saved['id'] != snapshot_id or object_digest({k:v for k,v in saved.items() if k != 'sha256'}) != saved['sha256']:
        raise ValueError('预览快照校验失败')
    if saved['analysis_sha256'] != load_project(path)['raw_sha256']:
        raise ValueError('预览快照与原始分析不匹配')
    revision = {**saved, 'id':snapshot_id,'snapshot_id':snapshot_id}
    return render(path, options={**saved['options'],**(options or {})},progress=progress,cancel=cancel,_snapshot=revision)


def render(path, revision_id=None, options=None, progress=None, cancel=None,
           allow_silent=False, iterations=200, _snapshot=None):
    from .fingerprints import stage_fingerprint
    path = Path(path)
    p = load_project(path)
    revision = load_revision(path, revision_id) if _snapshot is None else _snapshot
    smoothing = validate_smoothing(revision.get('smoothing'))
    from motion_schema import validate_config, PARAM_BOUNDS
    from .motion import build_motion_plan, sample_frame, frame_summary
    from .motion_features import MotionFeatureCancelled
    motion = validate_config(revision.get('motion'), p['duration_us'])
    settings = validate_render_options(p, options or {})
    # Conservative working-space estimate for intermediate MJPEG + MP4.
    estimated_bytes = int(settings['width'] * settings['height'] * 3 * settings['fps'] *
                          (settings['end'] - settings['start']) * .2 + 64 * 1024**2)
    if shutil.disk_usage(path).free < estimated_bytes:
        raise ValueError('输出磁盘可用空间不足：请预留约 %.1f MB 后重试' % (estimated_bytes / 1024**2))
    with tempfile.TemporaryFile(dir=path / 'renders'):
        pass  # Fail before expensive planning if the destination is not writable.
    audio = path / p['audio_file']
    if digest(audio) != p['audio_sha256']:
        raise ValueError('音频内容已变化，请重新分析建立新项目')
    ff = ffmpeg_path()
    if not allow_silent:
        environment = doctor()
        if not environment['h264'] or not environment['aac']:
            raise ValueError('缺少 FFmpeg H.264/AAC 编码器。运行 doctor；设置 STUDIO_FFMPEG 后重试。')
    render_id = 'v-' + uuid.uuid4().hex[:12]
    folder = path / 'renders' / render_id
    folder.mkdir()
    metadata = {'id': render_id, 'revision': revision['id'], 'revision_sha256': revision['sha256'],
                'status': 'planning', 'settings': settings, 'created_at': now(),
                'engine_hash': engine_hash(), 'packages': packages(), 'seed': p['seed'],
                'kind':'preview' if _snapshot else 'render', 'smoothing':smoothing,
                'motion':motion, 'motion_engine':motion['engine'],
                'stage_fingerprints': {k:stage_fingerprint(k) for k in ('planning','render')},
                'approximate': settings['fps'] != 30 or settings['width'] != 1280 or settings['height'] != 720}
    if _snapshot:
        metadata.update(snapshot_id=revision['snapshot_id'], revision=revision['base_revision'])
    atomic_json(folder / 'render.json', metadata)
    atomic_json(folder / 'revision_snapshot.json', revision)
    def report(stage, ratio, details=None):
        if cancel and cancel():
            raise Cancelled('用户取消生成')
        if progress:
            if getattr(progress, 'accepts_details', False) is True:
                progress(stage, ratio, details)
            else:
                progress(stage, ratio)
    writer = None
    try:
        report('planning', 0.)
        raw = load_raw(path)
        times, states, plan_key = plan(path, revision, iterations=iterations,
                                      progress=progress, cancel=cancel)
        ts = frame_times(p['duration_us'], settings['fps'])
        # Do not lose an edit that lies between coarse planning nodes.
        check = sorted(set(validation_times(len(raw), p['duration_us'], revision['edits']).tolist())
                       | set(times.tolist()) | set(ts.tolist()))
        from .timeline import visual_validation_times
        check = sorted(set(check) | set(visual_validation_times(
            {'times_us': times.tolist(), 'states': [asdict(s) for s in states]},
            p['duration_us'], revision['edits']).tolist()))
        visual_at(states, times, check, revision['edits'])
        emotion_values(raw, check_times(raw,p['duration_us'],revision['edits'],p['step_us'],smoothing),
                       revision['edits'],p['step_us'],smoothing)
        effective = emotion_values(raw, ts, revision['edits'], p['step_us'], smoothing)
        visual = visual_at(states, times, ts, revision['edits'])
        try:
            motion_plan = build_motion_plan(path,raw,revision['edits'],smoothing,motion,progress,cancel)
        except MotionFeatureCancelled as exc:
            raise Cancelled(str(exc)) from exc
        motion_enabled = motion['engine']=='flow-v1'
        motion_fields = ['motion_mode','motion_source','motion_reason','motion_components_json',
                         *['motion_'+key for key in PARAM_BOUNDS],
                         'audio_rms','audio_onset','audio_activity','audio_pulse','beat_impulse']
        feature_samples = motion_plan.get('audio_features',{})
        audio_controls = {key:np.interp(ts,feature_samples['times_us'],feature_samples[key])
                          for key in ('rms','onset','activity','pulse_strength')} if motion_enabled else {}
        metadata['motion_feature_key'] = motion_plan.get('feature_key')
        metadata['motion_note'] = '艺术规则匹配，不是模型置信度；motion_*标量描述主成分，完整混合见motion_components_json'
        if motion_enabled:
            atomic_json(folder / 'motion_plan.json', motion_plan)
            with open(folder / 'motion_plan.csv','w',encoding='utf-8-sig',newline='') as motion_csv:
                motion_writer=csv.writer(motion_csv)
                motion_writer.writerow(['time_us',*motion_fields[:-5]])
                for time_us in ts:
                    record=frame_summary(sample_frame(motion_plan,int(time_us)))
                    motion_writer.writerow([int(time_us),record['mode'],record['source'],record['reason'],
                        json.dumps(record['components'],ensure_ascii=False,separators=(',',':')),
                        *[record['params'][key] for key in PARAM_BOUNDS]])
        # Publish the complete numerical plan BEFORE video encoding begins.
        np.savez_compressed(folder / 'predictions_effective.npz', times_us=ts, values=effective)
        with open(folder / 'predictions_effective.csv', 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time_us', *EMOTIONS])
            w.writerows([int(t), *v] for t, v in zip(ts, effective))
        if smoothing['enabled']:
            with open(folder / 'predictions_baseline.csv', 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f)
                w.writerow(['time_us', *EMOTIONS])
                w.writerows([int(t), *v] for t, v in zip(ts, baseline_at(raw,ts,p['step_us'],smoothing)))
        with open(folder / 'visual_plan.csv', 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time_us', *VISUAL_FIELDS])
            w.writerows([int(t), *v] for t, v in zip(ts, visual))
        from feature_extraction import load_audio
        from mcts import VisualState
        from renderer import VideoRenderer
        from prediction_overlay import PredictionOverlay
        y, sr = load_audio(str(audio))
        hud = None
        if settings['mode'] == 'analysis':
            baseline = raw_at(raw, ts, p['step_us'])
            effective_states = [dict(zip(EMOTIONS, row)) for row in effective]
            original_states = [dict(zip(EMOTIONS, row)) for row in baseline]
            label = ('MANUAL REVISION' if any(e['enabled'] for e in revision['edits']) else 'MODEL PREDICTION')
            if smoothing['enabled']:
                label += ' / SMOOTHED BASELINE'
            if p['provenance']['kind'] != 'model':
                label = 'SYNTHETIC TEST / ' + label
            hud = PredictionOverlay(p['name'], p['duration'], effective_states,
                                    frame_duration=1 / settings['fps'], times=np.arange(len(ts)) / settings['fps'],
                                    raw_emotion_states=original_states,
                                    source_label=f"{label} / {revision['id']}")
        start_frame = int(math.floor(settings['start'] * settings['fps']))
        end_frame = min(len(ts), int(math.ceil(settings['end'] * settings['fps'])))
        metadata.update({'actual_start': start_frame / settings['fps'], 'actual_end': end_frame / settings['fps'],
                         'plan_key': plan_key, 'status': 'rendering'})
        atomic_json(folder / 'render.json', metadata)
        silent = folder / 'video_silent.avi'
        writer = VideoRenderer(str(silent), fps=settings['fps'], width=settings['width'],
                               height=settings['height'], prediction_overlay=hud,
                               rng_seed=p['seed'] + 1)
        from .checkpoints import save_checkpoint, load_checkpoint, find_checkpoint
        checkpoint_key = object_digest({'revision':revision['sha256'], 'plan':plan_key,
            'audio':p['audio_sha256'], 'render':metadata['stage_fingerprints']['render'],
            'packages':metadata['packages'], 'fps':settings['fps'], 'width':settings['width'],
            'height':settings['height'],'mode':settings['mode'],'seed':p['seed'],
            'motion_engine':motion['engine'],'motion_features':motion_plan.get('feature_key')})
        checkpoint_dir = path / 'checkpoints' / checkpoint_key
        resume_frame = 0
        checkpoint = find_checkpoint(checkpoint_dir, checkpoint_key, start_frame)
        if checkpoint:
            try:
                resume_frame = load_checkpoint(checkpoint, checkpoint_key, writer)
            except (ValueError, OSError, KeyError) as exc:
                # Keep corrupt checkpoint for diagnosis; full replay is safe.
                metadata['checkpoint_warning'] = str(exc)
        metadata['resumed_from_frame'] = resume_frame
        metadata['checkpoint_key'] = checkpoint_key
        columns = ['frame_index', 'time_us', *[k + '_raw' for k in EMOTIONS],
                   *[k + '_effective' for k in EMOTIONS], *['visual_' + k for k in VISUAL_FIELDS], 'revision']
        if motion_enabled:columns += motion_fields
        with open(folder / 'frame_values.csv', 'w', encoding='utf-8-sig', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(columns)
            for i in range(resume_frame, end_frame):
                if i % 30 == 0:
                    report('rendering', i / max(1, end_frame),{'done':i,'total':end_frame,'unit':'帧'})
                t = i / settings['fps']
                vs = VisualState(**{k: int(v) if k in INTEGER_FIELDS else float(v)
                                    for k, v in zip(VISUAL_FIELDS, visual[i])})
                offset, length = round(t * sr), max(1, round(sr / settings['fps']))
                chunk = y[offset:offset + length]
                if len(chunk) < length:
                    chunk = np.pad(chunk, (0, length - len(chunk)))
                beat = float(any(abs(t - b) < 1 / settings['fps'] for b in p['beat_times']))
                period = 60 / max(1., p['tempo'])
                motion_frame = sample_frame(motion_plan,int(ts[i])) if motion_enabled else None
                if motion_enabled:
                    beat=max(beat,float(audio_controls['pulse_strength'][i]))
                    frame = writer.render_frame(vs,chunk,beat,t*.05,(t%period)/period,motion=motion_frame)
                else:
                    frame = writer.render_frame(vs, chunk, beat, t * .05, (t % period) / period)
                if (i + 1) % (settings['fps'] * 10) == 0:
                    save_checkpoint(checkpoint_dir, checkpoint_key, writer)
                if i >= start_frame:
                    writer.write_frame(frame)
                    row=[i, ts[i], *raw_at(raw, [ts[i]], p['step_us'])[0],
                         *effective[i], *[getattr(vs, k) for k in VISUAL_FIELDS], revision['id']]
                    if motion_enabled:
                        record=frame_summary(motion_frame)
                        row += [record['mode'],record['source'],record['reason'],
                                json.dumps(record['components'],ensure_ascii=False,separators=(',',':')),
                                *[record['params'][key] for key in PARAM_BOUNDS],
                                *[audio_controls[key][i] for key in ('rms','onset','activity','pulse_strength')],beat]
                    csv_writer.writerow(row)
                if i == start_frame:
                    import cv2
                    ok, png = cv2.imencode('.png', frame)
                    if not ok:
                        raise RuntimeError('预览截图编码失败')
                    (folder / 'preview.png').write_bytes(png.tobytes())
        writer.release()
        writer = None
        if allow_silent:
            metadata.update(status='diagnostic_only', video_file='video_silent.avi',
                            note='无音轨测试文件，不是最终 MP4')
        else:
            report('encoding', 0.)
            output = folder / 'output.pending.mp4'
            encoding_progress = folder / 'encoding-progress.txt'
            argv = [ff, '-nostdin', '-hide_banner', '-y', '-progress', str(encoding_progress), '-i', str(silent),
                    '-ss', str(metadata['actual_start']), '-i', str(audio),
                    '-t', str((end_frame - start_frame) / settings['fps']),
                    '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-preset', 'fast',
                    '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-af', 'apad',
                    '-movflags', '+faststart', str(output)]
            with open(folder / 'encoder.log', 'w', encoding='utf-8') as log:
                proc = subprocess.Popen(argv, stdout=log, stderr=log,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                try:
                    while True:
                        try:
                            code = proc.wait(timeout=.5)
                            break
                        except subprocess.TimeoutExpired:
                            ratio = 0.
                            try:
                                for line in encoding_progress.read_text().splitlines():
                                    if line.startswith('out_time_us='):
                                        ratio = float(line.split('=',1)[1]) / 1e6 / ((end_frame-start_frame)/settings['fps'])
                            except (OSError, ValueError):
                                pass
                            report('encoding', max(0., min(1.,ratio)))
                except BaseException:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise
            if code != 0 or not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f'视频合并失败，请查看 {folder / "encoder.log"}')
            output.rename(folder / 'output.mp4')
            metadata.update(status='succeeded', video_file='output.mp4',
                            video_sha256=digest(folder / 'output.mp4'))
        metadata.update(completed_at=now(), frames=end_frame - start_frame,
                        csv_file='frame_values.csv', csv_sha256=digest(folder / 'frame_values.csv'))
        atomic_json(folder / 'render.json', metadata)
        report(metadata['status'], 1.)
        return metadata
    except BaseException as exc:
        metadata.update(status='cancelled' if isinstance(exc, Cancelled) else 'failed',
                        error=str(exc), completed_at=now())
        atomic_json(folder / 'render.json', metadata)
        raise
    finally:
        if writer is not None:
            writer.release()
