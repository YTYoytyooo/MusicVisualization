"""Atomic local projects and immutable revision snapshots. No source model writes."""
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
import numpy as np
from .timeline import EMOTIONS, TimedValues, validate_edits

from .paths import ROOT, DEFAULT_PROJECTS
SAFE_ID = re.compile(r'^[a-zA-Z0-9_-]{1,100}$')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def read_json(path):
    with open(path, encoding='utf-8-sig') as stream:
        return json.load(stream)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers/virus scanners may briefly deny replace while the
        # destination handle is open. Keep atomic publication; never truncate
        # the old record or retry persistent permissions indefinitely.
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(.01 * (2 ** attempt), .2))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def project_lock(project):
    # OS-owned advisory locks are released on crashes. The persistent file is
    # only diagnostic metadata, not evidence that a writer is still alive.
    path = Path(project) / '.write.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        if os.fstat(fd).st_size == 0:
            os.write(fd, b' ')
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError:
            raise ValueError('项目正在写入，请等待当前操作完成') from None
        # Refuse to bypass a still-running pre-v2 owner using the old format.
        previous = os.read(fd, 2048).decode('utf-8', errors='replace').strip()
        if previous and previous[0].isdigit():
            from .jobs import process_identity
            if process_identity(int(previous.split()[0])) is not None:
                raise ValueError('旧版服务仍持有项目锁，请先关闭该服务')
        record = json.dumps({'pid': os.getpid(), 'acquired_at': now(), 'lock': 'os-advisory'}).encode()
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, record)
        os.ftruncate(fd, len(record))
        yield
    finally:
        if locked:
            os.lseek(fd, 0, os.SEEK_SET)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def resolve_project(root, project_id):
    if not SAFE_ID.fullmatch(project_id):
        raise ValueError('非法项目 ID')
    path = (Path(root) / project_id).resolve()
    if path.parent != Path(root).resolve():
        raise ValueError('项目路径越界')
    if not (path / 'project.json').is_file():
        raise FileNotFoundError('项目不存在或分析尚未完成')
    return path


def _metadata(path):
    project = read_json(Path(path) / 'project.json')
    if project.get('schema_version') not in (1, 2):
        raise ValueError('不支持的项目格式版本')
    if project.get('schema_version') == 2:
        if object_digest(project['systems']) != project.get('systems_sha256'):
            raise ValueError('作品模型注册记录已改变')
    return project


def _system(project, path, system=None):
    if project.get('schema_version') == 1:
        if system not in (None, 'v2'):
            raise ValueError('此旧项目仅有 V2 分析')
        return 'v2', dict(file='analysis/predictions_raw.npz', label='V2 历史预测',
                          hold_last_to_end=True,
                          step_us=project['step_us'], raw_sha256=project['raw_sha256'],
                          model_path=project.get('model_path', ''), model_sha256=project.get('model_sha256'),
                          alignment=project.get('alignment', 'window-last-frame-v2'))
    if system is None and project.get('current_revision'):
        if not isinstance(project['current_revision'], str) or not re.fullmatch(r'r\d{6}', project['current_revision']):
            raise ValueError('非法当前修订编号')
        current = read_json(Path(path)/'revisions'/project['current_revision']/'revision.json')
        if object_digest({k: v for k, v in current.items() if k != 'sha256'}) != current.get('sha256'):
            raise ValueError('当前修订记录已被外部修改')
        system = current.get('source_system')
    system = system or project['default_system']
    if system not in project['systems']:
        raise ValueError('作品没有此模型结果；请先完成该模型的独立分析')
    return system, project['systems'][system]


def load_project(path, system=None):
    """Project view for a driver; the immutable registry is never rewritten."""
    project = _metadata(path)
    selected, record = _system(project, path, system)
    project.update({k: record.get(k) for k in ('raw_sha256', 'step_us', 'model_path', 'model_sha256', 'alignment')})
    project['hold_last_to_end'] = record.get('hold_last_to_end', False)
    project['source_system'] = selected
    return project


def _contained_file(path, relative):
    candidate = (Path(path)/relative).resolve()
    if not candidate.is_relative_to(Path(path).resolve()):
        raise ValueError('作品文件路径越界')
    return candidate


def load_raw(path, verify=True, system=None):
    path = Path(path)
    p = _metadata(path)
    _, record = _system(p, path, system)
    file = _contained_file(path, record['file'])
    if verify and digest(file) != record['raw_sha256']:
        raise ValueError('原始预测校验失败；拒绝继续生成')
    with np.load(file, allow_pickle=False) as data:
        values = data['values'].copy()
        if p['schema_version'] == 1:
            values = values[:, :2]
        clock = data['times_us'].copy() if 'times_us' in data else np.arange(len(values))*record['step_us']
        mask = data['mask'].copy() if 'mask' in data else np.isfinite(values)
    return TimedValues(values, clock, mask, record['step_us'], p['duration_us'], record.get('hold_last_to_end', False))


def systems_metadata(path, include_values=False):
    p = _metadata(path)
    names = list(p['systems']) if p['schema_version'] == 2 else ['v2']
    result = []
    for name in names:
        _, record = _system(p, path, name)
        row = {k: record.get(k) for k in ('label', 'step_us', 'raw_sha256', 'model_sha256', 'model_path', 'alignment')}
        row['id'] = name
        row['hold_last_to_end'] = record.get('hold_last_to_end', False)
        if include_values:
            raw = load_raw(path, system=name)
            row.update(times_us=raw.times_us.tolist(), mask=raw.mask.tolist(),
                       values=[[float(v) if valid else None for v, valid in zip(values, mask)]
                               for values, mask in zip(np.asarray(raw), raw.mask)])
        result.append(row)
    return result


def load_revision(path, revision_id=None):
    p = _metadata(path)
    revision_id = revision_id or p['current_revision']
    if not re.fullmatch(r'r\d{6}', revision_id):
        raise ValueError('非法修订版本')
    r = read_json(Path(path) / 'revisions' / revision_id / 'revision.json')
    if r.get('id') != revision_id:
        raise ValueError('修订编号与存储位置不匹配')
    if object_digest({k: v for k, v in r.items() if k != 'sha256'}) != r['sha256']:
        raise ValueError('修订记录已被外部修改')
    _, source = _system(p, path, r.get('source_system', 'v2'))
    if r['analysis_sha256'] != source['raw_sha256']:
        raise ValueError('修订引用的分析结果不匹配')
    if p['schema_version'] == 2 and r.get('model_sha256') != source.get('model_sha256'):
        raise ValueError('修订模型身份不匹配')
    if r.get('parent_sha256'):
        parent = r.get('parent')
        if not isinstance(parent, str) or not re.fullmatch(r'r\d{6}', parent):
            raise ValueError('修订父版本非法')
        parent_row = read_json(Path(path)/'revisions'/parent/'revision.json')
        if (parent_row.get('sha256') != r['parent_sha256']
                or object_digest({k: v for k, v in parent_row.items() if k != 'sha256'}) != r['parent_sha256']):
            raise ValueError('修订父版本已改变')
    return r


def save_revision(path, base_revision, edits, smoothing=None, motion=None, source_system=None):
    from .smoothing import validate_smoothing, emotion_values, check_times
    from tools.train_next_v2.v3_workbench.motion_schema import validate_config
    path = Path(path)
    with project_lock(path):
        meta = _metadata(path)
        p = load_project(path, source_system)
        if base_revision != p['current_revision']:
            raise ValueError('项目版本已变化，请重新加载后再保存；不会覆盖新修改')
        base = load_revision(path, base_revision) if base_revision is not None else None
        if motion is None and base:
            motion = base.get('motion')
        motion = validate_config(motion, p['duration_us'])
        edits = validate_edits(edits, p['duration_us'])
        raw = load_raw(path, system=p['source_system'])
        smoothing = validate_smoothing(smoothing)
        ts = check_times(raw, p['duration_us'], edits, p['step_us'], smoothing)
        emotion_values(raw, ts, edits, p['step_us'], smoothing)
        numbers = [int(f.name[1:]) for f in (path / 'revisions').glob('r[0-9]*')
                   if re.fullmatch(r'r\d{6}', f.name)]
        revision_id = f'r{max(numbers, default=-1) + 1:06d}'
        r = {'schema_version': 2, 'id': revision_id, 'parent': base_revision,
             'parent_sha256': base['sha256'] if base else None,
             'created_at': now(), 'analysis_sha256': p['raw_sha256'], 'edits': edits,
             'source_system': p['source_system'], 'model_sha256': p.get('model_sha256'),
             'smoothing': smoothing, 'motion': motion}
        r['sha256'] = object_digest(r)
        folder = path / 'revisions' / revision_id
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(folder / 'revision.json', r)
        meta['current_revision'] = revision_id
        meta['updated_at'] = now()
        atomic_json(path / 'project.json', meta)
    return r


def create_project(root, name, audio_path, model_path, raw, global_info,
                   seed=42, provenance=None, systems=None, source_system=None, publish=True):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32-1:
        raise ValueError('视觉随机种子必须是 0 到 4294967295 的整数')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    audio_path = Path(audio_path).resolve()
    audio_hash = digest(audio_path)
    duration = float(global_info['duration'])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('音频时长必须大于零')
    duration_us = int(round(duration * 1e6))
    if systems is None:
        original = np.asarray(raw, dtype=np.float64)
        if original.ndim != 2 or original.shape[1] not in (2, 5) or not len(original):
            raise ValueError('原始输入需要 N×2 V/A 或可归档的历史 N×5')
        source_system = source_system or 'v2'
        systems = {source_system: dict(values=original[:, :2],
                    times_us=getattr(raw, 'times_us', None), mask=getattr(raw, 'mask', None),
                    step_us=getattr(raw, 'step_us', None) or 100000,
                    hold_last_to_end=getattr(raw, 'hold_last_to_end', source_system == 'v2'),
                    model_path=str(model_path or ''), model_sha256=digest(model_path) if model_path else None,
                    label='合成工程样例' if (provenance or {}).get('kind') == 'synthetic' else source_system.upper(),
                    alignment='explicit-native-clock')}
    source_system = source_system or next(iter(systems))
    if source_system not in systems:
        raise ValueError('初始驱动模型不存在')
    checked = {}
    for key, system in systems.items():
        if not SAFE_ID.fullmatch(key):
            raise ValueError('非法模型标识')
        values, step = system['values'], system.get('step_us', 500000)
        clock = system.get('times_us')
        if clock is None:
            clock = np.arange(len(values))*step
        checked[key] = TimedValues(values, clock, system.get('mask'), step, duration_us, system.get('hold_last_to_end', False))
    project_id = 'p-' + uuid.uuid4().hex[:12]
    # Publish only when the audio, numerical data and initial revision all exist.
    # On failure preserve the hidden staging directory for diagnosis/recovery.
    path = root / ('.pending-' + project_id)
    path.mkdir()
    (path / 'analysis').mkdir()
    (path / 'revisions').mkdir()
    (path / 'renders').mkdir()
    registry = {}
    for key, values in checked.items():
        relative = f'analysis/{key}_raw.npz'
        np.savez_compressed(path/relative, values=np.asarray(values), times_us=values.times_us, mask=values.mask)
        registry[key] = dict(file=relative, raw_sha256=digest(path/relative), step_us=values.step_us,
                             hold_last_to_end=values.hold_last_to_end,
                             label=systems[key].get('label', key.upper()),
                             model_path=systems[key].get('model_path', ''), model_sha256=systems[key].get('model_sha256'),
                             alignment=systems[key].get('alignment', 'explicit-native-clock'))
        with open(path/f'analysis/{key}_raw.csv', 'w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_index', 'time_us', *EMOTIONS, 'valence_valid', 'arousal_valid'])
            writer.writerows([i, int(values.times_us[i]),
                               *[float(v) if valid else '' for v, valid in zip(row, values.mask[i])],
                               *values.mask[i].tolist()] for i, row in enumerate(np.asarray(values)))
    if systems is not None and raw is not None:
        historical = np.asarray(raw)
        if historical.ndim == 2 and historical.shape[1] == 5:
            np.savez_compressed(path/'analysis/legacy_five_dimensions.npz', values=historical)
    # Portable audio copy is local to this new project; never mutate the source.
    import shutil
    asset = path / ('audio' + audio_path.suffix.lower())
    shutil.copy2(audio_path, asset)
    if digest(asset) != audio_hash or digest(audio_path) != audio_hash:
        raise ValueError('复制期间音频发生改变；保留暂存作品，不发布')
    # Compact waveform for the editor. Read with SoundFile, never run ML here.
    import soundfile as sf
    with sf.SoundFile(str(asset)) as stream:
        rate = stream.samplerate
        hop = max(1, int(rate * .05))
        # MP3/VBR header frame counts are estimates. Count decoded PCM through
        # native EOF, as the frozen inference contract does; never scale time.
        peaks, decoded_frames = [], 0
        while True:
            block = stream.read(hop, dtype='float32', always_2d=True)
            if not len(block):break
            peaks.append(float(np.max(np.abs(block))))
            decoded_frames += len(block)
        if abs(decoded_frames / rate - duration) > max(1 / rate, 1 / 22050) + 1e-6:
            raise ValueError('分析时长与音频真实时长不一致；禁止拉伸时间轴')
    atomic_json(path / 'analysis/waveform.json',
                {'times': [i * hop / rate for i in range(len(peaks))], 'peaks': peaks})
    meta = {'schema_version': 2, 'id': project_id, 'name': name or audio_path.stem,
            'created_at': now(), 'updated_at': now(), 'duration': duration,
            'duration_us': duration_us,
            'seed': int(seed), 'audio_file': asset.name, 'source_audio': str(audio_path),
            'audio_sha256': digest(asset), 'systems': registry, 'systems_sha256': object_digest(registry),
            'default_system': source_system, 'current_revision': None,
            'provenance': provenance or {'kind': 'model'},
            'tempo': float(global_info.get('tempo', 0)),
            'beat_times': np.asarray(global_info.get('beat_times', [])).tolist()}
    historical_file = path/'analysis/legacy_five_dimensions.npz'
    if historical_file.is_file():
        meta['provenance'] = {**meta['provenance'],
            'legacy_reference_file':'analysis/legacy_five_dimensions.npz',
            'legacy_reference_sha256':digest(historical_file),
            'legacy_reference_step_us':registry[source_system]['step_us']}
    atomic_json(path / 'project.json', meta)
    save_revision(path, None, [])
    if not publish:
        return path
    published = root / project_id
    path.rename(published)
    return published


def list_projects(root):
    results = []
    for path in Path(root).glob('p-*/project.json'):
        try:
            p = load_project(path.parent)
            if not p.get('current_revision'):
                continue  # Compatibility: hide interrupted pre-staging projects.
            results.append({k: p[k] for k in ('id', 'name', 'duration', 'current_revision')})
        except (ValueError, KeyError, OSError):
            continue
    return results
