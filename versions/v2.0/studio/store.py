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
from .timeline import EMOTIONS, emotion_at, validate_edits, validation_times

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


def load_project(path):
    project = read_json(Path(path) / 'project.json')
    if project.get('schema_version') != 1:
        raise ValueError('不支持的项目格式版本')
    return project


def load_raw(path, verify=True):
    path = Path(path)
    p = load_project(path)
    if verify and digest(path / 'analysis' / 'predictions_raw.npz') != p['raw_sha256']:
        raise ValueError('原始预测校验失败；拒绝继续生成')
    with np.load(path / 'analysis' / 'predictions_raw.npz', allow_pickle=False) as data:
        return data['values'].copy()


def load_revision(path, revision_id=None):
    p = load_project(path)
    revision_id = revision_id or p['current_revision']
    if not re.fullmatch(r'r\d{6}', revision_id):
        raise ValueError('非法修订版本')
    r = read_json(Path(path) / 'revisions' / revision_id / 'revision.json')
    if r['analysis_sha256'] != p['raw_sha256']:
        raise ValueError('修订引用的分析结果不匹配')
    if object_digest({k: v for k, v in r.items() if k != 'sha256'}) != r['sha256']:
        raise ValueError('修订记录已被外部修改')
    return r


def save_revision(path, base_revision, edits, smoothing=None, motion=None):
    from .smoothing import validate_smoothing, emotion_values, check_times
    from motion_schema import validate_config
    path = Path(path)
    with project_lock(path):
        p = load_project(path)
        if base_revision != p['current_revision']:
            raise ValueError('项目版本已变化，请重新加载后再保存；不会覆盖新修改')
        if motion is None and base_revision is not None:
            motion = load_revision(path, base_revision).get('motion')
        motion = validate_config(motion, p['duration_us'])
        edits = validate_edits(edits, p['duration_us'])
        raw = load_raw(path)
        ts = validation_times(len(raw), p['duration_us'], edits, p['step_us'])
        smoothing = validate_smoothing(smoothing)
        ts = check_times(raw, p['duration_us'], edits, p['step_us'], smoothing)
        emotion_values(raw, ts, edits, p['step_us'], smoothing)
        numbers = [int(f.name[1:]) for f in (path / 'revisions').glob('r[0-9]*')
                   if re.fullmatch(r'r\d{6}', f.name)]
        revision_id = f'r{max(numbers, default=-1) + 1:06d}'
        r = {'schema_version': 1, 'id': revision_id, 'parent': base_revision,
             'created_at': now(), 'analysis_sha256': p['raw_sha256'], 'edits': edits,
             'smoothing': smoothing, 'motion': motion}
        r['sha256'] = object_digest(r)
        folder = path / 'revisions' / revision_id
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(folder / 'revision.json', r)
        p['current_revision'] = revision_id
        p['updated_at'] = now()
        atomic_json(path / 'project.json', p)
    return r


def create_project(root, name, audio_path, model_path, raw, global_info,
                   seed=42, provenance=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    audio_path = Path(audio_path).resolve()
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 5 or not len(raw) or not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1:
        raise ValueError('原始预测必须为非空 N×5 有限数值，范围 [-1,1]')
    duration = float(global_info['duration'])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('音频时长必须大于零')
    project_id = 'p-' + uuid.uuid4().hex[:12]
    # Publish only when the audio, numerical data and initial revision all exist.
    # On failure preserve the hidden staging directory for diagnosis/recovery.
    path = root / ('.pending-' + project_id)
    path.mkdir()
    (path / 'analysis').mkdir()
    (path / 'revisions').mkdir()
    (path / 'renders').mkdir()
    np.savez_compressed(path / 'analysis/predictions_raw.npz', values=raw)
    # Portable audio copy is local to this new project; never mutate the source.
    import shutil
    asset = path / ('audio' + audio_path.suffix.lower())
    shutil.copy2(audio_path, asset)
    # Compact waveform for the editor. Read with SoundFile, never run ML here.
    import soundfile as sf
    with sf.SoundFile(str(asset)) as stream:
        rate = stream.samplerate
        hop = max(1, int(rate * .05))
        peaks = [float(np.max(np.abs(block)))
                 for block in stream.blocks(blocksize=hop, dtype='float32', always_2d=True)]
    atomic_json(path / 'analysis/waveform.json',
                {'times': [i * hop / rate for i in range(len(peaks))], 'peaks': peaks})
    meta = {'schema_version': 1, 'id': project_id, 'name': name or audio_path.stem,
            'created_at': now(), 'updated_at': now(), 'duration': duration,
            'duration_us': int(round(duration * 1000000)), 'step_us': 100000,
            'seed': int(seed), 'audio_file': asset.name, 'source_audio': str(audio_path),
            'audio_sha256': digest(asset), 'model_path': str(model_path or ''),
            'model_sha256': digest(model_path) if model_path else None,
            'raw_sha256': digest(path / 'analysis/predictions_raw.npz'),
            'alignment': 'window-last-frame-v2', 'current_revision': None,
            'provenance': provenance or {'kind': 'model'},
            'tempo': float(global_info['tempo']),
            'beat_times': np.asarray(global_info['beat_times']).tolist()}
    with open(path / 'analysis/predictions_raw.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_index', 'time_us', *EMOTIONS])
        writer.writerows([i, i * 100000, *row] for i, row in enumerate(raw))
    atomic_json(path / 'project.json', meta)
    save_revision(path, None, [])
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
