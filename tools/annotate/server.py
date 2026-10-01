"""Local human V/A annotation and local audio imports. No model predictions."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent / 'web'
EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
PROTOCOL = 'perceived-musical-expression-va-segment-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def reviewer_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', value):
        raise ValueError('标注者代号请使用 1–40 位英文字母、数字、下划线或短横线')
    # Also avoid Windows reserved directory names.
    if value.upper() in {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(n) for n in range(1,10)), *('LPT'+str(n) for n in range(1,10))}:
        raise ValueError('请使用其他标注者代号')
    return value.lower()


def number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{label} 必须在 {low} 到 {high} 之间')
    return float(value)


def bounded_text(value, limit):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError('文字字段格式错误或过长')
    return value.strip()


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, audio_root, output):
        self.audio_root = Path(audio_root).resolve()
        self.output = Path(output).resolve()
        if not self.audio_root.is_dir():
            raise ValueError('音频目录不存在')
        if self.output == self.audio_root or self.output.is_relative_to(self.audio_root):
            raise ValueError('标注输出目录不能放在音频目录内')
        self.output.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.RLock()
        self.files = {}
        self.hashes = {}
        for p in sorted(self.audio_root.rglob('*')):
            if p.is_file() and p.suffix.lower() in EXTENSIONS and p.resolve().is_relative_to(self.audio_root):
                relative = p.relative_to(self.audio_root).as_posix()
                key = hashlib.sha256(relative.encode()).hexdigest()[:24]
                self.files[key] = (p, relative)
        # Prevent independent servers from concurrently editing this output tree.
        self.lock = open(self.output / '.server.lock', 'a+b')
        try:
            self.lock.seek(0)
            if os.name == 'nt':
                import msvcrt
                if self.lock.read(1) == b'':
                    self.lock.write(b'1'); self.lock.flush()
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ValueError('此标注目录已有服务运行，请使用现有网页或先关闭旧服务') from None

    def close(self):
        self.lock.close()

    def source(self, key):
        if key not in self.files:
            raise ValueError('音乐不存在；添加新文件后请重启标注工具')
        path, relative = self.files[key]
        if not path.resolve().is_relative_to(self.audio_root):
            raise ValueError('音频路径已改变')
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        old = self.hashes.get(key)
        if old and old[0] == signature:
            return old[1].copy()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if signature != (path.stat().st_size, path.stat().st_mtime_ns):
            raise Conflict('音频正在变化，请稍后重新打开')
        duration = None
        try:
            import soundfile as sf
            duration = float(sf.info(path).duration)
        except (ImportError, OSError, RuntimeError):
            pass  # Browser metadata is the fallback for unsupported formats.
        value = {'song_id': key, 'audio_relative_path': relative, 'audio_root': str(self.audio_root),
                 'audio_sha256': digest.hexdigest(), 'duration': duration}
        self.hashes[key] = signature, value
        return value.copy()

    def file(self, reviewer, key):
        reviewer = reviewer_id(reviewer)
        if key not in self.files:
            raise ValueError('音乐不存在')
        return self.output / reviewer / (key + '.json')

    def load(self, reviewer, key):
        with self.mutex:
            path = self.file(reviewer, key)
            source = self.source(key)
            if path.is_file():
                doc = json.loads(path.read_text(encoding='utf-8'))
                if doc['source']['audio_sha256'] != source['audio_sha256']:
                    raise Conflict('音频内容与旧标注不一致。请恢复原音频，或使用新的标注输出目录')
                return doc
            return {'schema_version': 1, 'protocol': PROTOCOL, 'label_scope': 'segment_mean', 'revision': 0,
                    'reviewer': reviewer_id(reviewer), 'source': source,
                    'duration': source['duration'], 'artist': '', 'style': '', 'segments': []}

    def catalog(self, reviewer):
        reviewer = reviewer_id(reviewer)
        items = []
        with self.mutex:
            for key, (_, relative) in self.files.items():
                path = self.file(reviewer, key)
                counts = {'annotated': 0, 'uncertain': 0, 'skip': 0, 'pending': 0}
                if path.is_file():
                    doc = json.loads(path.read_text(encoding='utf-8'))
                    for segment in doc['segments']:
                        counts[segment['status']] += 1
                items.append({'id': key, 'name': relative, 'counts': counts})
        return items

    def save(self, reviewer, key, body):
        with self.mutex:
            current = self.load(reviewer, key)
            if type(body.get('base_revision')) is not int or body['base_revision'] != current['revision']:
                raise Conflict('此标注已在其他页面更新。请保留当前草稿，刷新后核对；未覆盖磁盘数据')
            source = self.source(key)
            if body.get('audio_sha256') != source['audio_sha256']:
                raise Conflict('音频身份不一致，请重新打开音乐')
            duration = number(body.get('duration'), '音乐时长', .01, 86400)
            expected = source['duration'] or current['duration']
            if expected is not None and abs(duration - expected) > .15:
                raise ValueError('音乐时长与已读取的音频不一致，请重新加载')
            duration = expected or duration
            values = body.get('segments')
            if not isinstance(values, list) or len(values) > 20000:
                raise ValueError('片段列表无效或过长')
            segments, ids = [], set()
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError('片段格式错误')
                segment_id = value.get('id')
                if not isinstance(segment_id, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', segment_id) or segment_id in ids:
                    raise ValueError('片段 ID 错误或重复')
                ids.add(segment_id)
                start = number(value.get('start'), '开始时间', 0, duration)
                end = number(value.get('end'), '结束时间', 0, duration)
                if end - start < .01:
                    raise ValueError('片段结束必须晚于开始，且至少 0.01 秒')
                status = value.get('status')
                if status not in ('pending', 'annotated', 'uncertain', 'skip'):
                    raise ValueError('片段状态错误')
                valence = arousal = None
                confidence = value.get('confidence', '')
                if status == 'annotated':
                    valence = number(value.get('valence'), '愉悦度', -1, 1)
                    arousal = number(value.get('arousal'), '激烈程度', -1, 1)
                    if confidence not in ('high', 'medium', 'low'):
                        raise ValueError('请明确选择把握程度')
                elif value.get('valence') is not None or value.get('arousal') is not None:
                    raise ValueError('不确定、跳过和待标片段不能携带数值标签')
                vocals = value.get('vocals', '')
                if vocals not in ('', 'instrumental', 'vocals', 'mixed'):
                    raise ValueError('人声类型错误')
                segments.append({'id': segment_id, 'start': start, 'end': end, 'status': status,
                                 'valence': valence, 'arousal': arousal,
                                 'confidence': confidence if status == 'annotated' else '',
                                 'vocals': vocals, 'note': bounded_text(value.get('note', ''), 2000)})
            segments.sort(key=lambda s: (s['start'], s['end']))
            if any(a['end'] > b['start'] + 1e-6 for a, b in zip(segments, segments[1:])):
                raise ValueError('片段不能重叠；请先调整边界')
            doc = {'schema_version': 1, 'protocol': PROTOCOL, 'label_scope': 'segment_mean', 'revision': current['revision']+1,
                   'reviewer': reviewer_id(reviewer), 'source': source, 'duration': duration,
                   'artist': bounded_text(body.get('artist', ''), 200), 'style': bounded_text(body.get('style', ''), 200),
                   'segments': segments, 'created_at': current.get('created_at', now()), 'updated_at': now()}
            path = self.file(reviewer, key)
            if current['revision']:
                history = path.parent/'history'/key/f"r{current['revision']:06}.json"
                if not history.exists():
                    atomic_json(history, current)
            atomic_json(path, doc)
            return doc

    def export(self, reviewer, kind):
        if kind not in ('all', 'valid', 'json'):
            raise ValueError('导出类型错误')
        reviewer = reviewer_id(reviewer)
        with self.mutex:
            docs = [self.load(reviewer, key) for key in self.files if self.file(reviewer, key).is_file()]
        if kind == 'json':
            return json.dumps({'schema_version': 1, 'protocol': PROTOCOL, 'exported_at': now(), 'songs': docs}, ensure_ascii=False, indent=2).encode('utf-8')
        columns = ['protocol', 'label_scope', 'reviewer', 'song_id', 'audio_sha256', 'audio_relative_path', 'artist', 'style',
                   'segment_id', 'start_seconds', 'end_seconds', 'status', 'valence', 'arousal', 'confidence', 'vocals', 'note', 'revision']
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(columns)
        for doc in docs:
            src = doc['source']
            for s in doc['segments']:
                if kind == 'valid' and (s['status'] != 'annotated' or s['confidence'] == 'low'):
                    continue
                row = [PROTOCOL, 'segment_mean', reviewer, src['song_id'], src['audio_sha256'], src['audio_relative_path'], doc['artist'], doc['style'],
                       s['id'], s['start'], s['end'], s['status'], s['valence'], s['arousal'], s['confidence'], s['vocals'], s['note'], doc['revision']]
                # Avoid executing notes or filenames as formulas when opened in Excel.
                writer.writerow([("'"+v if v and v.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else v) if isinstance(v, str) else v for v in row])
        return stream.getvalue().encode('utf-8-sig')


CONTINUOUS_PROTOCOL = 'perceived-musical-expression-va-continuous-v1'


def overwrite_tracks(takes):
    """Later observations replace only their axis and observed closed time span."""
    tracks = []
    for take in takes:
        for axis in take.get('axes', ['valence', 'arousal']):
            start, end = take['points'][0]['time'], take['points'][-1]['time']
            remaining = []
            for old in tracks:
                if old['axes'] != [axis] or old['points'][-1]['time'] < start or old['points'][0]['time'] > end:
                    remaining.append(old)
                    continue
                for side, points in [('left', [p for p in old['points'] if p['time'] < start]),
                                     ('right', [p for p in old['points'] if p['time'] > end])]:
                    if points:
                        fragment = hashlib.sha256(f"{old['id']}:{side}:{start}:{end}".encode()).hexdigest()[:32]
                        remaining.append(dict(id=fragment, axes=[axis], points=points))
            key = take['id'] if len(take.get('axes', ['valence', 'arousal'])) == 1 else hashlib.sha256(f"{take['id']}:{axis}".encode()).hexdigest()[:32]
            points = [{**p, 'valence': p['valence'] if axis == 'valence' else None,
                       'arousal': p['arousal'] if axis == 'arousal' else None} for p in take['points']]
            tracks = remaining + [dict(id=key, axes=[axis], points=points)]
    return tracks


class ContinuousStore(Store):
    """Whole-song trajectories with latest-observation precedence per axis."""
    def import_audio(self, name, stream, length):
        if not isinstance(name, str) or not name or len(name) > 240 or any(c in name for c in '/\\:'):
            raise ValueError('文件名无效')
        suffix = Path(name).suffix.lower()
        if suffix not in EXTENSIONS or not 0 < length <= 512*1024*1024:
            raise ValueError('请选择支持的音频文件，每个文件不超过 512 MB')
        folder = self.audio_root/'imports'
        if not folder.resolve().is_relative_to(self.audio_root):
            raise ValueError('导入路径无效')
        folder.mkdir(exist_ok=True)
        fd, temporary = tempfile.mkstemp(suffix='.part', dir=folder)
        try:
            digest = hashlib.sha256()
            with os.fdopen(fd, 'wb') as output:
                remaining = length
                while remaining:
                    block = stream.read(min(1024*1024, remaining))
                    if not block: raise ValueError('文件传输不完整，请重试')
                    output.write(block); digest.update(block); remaining -= len(block)
                output.flush(); os.fsync(output.fileno())
            # Recognize supported containers without loading model dependencies.
            with open(temporary, 'rb') as source: header = source.read(16)
            if not (header.startswith((b'RIFF', b'fLaC', b'OggS', b'ID3')) or header[4:8] == b'ftyp' or
                    (len(header)>1 and header[0] == 255 and header[1] & 224 == 224)):
                raise ValueError('文件内容不是可识别的音频格式')
            stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', Path(name).stem).strip(' .')[:80] or 'audio'
            target = folder/f"{stem}-{digest.hexdigest()}{suffix}"
            if not target.resolve().is_relative_to(folder.resolve()): raise ValueError('导入路径无效')
            with self.mutex:
                if not target.exists(): os.replace(temporary, target)
                relative = target.relative_to(self.audio_root).as_posix()
                key = hashlib.sha256(relative.encode()).hexdigest()[:24]
                self.files[key] = target, relative
            return {'id': key, 'name': relative}
        finally:
            if os.path.exists(temporary): os.unlink(temporary)

    def load(self, reviewer, key):
        doc = super().load(reviewer, key)
        if doc['revision']:
            if doc.get('protocol') != CONTINUOUS_PROTOCOL:
                raise Conflict('此目录含旧分段数据，请使用独立的连续标注输出目录')
            return doc
        doc.pop('segments')
        doc.update(schema_version=2, protocol=CONTINUOUS_PROTOCOL,
                   label_scope='continuous_raw', takes=[], transitions=[], note='', completed={'valence': False, 'arousal': False})
        return doc

    def catalog(self, reviewer):
        reviewer = reviewer_id(reviewer)
        with self.mutex:
            items = []
            for key, (_, relative) in self.files.items():
                path = self.file(reviewer, key)
                doc = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
                if doc and doc.get('protocol') != CONTINUOUS_PROTOCOL:
                    raise Conflict('此目录含旧分段数据，请使用独立的连续标注输出目录')
                items.append({'id': key, 'name': relative,
                              'completed': doc.get('completed', {'valence': False, 'arousal': False}),
                              'has_annotations': bool(doc.get('takes') or doc.get('transitions')),
                              'unrated': sum(p.get('status') == 'annotated' and not p.get('confidence') for t in doc.get('takes', []) for p in t['points']),
                              'counts': {'takes': len(doc.get('takes', [])),
                                         'transitions': len(doc.get('transitions', []))}})
            return items

    def save(self, reviewer, key, body):
        with self.mutex:
            current = self.load(reviewer, key)
            if type(body.get('base_revision')) is not int or body['base_revision'] != current['revision']:
                raise Conflict('其他页面已更新此歌，请保留草稿并刷新核对；未覆盖磁盘数据')
            if body.get('audio_sha256') != current['source']['audio_sha256']:
                raise Conflict('音频身份不一致')
            duration = number(body.get('duration'), '音乐时长', .01, 86400)
            expected = current['source']['duration'] or current['duration']
            if expected is not None and abs(duration-expected) > .15:
                raise ValueError('音乐时长不一致')
            ids = set()
            def identity(value):
                key = value.get('id')
                if not isinstance(key, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', key) or key in ids:
                    raise ValueError('记录 ID 错误或重复')
                ids.add(key)
                return key
            takes, markers = body.get('takes'), body.get('transitions')
            if not isinstance(takes, list) or not isinstance(markers, list) or len(takes)>2000 or len(markers)>10000:
                raise ValueError('记录列表无效')
            clean_takes, clean_markers, count = [], [], 0
            for take in takes:
                if not isinstance(take, dict): raise ValueError('记录格式错误')
                take_id = identity(take)
                axes = take.get('axes', ['valence', 'arousal'])
                if not isinstance(axes, list) or not axes or len(set(axes)) != len(axes) or any(a not in ('valence', 'arousal') for a in axes):
                    raise ValueError('请选择效价、唤醒度或两者')
                points = take.get('points')
                if not isinstance(points, list) or not points: raise ValueError('记录不能为空')
                cleaned, previous = [], -1
                for point in points:
                    count += 1
                    if count > 50000 or not isinstance(point, dict): raise ValueError('采样数据无效或过长')
                    t = number(point.get('time'), '时间', 0, duration)
                    if t <= previous: raise ValueError('同次记录的时间必须严格递增')
                    previous = t
                    status = point.get('status')
                    if status not in ('annotated', 'uncertain'): raise ValueError('采样状态错误')
                    v = a = None
                    confidence = ''
                    if status == 'annotated':
                        v = number(point.get('valence'), '效价', -1, 1) if 'valence' in axes else None
                        a = number(point.get('arousal'), '唤醒度', -1, 1) if 'arousal' in axes else None
                        if any(point.get(axis) is not None for axis in ('valence', 'arousal') if axis not in axes):
                            raise ValueError('未选维度不能携带数值')
                        confidence = point.get('confidence', '')
                        if confidence not in ('', 'high', 'medium', 'low'): raise ValueError('把握程度无效')
                    elif point.get('valence') is not None or point.get('arousal') is not None:
                        raise ValueError('不确定时不能携带数值')
                    cleaned.append(dict(time=t, status=status, valence=v, arousal=a, confidence=confidence))
                clean_takes.append(dict(id=take_id, axes=axes, points=cleaned))
            for marker in markers:
                if not isinstance(marker, dict): raise ValueError('转折格式错误')
                marker_id = identity(marker)
                kind = marker.get('kind', '')  # Preserve legacy metadata; new markers need no type.
                if kind not in ('', 'sudden', 'gradual'): raise ValueError('转折类型错误')
                cleaned_marker = dict(id=marker_id, time=number(marker.get('time'), '转折时间', 0, duration),
                                      note=bounded_text(marker.get('note', ''), 2000))
                if kind: cleaned_marker['kind'] = kind
                clean_markers.append(cleaned_marker)
            clean_takes = overwrite_tracks(clean_takes)
            completed = body.get('completed', current.get('completed', {'valence': False, 'arousal': False}))
            if not isinstance(completed, dict) or set(completed) != {'valence', 'arousal'} or any(type(v) is not bool for v in completed.values()):
                raise ValueError('完成状态无效')
            for axis, done in completed.items():
                if done and not any(axis in t['axes'] and any(p['status']=='annotated' for p in t['points']) for t in clean_takes):
                    raise ValueError('请先记录此维度，再标记完成')
            doc = {**current, 'revision': current['revision']+1, 'duration': duration, 'completed': completed,
                   'takes': clean_takes, 'transitions': sorted(clean_markers, key=lambda x:x['time']),
                   'artist': bounded_text(body.get('artist', ''), 200),
                   'style': bounded_text(body.get('style', ''), 200),
                   'note': bounded_text(body.get('note', ''), 2000),
                   'capture': {'target_interval_seconds': .25, 'clock': 'audio_currentTime',
                               'reaction_delay_correction': None, 'overlap_policy': 'latest_per_axis_interval'},
                   'created_at': current.get('created_at', now()), 'updated_at': now()}
            path = self.file(reviewer, key)
            if current['revision']:
                history = path.parent/'history'/key/f"r{current['revision']:06}.json"
                if not history.exists(): atomic_json(history, current)
            atomic_json(path, doc)
            return doc

    def export(self, reviewer, kind):
        if kind not in ('valid', 'all', 'json'): raise ValueError('导出类型错误')
        with self.mutex:
            docs = [self.load(reviewer, key) for key in self.files if self.file(reviewer, key).is_file()]
        docs = [{**doc, 'takes': overwrite_tracks(doc['takes']),
                 'capture': {**doc.get('capture', {}), 'overlap_policy': 'latest_per_axis_interval'}} for doc in docs]
        if kind == 'json':
            return json.dumps(dict(schema_version=2, protocol=CONTINUOUS_PROTOCOL, songs=docs), ensure_ascii=False).encode('utf-8')
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['protocol','reviewer','song_id','audio_sha256','audio_relative_path','artist','style',
                         'record_type','take_id','time_seconds','status','valence','arousal','confidence','transition_kind','note','revision','axis'])
        for doc in docs:
            source = doc['source']
            prefix = [CONTINUOUS_PROTOCOL, doc['reviewer'], source['song_id'], source['audio_sha256'], source['audio_relative_path'], doc['artist'], doc['style']]
            rows = []
            for take in overwrite_tracks(doc['takes']):
                for p in take['points']:
                    if kind == 'valid' and (p['status'] != 'annotated' or p['confidence'] not in ('high', 'medium')): continue
                    rows.append((['sample',take['id'],p['time'],p['status'],p['valence'],p['arousal'],p['confidence'],'',doc['note']], take['axes'][0]))
            for m in doc['transitions']:
                rows.append((['transition','',m['time'],'','','','',m.get('kind',''),m['note']], ''))
            for row, axis in rows:
                writer.writerow([("'"+v if v and v.lstrip().startswith(('=','+','-','@','\t','\r')) else v) if isinstance(v,str) else v for v in prefix+row+[doc['revision'], axis]])
        return stream.getvalue().encode('utf-8-sig')


def make_server(audio_root=ROOT/'assets/audio', output=ROOT/'data/annotations/continuous', port=8766, mode='continuous'):
    store = (Store if mode == 'segments' else ContinuousStore)(audio_root, output)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_bytes(self, value, content_type='application/json; charset=utf-8', status=200, filename=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(value)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(value)

        def json(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'), status=status)

        def allowed(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def media(self, path):
            total = path.stat().st_size
            start, end = 0, total-1
            ranged = self.headers.get('Range')
            if ranged:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', ranged)
                if not match or not any(match.groups()):
                    self.send_error(416); return
                a, b = match.groups()
                if a:
                    start = int(a); end = min(int(b), end) if b else end
                else:
                    start = max(0, total-int(b))
                if start > end or start >= total:
                    self.send_error(416); return
            self.send_response(206 if ranged else 200)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end-start+1))
            if ranged:
                self.send_header('Content-Range', f'bytes {start}-{end}/{total}')
            self.end_headers()
            with path.open('rb') as stream:
                stream.seek(start)
                remaining = end-start+1
                while remaining > 0:
                    block = stream.read(min(65536, remaining))
                    if not block: break
                    self.wfile.write(block); remaining -= len(block)

        def do_GET(self):
            if not self.allowed():
                self.json({'error': '仅允许本机访问'}, 403); return
            try:
                url = urlparse(self.path)
                args = parse_qs(url.query)
                reviewer = args.get('reviewer', ['me'])[0]
                if url.path == '/api/library':
                    self.json({'songs': store.catalog(reviewer), 'token': token, 'output': str(store.output), 'protocol': CONTINUOUS_PROTOCOL if mode == 'continuous' else PROTOCOL})
                elif url.path.startswith('/api/song/'):
                    self.json(store.load(reviewer, url.path.rsplit('/',1)[1]))
                elif url.path == '/api/export':
                    kind = args.get('kind',['valid'])[0]
                    data = store.export(reviewer, kind)
                    self.send_bytes(data, 'application/json; charset=utf-8' if kind == 'json' else 'text/csv; charset=utf-8',
                                    filename='annotations-'+kind+('.json' if kind == 'json' else '.csv'))
                elif url.path.startswith('/audio/'):
                    key = url.path.rsplit('/',1)[1]
                    store.source(key)
                    self.media(store.files[key][0])
                elif url.path in ('/', '/app.js', '/style.css'):
                    name = 'index.html' if url.path == '/' else url.path[1:]
                    self.send_bytes((WEB/name).read_bytes(), {'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','style.css':'text/css; charset=utf-8'}[name])
                else:
                    self.json({'error':'未找到'},404)
            except Conflict as exc:
                self.json({'error':str(exc)},409)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self.json({'error':str(exc)},400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.json({'error':str(exc)},500)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if not self.allowed() or origin not in (None, f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}') or not secrets.compare_digest(self.headers.get('X-Annotation-Token',''),token):
                self.json({'error':'会话已失效，请刷新网页'},403); return
            try:
                length = int(self.headers.get('Content-Length','0'))
                url = urlparse(self.path)
                if url.path == '/api/import' and isinstance(store, ContinuousStore):
                    name = parse_qs(url.query).get('name', [''])[0]
                    self.json(store.import_audio(name, self.rfile, length))
                    return
                if not 0 < length <= 8_000_000:
                    raise ValueError('请求大小无效')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('请求格式错误')
                if self.path != '/api/save':
                    self.json({'error':'未找到'},404); return
                self.json(store.save(body.get('reviewer'), body.get('song_id'), body))
            except Conflict as exc:
                self.json({'error':str(exc)},409)
            except (ValueError, KeyError, TypeError) as exc:
                self.json({'error':str(exc)},400)
            except Exception as exc:
                self.json({'error':str(exc)},500)

    try:
        server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    except Exception:
        store.close(); raise
    server.store = store
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description='本地音乐情绪标注工具；音乐与标注不会上传')
    parser.add_argument('--audio-dir', type=Path, default=ROOT/'assets/audio')
    parser.add_argument('--output', type=Path, default=ROOT/'data/annotations/continuous')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server = None
    try:
        server = make_server(args.audio_dir, args.output, args.port)
        print(f'Annotation: http://127.0.0.1:{server.server_port}', flush=True)
        print(f'Audio files: {len(server.store.files)} | Labels: {server.store.output}', flush=True)
        print('Keep this terminal open. Ctrl+C stops the server.', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError) as exc:
        print(f'ERROR: {exc}', flush=True)
        return 1
    finally:
        if server:
            server.server_close(); server.store.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
