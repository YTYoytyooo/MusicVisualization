"""Loopback-only standard-library HTTP adapter; no framework install required."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import mimetypes
from pathlib import Path
import secrets
from urllib.parse import urlparse, parse_qs, unquote
from .store import (ROOT, DEFAULT_PROJECTS, list_projects, load_project, load_revision,
                    read_json, resolve_project, save_revision, project_lock)
from .pipeline import (timeline_payload, validate_render_options, validate_preview,
                       cached_visual_plan, planning_key)
from .jobs import JobManager, validate_consumer, validate_consumer_seq


def project_payload(path, revision_id=None):
    p = load_project(path)
    r = load_revision(path, revision_id)
    revisions = [load_revision(path, f.name) for f in sorted((path / 'revisions').glob('r[0-9]*'))]
    renders = []
    for f in sorted((path / 'renders').glob('v-*/render.json'), reverse=True):
        item = read_json(f)
        prefix = f"/media/{p['id']}/renders/{item['id']}"
        if item.get('video_file'):
            item['video_url'] = prefix + '/' + item['video_file']
        if item.get('csv_file'):
            item['csv_url'] = prefix + '/' + item['csv_file']
        if (f.parent / 'preview.png').is_file():
            item['preview_url'] = prefix + '/preview.png'
        renders.append(item)
    # Consumers choose the last matching item as the newest. UUID order is random.
    renders.sort(key=lambda item: (item.get('created_at', ''), item['id']))
    return {'project': p, 'revision': r, 'revisions': revisions, 'renders': renders,
            'timeline': timeline_payload(path, revision=r['id'])}


def make_server(projects=DEFAULT_PROJECTS, port=8765, start_queue=True):
    root = Path(projects).resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    manager = JobManager(root) if start_queue else None
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def allowed_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',
                                               f'localhost:{self.server.server_port}')

        def json_response(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path):
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            requested = self.headers.get('Range')
            if requested:
                import re
                match = re.fullmatch(r'bytes=(\d+)-(\d*)', requested)
                if not match:
                    self.json_response({'error': 'Unsupported range'}, 416)
                    return
                start = int(match[1])
                end = min(size - 1, int(match[2]) if match[2] else size - 1)
                if not 0 <= start <= end:
                    self.json_response({'error': 'Range outside file'}, 416)
                    return
                status = 206
            self.send_response(status)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(max(0, end - start + 1)))
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            if path.suffix == '.html':
                self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            with open(path, 'rb') as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self):
            try:
                if not self.allowed_host():
                    self.json_response({'error': '只允许本机访问'}, 403)
                    return
                url = urlparse(self.path)
                parts = unquote(url.path).strip('/').split('/')
                if url.path == '/api/bootstrap':
                    self.json_response({'projects': list_projects(root), 'token': token})
                elif parts == ['api', 'jobs']:
                    self.json_response({'jobs': manager.list() if manager else []})
                elif len(parts) == 4 and parts[:2] == ['api', 'jobs'] and parts[3] == 'log':
                    if manager is None:
                        self.json_response({'error': '当前服务未启动任务队列'}, 503)
                        return
                    limit = int(parse_qs(url.query).get('limit', ['16384'])[0])
                    self.json_response(manager.log(parts[2], limit))
                elif len(parts) == 3 and parts[:2] == ['api', 'projects']:
                    project = resolve_project(root, parts[2])
                    revision = parse_qs(url.query).get('revision', [None])[0]
                    self.json_response(project_payload(project, revision))
                elif len(parts) >= 3 and parts[0] == 'media':
                    project = resolve_project(root, parts[1])
                    if parts[2:] == ['audio']:
                        path = project / load_project(project)['audio_file']
                    else:
                        if len(parts) != 5 or parts[2] != 'renders' or parts[4] not in ('output.mp4', 'video_silent.avi', 'frame_values.csv', 'preview.png', 'predictions_effective.csv', 'motion_plan.csv', 'motion_plan.json'):
                            raise ValueError('非法媒体路径')
                        path = (project / 'renders' / parts[3] / parts[4]).resolve()
                    if not path.resolve().is_relative_to(project.resolve()):
                        raise ValueError('媒体路径越界')
                    self.send_file(path)
                elif url.path in ('/', '/index.html', '/app.js', '/timeline-math.js', '/motion-math.js', '/style.css'):
                    self.send_file(ROOT / 'web' / ('index.html' if url.path == '/' else url.path[1:]))
                else:
                    self.json_response({'error': 'Not found'}, 404)
            except (ValueError, KeyError) as exc:
                self.json_response({'error': str(exc)}, 400)
            except FileNotFoundError as exc:
                self.json_response({'error': str(exc)}, 404)
            except (ConnectionError, BrokenPipeError):
                pass
            except Exception as exc:
                self.json_response({'error': str(exc)}, 500)

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                allowed = (None, f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}')
                if not self.allowed_host() or origin not in allowed or not secrets.compare_digest(self.headers.get('X-Studio-Token', ''), token):
                    self.json_response({'error': '请求未授权；请刷新本机编辑器'}, 403)
                    return
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('需要 application/json')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError('请求体大小非法')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('请求体必须为对象')
                parts = urlparse(self.path).path.strip('/').split('/')
                if parts == ['api', 'plan-release']:
                    if manager is None:
                        self.json_response({'error': '当前服务未启动规划队列'}, 503)
                        return
                    self.json_response(manager.release_plan(body.get('consumer_id'), body.get('consumer_seq')))
                elif parts == ['api', 'projects']:
                    if manager is None:
                        self.json_response({'error': '当前服务未启动任务队列'}, 503)
                        return
                    from .timeline import integer
                    integer(body.get('seed', 42), 'seed')
                    for field in ('audio_path', 'model_path'):
                        if not isinstance(body.get(field), str) or not Path(body[field]).is_file():
                            raise ValueError(f'{field} 文件不存在')
                    self.json_response(manager.submit('analyze', body), 202)
                elif len(parts) == 4 and parts[:2] == ['api', 'projects']:
                    path = resolve_project(root, parts[2])
                    if parts[3] == 'revisions':
                        base = load_revision(path, body['base_revision'])
                        smoothing = body.get('smoothing')
                        motion_args = {'motion':base.get('motion') if body['motion'] is None else body['motion']} if 'motion' in body else (
                            {'motion':base['motion']} if base.get('motion',{}).get('engine')=='flow-v1' else {})
                        validate_preview(path, body['edits'], smoothing=smoothing, **motion_args)
                        self.json_response(save_revision(path, body['base_revision'], body['edits'],
                                                         smoothing=smoothing, **motion_args), 201)
                    elif parts[3] == 'preview':
                        base = load_revision(path, body['base_revision'])
                        motion_args = {'motion':base.get('motion') if body['motion'] is None else body['motion']} if 'motion' in body else (
                            {'motion':base['motion']} if base.get('motion',{}).get('engine')=='flow-v1' else {})
                        self.json_response(validate_preview(path, body['edits'], smoothing=body.get('smoothing'), **motion_args))
                    elif parts[3] == 'motion-plan':
                        base = load_revision(path, body['base_revision'])
                        from .store import load_raw
                        from .timeline import validate_edits
                        from .smoothing import validate_smoothing, emotion_values, check_times
                        from .motion import build_motion_plan
                        p = load_project(path)
                        raw = load_raw(path)
                        edits = validate_edits(body['edits'],p['duration_us'])
                        smoothing = validate_smoothing(body.get('smoothing',base.get('smoothing')))
                        emotion_values(raw,check_times(raw,p['duration_us'],edits,p['step_us'],smoothing),edits,p['step_us'],smoothing)
                        chosen_motion = base.get('motion') if body.get('motion') is None else body['motion']
                        self.json_response(build_motion_plan(path,raw,edits,smoothing,chosen_motion))
                    elif parts[3] == 'visual-plan':
                        from .timeline import validate_edits
                        load_revision(path, body['base_revision'])
                        p = load_project(path)
                        purpose = body.get('purpose', 'confirmed')
                        consumer_id = body.get('consumer_id')
                        validate_consumer(consumer_id, purpose)
                        consumer_seq = body.get('consumer_seq')
                        validate_consumer_seq(consumer_id, consumer_seq)
                        smoothing = body.get('smoothing')
                        edits = validate_edits(body['edits'], p['duration_us'])
                        timeline_payload(path, edits=edits, smoothing=smoothing, motion={'engine':'legacy'})
                        cached = cached_visual_plan(path, edits, smoothing=smoothing)
                        if cached is not None:
                            if manager is not None:
                                accepted = manager.accept_cached_plan(p['id'] + ':' + cached['key'],
                                                                      consumer_id, purpose, consumer_seq)
                                if accepted['status'] == 'superseded':
                                    self.json_response(accepted)
                                    return
                            self.json_response({'status': 'ready', 'plan': cached})
                        elif manager is None:
                            self.json_response({'error': '当前服务未启动规划队列'}, 503)
                        else:
                            key = planning_key(path, edits, smoothing=smoothing)
                            job = manager.request_plan({'project_id': p['id'],
                                'base_revision': body['base_revision'], 'edits': edits,
                                'smoothing': smoothing}, p['id'] + ':' + key,
                                consumer_id=consumer_id, purpose=purpose, consumer_seq=consumer_seq)
                            if job['status'] == 'superseded':
                                self.json_response(job)
                                return
                            self.json_response({'status': 'queued', 'job_id': job['id'], 'emotion_key': key}, 202)
                    elif parts[3] == 'previews':
                        if manager is None:
                            self.json_response({'error': '当前服务未启动任务队列'}, 503)
                            return
                        from .pipeline import create_preview_snapshot
                        option_keys = {'mode', 'fps', 'width', 'height', 'start', 'end'}
                        if set(body) - (option_keys | {'base_revision', 'edits', 'smoothing', 'edit_id', 'motion', 'motion_edit_id'}):
                            raise ValueError('局部预览含未知参数；上下文范围固定为前后 2 秒')
                        snapshot = create_preview_snapshot(path, body['base_revision'], body['edits'],
                                                           smoothing=body.get('smoothing'), edit_id=body.get('edit_id'),
                                                           **{k:body[k] for k in ('motion','motion_edit_id') if k in body})
                        requested = dict(snapshot['options'])
                        requested.update({k: body[k] for k in option_keys if k in body})
                        options = validate_render_options(load_project(path), requested)
                        job = manager.submit('preview', {'project_id': parts[2],
                                             'snapshot_id': snapshot['id'], 'options': options})
                        self.json_response(job, 202)
                    elif parts[3] == 'renders':
                        if manager is None:
                            self.json_response({'error': '当前服务未启动任务队列'}, 503)
                            return
                        p = load_project(path)
                        revision = load_revision(path, body.get('revision'))
                        options = validate_render_options(p, body)
                        self.json_response(manager.submit('render', {'project_id': parts[2], 'revision': revision['id'], 'options': options}), 202)
                    else:
                        self.json_response({'error': 'Not found'}, 404)
                elif len(parts) == 4 and parts[:2] == ['api', 'jobs'] and parts[3] == 'cancel':
                    if manager is None:
                        self.json_response({'error': '当前服务未启动任务队列'}, 503)
                        return
                    self.json_response(manager.cancel(parts[2]))
                elif len(parts) == 4 and parts[:2] == ['api', 'jobs'] and parts[3] == 'retry':
                    if manager is None:
                        self.json_response({'error': '当前服务未启动任务队列'}, 503)
                        return
                    self.json_response(manager.retry(parts[2]), 202)
                else:
                    self.json_response({'error': 'Not found'}, 404)
            except (ValueError, KeyError, TypeError) as exc:
                self.json_response({'error': str(exc)}, 400)
            except FileNotFoundError as exc:
                self.json_response({'error': str(exc)}, 404)
            except Exception as exc:
                self.json_response({'error': str(exc)}, 500)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.manager = manager
    return server


def serve(projects=DEFAULT_PROJECTS, port=8765):
    # Prevent two queue schedulers for the same project collection.
    root = Path(projects).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with project_lock(root):
        server = make_server(root, port)
        print(f'MusicVisualization Studio: http://127.0.0.1:{server.server_port}', flush=True)
        print('仅本机访问；关闭网页不会停止生成。Ctrl+C 关闭服务。', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.manager.close()
            server.server_close()
