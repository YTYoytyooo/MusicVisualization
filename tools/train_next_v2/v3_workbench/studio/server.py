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
from .paths import WORKSPACE, WEB_ROOT


def source_catalog():
    """Discovery is read-only; import is an explicit queued operation."""
    from tools.train_next_v2.v3_studio import store as analysis_store
    analyses = []
    for item in analysis_store.projects():
        if item.get('has_result'):
            try:
                loaded = analysis_store.load_project(item['id'])
                analyses.append({'id': item['id'], 'name': item['name'],
                                 'systems': list(loaded['result']['systems'])})
            except (OSError,ValueError,KeyError):
                continue  # A source changed between discovery and validation.
    legacy = []
    for file in (WORKSPACE / 'data/v2.0/projects').glob('p-*/project.json'):
        try:
            p = read_json(file)
            legacy.append({'id': p['id'], 'name': p['name'], 'path': str(file.parent)})
        except (OSError, ValueError, KeyError):
            continue
    return {'analyses': analyses, 'v2_projects': legacy, 'library': analysis_store.library()}


def project_payload(path, revision_id=None, source_system=None):
    r = load_revision(path, revision_id)
    driver = source_system or r.get('source_system')
    p = load_project(path, system=driver)
    from .store import systems_metadata
    revisions = [load_revision(path, f.parent.name) for f in sorted((path / 'revisions').glob('r[0-9]*/revision.json'))]
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
        for filename, key in (('predictions_effective.csv','effective_csv_url'),
                              ('predictions_baseline.csv','baseline_csv_url'),
                              ('visual_plan.csv','visual_csv_url'), ('render.json','manifest_url')):
            if (f.parent / filename).is_file():
                item[key] = prefix + '/' + filename
        item.update({key: value for key, value in item.get('settings', {}).items()
                     if key in ('width','height','fps')})
        renders.append(item)
    # Consumers choose the last matching item as the newest. UUID order is random.
    renders.sort(key=lambda item: (item.get('created_at', ''), item['id']))
    systems = systems_metadata(path)
    for item in systems:
        item['raw_csv_url'] = f"/api/projects/{p['id']}/raw?source_system={item['id']}"
    return {'project': p, 'revision': r, 'revisions': revisions, 'renders': renders,
            'systems': systems,
            'timeline': timeline_payload(path, revision=r['id'], source_system=driver)}


def make_server(projects=DEFAULT_PROJECTS, port=8768, start_queue=True):
    root = Path(projects).resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    manager = JobManager(root) if start_queue else None
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

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
            if self.command != 'HEAD':
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
            if self.command == 'HEAD':
                return
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
                    self.json_response({'projects': list_projects(root), 'token': token,
                                        'version': 'V3 workbench', 'training_enabled': False})
                elif url.path == '/api/sources':
                    self.json_response(source_catalog())
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
                    query = parse_qs(url.query)
                    revision = query.get('revision', [None])[0]
                    self.json_response(project_payload(project, revision, query.get('source_system', [None])[0]))
                elif len(parts) == 4 and parts[:2] == ['api', 'projects'] and parts[3] == 'raw':
                    import csv
                    import io
                    from .store import load_raw
                    path = resolve_project(root, parts[2])
                    system = parse_qs(url.query).get('source_system',[None])[0]
                    raw = load_raw(path,system=system)
                    stream = io.StringIO(newline='')
                    writer = csv.writer(stream)
                    writer.writerow(['time_seconds','valence','arousal','valence_valid','arousal_valid'])
                    for t, values, mask in zip(raw.times_us,raw,raw.mask):
                        writer.writerow([int(t)/1e6,*[float(v) if valid else '' for v,valid in zip(values,mask)],*map(bool,mask)])
                    data = stream.getvalue().encode('utf-8-sig')
                    self.send_response(200)
                    self.send_header('Content-Type','text/csv; charset=utf-8')
                    self.send_header('Content-Disposition','attachment; filename="original_predictions.csv"')
                    self.send_header('Content-Length',str(len(data)))
                    self.send_header('Cache-Control','no-store')
                    self.end_headers()
                    if self.command != 'HEAD':self.wfile.write(data)
                elif len(parts) >= 3 and parts[0] == 'media':
                    project = resolve_project(root, parts[1])
                    if parts[2:] == ['audio']:
                        path = project / load_project(project)['audio_file']
                    else:
                        if len(parts) != 5 or parts[2] != 'renders' or parts[4] not in ('output.mp4', 'video_silent.avi', 'frame_values.csv', 'preview.png', 'predictions_effective.csv', 'predictions_baseline.csv', 'visual_plan.csv', 'motion_plan.csv', 'motion_plan.json', 'render.json'):
                            raise ValueError('非法媒体路径')
                        path = (project / 'renders' / parts[3] / parts[4]).resolve()
                    if not path.resolve().is_relative_to(project.resolve()):
                        raise ValueError('媒体路径越界')
                    self.send_file(path)
                elif url.path in ('/', '/index.html', '/app.js', '/timeline-math.js', '/motion-math.js', '/workbench-state.js', '/style.css'):
                    self.send_file(WEB_ROOT / ('index.html' if url.path == '/' else url.path[1:]))
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

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            try:
                # Drain a bounded body before rejecting auth (Windows avoids a
                # reset that would hide the HTTP rejection from the browser).
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError('请求体大小非法')
                encoded = self.rfile.read(length)
                if len(encoded) != length:
                    raise ValueError('请求体不完整')
                origin = self.headers.get('Origin')
                allowed = (None, f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}')
                if not self.allowed_host() or origin not in allowed or not secrets.compare_digest(self.headers.get('X-Studio-Token', ''), token):
                    self.json_response({'error': '请求未授权；请刷新本机编辑器'}, 403)
                    return
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('需要 application/json')
                body = json.loads(encoded)
                if not isinstance(body, dict):
                    raise ValueError('请求体必须为对象')
                parts = urlparse(self.path).path.strip('/').split('/')
                if parts == ['api', 'plan-release']:
                    if manager is None:
                        self.json_response({'error': '当前服务未启动规划队列'}, 503)
                        return
                    self.json_response(manager.release_plan(body.get('consumer_id'), body.get('consumer_seq')))
                elif parts in (['api', 'import-analysis'], ['api', 'import-v2']):
                    if manager is None:
                        self.json_response({'error': '当前服务未启动任务队列'}, 503)
                        return
                    if parts[-1] == 'import-analysis':
                        from tools.train_next_v2.v3_studio.store import project_path
                        source = project_path(body['analysis_id'])
                        args = {'analysis_dir': str(source), 'name': body.get('name'),
                                'source_system': body.get('source_system', 'b01'), 'seed':body.get('seed',42)}
                        kind = 'import_analysis'
                    else:
                        source = Path(body['path']).resolve()
                        if not source.is_relative_to(WORKSPACE / 'data/v2.0/projects'):
                            raise ValueError('只允许导入现有 V2 项目目录')
                        args = {'source_dir': str(source), 'name': body.get('name')}
                        kind = 'import_v2'
                    self.json_response(manager.submit(kind, args), 202)
                elif parts == ['api', 'projects']:
                    if manager is None:
                        self.json_response({'error': '当前服务未启动任务队列'}, 503)
                        return
                    from .timeline import integer
                    integer(body.get('seed', 42), 'seed')
                    if not isinstance(body.get('audio_path'), str):
                        raise ValueError('audio_path 文件不存在')
                    audio = Path(body['audio_path']).resolve()
                    if not audio.is_relative_to(WORKSPACE) or not audio.is_file():
                        raise ValueError('请选择 Vm 内已有音频文件')
                    if body.get('device', 'cpu') not in ('cpu', 'cuda') or type(body.get('include_v2', False)) is not bool:
                        raise ValueError('计算设备或旧版对照选项非法')
                    payload = {key:body[key] for key in ('name','device','include_v2','seed') if key in body}
                    payload['audio_path'] = str(audio)
                    self.json_response(manager.submit('analyze', payload), 202)
                elif len(parts) == 4 and parts[:2] == ['api', 'projects']:
                    path = resolve_project(root, parts[2])
                    if parts[3] == 'revisions':
                        base = load_revision(path, body['base_revision'])
                        source_system = body.get('source_system', base.get('source_system'))
                        smoothing = body.get('smoothing')
                        motion_args = {'motion':base.get('motion') if body['motion'] is None else body['motion']} if 'motion' in body else (
                            {'motion':base['motion']} if base.get('motion',{}).get('engine')=='flow-v1' else {})
                        validate_preview(path, body['edits'], smoothing=smoothing, source_system=source_system, **motion_args)
                        self.json_response(save_revision(path, body['base_revision'], body['edits'],
                                                         smoothing=smoothing, source_system=source_system, **motion_args), 201)
                    elif parts[3] == 'preview':
                        base = load_revision(path, body['base_revision'])
                        motion_args = {'motion':base.get('motion') if body['motion'] is None else body['motion']} if 'motion' in body else (
                            {'motion':base['motion']} if base.get('motion',{}).get('engine')=='flow-v1' else {})
                        self.json_response(validate_preview(path, body['edits'], smoothing=body.get('smoothing'),
                            source_system=body.get('source_system',base.get('source_system')), **motion_args))
                    elif parts[3] == 'motion-plan':
                        base = load_revision(path, body['base_revision'])
                        from .store import load_raw
                        from .timeline import validate_edits
                        from .smoothing import validate_smoothing, emotion_values, check_times
                        from .motion import build_motion_plan
                        system = body.get('source_system',base.get('source_system'))
                        p = load_project(path, system=system)
                        raw = load_raw(path, system=system)
                        edits = validate_edits(body['edits'],p['duration_us'])
                        smoothing = validate_smoothing(body.get('smoothing',base.get('smoothing')))
                        emotion_values(raw,check_times(raw,p['duration_us'],edits,p['step_us'],smoothing),edits,p['step_us'],smoothing)
                        chosen_motion = base.get('motion') if body.get('motion') is None else body['motion']
                        self.json_response(build_motion_plan(path,raw,edits,smoothing,chosen_motion,source_system=system))
                    elif parts[3] == 'visual-plan':
                        from .timeline import validate_edits
                        base = load_revision(path, body['base_revision'])
                        system = body.get('source_system',base.get('source_system'))
                        p = load_project(path, system=system)
                        purpose = body.get('purpose', 'confirmed')
                        consumer_id = body.get('consumer_id')
                        validate_consumer(consumer_id, purpose)
                        consumer_seq = body.get('consumer_seq')
                        validate_consumer_seq(consumer_id, consumer_seq)
                        smoothing = body.get('smoothing')
                        edits = validate_edits(body['edits'], p['duration_us'])
                        timeline_payload(path, edits=edits, smoothing=smoothing, motion={'engine':'legacy'},source_system=system)
                        cached = cached_visual_plan(path, edits, smoothing=smoothing,source_system=system)
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
                            key = planning_key(path, edits, smoothing=smoothing,source_system=system)
                            job = manager.request_plan({'project_id': p['id'],
                                'base_revision': body['base_revision'], 'edits': edits,
                                'smoothing': smoothing, 'source_system':system}, p['id'] + ':' + key,
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
                        if set(body) - (option_keys | {'base_revision', 'edits', 'smoothing', 'edit_id', 'motion', 'motion_edit_id','source_system'}):
                            raise ValueError('局部预览含未知参数；上下文范围固定为前后 2 秒')
                        snapshot = create_preview_snapshot(path, body['base_revision'], body['edits'],
                                                           smoothing=body.get('smoothing'), edit_id=body.get('edit_id'),
                                                           source_system=body.get('source_system'),
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


def serve(projects=DEFAULT_PROJECTS, port=8768):
    # Prevent two queue schedulers for the same project collection.
    root = Path(projects).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with project_lock(root):
        server = make_server(root, port)
        print(f'MusicVisualization Studio V3: http://127.0.0.1:{server.server_port}', flush=True)
        print('仅本机访问；关闭网页不会停止生成。Ctrl+C 关闭服务。', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.manager.close()
            server.server_close()
