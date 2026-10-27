"""Loopback-only Studio API. All audio is referenced by a frozen library identity."""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import secrets
from urllib.parse import parse_qs,urlsplit
from . import store,jobs,render

WEB=store.ROOT/'versions/v3.0/web'


def make_server(port=8768):
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'

        def log_message(self, fmt, *args):
            # No tokens, user audio paths, or request bodies in access logs.
            pass

        def trusted(self):
            port=self.server.server_port
            allowed={f'127.0.0.1:{port}',f'localhost:{port}'}
            if self.headers.get('Host') not in allowed:
                raise PermissionError('Invalid loopback Host')
            origin=self.headers.get('Origin')
            if origin and origin not in {f'http://{x}' for x in allowed}:
                raise PermissionError('Cross-origin requests are not allowed')

        def send(self,code,data,content='application/json; charset=utf-8',headers=None):
            if not isinstance(data,bytes):
                data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(data)))
            if self.close_connection:self.send_header('Connection','close')
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
            for k,v in (headers or {}).items(): self.send_header(k,v)
            self.end_headers()
            if self.command!='HEAD': self.wfile.write(data)

        def file(self,path,download=False):
            path=store.inside(path);size=path.stat().st_size;start=0;end=size-1;code=200
            raw=self.headers.get('Range')
            if raw:
                try:
                    if not raw.startswith('bytes=') or ',' in raw: raise ValueError()
                    a,b=raw[6:].split('-',1)
                    if not a:
                        count=int(b)
                        if count<=0: raise ValueError()
                        start=max(0,size-count)
                    else:
                        start=int(a);end=min(size-1,int(b)) if b else size-1
                    if not 0<=start<=end<size: raise ValueError()
                    code=206
                except ValueError:
                    self.send(416,{'error':'Unsatisfiable range'},headers={'Content-Range':f'bytes */{size}'});return
            self.send_response(code)
            self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(max(0,end-start+1)));self.send_header('Accept-Ranges','bytes')
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if code==206: self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            if download: self.send_header('Content-Disposition','attachment; filename="'+path.name+'"')
            self.end_headers()
            if self.command=='HEAD':return
            with path.open('rb') as f:
                f.seek(start);remaining=end-start+1
                while remaining>0:
                    chunk=f.read(min(65536,remaining))
                    if not chunk:break
                    self.wfile.write(chunk);remaining-=len(chunk)

        def do_HEAD(self): self.do_GET()

        def do_GET(self):
            try:
                self.trusted();url=urlsplit(self.path);q=parse_qs(url.query)
                value=lambda k:q.get(k,[''])[0]
                if url.path=='/api/state':
                    m=store.read(store.MANIFEST) if store.MANIFEST.exists() else {}
                    self.send(200,{'token':token,'library':store.library(),'projects':store.projects(),
                        'jobs':jobs.all_jobs(),'model':{'id':m.get('id','not-prepared'),
                        'description':'B01 三种子等权集成 · 离线研究预览 · 未通过正式替换门槛'},'port':self.server.server_port})
                elif url.path=='/api/project': self.send(200,store.load_project(value('id')))
                elif url.path in ('/audio','/library-audio'):
                    if url.path=='/audio':
                        audio=store.load_project(value('id'))['project']['audio']
                        path=store.inside(store.ROOT/audio['path'])
                        if store.digest(path)!=audio['sha256']:raise ValueError('Audio identity changed')
                    else:
                        row=store.audio_entry(value('id'));path=store.ROOT/row['path']
                    self.file(path)
                elif url.path=='/download':
                    project_id=value('id');loaded=store.load_project(project_id);result=loaded['result']
                    if not result: raise ValueError('Analysis is incomplete')
                    name=value('file')
                    if name in ('raw.csv','display.csv'):
                        seconds=None if name=='raw.csv' else float(value('smoothing_seconds') or 0)
                        self.send(200,render.csv_text(result,seconds).encode('utf-8'),'text/csv; charset=utf-8',
                                  {'Content-Disposition':f'attachment; filename="{name}"'})
                    elif name=='bundle.json':
                        latest=store.project_path(project_id)/'exports/latest.json'
                        self.send(200,{**loaded,'export':store.read(latest) if latest.exists() else None},
                                  headers={'Content-Disposition':'attachment; filename="bundle.json"'})
                    elif name in ('preview.mp4','frame_values.csv'):
                        meta=store.read(store.project_path(project_id)/'exports/latest.json')
                        path=store.inside(store.ROOT/meta['directory']/name,store.project_path(project_id))
                        if store.digest(path)!=meta['files'][name]:raise ValueError('Export integrity mismatch')
                        self.file(path,True)
                    else: raise ValueError('Unsupported export file')
                elif url.path in ('/','/index.html','/app.js','/style.css'):
                    name='index.html' if url.path=='/' else url.path[1:]
                    self.send(200,(WEB/name).read_bytes(),mimetypes.guess_type(name)[0]+'; charset=utf-8')
                else:self.send(404,{'error':'Not found'})
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
            except PermissionError as e:self.send(403,{'error':str(e)})
            except FileNotFoundError as e:self.send(404,{'error':str(e)})
            except (ValueError,KeyError,TypeError) as e:self.send(400,{'error':str(e)})
            except Exception as e:self.send(500,{'error':str(e)})

        def do_POST(self):
            # Close after each mutation, including rejected unread bodies.
            self.close_connection=True
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:raise ValueError('Invalid request size')
                # Consume a bounded body before returning an auth rejection.
                # Otherwise Windows may reset a socket with unread inbound data
                # and discard the 403 response. No JSON or action is trusted yet.
                self.connection.settimeout(5)
                raw=self.rfile.read(length)
                if len(raw)!=length:raise ValueError('Incomplete request body')
                self.trusted()
                if not secrets.compare_digest(self.headers.get('X-V3-Token',''),token):
                    raise PermissionError('Missing or invalid session token')
                if self.headers.get('Content-Type','').split(';',1)[0].strip().lower()!='application/json':
                    raise ValueError('Expected application/json')
                data=json.loads(raw)
                if not isinstance(data,dict):raise ValueError('Expected JSON object')
                if self.path=='/api/analyze':
                    result=jobs.submit('analyze',{'audio_id':data['audio_id'],'device':data.get('device','cuda'),
                                                 'include_v2':data.get('include_v2',False)})
                elif self.path=='/api/cancel':result=jobs.cancel(data['job_id'])
                elif self.path=='/api/resume':result=jobs.resume(data['job_id'])
                elif self.path=='/api/export':
                    # Validate before registration so malformed requests do not occupy the job slot.
                    project=store.load_project(data['project_id'])
                    window=data.get('smoothing_seconds',0)
                    if type(window) not in (int,float) or window not in render.WINDOWS:raise ValueError('Invalid display window')
                    start,end=data['start'],data['end'];duration=project['project']['audio']['duration_seconds']
                    if any(type(v) not in (int,float) for v in (start,end)) or not 0<=start<end<=duration or end-start>60:
                        raise ValueError('Invalid preview range (maximum 60 seconds)')
                    result=jobs.submit('export',{'smoothing_seconds':data.get('smoothing_seconds',0),'start':start,'end':end},data['project_id'])
                else:self.send(404,{'error':'Not found'});return
                self.send(200,result)
            except PermissionError as e:self.send(403,{'error':str(e)})
            except (ValueError,KeyError,TypeError) as e:self.send(400,{'error':str(e)})
            except RuntimeError as e:self.send(409,{'error':str(e)})
            except Exception as e:self.send(500,{'error':str(e)})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    return server
