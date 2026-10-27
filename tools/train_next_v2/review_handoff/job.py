"""Register/build/test the independent 58-song handoff without serving in background."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import traceback
from urllib.request import urlopen, Request

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, run_lease, lock_path
from tools.train_next_v2.common import ROOT, BASE, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.research_inference.benchmark import process_alive
from .server import create_server
from .validate import inspect_document, validate

JOB='e-pilot58-review-handoff-v1'
DIRECTORY=BASE/'annotation_pilot'/JOB
POOL=BASE/'annotation_pilot/e-fabio-pilot-supplement-20-v1/review-pool.json'
SOURCE=BASE/'annotation_pilot/e-fabio-intake-f26ea031b964-v1/sources/continuous-fabio-json.json'
OUTPUT=ROOT/'data/annotations/pilot58-v1'
LAUNCHER=ROOT/'Start Pilot58 Annotation.cmd'


def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()


def status():
    p=read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity=p.get('process_identity')
    return dict(job_id=JOB,checked_at=now(),process_identity=identity,process_alive=bool(identity and process_alive(identity)),
        os_lease_active=is_running(JOB),progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None)


def state(stage,**extra):
    row=dict(stage=stage,checked_at=now(),**extra);atomic_json(DIRECTORY/'status.json',row)
    text=json.dumps(row,ensure_ascii=False)
    with (DIRECTORY/'console.log').open('a',encoding='utf-8') as f:f.write(text+'\n')
    print(text,flush=True)


def prepare():
    if DIRECTORY.exists():raise ValueError('Preserve existing registered handoff')
    previous=read_json(POOL.parent/'launch.json')
    files=list(Path(__file__).parent.glob('*.py'))+[POOL,SOURCE,POOL.parent/'report.json',POOL.parent/'verification.json',ROOT/'tools/annotate/server.py']
    files+=list((ROOT/'tools/annotate/web').glob('*'))
    pins={rel(p):sha256(p) for p in files if p.is_file()}
    pins.update({r['audio_path']:r['audio_sha256'] for r in read_json(POOL)['candidates']})
    for name in ('tools/train_next/core.py','tools/train_next/runtime.py','tools/train_next_v2/common.py','tools/train_next_v2/research_inference/benchmark.py'):
        pins[name]=sha256(ROOT/name)
    protected_models={k:v for k,v in previous['protected_hashes'].items() if k.startswith('models/')}
    pins.update(protected_models);verify_hashes(pins)
    argv=[previous['python'],'-u','-m','tools.train_next_v2.review_handoff.job','--launch',str(DIRECTORY/'launch.json')]
    config=dict(job_id=JOB,registered_at=now(),original_argv=argv,resume_argv=argv+['--resume'],cwd=str(ROOT),python=previous['python'],environment=previous['environment'],
        protected_hashes=pins,pool=rel(POOL),source=rel(SOURCE),annotation_output=rel(OUTPUT),audio_root='data/datasets/external/fabio/audio',
        log=rel(DIRECTORY/'console.log'),runtime_lock=str(lock_path(JOB)),
        recovery='After PID creation identity is dead and lease free, diagnose failure and use resume_argv. Never overwrite modified annotations or launcher.',
        scope='Independent filtered annotation entry point, exact-value copy of nine existing raw documents into separate reviewer output, finite local HTTP smoke tests in temporary output and read-only validation. No new real labels, model changes or persistent server.')
    DIRECTORY.mkdir(parents=True);atomic_json(DIRECTORY/'launch.json',config)
    record_event(dict(event='E_review_handoff_registered',job_id=JOB),run=dict(run_id=JOB,stage='E',status='registered',launch=rel(DIRECTORY/'launch.json'),launch_sha256=sha256(DIRECTORY/'launch.json'),original_argv=argv,cwd=str(ROOT),runtime_lock=str(lock_path(JOB)),log=rel(DIRECTORY/'console.log')))
    print(json.dumps(dict(job_id=JOB,pinned_files=len(pins))))


def seed(output):
    sources=read_json(SOURCE)['songs']
    for song in sources:
        path=output/song['reviewer']/f"{song['source']['song_id']}.json"
        if path.exists():
            if read_json(path)!=song:raise ValueError('Existing human output changed; never overwrite it')
        else:path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,song)
    return sources


def smoke(config):
    checks=[]
    with tempfile.TemporaryDirectory(dir=BASE/'tmp') as temporary:
        output=Path(temporary)/'annotations';original=seed(output)
        before={str(p):sha256(p) for p in output.rglob('*.json')}
        server=create_server(config,output,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            def get(path):
                with urlopen(base+path,timeout=10) as response:return json.load(response)
            library=get('/api/library?reviewer=fabio');assert len(library['songs'])==58
            assert sum(s['has_annotations'] for s in library['songs'])==9;checks.append('HTTP library bounded to 58 with nine raw Fabio documents')
            for song in original:
                assert get('/api/song/'+song['source']['song_id']+'?reviewer=fabio')==song
            assert {str(p):sha256(p) for p in output.rglob('*.json')}==before;checks.append('Raw takes, timestamps, source provenance and completion flags unchanged')
            key=max(server.pilot_keys,key=lambda k:abs(server.pilot_keys[k][0]['native_audio']['header_minus_actual_frames']))
            doc=get('/api/song/'+key+'?reviewer=synthetic_check');row=server.pilot_keys[key][0]
            assert doc['source']['duration']==row['native_audio']['frames']/row['native_audio']['sample_rate']
            assert abs(row['native_audio']['header_minus_actual_frames']/row['native_audio']['sample_rate'])>.15
            body=dict(reviewer='synthetic_check',song_id=key,base_revision=0,audio_sha256=doc['source']['audio_sha256'],duration=doc['source']['duration'],
                takes=[dict(id='synthetic-fixture',axes=['valence'],points=[dict(time=0.,status='annotated',valence=.1,arousal=None,confidence='')])],
                transitions=[],completed=dict(valence=False,arousal=False),artist='',style='',note='ISOLATED TEMPORARY SOFTWARE TEST; NOT HUMAN STUDY DATA')
            request=Request(base+'/api/save',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Annotation-Token':library['token']},method='POST')
            with urlopen(request,timeout=10) as response: saved=json.load(response)
            assert saved['takes'][0]['points'][0]['arousal'] is None
            checks.append('Actual HTTP save uses native EOF duration despite >0.15s header error; independent missing axis retained')
            audio_request=Request(base+'/audio/'+key,headers={'Range':'bytes=0-15'})
            with urlopen(audio_request,timeout=10) as response:
                assert response.status==206;actual=response.read()
            with server.pilot_keys[key][1].open('rb') as f:assert actual==f.read(16)
            checks.append('HTTP audio range serves exact source bytes')
            try:server.store.import_audio('extra.mp3',None,5)
            except ValueError:checks.append('Fixed pool rejects additions')
            else:raise AssertionError('Unexpected pool mutation')
            for mutator in (
                lambda d:d['source'].update(audio_sha256='0'*64),
                lambda d:d['takes'][0]['points'][0].update(time=row['native_audio']['duration_seconds']+1),
                lambda d:d['takes'][0]['points'][0].update(arousal=.4),
                lambda d:d['completed'].update(arousal=True)):
                bad=copy.deepcopy(saved);mutator(bad)
                try:inspect_document(bad,row)
                except ValueError:pass
                else:raise AssertionError('Malformed return accepted')
            checks.append('Return validator rejects wrong hash, out-of-bounds clock, fabricated missing axis and empty completed axis')
            # Repeated exports from one person must not count as extra independent people.
            good=copy.deepcopy(saved);good['completed']['valence']=True
            a=Path(temporary)/'one.json';b=Path(temporary)/'two.json';atomic_json(a,good);atomic_json(b,good)
            result=validate({'candidates':[row]},[a,b]);assert result['input_documents_after_duplicate_collapse']==1 and not result['songs'][0]['meets_two_ids_per_axis']
            good['reviewer']='SYNTHETIC_CHECK';good['revision']+=1;atomic_json(b,good)
            result=validate({'candidates':[row]},[a,b]);assert result['errors'] and result['songs'][0]['distinct_declared_complete_ids']['valence']==0
            checks.append('Duplicate exports and case-alias/conflicting returns cannot create multiple completed reviewers')
        finally:server.shutdown();thread.join(timeout=5);server.server_close();server.store.close()
    return checks


def run(config):
    checks=smoke(config);state('isolated_http_and_validator_checks_passed',checks=len(checks))
    seed(OUTPUT)
    launcher=['@echo off','setlocal',f'cd /d "{ROOT}"']
    launcher += [f'set "{k}={v}"' for k,v in config['environment'].items()]
    launcher += [f'"{config["python"]}" -u -m tools.train_next_v2.review_handoff.server --config "{DIRECTORY/"launch.json"}"','if errorlevel 1 pause']
    text='\r\n'.join(launcher)+'\r\n'
    if LAUNCHER.exists() and LAUNCHER.read_bytes()!=text.encode('utf-8'): raise ValueError('Launcher exists with different content')
    LAUNCHER.write_bytes(text.encode('utf-8'))
    result=validate(read_json(POOL),sorted(OUTPUT.glob('*/*.json')))
    if result['errors'] or result['input_documents_after_duplicate_collapse']!=9:raise ValueError('Real raw return accounting changed')
    atomic_json(DIRECTORY/'current-return-validation.json',result)
    instructions='固定58首音乐连续标注试点\n\n双击 Vm 根目录 Start Pilot58 Annotation.cmd，然后打开 http://127.0.0.1:8767 。此入口只显示已核验的58首，使用实际解码时长；保留原 Start Annotation.cmd 的行为。\n\n已有9首原始标注已按原值复制到独立目录 data/annotations/pilot58-v1/fabio。用 fabio 代号可继续复核；未替你勾选完成，也未补把握程度。完整原始导出仍保留。\n\n其他真实标注者使用各自不同代号，独立标注音乐表达的效价与唤醒度；不要查看或复制他人的答案。每首至少2人，目标3人。不同代号本身不能证明是不同人。\n\nV：-1负面/忧伤，0中性，+1正面/愉悦；A：-1平静，0中等，+1高唤醒。可先标V，再从歌首标A。不确定请记录空值，不用0代替。保留原时间轴，不拉伸、不补齐空白。\n\n完成实际复核后分别确认两维完成状态；缺失歌手、别名、是否完整原曲仍需核实。31首缺少歌手元数据。暂不分配训练/验证角色，也不自动训练。\n\n新标注保存在 data/annotations/pilot58-v1/<真实代号>/ 下。导出完整JSON保留原始采样；只读校验器 tools.train_next_v2.review_handoff.validate 会检查身份、时间、完成声明与重复返回，不替代人工质量/一致性核验。\n'
    (DIRECTORY/'使用说明.txt').write_text(instructions,encoding='utf-8')
    verify_hashes(config['protected_hashes'])
    report=dict(job_id=JOB,status='completed_verified_human_returns_pending',completed_at=now(),checks=checks,
        seeded_raw_documents=9,declared_complete_two_reviewer_songs=result['songs_with_two_declared_ids_per_axis'],
        annotation_output=rel(OUTPUT),launcher=rel(LAUNCHER),launcher_sha256=sha256(LAUNCHER),instructions=rel(DIRECTORY/'使用说明.txt'),
        original_app_unchanged=True,persistent_server_running=False,synthetic_fixtures_removed=True,training_admitted=False,formal_evaluation_admitted=False)
    atomic_json(DIRECTORY/'report.json',report);state(report['status'])
    record_event(dict(event='E_review_handoff_verified',job_id=JOB,checks=len(checks),seeded_raw_documents=9),run=dict(run_id=JOB,stage='E',status='completed_verified',report=rel(DIRECTORY/'report.json')))


def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--launch',type=Path);g.add_argument('--status',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return
    if args.status:print(json.dumps(status()));return
    if args.launch.resolve()!=(DIRECTORY/'launch.json').resolve():raise ValueError('Wrong launch')
    config=read_json(args.launch);verify_hashes(config['protected_hashes'])
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(config['python']).resolve():raise ValueError('Wrong runtime/cwd')
    if any(os.environ.get(k)!=v for k,v in config['environment'].items()):raise ValueError('Environment drift')
    previous=status()
    if previous['process_alive'] or previous['os_lease_active']:raise ValueError('Job active')
    if (DIRECTORY/'report.json').exists():raise ValueError('Completed job must not rerun')
    if (DIRECTORY/'process.json').exists() and not args.resume:raise ValueError('Diagnose prior attempt and use --resume')
    with run_lease(JOB):
        atomic_json(DIRECTORY/'process.json',dict(process_identity=process_identity(),argv=sys.argv,started_at=now()))
        record_event(dict(event='E_review_handoff_started',job_id=JOB,process_identity=process_identity()),run=dict(run_id=JOB,stage='E',status='running'))
        try:run(config)
        except BaseException:
            state('failed',error=traceback.format_exc());record_event(dict(event='E_review_handoff_failed',job_id=JOB),run=dict(run_id=JOB,stage='E',status='failed'));raise


if __name__=='__main__':main()
