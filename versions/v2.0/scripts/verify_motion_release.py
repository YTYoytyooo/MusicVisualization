"""Finite, isolated motion acceptance: real browser flows and encoded MP4s.

No downloads or fresh model inference. Existing predictions are copied with
explicit provenance; source audio, weights, revisions and outputs stay read-only.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import DATA_ROOT, DEFAULT_PROJECTS, DEFAULT_MODEL
from studio.store import (atomic_json, create_project, digest, load_project,
                          load_raw, now, save_revision)
from studio.pipeline import render
from studio.paths import migrated_path
from validate_release import inspect_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('all', 'browser', 'videos'), default='all')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source', type=Path, default=DEFAULT_PROJECTS/'p-ccc83cd185cb')
    parser.add_argument('--browser-script', choices=('live', 'upgrade', 'motion'))
    args = parser.parse_args()
    output = (args.output or DATA_ROOT / 'validation-output'/('motion-'+uuid.uuid4().hex[:8])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    projects = output/'projects'
    report_path = output/'report.json'
    report = {'status':'running', 'started_at':now(), 'steps':[], 'output':str(output)}
    atomic_json(report_path, report)

    def record(name, details):
        report['steps'].append({'step':name, 'at':now(), 'details':details})
        atomic_json(report_path, report)
        print(name+': '+json.dumps(details, ensure_ascii=True), flush=True)

    def progress(stage, ratio):
        bucket = int(ratio*10)
        if progress.last != (stage, bucket):
            print(f'{stage}: {ratio:.0%}', flush=True)
            progress.last = (stage, bucket)
    progress.last = None

    def video(project, revision, options):
        started = time.monotonic()
        result = render(project, revision, options, progress=progress)
        checked = inspect_video(project, result)
        checked.update(seconds_to_render=round(time.monotonic()-started, 2),
                       revision=revision, resumed_from_frame=result['resumed_from_frame'])
        return result, checked

    try:
        if args.part in ('all', 'browser'):
            from studio.server import make_server
            server = make_server(projects, 0)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            env = {**os.environ, 'STUDIO_URL':f'http://127.0.0.1:{server.server_port}',
                   'STUDIO_PROJECTS':str(projects), 'PYTHONUTF8':'1'}
            try:
                for name in ([args.browser_script] if args.browser_script else ['live', 'upgrade', 'motion']):
                    log = output/(name+'-browser.log')
                    with open(log, 'w', encoding='utf-8') as stream:
                        completed = subprocess.run(['node', f'scripts/browser_{name}_check.cjs'],
                            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                            timeout=600, creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    if completed.returncode:
                        raise RuntimeError(f'{name} browser check failed: {log}\n'+log.read_text(encoding='utf-8')[-7000:])
                    record('browser-'+name, {'passed':True, 'log':str(log)})
                # A frozen-preview click queues an actual worker. Validate its
                # terminal result too, not merely the accepted HTTP response.
                deadline = time.monotonic()+300
                while any(j['status'] in ('queued','running','cancelling') for j in server.manager.list()):
                    if time.monotonic()>deadline:
                        raise TimeoutError('Acceptance preview queue did not finish')
                    time.sleep(.5)
                jobs = server.manager.list()
                failed = [j for j in jobs if j['status'] in ('failed','interrupted','diagnostic_only')]
                assert not failed, failed
                rendered=[]
                for job in jobs:
                    if job['status']=='succeeded' and job['kind'] in ('render','preview'):
                        rendered.append(inspect_video(projects/job['payload']['project_id'], job['result']))
                record('browser-queued-videos', {'videos':rendered, 'jobs':len(jobs)})
            finally:
                server.shutdown()
                server.manager.close()
                server.server_close()

        if args.part in ('all', 'videos'):
            import cv2
            import numpy as np
            from motion_schema import MODE_LABELS
            from studio.demo import create_demo

            # Six-second slots: one-second entry, four-second full strength,
            # one-second exit. Mode changes preserve the live particle state.
            gallery = create_demo(projects, 54.)
            overrides = [dict(id='gallery-'+mode, mode=mode, start_us=(i*6+1)*1000000,
                end_us=(i*6+5)*1000000, transition_in_us=1000000, transition_out_us=1000000)
                for i,mode in enumerate(MODE_LABELS)]
            revision = save_revision(gallery,'r000000',[],motion={'engine':'flow-v1','overrides':overrides})
            result, checked = video(gallery, revision['id'],dict(width=640,height=360,fps=30,mode='presentation'))
            with open(gallery/'renders'/result['id']/'frame_values.csv',encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            samples=[]
            reader=cv2.VideoCapture(checked['video'])
            for i,mode in enumerate(MODE_LABELS):
                row=rows[(i*6+4)*30]
                assert row['motion_mode']==mode and row['motion_source']=='manual', row
                components=json.loads(row['motion_components_json'])
                assert len(components)==1 and components[0]['weight']==1
                assert max(c['params']['trail_seconds'] for c in components)<=2
                reader.set(cv2.CAP_PROP_POS_FRAMES,(i*6+4)*30)
                ok,frame=reader.read();assert ok
                frame=cv2.resize(frame,(480,270))
                cv2.rectangle(frame,(0,0),(480,31),(12,16,22),-1)
                cv2.putText(frame,f'{i+1:02d} {mode} / {i*6+4}s',(12,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(235,240,255),1,cv2.LINE_AA)
                samples.append(frame)
            reader.release()
            sheet=np.concatenate([np.concatenate(samples[i:i+3],axis=1) for i in range(0,9,3)],axis=0)
            sheet_path=output/'nine-modes.png'
            ok,encoded=cv2.imencode('.png',sheet);assert ok;sheet_path.write_bytes(encoded.tobytes())
            record('nine-mode-gallery', {**checked,'contact_sheet':str(sheet_path),'manual_modes':list(MODE_LABELS)})

            source=args.source.resolve()
            original=load_project(source)
            protected=[source/'project.json',source/'analysis/predictions_raw.npz',source/original['audio_file']]
            if original.get('model_path'):
                protected.append(migrated_path(original['model_path']))
            original_hashes={str(path):digest(path) for path in protected}
            source_revisions={str(path):digest(path) for path in (source/'revisions').glob('*/revision.json')}
            real=create_project(projects,original['name']+' / MOTION ACCEPTANCE',source/original['audio_file'],
                migrated_path(original.get('model_path')),load_raw(source),original,seed=original['seed'],
                provenance={**original['provenance'],'analysis_reused_from':str(source),
                            'analysis_reused_sha256':original['raw_sha256'],'fresh_model_inference':False})
            auto=save_revision(real,'r000000',[],motion={'engine':'flow-v1'})
            options=dict(width=1280,height=720,fps=30,mode='analysis')
            full,checked=video(real,auto['id'],options)
            plan=json.loads((real/'renders'/full['id']/'motion_plan.json').read_text(encoding='utf-8'))
            record('real-full-song-auto', {**checked,'source_prediction_reused':str(source),
                'duration':original['duration'],'automatic_modes':sorted(set(f['mode'] for f in plan['auto']))})
            tail,tail_checked=video(real,auto['id'],{**options,'start':90})
            assert tail['resumed_from_frame']==2700,tail
            def rows_for(result):
                with open(real/'renders'/result['id']/'frame_values.csv',encoding='utf-8-sig') as stream:
                    return list(csv.DictReader(stream))
            assert rows_for(tail)==rows_for(full)[2700:]
            record('real-checkpoint-tail', {**tail_checked,'frame_values_equal_full_tail':True})
            manual=save_revision(real,auto['id'],[],motion={'engine':'flow-v1','overrides':[
                dict(id='real-orbit',mode='orbit',start_us=4000000,end_us=8000000,
                     transition_in_us=2000000,transition_out_us=2000000),
                dict(id='real-meteor',mode='meteor',start_us=12000000,end_us=16000000,
                     transition_in_us=2000000,transition_out_us=2000000)]})
            edited,edited_checked=video(real,manual['id'],{**options,'end':20})
            assert edited['plan_key']==full['plan_key'],'motion-only revision reran MCTS'
            checked_rows=rows_for(edited)
            assert checked_rows[6*30]['motion_mode']=='orbit'
            assert checked_rows[14*30]['motion_mode']=='meteor'
            record('real-manual-regeneration', {**edited_checked,'visual_plan_reused':True})
            for path,expected in {**original_hashes,**source_revisions}.items():
                assert digest(path)==expected,path
            record('source-protection', {'audio_model_raw_revisions_unchanged':True,'files':len(original_hashes)+len(source_revisions)})
        report.update(status='passed',completed_at=now())
        atomic_json(report_path,report)
        print('PASS: '+str(report_path),flush=True)
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc(),completed_at=now())
        atomic_json(report_path,report)
        raise


if __name__=='__main__':
    main()
