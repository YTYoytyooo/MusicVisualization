"""Explain failed MP3 header gates; preserve their rejection and all raw timestamps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import soundfile as sf

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running,run_lease
from tools.train_next_v2.common import ROOT,BASE,PLAN,now,read_json,record_event,verify_hashes,process_identity
from tools.train_next_v2.research_inference.benchmark import process_alive
from tools.train_next_v2.jamendo_audit_v2.native import read_unseeked_block,FFMPEG

JOB='e-fabio-clock-diagnosis-v1'
DIRECTORY=BASE/'annotation_pilot'/JOB


def state(stage,**values):
    value=dict(job_id=JOB,stage=stage,updated_at=now(),**values)
    atomic_json(DIRECTORY/'status.json',value)
    line=json.dumps(value,ensure_ascii=False)
    with (DIRECTORY/'console.log').open('a',encoding='utf-8') as log:log.write(line+'\n')
    print(line,flush=True)


def status():
    p=read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity=p.get('process_identity')
    return dict(job_id=JOB,checked_at=now(),process_identity=identity,process_alive=process_alive(identity) if identity else False,
        os_lease_active=is_running(JOB),progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None)


def run(launch):
    annotations={s['source']['song_id']:s for s in read_json(ROOT/launch['annotation_snapshot'])['songs']}
    rows=[];(DIRECTORY/'records').mkdir(exist_ok=True)
    config_hash=sha256(DIRECTORY/'launch.json')
    for item in launch['songs']:
        output=DIRECTORY/'records'/(item['annotation_song_id']+'.json')
        if output.exists():
            row=read_json(output)
            if row['config_sha256']!=config_hash or row['audio_sha256']!=item['audio_sha256']:raise ValueError('Saved diagnosis changed')
        else:
            path=ROOT/item['audio_path'];samples,rate=sf.read(path,dtype='float32',always_2d=True)
            if not len(samples) or not np.isfinite(samples).all():raise ValueError('Invalid native decode')
            frames=len(samples);channels=samples.shape[1];bulk_hash=hashlib.sha256(samples.tobytes(order='C')).hexdigest()
            del samples
            streamed=hashlib.sha256();stream_frames=0
            with sf.SoundFile(path) as stream:
                while True:
                    block=read_unseeked_block(stream,65536)
                    if not len(block):break
                    if not np.isfinite(block).all():raise ValueError('Non-finite sequential samples')
                    streamed.update(block.tobytes(order='C'));stream_frames+=len(block)
            command=[str(FFMPEG),'-nostdin','-hide_banner','-v','error','-xerror','-err_detect','explode','-threads','1',
                '-i',str(path),'-map','0:a:0','-f','f32le','-acodec','pcm_f32le','-']
            ff=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if ff.returncode or ff.stderr.strip() or len(ff.stdout)%(4*channels):raise ValueError('Strict second decode failed: '+ff.stderr.decode('utf-8','replace')[-500:])
            ff_frames=len(ff.stdout)//(4*channels);ff_hash=hashlib.sha256(ff.stdout).hexdigest();del ff
            annotation=annotations[item['annotation_song_id']];bounds={}
            for axis in ('valence','arousal'):
                takes=[]
                for take in annotation['takes']:
                    if axis not in take['axes']:continue
                    points=[p for p in take['points'] if p.get(axis) is not None and p.get('status')=='annotated']
                    times=[p['time'] for p in points]
                    takes.append(dict(take_id=take['id'],points=len(points),start=min(times) if times else None,
                        end=max(times) if times else None,out_of_bounds=sum(t<0 or t>frames/rate for t in times)))
                bounds[axis]=dict(takes=takes,declared_completed=annotation['completed'][axis])
            row=dict(**item,config_sha256=config_hash,actual_frames=frames,actual_sample_rate=rate,actual_channels=channels,
                actual_duration_seconds=frames/rate,header_minus_actual_frames=item['header_frames']-frames,
                browser_minus_actual_seconds=item['browser_duration_seconds']-frames/rate,
                bulk_stream_exact=(stream_frames==frames and streamed.hexdigest()==bulk_hash),
                bulk_float32_sha256=bulk_hash,stream_float32_sha256=streamed.hexdigest(),stream_frames=stream_frames,
                ffmpeg_argv=command,ffmpeg_returncode=0,ffmpeg_stderr_empty=True,ffmpeg_frames=ff_frames,
                ffmpeg_minus_native_frames=ff_frames-frames,ffmpeg_float32_sha256=ff_hash,
                independent_axis_bounds=bounds,original_native_gate_overridden=False,timeline_scaled=False,
                labels_shifted=False,labels_modified=False,training_admitted=False)
            atomic_json(output,row)
        rows.append(row);state('diagnosing',done=len(rows),total=9)
    verify_hashes(launch['protected_hashes'])
    result=dict(job_id=JOB,status='completed_diagnosis_no_gate_override',completed_at=now(),songs=9,
        bulk_stream_exact_songs=sum(r['bulk_stream_exact'] for r in rows),
        second_decoder_equal_frame_count_songs=sum(r['ffmpeg_minus_native_frames']==0 for r in rows),
        maximum_abs_browser_minus_actual_seconds=max(abs(r['browser_minus_actual_seconds']) for r in rows),
        header_count_mismatch_songs=sum(r['header_minus_actual_frames']!=0 for r in rows),
        out_of_bounds_annotation_points=sum(t['out_of_bounds'] for r in rows for a in r['independent_axis_bounds'].values() for t in a['takes']),
        evidence=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha256(p)) for p in sorted((DIRECTORY/'records').glob('*.json'))],
        original_native_gate_overridden=False,source_files_unchanged=True,training_admitted=False,final_test_targets_read=0,
        conclusion='Header-estimated frame counts must not be used to stretch annotations. Actual end-of-file sample counts and separate V/A timestamps are preserved. This diagnostic does not overrule the frozen native gate or certify browser decoder priming/psychological label quality.')
    atomic_json(DIRECTORY/'report.json',result);state(result['status'],bulk_stream_exact_songs=result['bulk_stream_exact_songs'],second_decoder_equal_frame_count_songs=result['second_decoder_equal_frame_count_songs'])
    record_event(dict(event='E_fabio_clock_diagnosis_completed',**result),run=dict(run_id=JOB,stage='E',status='completed',report=(DIRECTORY/'report.json').relative_to(ROOT).as_posix()))


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch',type=Path);group.add_argument('--status',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.status:print(json.dumps(status()));return
    if args.launch.resolve()!=(DIRECTORY/'launch.json').resolve():raise ValueError('Wrong launch file')
    launch=read_json(args.launch);verify_hashes(launch['protected_hashes'])
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(launch['python']).resolve():raise ValueError('Wrong cwd/runtime')
    if any(os.environ.get(k)!=v for k,v in launch['environment'].items()):raise ValueError('Frozen environment changed')
    if (DIRECTORY/'report.json').exists():raise ValueError('Completed diagnostic must not rerun')
    previous=status()
    if previous['process_alive'] or previous['os_lease_active']:raise ValueError('Existing process still active')
    if (DIRECTORY/'process.json').exists() and not args.resume:raise ValueError('Diagnose previous attempt and use --resume')
    with run_lease(JOB):
        atomic_json(DIRECTORY/'process.json',dict(process_identity=process_identity(),started_at=now(),argv=sys.argv))
        record_event(dict(event='E_fabio_clock_diagnosis_started',job_id=JOB,process_identity=process_identity()),run=dict(run_id=JOB,stage='E',status='running'))
        try:run(launch)
        except BaseException:
            state('failed',error=traceback.format_exc())
            record_event(dict(event='E_fabio_clock_diagnosis_failed',job_id=JOB),run=dict(run_id=JOB,stage='E',status='failed'))
            raise


if __name__=='__main__':main()
