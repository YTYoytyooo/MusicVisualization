"""Finite isolated upgrade acceptance; synthetic long-song stress + real model smoke.

Run with --stress for complete 3/10 minute synthetic rendering (approximate
320x180, 10 fps); --model-smoke uses existing local model and 6s WAV fixture.
Neither mode edits user projects or downloads dependencies/models.
"""
import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch
import cv2
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from studio.paths import DATA_ROOT, DEFAULT_PROJECTS, DEFAULT_MODEL
from studio.demo import create_demo
from studio.pipeline import render, analyze, ffmpeg_path
from studio.store import atomic_json, digest, load_project, save_revision, create_project
from renderer import VideoRenderer


def peak_memory():
    if os.name!='nt':return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    value=Counters();value.cb=ctypes.sizeof(value)
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
    memory=ctypes.WinDLL('psapi').GetProcessMemoryInfo
    memory.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    memory.restype=wintypes.BOOL
    if not memory(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb):return None
    return value.PeakWorkingSetSize


def inspect_video(project,result):
    import subprocess
    folder=project/'renders'/result['id']
    cap=cv2.VideoCapture(str(folder/result['video_file']))
    count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            count+=1
    finally:cap.release()
    with open(folder/'frame_values.csv',encoding='utf-8-sig') as stream:
        rows=sum(1 for _ in csv.DictReader(stream))
    assert count==rows==result['frames'],(count,rows,result['frames'])
    audio=subprocess.run([ffmpeg_path(),'-v','error','-i',str(folder/result['video_file']),'-map','0:a:0',
        '-ac','1','-ar','8000','-f','s16le','-'],capture_output=True,check=True,timeout=30)
    audio_seconds=len(audio.stdout)/16000
    assert abs(audio_seconds-count/result['settings']['fps'])<.15
    return {'decoded_frames':count,'csv_rows':rows,'audio_seconds':audio_seconds,'video':str(folder/result['video_file'])}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stress',action='store_true');parser.add_argument('--model-smoke',action='store_true')
    args=parser.parse_args()
    target=DATA_ROOT / 'validation-output'/'upgrade-acceptance';target.mkdir(parents=True,exist_ok=True)
    report={'status':'running','synthetic_stress':[],'real_model':None,'started':time.time()}
    output=target/'report.json'
    def progress(stage,ratio):
        bucket=int(ratio*10)
        if progress.last!=(stage,bucket):
            print(stage,round(ratio*100),'%',flush=True);progress.last=(stage,bucket)
    progress.last=None
    try:
        if args.stress:
            for duration in (180,600):
                audio=target/f'synthetic-{duration}s.wav'
                t=np.arange(duration*22050)/22050
                sf.write(audio,(.12*np.sin(t*2*np.pi*220)).astype(np.float32),22050)
                clock=np.arange(duration*10)/10
                raw=np.zeros((len(clock),5));raw[:,0]=.2*np.sin(clock/8);raw[:,1]=.3+.1*np.cos(clock/13)
                project=create_project(target,f'SYNTHETIC STRESS / {duration}s',audio,None,raw,
                    {'duration':duration,'tempo':120.,'beat_times':np.arange(0,duration,.5)},
                    provenance={'kind':'synthetic'})
                before=digest(project/'analysis/predictions_raw.npz')
                samples=[];original=VideoRenderer.render_frame
                def measured(renderer,*a,**kw):
                    started=time.perf_counter();frame=original(renderer,*a,**kw);samples.append(time.perf_counter()-started);return frame
                begin=time.perf_counter()
                with patch.object(VideoRenderer,'render_frame',measured), patch('feature_extraction.load_audio',side_effect=lambda p:sf.read(p,dtype='float32')):
                    result=render(project,options={'width':320,'height':180,'fps':10},progress=progress)
                inspected=inspect_video(project,result)
                midpoint=len(samples)//2
                record={'seconds':duration,'source':'synthetic; not real model prediction','project':str(project),
                    'wall_seconds':time.perf_counter()-begin,'first_half_frame_ms':statistics.median(samples[:midpoint])*1000,
                    'second_half_frame_ms':statistics.median(samples[midpoint:])*1000,'process_peak_working_set_bytes':peak_memory(),**inspected}
                with patch('feature_extraction.load_audio',side_effect=lambda p:sf.read(p,dtype='float32')):
                    start=time.perf_counter()
                    clip=render(project,options={'width':320,'height':180,'fps':10,'start':duration-2},progress=progress)
                    record['tail_preview_seconds']=time.perf_counter()-start
                    record['tail_resumed_from_frame']=clip['resumed_from_frame']
                assert clip['resumed_from_frame']>0
                assert digest(project/'analysis/predictions_raw.npz')==before
                report['synthetic_stress'].append(record);atomic_json(output,report)
        if args.model_smoke:
            model=DEFAULT_MODEL;audio=DATA_ROOT / 'validation-output/battleThemeA-validation-6s.wav'
            hashes=[digest(model),digest(audio)]
            project=analyze(audio,model,target,name='REAL MODEL / UPGRADE ACCEPTANCE',progress=progress)
            raw_hash=digest(project/'analysis/predictions_raw.npz')
            initial=render(project,options={'width':640,'height':360,'fps':30},progress=progress)
            edit=dict(id='acceptance',layer='emotion',field='arousal',operation='point_target',time_us=3000000,
                value=.8,transition_in_us=1000000,transition_out_us=1000000,interpolation='smootherstep')
            revision=save_revision(project,'r000000',[edit],smoothing={'enabled':True,'window_seconds':.5})
            revised=render(project,revision['id'],options={'width':640,'height':360,'fps':30},progress=progress)
            videos=[inspect_video(project,r) for r in (initial,revised)]
            with open(project/'renders'/revised['id']/'frame_values.csv',encoding='utf-8-sig') as stream:
                row=next(row for row in csv.DictReader(stream) if row['time_us']=='3000000')
            assert float(row['arousal_effective'])==.8
            assert hashes==[digest(model),digest(audio)] and raw_hash==digest(project/'analysis/predictions_raw.npz')
            report['real_model']={'project':str(project),'videos':videos,'raw_model_audio_preserved':True,'target':.8}
        report['status']='passed';report['finished']=time.time();atomic_json(output,report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));atomic_json(output,report);raise


if __name__=='__main__':main()
