"""Derived display/export artifacts; immutable raw model values remain untouched."""
from __future__ import annotations
import bisect
import colorsys
import csv
import io
import math
from pathlib import Path
import subprocess
import uuid
from . import store

WINDOWS = (0., .5, 1., 2.)


def smooth(times, values, seconds):
    if isinstance(seconds, bool) or seconds not in WINDOWS:
        raise ValueError('Display window must be 0, 0.5, 1 or 2 seconds')
    if len(times) != len(values) or not times or any(not math.isfinite(t) for t in times):
        raise ValueError('Invalid display time axis')
    if any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError('Display time axis must strictly increase')
    if any(not isinstance(v,(list,tuple)) or len(v)!=2 or any(x is not None and
               (isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x)) for x in v) for v in values):
        raise ValueError('Display values must be finite N by 2 values or explicit nulls')
    if not seconds:
        return [list(v) for v in values]
    hop = min((b-a for a,b in zip(times,times[1:])),default=.5)
    output = [[None,None] for _ in values]
    for axis in range(2):
        begin = 0
        while begin < len(times):
            if values[begin][axis] is None:
                begin += 1; continue
            end = begin + 1
            while end < len(times) and values[end][axis] is not None and times[end]-times[end-1] <= hop*1.5+1e-8:
                end += 1
            for i in range(begin,end):
                left=bisect.bisect_left(times,times[i]-seconds/2-1e-8,begin,end)
                right=bisect.bisect_right(times,times[i]+seconds/2+1e-8,begin,end)
                output[i][axis]=sum(values[j][axis] for j in range(left,right))/(right-left)
            begin=end
    return output


def csv_text(result, seconds=None):
    buf=io.StringIO(newline=''); writer=csv.writer(buf)
    writer.writerow(['system','time_seconds','valence_raw' if seconds is None else 'valence_display',
                     'arousal_raw' if seconds is None else 'arousal_display','display_window_seconds'])
    for name, system in result['systems'].items():
        vals=system['raw_va'] if seconds is None else smooth(system['times_seconds'],system['raw_va'],seconds)
        for t,v in zip(system['times_seconds'],vals):
            writer.writerow([name,t,*v,'' if seconds is None else seconds])
    return '\ufeff'+buf.getvalue()


def interp(times, values, t):
    index=bisect.bisect_right(times,t)-1
    if index < 0: return values[0]
    if index >= len(times)-1: return values[-1]
    fraction=(t-times[index])/(times[index+1]-times[index])
    return values[index]*(1-fraction)+values[index+1]*fraction


def particles(t, valence, arousal, dbfs, width, height):
    energy=max(0.,min(1.,(dbfs+60)/60)); a=(arousal+1)/2
    out=[]
    for i in range(80):
        angle=i*2.3999632297+t*(.12+.2*a)
        radius=(30+(i%17)/17*min(width,height)*.34)*(.7+.3*energy)
        out.append((width/2+math.cos(angle)*radius,height/2+math.sin(angle*.87)*radius,
                    1+i%3+3*energy,((valence+1)*120+i%30)%360,.45+.5*energy))
    return out


def export(project_id, seconds, start, end, progress=None, cancelled=None):
    loaded=store.load_project(project_id);result=loaded['result']
    if not result: raise ValueError('Analysis is not complete')
    duration=result['audio']['duration_seconds']
    if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in [start,end])
        or not 0 <= start < end <= duration or end-start > 60):
        raise ValueError('Choose a valid preview range of at most 60 seconds')
    system=result['systems']['b01'];vals=smooth(system['times_seconds'],system['raw_va'],seconds)
    ffmpeg=store.ROOT/'versions/v2.0/.runtime/bin/ffmpeg.exe'
    if not ffmpeg.is_file(): raise FileNotFoundError('Existing local FFmpeg is unavailable')
    directory=store.project_path(project_id)/'exports'/('e-'+uuid.uuid4().hex[:12]);directory.mkdir(parents=True)
    raw_hash=store.digest(store.project_path(project_id)/'analysis/result.json')
    audio=store.inside(store.ROOT/result['audio']['path'])
    if store.digest(audio)!=result['audio']['sha256']: raise ValueError('Audio identity changed')
    width,height,fps=640,360,24;frames=math.ceil((end-start)*fps);seconds_out=frames/fps
    video=directory/'preview.mp4';temp=directory/'preview.partial.mp4'
    argv=[str(ffmpeg),'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
          '-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-ss',str(start),'-i',str(audio),
          '-map','0:v:0','-map','1:a:0','-af','apad','-t',str(seconds_out),'-c:v','libx264',
          '-preset','fast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(temp)]
    store.atomic(directory/'launch.json',{'argv':argv,'cwd':str(store.ROOT),'raw_sha256':raw_hash,
        'start':start,'end':end,'frame_count':frames,'fps':fps,'smoothing_seconds':seconds,
        'visual_mapping':'V3 analytic 80-particle artistic mapping v1; not psychological classification'})
    import numpy as np
    acoustic=result['acoustic'];v=[p[0] for p in vals];a=[p[1] for p in vals]
    with (directory/'ffmpeg.log').open('wb') as log, (directory/'frame_values.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['frame','source_time_seconds','valence_display','arousal_display','rms_dbfs'])
        proc=subprocess.Popen(argv,cwd=store.ROOT,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            for i in range(frames):
                if cancelled and cancelled(): raise InterruptedError('Export cancelled')
                t=start+i/fps
                vi=interp(system['times_seconds'],v,t);ai=interp(system['times_seconds'],a,t)
                energy=interp(acoustic['times_seconds'],acoustic['rms_dbfs'],t)
                im=np.empty((height,width,3),dtype=np.uint8);im[:]=(13,22,37)
                for x,y,r,h,opacity in particles(t,vi,ai,energy,width,height):
                    rgb=tuple(round(c*255) for c in colorsys.hsv_to_rgb(h/360,.65,.95))
                    left,right=max(0,int(x-r)),min(width,math.ceil(x+r)+1)
                    top,bottom=max(0,int(y-r)),min(height,math.ceil(y+r)+1)
                    yy,xx=np.ogrid[top:bottom,left:right]
                    mask=(xx-x)**2+(yy-y)**2<=r*r
                    patch=im[top:bottom,left:right]
                    patch[mask]=np.rint(patch[mask]*(1-opacity)+np.asarray(rgb)*opacity).astype(np.uint8)
                proc.stdin.write(im.tobytes());writer.writerow([i,t,vi,ai,energy])
                if progress and i%24==0: progress({'stage':'正在导出视频','progress':i/frames})
            proc.stdin.close();code=proc.wait(timeout=60)
            if code: raise RuntimeError(f'FFmpeg failed ({code}); inspect {directory / "ffmpeg.log"}')
        except BaseException:
            if proc.poll() is None: proc.terminate();proc.wait(timeout=10)
            raise
    temp.rename(video)
    (directory/'raw.csv').write_text(csv_text(result),encoding='utf-8',newline='')
    (directory/'display.csv').write_text(csv_text(result,seconds),encoding='utf-8',newline='')
    if store.digest(store.project_path(project_id)/'analysis/result.json')!=raw_hash:
        raise ValueError('Raw predictions changed during export')
    metadata={'status':'completed','created_at':store.now(),'directory':directory.relative_to(store.ROOT).as_posix(),
              'raw_sha256':raw_hash,'manifest_sha256':result['manifest_sha256'],'start':start,'requested_end':end,
              'actual_end':start+seconds_out,'fps':fps,'frames':frames,'smoothing_seconds':seconds,
              'audio_video_boundary_tolerance_seconds':1/fps,'artistic_mapping_not_model_prediction':True,
              'files':{p.name:store.digest(p) for p in [video,directory/'frame_values.csv',directory/'raw.csv',directory/'display.csv']}}
    store.atomic(directory/'export.json',metadata)
    store.atomic(store.project_path(project_id)/'exports/latest.json',metadata)
    return metadata
