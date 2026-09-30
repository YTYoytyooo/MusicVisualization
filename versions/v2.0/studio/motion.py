"""Deterministic emotion-to-motion planning and editable interval overlays.

Scores express an artistic mapping, not model confidence. Original predictions
are never normalized or overwritten. No renderer or neural model is imported.
"""
from bisect import bisect_right
from copy import deepcopy
import math
import numpy as np
from motion_schema import (MODE_LABELS, MOTION_DEFAULTS, PARAM_BOUNDS, catalog,
                           mode_params, validate_config)


def ease(value, interpolation='smootherstep'):
    u = min(1.,max(0.,float(value)))
    if interpolation=='linear':return u
    if interpolation=='smoothstep':return u*u*(3-2*u)
    return u*u*u*(u*(6*u-15)+10)


def override_weight(edit, time_us):
    t = int(time_us)
    start,end = edit['start_us'],edit['end_us']
    if start<=t<=end:return 1.
    if t<start:
        width = edit['transition_in_us']
        return ease((t-start+width)/width,edit['interpolation']) if width and t>=start-width else 0.
    width = edit['transition_out_us']
    return 1-ease((t-end)/width,edit['interpolation']) if width and t<=end+width else 0.


def blend_frames(left, right, amount):
    """Mix mode weights, merge equal modes, keep circular/discrete fields typed."""
    amount = min(1.,max(0.,float(amount)))
    groups = {}
    for frame,factor in ((left,1-amount),(right,amount)):
        for component in frame['components']:
            weight = factor*component['weight']
            if weight>1e-12:
                groups.setdefault(component['mode'],[]).append((weight,component['params']))
    components = []
    total = sum(w for entries in groups.values() for w,_ in entries)
    if total<=0:raise ValueError('运动混合权重为空')
    for mode,entries in groups.items():
        weight = sum(w for w,_ in entries)
        first = entries[0][1]
        params = {}
        for key in PARAM_BOUNDS:
            if key=='direction_deg':
                base=first[key]
                params[key]=(base+sum(w*((p[key]-base+180)%360-180) for w,p in entries)/weight)%360
            elif key=='rotation':
                params[key]=max(entries,key=lambda item:item[0])[1][key]
            else:
                low,high=PARAM_BOUNDS[key]
                params[key]=min(high,max(low,sum(w*p[key] for w,p in entries)/weight))
        components.append({'mode':mode,'weight':weight/total,'params':params})
    result = deepcopy(right if amount>=1 else left)
    result['components']=components
    result['mode']=max(components,key=lambda c:c['weight'])['mode']
    return result


def sample_frame(plan, time_us, kind='effective'):
    times=plan['times_us']
    frames=plan.get(kind,plan.get('auto',[]))
    if not len(times) or len(frames)!=len(times):
        raise ValueError('运动时间线为空或长度不匹配')
    index=max(0,min(len(times)-1,bisect_right(times,int(time_us))-1))
    if index==len(times)-1 or time_us<=times[index]:return deepcopy(frames[index])
    ratio=(time_us-times[index])/(times[index+1]-times[index])
    return blend_frames(frames[index],frames[index+1],ratio)


def segments_for(times, frames, duration_us):
    segments=[]
    for i,frame in enumerate(frames):
        t=int(times[i]);end=int(times[i+1]) if i+1<len(times) else int(duration_us)
        if end<=t:continue
        if segments and (segments[-1]['mode'],segments[-1]['source'])==(frame['mode'],frame['source']):
            segments[-1]['end_us']=end
        else:
            segments.append({'start_us':t,'end_us':end,'mode':frame['mode'],
                             'source':frame['source'],'reason':frame['reason']})
    return segments


def apply_overrides(plan, config, duration_us):
    config=validate_config(config,int(duration_us))
    if config['engine']=='legacy':
        return {'config':config,'times_us':[],'auto':[],'effective':[],'segments':[],'modes':catalog()}
    points=set(int(t) for t in plan['times_us'])
    for edit in config['overrides']:
        if not edit['enabled']:continue
        start,end=edit['start_us'],edit['end_us']
        left,right=start-edit['transition_in_us'],end+edit['transition_out_us']
        points.update((left,start,end,right))
        for a,b in ((left,start),(end,right)):
            points.update(a+(b-a)*i//8 for i in range(1,8))
        if not edit['transition_in_us']:points.add(max(0,start-1))
        if not edit['transition_out_us']:points.add(min(duration_us,end+1))
    times=sorted(t for t in points if 0<=t<=duration_us)
    automatic=[sample_frame(plan,t,'auto') for t in times]
    effective=[]
    for t,base in zip(times,automatic):
        current=deepcopy(base)
        for edit in config['overrides']:
            if not edit['enabled']:continue
            weight=override_weight(edit,t)
            if weight<=1e-12:continue
            target={'mode':edit['mode'],'source':'manual','reason':edit['note'] or '人工指定：'+MODE_LABELS[edit['mode']],
                    'components':[{'mode':edit['mode'],'weight':1.,'params':edit['params']}]}
            current=blend_frames(base,target,weight)
            current['source']='manual' if weight>=1-1e-12 else 'transition'
            current['reason']=target['reason'] if weight>=1-1e-12 else '人工过渡：'+MODE_LABELS[edit['mode']]
            current['override_id']=edit['id']
            break
        effective.append(current)
    return {'config':config,'times_us':times,'auto':automatic,'effective':effective,
            'segments':segments_for(times,effective,duration_us),'modes':catalog()}


def _scores(v,a,activity,pulse,trend):
    # Configurable artistic prototypes. Relative changes complement, but never
    # replace, the absolute V/A coordinates; near-constant songs are not stretched.
    prototypes={'rise':(.65,.35),'fall':(-.65,-.60),'orbit':(.35,-.65),
                'wave':(-.05,-.20),'turbulent':(-.60,.75),'spiral':(.05,.45),
                'expand':(.65,.85),'gather':(-.45,.05),'meteor':(.35,.65)}
    scores={mode:math.exp(-((v-x)**2/.6**2+(a-y)**2/.6**2)/2)
            for mode,(x,y) in prototypes.items()}
    scores['meteor']*=.55+.35*activity+.25*pulse
    scores['expand']*=.7+.3*activity+.2*pulse
    scores['spiral']+=max(0.,trend)*.2
    scores['rise']+=max(0.,trend)*.1
    scores['gather']+=max(0.,trend)*.12 if v<.2 else 0
    return scores


def automatic_plan(times_us, values, features, config, seed=42, beat_times_us=()):
    config=validate_config(config)
    times=np.asarray(times_us,dtype=np.int64)
    values=np.asarray(values,dtype=float)
    if (not len(times) or len(values)!=len(times) or values.ndim!=2 or values.shape[1]<2
            or not np.isfinite(values).all() or np.any(np.diff(times)<=0)):
        raise ValueError('自动运动规划的时间和数值不合法')
    ft=np.asarray(features['times_us'])
    energy=np.interp(times,ft,features['activity'])
    pulse=np.interp(times,ft,features['pulse_strength'])
    rms=np.interp(times,ft,features['rms'])
    beats=np.asarray(beat_times_us,dtype=np.int64)
    output=[];v,a=map(float,values[0,:2]);previous_a=a
    current=prior=candidate=None
    candidate_since=0;switched=-int(config['min_hold_seconds']*1e6)
    wait_since=None;previous_time=int(times[0]);pulse_mean=0.;activity=float(energy[0])
    for index,t in enumerate(times):
        t=int(t);dt=max(0.,(t-previous_time)/1e6);previous_time=t
        alpha=1-math.exp(-dt/.65)
        v+=(float(values[index,0])-v)*alpha
        a+=(float(values[index,1])-a)*alpha
        activity+=(float(energy[index])-activity)*(1-math.exp(-dt/.25))
        pulse_mean+=(float(pulse[index])-pulse_mean)*(1-math.exp(-dt/.6))
        # Bounded derivative, not per-song min/max scaling.
        trend=max(-1.,min(1.,(a-previous_a)/max(.1,dt)*config['sensitivity']))
        previous_a=a
        scores=_scores(v,a,activity,pulse_mean,trend)
        best=max(scores,key=scores.get)
        if current is None:current=prior=best;switched=t
        if best!=current and scores[best]>scores[current]+.10:
            if best!=candidate:candidate=best;candidate_since=t;wait_since=None
            ready=t-candidate_since>=1000000 and t-switched>=config['min_hold_seconds']*1e6
            if ready:
                if wait_since is None:wait_since=t
                beat_index=int(np.searchsorted(beats,t-100000)) if len(beats) else 0
                near_beat=bool(len(beats) and beat_index<len(beats) and abs(int(beats[beat_index])-t)<=100000)
                if near_beat or t-wait_since>=400000 or not len(beats):
                    prior,current=current,best;switched=t;candidate=None;wait_since=None
        else:candidate=None;wait_since=None
        amount=ease((t-switched)/max(1,config['transition_seconds']*1e6)) if prior!=current else 1.
        components=[]
        for mode,weight in ((prior,1-amount),(current,amount)):
            if weight<=1e-12:continue
            params=mode_params(mode)
            an=(a+1)/2
            scale=.45+.7*an+.45*activity
            params['speed']=min(.5,params['speed']*scale)
            # Physical silence should not become an energetic "local maximum".
            params['speed']*=min(1.,max(0.,float(rms[index]))/.008)
            params['turbulence']=min(1.,params['turbulence']*(.6+.5*an))
            params['rotation']=1. if int(seed)%2==0 else -1.
            if mode=='spiral':params['radial']=max(-.8,min(.8,.25+.6*trend))
            params['pulse']*=min(1.,activity*2)
            components.append({'mode':mode,'weight':weight,'params':params})
        chosen=max(components,key=lambda c:c['weight'])['mode']
        reason=f'艺术映射：V={v:+.2f}，A={a:+.2f}；强弱={activity:.2f}'
        if prior!=current and amount<1:reason+='；平滑切换中'
        output.append({'mode':chosen,'source':'auto','reason':reason,'components':components,
                       'scores':{k:round(value,6) for k,value in scores.items()}})
    return output


def build_motion_plan(path, raw, edits, smoothing, config=None, progress=None, cancel=None):
    """Build a small control timeline independent of the visual MCTS queue."""
    from .store import load_project
    p=load_project(path)
    config=validate_config(config,p['duration_us'])
    if config['engine']=='legacy':
        return apply_overrides({'times_us':[],'auto':[]},config,p['duration_us'])
    from .motion_features import load_motion_features, MotionFeatureCancelled
    from .smoothing import emotion_values
    from .timeline import event_times
    features=load_motion_features(path,progress=progress,cancel=cancel)
    if cancel and cancel():raise MotionFeatureCancelled('用户取消运动规划')
    times=np.array(sorted(set(range(0,p['duration_us'],100000))|{p['duration_us']}|event_times(edits,'emotion')),dtype=np.int64)
    values=emotion_values(raw,times,edits,p['step_us'],smoothing)
    auto=automatic_plan(times,values,features,config,p['seed'],np.asarray(p['beat_times'])*1e6)
    result=apply_overrides({'times_us':times.tolist(),'auto':auto},config,p['duration_us'])
    result['feature_key']=features['key']
    result['feature_kind']=features['kind']
    result['audio_features']={'times_us':features['times_us'].tolist(),
                              **{k:features[k].tolist() for k in ('rms','onset','activity','pulse_strength')}}
    return result


def frame_summary(frame):
    """Serializable actual control values; categorical mixtures stay explicit."""
    dominant=max(frame['components'],key=lambda c:c['weight'])
    return {'mode':frame['mode'],'source':frame['source'],'reason':frame['reason'],
            'components':frame['components'],'params':dominant['params']}
