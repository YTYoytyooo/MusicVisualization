"""Exact clock scoring and paired recording-group uncertainty, without tuning."""
from collections import defaultdict

import numpy as np

from tools.train_next.core import attach_targets
from tools.train_next.diagnostics import correlation
from tools.train_next_v2.cross_version.adapters import common_clock
from tools.train_next_v2.evaluation import metrics_for_axis

AXES = ('valence', 'arousal')
SYSTEMS = ('v2', 'mixed', 'b01_s20261003', 'b01_s20261004', 'b01_s20261005', 'b01_ensemble')


def align_labels(times, v2_times, label_times, labels, mask, duration):
    lt, y, m = np.asarray(label_times), np.asarray(labels), np.asarray(mask)
    if (lt.ndim != 1 or len(lt)<2 or not np.isfinite(lt).all() or np.any(np.diff(lt)<=0)
            or y.shape!=(len(lt),2) or m.shape!=y.shape
            or m.dtype!=np.bool_ or not np.isfinite(y[m]).all() or np.any(np.abs(y[m])>1)
            or not m.any()):
        raise ValueError('Invalid validation label clock, values, or independent masks')
    observed=m.any(axis=1)
    if np.any(lt[observed]<0) or np.any(lt[observed]>duration+1e-7):
        raise ValueError('Observed label outside native audio time')
    native = float(np.min(np.diff(lt)))
    if native not in (.5,1.) or not np.allclose(np.diff(lt)/native,np.rint(np.diff(lt)/native),rtol=0,atol=1e-7):
        raise ValueError('Unexpected native label cadence')
    target, original = attach_targets(times,lt[observed],np.where(m,y,0)[observed],m[observed])
    indices, common = common_clock(times,v2_times,original)
    if not np.array_equal(original.sum(0),m.sum(0)):
        raise ValueError('Labels lost during exact join')
    return dict(y=target, original_mask=original, common_mask=common, v2_indices=indices,
                excluded_values=(original.sum(0)-common.sum(0)).astype(np.int64), native_seconds=native)


def lag_diagnostic(times, truth, pred, mask, native):
    """Same central targets at all shifts; positive lag means prediction follows."""
    observed=np.flatnonzero(mask)
    out=dict(best_seconds=None, best_correlation=None, zero_correlation=None, improvement=None,
             credible=False, status='insufficient_points', points=0)
    if len(observed)<12: return out
    stride=round(native/.5)
    radius=int(min(8/.5,(observed[-1]-observed[0])/4))//stride*stride
    if not radius: return out
    core=observed[(observed>=observed[0]+radius)&(observed<=observed[-1]-radius)]
    if len(core)<12: return out
    shifts=np.arange(-radius,radius+1,stride,dtype=int)
    cc=[correlation(truth[core],pred[core+s]) for s in shifts]
    finite=[i for i,v in enumerate(cc) if v is not None and np.isfinite(v)]
    if not finite: return dict(out,status='constant_unidentifiable')
    best=max(finite,key=lambda i:(cc[i],-abs(shifts[i])))
    zero=cc[list(shifts).index(0)]; gain=cc[best]-zero if zero is not None else None
    if np.std(truth[core])<.02 or np.std(pred[core])<.005: flag='low_variation'
    elif cc[best]<.3: flag='weak_correlation'
    elif abs(shifts[best])==radius: flag='search_boundary'
    elif sum(abs(cc[i]-cc[best])<1e-8 for i in finite)>1: flag='ambiguous_peak'
    elif gain is None or gain<.05: flag='no_clear_delay'
    else: flag='candidate_requires_curve_review'
    return dict(out,best_seconds=float(shifts[best]*.5),best_correlation=cc[best],zero_correlation=zero,
                improvement=gain,credible=flag=='candidate_requires_curve_review',status=flag,points=len(core),
                search_seconds=(shifts*.5).tolist(),correlations=cc)


def song_scores(times, truth, mask, predictions, native):
    t,y,m,p=map(np.asarray,(times,truth,mask,predictions))
    if (p.shape!=(len(SYSTEMS),len(t),2) or y.shape!=(len(t),2) or m.shape!=y.shape
            or m.dtype!=np.bool_ or not np.isfinite(y[m]).all()
            or not np.isfinite(p[:,m]).all() or np.any(np.abs(p[:,m])>1)
            or native not in (.5,1.)):
        raise ValueError('Invalid scoring arrays')
    song=dict(times=t,y=y,mask=m,native_seconds=native)
    results={}
    for s,pred in zip(SYSTEMS,p):
        results[s]={}
        for a,axis in enumerate(AXES):
            row=metrics_for_axis(pred,song,a)
            if row is not None:
                row.update(truth_mean=float(y[m[:,a],a].astype(float).mean()),
                           prediction_mean=float(pred[m[:,a],a].astype(float).mean()),
                           lag=lag_diagnostic(t,y[:,a],pred[:,a],m[:,a],native))
            results[s][axis]=row
    return results


def recording_groups(rows):
    if len({r['song_id'] for r in rows})!=len(rows): raise ValueError('Duplicate song IDs')
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    seen={}
    for i,r in enumerate(rows):
        for field in ('group_id','audio_sha256'):
            if not r.get(field): raise ValueError('Missing grouping identity')
            key=(field,r[field])
            if key in seen: parent[root(i)]=root(seen[key])
            else: seen[key]=i
    groups={}; ids=[]
    for i in range(len(rows)):
        k=root(i)
        if k not in groups: groups[k]=len(groups)
        ids.append(groups[k])
    ids=np.asarray(ids,dtype=np.int64)
    members=[[i for i,g in enumerate(ids) if g==j] for j in range(len(groups))]
    domains=[sorted({rows[i]['source_dataset'] for i in group}) for group in members]
    return ids,dict(groups=len(groups),cross_dataset_components=any(len(d)>1 for d in domains),
                    members=[[rows[i]['song_id'] for i in group] for group in members],domains=domains)


def nullable(v):
    return float(v) if np.isfinite(v) else None


def interval(values, point, confidence):
    finite=values[np.isfinite(values)]
    q=(1-confidence)/2
    return dict(point=nullable(point),lower=nullable(np.quantile(finite,q)) if len(finite) else None,
                upper=nullable(np.quantile(finite,1-q)) if len(finite) else None,
                valid_replicates=len(finite),total_replicates=len(values))


def summarize(rows, bootstrap, metrics):
    """Song macro within each of six cells; identical cluster draws for all systems."""
    if not rows: raise ValueError('No completed validation songs')
    domains=sorted({r['source_dataset'] for r in rows})
    cells=[(d,a) for d in domains for a in AXES]
    ids,grouping=recording_groups(rows)
    n_groups=grouping['groups']; n_sys=len(SYSTEMS); n_cells=len(cells); n_metrics=len(metrics)
    sums=np.zeros((n_groups,n_sys,n_cells,n_metrics),np.float64); counts=np.zeros_like(sums)
    songs_per_cell={f'{d}/{a}':0 for d,a in cells}
    for i,row in enumerate(rows):
        for a in AXES:
            c=cells.index((row['source_dataset'],a))
            if row['scores']['v2'][a] is not None: songs_per_cell[f'{row["source_dataset"]}/{a}']+=1
            for s,system in enumerate(SYSTEMS):
                data=row['scores'][system][a]
                if data is None: continue
                for k,metric in enumerate(metrics):
                    value=data.get(metric)
                    if value is not None:
                        if not np.isfinite(value): raise ValueError('Non-finite saved metric')
                        sums[ids[i],s,c,k]+=value; counts[ids[i],s,c,k]+=1
    def divide(n,d):
        return np.divide(n,d,out=np.full_like(n,np.nan),where=d>0)
    points=divide(sums.sum(0),counts.sum(0))
    rng=np.random.default_rng(bootstrap['seed']); reps=bootstrap['replicates']
    weights=np.zeros((reps,n_groups),np.float64)
    blocks=[list(range(n_groups))] if grouping['cross_dataset_components'] else [
        [i for i,d in enumerate(grouping['domains']) if d==[domain]] for domain in domains]
    for block in blocks:
        if not block: raise ValueError('Empty bootstrap stratum')
        weights[:,block]=rng.multinomial(len(block),np.full(len(block),1/len(block)),size=reps)
    draws=divide(weights@sums.reshape(n_groups,-1),weights@counts.reshape(n_groups,-1)).reshape(reps,n_sys,n_cells,n_metrics)
    # Applicability (including constant-target CCC) depends only on the shared targets.
    if not all(np.array_equal(counts[:,0],counts[:,s]) for s in range(1,n_sys)):
        raise ValueError('Metric applicability differs between models')
    names=SYSTEMS+('b01_mean',)
    points=np.concatenate([points,points[2:5].mean(0,keepdims=True)],axis=0)
    draws=np.concatenate([draws,draws[:,2:5].mean(1,keepdims=True)],axis=1)
    overall=points.mean(1); overall_draws=draws.mean(2)
    systems={}
    for s,name in enumerate(names):
        systems[name]=dict(cells={f'{d}/{a}':{m:nullable(points[s,c,k]) for k,m in enumerate(metrics)} for c,(d,a) in enumerate(cells)},
                           overall={m:nullable(overall[s,k]) for k,m in enumerate(metrics)})
    diffs={}
    for candidate,reference in [('mixed','v2'),('b01_mean','v2'),('b01_mean','mixed'),('b01_ensemble','v2')]:
        ci,ri=names.index(candidate),names.index(reference)
        values=draws[:,ci]-draws[:,ri]; point=points[ci]-points[ri]
        diffs[candidate+'_minus_'+reference]=dict(
            cells={f'{d}/{a}':{m:interval(values[:,c,k],point[c,k],bootstrap['confidence']) for k,m in enumerate(metrics)} for c,(d,a) in enumerate(cells)},
            overall={m:interval(overall_draws[:,ci,k]-overall_draws[:,ri,k],overall[ci,k]-overall[ri,k],bootstrap['confidence']) for k,m in enumerate(metrics)})
    seed_std=points[2:5].std(0,ddof=1); overall_std=overall[2:5].std(0,ddof=1)
    return dict(systems=systems,paired_differences=diffs,
        b01_seed_sample_std=dict(cells={f'{d}/{a}':{m:nullable(seed_std[c,k]) for k,m in enumerate(metrics)} for c,(d,a) in enumerate(cells)},
                                  overall={m:nullable(overall_std[k]) for k,m in enumerate(metrics)}),
        songs_per_cell=songs_per_cell,metric_song_counts={f'{d}/{a}':{m:int(counts.sum(0)[0,c,k]) for k,m in enumerate(metrics)} for c,(d,a) in enumerate(cells)},
        bootstrap=dict(**bootstrap,mode='joint' if grouping['cross_dataset_components'] else 'stratified_by_dataset',pairing='same cluster multiplicities for all systems and seeds',grouping=grouping))
