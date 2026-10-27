"""Read labels without modifying them or granting psychological/training approval."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.annotate.server import reviewer_id, CONTINUOUS_PROTOCOL
from tools.train_next.core import atomic_json
from tools.train_next_v2.common import read_json, now


def inspect_document(doc,row):
    if doc.get('schema_version')!=2 or doc.get('protocol')!=CONTINUOUS_PROTOCOL or doc.get('label_scope')!='continuous_raw':
        raise ValueError('Only original continuous raw V/A documents are accepted')
    reviewer=reviewer_id(doc['reviewer'])
    if doc['source']['audio_sha256']!=row['audio_sha256']: raise ValueError('Source hash mismatch')
    native=row['native_audio'];duration=native['frames']/native['sample_rate']
    if not isinstance(doc.get('duration'),(int,float)) or isinstance(doc['duration'],bool) or not math.isfinite(doc['duration']) or abs(doc['duration']-duration)>.15:
        raise ValueError('Browser/native duration mismatch; never stretch timestamps')
    completed=doc.get('completed')
    if not isinstance(completed,dict) or set(completed)!={'valence','arousal'} or any(type(v)is not bool for v in completed.values()):
        raise ValueError('Explicit independent boolean completion flags required')
    ids=set();counts={a:0 for a in completed};points=0
    for take in doc['takes']:
        if not isinstance(take['id'],str) or not take['id'] or take['id'] in ids: raise ValueError('Duplicate/invalid take ID')
        ids.add(take['id']);axes=take['axes']
        if not isinstance(axes,list) or not axes or len(set(axes))!=len(axes) or any(a not in completed for a in axes): raise ValueError('Invalid independent axes')
        previous=-1
        for point in take['points']:
            t=point['time']
            if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or not 0<=t<=duration or t<=previous:
                raise ValueError('Invalid/out-of-bounds raw time within take')
            previous=t;points+=1
            if point['status'] not in ('annotated','uncertain'): raise ValueError('Invalid point status')
            for axis in completed:
                v=point.get(axis)
                if point['status']=='annotated' and axis in axes:
                    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not -1<=v<=1: raise ValueError('Invalid annotated axis value')
                    counts[axis]+=1
                elif v is not None: raise ValueError('Missing/inactive axis must stay missing')
    for marker in doc.get('transitions',[]):
        t=marker.get('time')
        if isinstance(t,bool) or not isinstance(t,(float,int)) or not math.isfinite(t) or not 0<=t<=duration: raise ValueError('Invalid raw transition time')
    if any(completed[a] and counts[a]==0 for a in completed): raise ValueError('Completion flag without actual axis samples')
    return dict(reviewer=reviewer,declared_completed=completed,valid_axis_points=counts,raw_points=points,
        psychological_quality_verified=False,independent_person_verified=False)


def validate(pool,inputs):
    by_hash={r['audio_sha256']:r for r in pool['candidates']};seen={};errors=[];outside=[]
    for path in inputs:
        document=read_json(path);docs=document.get('songs',[document])
        for doc in docs:
            digest=(doc.get('source') or {}).get('audio_sha256')
            if digest not in by_hash:
                outside.append(dict(path=str(path),audio_sha256=digest));continue
            try:
                result=inspect_document(doc,by_hash[digest]);key=(digest,result['reviewer'])
                canonical=hashlib.sha256(json.dumps(doc,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
                if key in seen:
                    if seen[key]['document_sha256']!=canonical:
                        seen[key]['conflicting_returns']=True
                        raise ValueError('Conflicting returns for same recording/reviewer; reconcile explicitly')
                    continue
                seen[key]=dict(**result,document_sha256=canonical,conflicting_returns=False)
            except (ValueError,KeyError,TypeError) as exc:
                errors.append(dict(path=str(path),audio_sha256=digest,error=str(exc)))
    songs=[]
    for digest,row in by_hash.items():
        returns=[v for (h,_),v in seen.items() if h==digest]
        counts={a:sum(v['declared_completed'][a] and not v['conflicting_returns'] for v in returns) for a in ('valence','arousal')}
        songs.append(dict(song_id=row['song_id'],audio_sha256=digest,returns=returns,distinct_declared_complete_ids=counts,
            meets_two_ids_per_axis=all(n>=2 for n in counts.values()),actual_independence_verified=False,
            artist_group_frozen=False,training_admitted=False,formal_evaluation_admitted=False))
    return dict(checked_at=now(),status='format_errors' if errors else 'format_checked_human_review_pending',
        songs=songs,input_documents_after_duplicate_collapse=len(seen),errors=errors,outside_pool=outside,
        songs_with_two_declared_ids_per_axis=sum(s['meets_two_ids_per_axis'] for s in songs),
        distinct_ids_do_not_prove_independent_people=True,training_admitted=False,formal_evaluation_admitted=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--input',type=Path,action='append',required=True);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.report.resolve() in {p.resolve() for p in args.input} | {args.pool.resolve()}: raise ValueError('Report cannot overwrite source data')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    result=validate(read_json(args.pool),args.input);atomic_json(args.report,result)
    print(json.dumps({k:v for k,v in result.items() if k!='songs'}))


if __name__=='__main__': main()
