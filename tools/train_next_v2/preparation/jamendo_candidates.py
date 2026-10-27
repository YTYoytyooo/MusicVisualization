"""Metadata-only, bounded Jamendo auxiliary selection; audio admission is separate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.preparation.jamendo_schema import literal_tsv, tags
from tools.train_next_v2.auxiliary_data.merge_candidates import normalized, connected_groups
from tools.train_next_v2.auxiliary_data.merge_inspect import csv_rows

JOB='d-jamendo-candidates-v1'
DIRECTORY=BASE/'analyses'/JOB
OUTPUT=ROOT/'data/datasets/mtg_jamendo/prepared/candidates-v1'
SOURCE=ROOT/'data/datasets/mtg_jamendo/raw/metadata-v2/source'
MANY=ROOT/'data/datasets/manymusic/raw/v1/identity-members'
EMOTION_TAGS=tuple('mood/theme---'+name for name in (
    'calm','dark','emotional','energetic','happy','hopeful','melancholic','relaxing','romantic','sad','positive','upbeat'))
SELECTION=dict(maximum_tracks=256, maximum_per_connected_artist_group=2,
               minimum_duration_seconds=60, maximum_duration_seconds=360, hash_seed='jamendo-pilot-20261004-v1',
               labels='Whole-track uploader positive tags; other tags remain missing/weak, not verified negative',
               allowed_audio_license_codes=['by','by-sa','by-nc','by-nc-sa'])


def relative(path):return Path(path).relative_to(ROOT).as_posix()


def licenses(path):
    result={}
    for block in Path(path).read_text(encoding='utf-8').split('\n\n'):
        lines=block.strip().splitlines()
        if not lines:continue
        if not re.fullmatch(r'\d+/\d+\.mp3',lines[0]):raise ValueError('Unknown license block format')
        urls=re.findall(r'https?://creativecommons\.org/licenses/([a-z-]+)/([0-9.]+)(/[^\s]*)?', '\n'.join(lines))
        if lines[0] in result:raise ValueError('Duplicate license declaration')
        result[lines[0]]=dict(codes=sorted({r[0] for r in urls}),versions=sorted({r[1] for r in urls}),
            declaration='\n'.join(lines))
    return result


def deterministic_selection(rows, count=256, cap=2):
    remaining={row['track_id']:row for row in rows}
    picked=[]; exposure=Counter(); groups=Counter()
    while remaining and len(picked)<count:
        possible=[r for r in remaining.values() if groups[r['group_id']]<cap]
        if not possible:break
        def key(row):
            balance=sum(1/(1+exposure[tag]) for tag in row['selected_positive_tags'])
            tie=hashlib.sha256((SELECTION['hash_seed']+row['track_id']).encode()).hexdigest()
            return (-balance,tie)
        chosen=min(possible,key=key)
        picked.append(chosen);groups[chosen['group_id']]+=1;exposure.update(chosen['selected_positive_tags'])
        del remaining[chosen['track_id']]
    return picked


def registration_files():
    files=[Path(__file__),ROOT/'tools/train_next_v2/preparation/jamendo_schema.py',
        ROOT/'tools/train_next_v2/auxiliary_data/merge_candidates.py',ROOT/'tools/train_next_v2/auxiliary_data/merge_inspect.py',
        ROOT/'tools/train_next_v2/common.py',ROOT/'tools/train_next/core.py',ROOT/'tools/train_next/runtime.py',
        BASE/'protocols/b-v1.json',ROOT/'data/datasets/combined/prepared/v1/manifest.jsonl',
        BASE/'analyses/d-merge-schema-v1/launch.json',BASE/'analyses/d-jamendo-schema-v1/report.json']
    meta=read_json(BASE/'resources/20261004/jobs/d-jamendo-metadata-v2/report.json')
    files += [ROOT/r['file'] for r in meta['files']]
    files += [MANY/name for name in ('jamendoA_ids.txt','jamendoB_ids.txt','audio_ids.txt')]
    merge=read_json(BASE/'analyses/d-merge-schema-v1/launch.json')['files']
    files += [ROOT/path for name,path in merge.items() if 'metadata.csv' in name or 'tvt_70_15_15_' in name]
    files += [ROOT/'data/datasets/mtg_jamendo/raw/source-details-v1/jamendo/data/download/autotagging_moodtheme_audio-low_sha256_tracks.txt']
    return sorted(set(files))


def register():
    if DIRECTORY.exists() or OUTPUT.exists():raise FileExistsError('Preserve existing candidate specification')
    env=read_json(BASE/'resources/20261004/jobs/d-jamendo-metadata-v2/launch.json')['environment']
    argv=[str(Path(sys.executable).resolve()),'-m','tools.train_next_v2.preparation.jamendo_candidates','--run']
    launch=dict(job_id=JOB,registered_at=now(),original_argv=argv,cwd=str(ROOT),python=argv[0],environment=env,
        protected_hashes={relative(p):sha256(p) for p in registration_files()},selection=SELECTION,emotion_tags=EMOTION_TAGS,
        os_lock=str(lock_path(JOB)),log=str(DIRECTORY/'report.json'),output=relative(OUTPUT),gpu_used=False,
        recovery='No automatic retries or overwritten selection. Inspect failed evidence and version changed logic.',
        scope='Only official Jamendo training positive tags can be selected. Main, MERGE and ManyMusic data are identity exclusions only. No audio or continuous target reads.')
    DIRECTORY.mkdir(parents=True);atomic_json(DIRECTORY/'launch.json',launch)
    record_event(dict(event='D_Jamendo_candidate_selection_registered',launch=relative(DIRECTORY/'launch.json')))
    print(json.dumps(dict(registered=True,job_id=JOB)))


def prepare():
    launch=read_json(DIRECTORY/'launch.json');verify_hashes(launch['protected_hashes'])
    if OUTPUT.exists():raise FileExistsError('Preserve candidate screen')
    if launch['selection']!=SELECTION or tuple(launch['emotion_tags'])!=EMOTION_TAGS:raise ValueError('Selection changed')
    atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=process_identity()))
    raw=literal_tsv(SOURCE/'data/raw.meta.tsv');header=raw.pop(0)
    meta={r[0]:dict(zip(header,r)) for r in raw if len(r)==8}
    mood={r[0]:r for r in tags(SOURCE/'data/autotagging_moodtheme.tsv')}
    splits={role:tags(SOURCE/f'data/splits/split-0/autotagging_moodtheme-{role}.tsv') for role in ('train','validation','test')}
    roles={r[0]:role for role,rows in splits.items() for r in rows}
    assert len(roles)==sum(len(rows) for rows in splits.values())
    many_ids=set()
    for name in ('jamendoA_ids.txt','jamendoB_ids.txt','audio_ids.txt'):
        lines=(MANY/name).read_text(encoding='utf-8').splitlines()
        if any(not re.fullmatch(r'\d+',value) for value in lines):raise ValueError('ManyMusic ID is not numeric')
        many_ids.update('track_'+str(int(value)).zfill(7) for value in lines)
    assignments=read_json(BASE/'protocols/b-v1.json')['split']['assignments']
    with (ROOT/'data/datasets/combined/prepared/v1/manifest.jsonl').open(encoding='utf-8') as stream:
        main=[json.loads(line) for line in stream]
    identities={};known_titles=defaultdict(list);known_pairs=defaultdict(list)
    def add(node,corpus,role,artist,title,extra):
        artist,title=normalized(artist),normalized(title)
        keys=list(extra)
        if artist:keys.append(('artist_name',artist))
        if artist and title:keys.append(('artist_title',artist,title))
        identities[node]=dict(corpus=corpus,role=role,identity_keys=keys)
        if corpus in ('main','merge'):
            if title:known_titles[title].append(node)
            if artist and title:known_pairs[artist,title].append(node)
    for row in main:
        add(row['song_id'],'main',assignments[row['song_id']] if row['split']=='train' else row['split'],
            row.get('artist'),row.get('title'),[('main_group',row['group_id']),('audio_sha256',row['audio_sha256'])])
    files=read_json(BASE/'analyses/d-merge-schema-v1/launch.json')['files']
    merge=csv_rows(ROOT/files['MERGE_Audio_Complete/merge_audio_complete_metadata.csv']);merge_roles={}
    for role in ('train','validate','test'):
        key=f'MERGE_Audio_Complete/tvt_dataframes/tvt_70_15_15/tvt_70_15_15_{role}_audio_complete.csv'
        for sid in csv_rows(ROOT/files[key]):
            if sid in merge_roles:raise ValueError('MERGE split overlap')
            merge_roles[sid]=role
    for sid,row in merge.items():
        allmusic=sorted(set(re.findall(r'MT\d+',row['AllMusic Id']+' '+row['AppearancesTrackIDs'])))
        add('merge:'+sid,'merge',merge_roles[sid],row['Artist'],row['Title'],[('allmusic_id',s) for s in allmusic])
    for sid in sorted(set(mood)|many_ids):
        row=meta.get(sid,{})
        role='external_reserved' if sid in many_ids else roles.get(sid,'excluded_no_official_split')
        extra=[('jamendo_track',sid)]
        if row.get('ARTIST_ID'):extra.append(('jamendo_artist',row['ARTIST_ID']))
        add('jamendo:'+sid,'jamendo',role,row.get('ARTIST_NAME'),row.get('TRACK_NAME'),extra)
    groups=connected_groups(identities);license_by_path=licenses(SOURCE/'audio_licenses.txt')
    checksum_path=ROOT/'data/datasets/mtg_jamendo/raw/source-details-v1/jamendo/data/download/autotagging_moodtheme_audio-low_sha256_tracks.txt'
    sums={line.split()[1]:line.split()[0] for line in checksum_path.read_text().splitlines()}
    rows=[]
    for row in splits['train']:
        sid,artist_id,album_id,audio_path,duration,*positive=row
        info=meta.get(sid,{});artist,title=info.get('ARTIST_NAME',''),info.get('TRACK_NAME','')
        selected_tags=sorted(set(positive)&set(EMOTION_TAGS));members=groups['jamendo:'+sid]
        heldout=[s for s in members if identities[s]['role'] not in ('fit','train')]
        reasons=[]
        if not artist.strip() or not title.strip():reasons.append('missing_or_ambiguous_identity')
        if heldout:reasons.append('linked_reserved_artist_or_recording_group')
        if not selected_tags:reasons.append('no_selected_emotion_tags')
        if not SELECTION['minimum_duration_seconds']<=float(duration)<=SELECTION['maximum_duration_seconds']:reasons.append('outside_full_track_duration_budget')
        exact=known_pairs.get((normalized(artist),normalized(title)),[])
        possible=known_titles.get(normalized(title),[])
        if exact:reasons.append('same_artist_title_in_main_or_MERGE')
        elif possible:reasons.append('possible_title_overlap_pending_audio')
        declared=license_by_path.get(audio_path,{})
        if not declared.get('codes') or not set(declared['codes'])<=set(SELECTION['allowed_audio_license_codes']):reasons.append('audio_license_outside_pilot_allowlist')
        low_path=audio_path.removesuffix('.mp3')+'.low.mp3'
        if not re.fullmatch(r'[a-f0-9]{64}',sums.get(low_path,'')):reasons.append('missing_official_low_quality_audio_checksum')
        rows.append(dict(track_id=sid,song_id='mtg_jamendo:'+sid,artist_id=artist_id,album_id=album_id,artist=artist,title=title,
            group_id='jamendo-group-'+hashlib.sha256('\n'.join(members).encode()).hexdigest()[:20],official_role='train',
            native_path=audio_path,low_quality_path=low_path,expected_audio_sha256=sums.get(low_path),
            metadata_duration_seconds=float(duration),source_song_offset_seconds=0,positive_tags=sorted(set(positive)),
            selected_positive_tags=selected_tags,audio_license=declared,heldout_identity_ids=heldout,
            possible_overlap_ids=exact or possible,exclusion_reasons=reasons,metadata_eligible=not reasons,training_admitted=False))
    eligible=[r for r in rows if r['metadata_eligible']]
    selected=deterministic_selection(eligible,SELECTION['maximum_tracks'],SELECTION['maximum_per_connected_artist_group'])
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT/'candidate-screen.json',rows);atomic_json(OUTPUT/'selected.json',selected)
    missing_many=sorted(many_ids-meta.keys())
    report=dict(status='completed_metadata_selection_only',job_id=JOB,completed_at=now(),official_train_tracks=len(rows),
        metadata_eligible=len(eligible),selected_tracks=len(selected),selected_groups=len({r['group_id'] for r in selected}),
        selected_artist_ids=len({r['artist_id'] for r in selected}),manymusic_reserved_ids=len(many_ids),
        manymusic_ids_without_artist_metadata=missing_many,exclusion_reason_counts=dict(Counter(k for r in rows for k in r['exclusion_reasons'])),
        selected_tag_counts=dict(Counter(t for r in selected for t in r['selected_positive_tags'])),
        selected_audio_hours=sum(r['metadata_duration_seconds'] for r in selected)/3600,
        selection=SELECTION,source_hashes=launch['protected_hashes'],outputs={relative(p):sha256(p) for p in OUTPUT.glob('*.json')},
        training_admitted=False,audio_downloaded=False,continuous_va_labels_created=False,
        main_heldout_targets_used=False,human_ManyMusic_ratings_used=False,
        limitations=['Metadata matches cannot exclude every re-recording/alias; native audio fingerprint audit remains required.',
                    'MusAV Spotify preview metadata/audio needs authorized access; independent external evaluation remains unverified.',
                    'Full-track uploader categories are weak supervision. Missing tags are not verified negative emotion judgments.',
                    'Dataset/metadata use remains noncommercial research; no app release license clearance is claimed.'],
        next_action='Probe bounded audio delivery, retrieve only selected tracks with official SHA256, audit native decoding/duration/time origin and cross-corpus fingerprint overlaps. Freeze class-only loss and paired training before GPU work.')
    verify_hashes(launch['protected_hashes']);atomic_json(DIRECTORY/'report.json',report)
    with run_lease('next-va-v2-d-ledger'):
        stages=read_json(PLAN)['stages'];stages['D']['jamendo_candidates']=dict(report=relative(DIRECTORY/'report.json'),
            selected=relative(OUTPUT/'selected.json'),tracks=len(selected),training_admitted=False)
        stages['D']['phase']='Jamendo_selected_audio_admission_pending'
        record_event(dict(event='D_Jamendo_metadata_candidates_selected',selected=len(selected)),updates=dict(stages=stages,next_action=report['next_action']))
    print(json.dumps({k:report[k] for k in ('status','metadata_eligible','selected_tracks','selected_groups','manymusic_reserved_ids',
        'manymusic_ids_without_artist_metadata','exclusion_reason_counts','selected_tag_counts','selected_audio_hours')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true');mode.add_argument('--run',action='store_true');args=p.parse_args()
    with run_lease(JOB):register() if args.register else prepare()
