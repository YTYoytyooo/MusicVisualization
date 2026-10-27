"""Freeze every continuous-label and observed-tag training input separately."""
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.train_next.core import atomic_json, fit_scaler, sha256
from tools.train_next.runtime import is_running
from tools.train_next_v2.common import BASE,ROOT,GPU_LEASE,now,read_json,verify_hashes,code_hashes
from tools.train_next_v2.data import load_internal
from tools.train_next_v2.jamendo_data.contract import validate_candidates, observed_tag_distribution, TAG_ORDER
from tools.train_next_v2.jamendo_data.features import read_cached, CACHE
from tools.train_next_v2.jamendo_data.protocol import PROTOCOL, SOURCE, AUDIT, B_PROTOCOL

CONTRACT=BASE/'protocols/d-jamendo-training-inputs-v1.json'
FEATURE_JOB=BASE/'feature_jobs/d-jamendo-clap-features-v1'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def array_digest(*arrays):
    h=hashlib.sha256()
    for value in arrays:
        a=np.ascontiguousarray(value)
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def training_code_hashes():
    result=code_hashes()
    files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'tools/train_next_v2/jamendo_data').glob('*.py'))
    files += [ROOT/'tools/train_next_v2/tests/test_experiments.py']
    result.update({relative(p):sha256(p) for p in files})
    return result


def main_membership(songs):
    return [dict(song_id=s['song_id'],role=role,source_dataset=s['source_dataset'],group_id=s['group_id'],
        audio_sha256=s['audio_sha256'],target_digest=array_digest(s['times'],s['y'],s['mask']),native_seconds=s['native_seconds'])
        for role in ('fit','development') for s in songs[role]]


def validate_pool(rows, protocol, main_hashes):
    summary=validate_candidates(rows)
    if summary!=protocol['candidate_summary'] or len(rows)!=protocol['auxiliary_songs'] or protocol['tag_order']!=list(TAG_ORDER):
        raise ValueError('Frozen Jamendo membership or tag vocabulary changed')
    if {r['audio']['audio_sha256'] for r in rows}&set(main_hashes):
        raise ValueError('Auxiliary audio overlaps protected main songs')


def read_auxiliary_feature(entry, row, feature_key):
    if entry['role']!='auxiliary_fit' or entry['song_id']!=row['song_id'] or entry['group_id']!=row['group_id']:
        raise ValueError('Wrong auxiliary role/song/artist group')
    if entry['observed_tags']!=row['selected_positive_tags'] or entry['tag_order']!=list(TAG_ORDER):
        raise ValueError('Frozen whole-track annotations changed')
    if set(entry)&{'y','mask','timepoint_targets','class_index'}:
        raise ValueError('No continuous auxiliary or missing-tag binary targets permitted')
    if (entry['duration_seconds']!=row['audio']['native_audio']['duration_seconds']
            or entry['local_time_origin_seconds']!=0 or entry['source_song_offset_seconds']!=0
            or entry['audio_path']!=row['audio']['audio_path'] or entry['audio_sha256']!=row['expected_audio_sha256']):
        raise ValueError('Native auxiliary clock/audio identity changed')
    verify_hashes({entry['audio_path']:entry['audio_sha256'],entry['path']:entry['sha256'],entry['receipt']:entry['receipt_sha256']})
    actual=read_cached(row,CACHE,feature_key)
    if actual is None:
        raise ValueError('Missing audited feature cache')
    for key in ('song_id','audio_sha256','feature_key','path','sha256','frames','dimensions','receipt','receipt_sha256','native_float32_sha256'):
        if actual[key]!=entry[key]: raise ValueError('Frozen feature identity changed: '+key)
    target=observed_tag_distribution(row['selected_positive_tags'])
    if entry['observed_distribution']!=target.tolist():
        raise ValueError('Observed-tag distribution differs from source annotations')
    with np.load(ROOT/entry['path'],allow_pickle=False) as saved:
        values=saved['features'].copy();times=saved['times'].copy()
    return dict(song_id=entry['song_id'],group_id=entry['group_id'],audio_sha256=entry['audio_sha256'],
        x=values,times=times,observed_tags=list(entry['observed_tags']))


def prepare_contract(device, progress=None):
    if CONTRACT.exists(): raise FileExistsError('Preserve the immutable Jamendo input contract')
    if is_running(GPU_LEASE) or is_running('d-jamendo-clap-features-v1'):
        raise RuntimeError('Feature producer/GPU still active; no partial input admission')
    protocol=read_json(PROTOCOL);b=read_json(B_PROTOCOL)
    verify_hashes({relative(B_PROTOCOL):protocol['source_B_protocol_sha256'],relative(SOURCE):protocol['auxiliary_candidates_sha256']})
    report,launch=read_json(FEATURE_JOB/'report.json'),read_json(FEATURE_JOB/'launch.json')
    verify_hashes(launch['protected_hashes'])
    if (report['status']!='completed_features_verified' or report['songs']!=protocol['auxiliary_songs']
            or report['launch_sha256']!=sha256(FEATURE_JOB/'launch.json') or report['source_sha256']!=sha256(SOURCE)
            or report['feature_key']!=b['feature_key'] or launch['feature_spec']!=b['feature_spec'] or report['labels_used']):
        raise ValueError('Feature completion contract not met')
    manifest=Path(b['source_manifest']);verify_hashes({relative(manifest):b['manifest_sha256']})
    # Identity metadata only; no calibration, original validation or test targets.
    main_hashes={json.loads(line)['audio_sha256'] for line in manifest.read_text(encoding='utf-8').splitlines() if line}
    rows=read_json(SOURCE);validate_pool(rows,protocol,main_hashes)
    features={r['song_id']:r for r in report['feature_manifest']}
    if len(features)!=len(rows) or set(features)!={r['song_id'] for r in rows}:
        raise ValueError('Feature manifest is not the exact approved auxiliary pool')
    admitted=[]
    for row in rows:
        f=features[row['song_id']];native=row['audio']['native_audio']
        entry={k:f[k] for k in ('song_id','audio_sha256','feature_key','path','sha256','frames','dimensions','receipt','receipt_sha256','native_float32_sha256')}
        entry.update(group_id=row['group_id'],role='auxiliary_fit',observed_tags=row['selected_positive_tags'],tag_order=list(TAG_ORDER),
            observed_distribution=observed_tag_distribution(row['selected_positive_tags']).tolist(),audio_path=row['audio']['audio_path'],
            duration_seconds=native['duration_seconds'],local_time_origin_seconds=0,source_song_offset_seconds=0)
        read_auxiliary_feature(entry,row,b['feature_key']);admitted.append(entry)
    songs=load_internal(b,device,progress)
    if len(songs['fit'])!=3070 or len(songs['development'])!=383: raise ValueError('Main fixed split changed')
    mean,std=fit_scaler(songs['fit'])
    history_path=ROOT/'data/training/next_va/completion-20261003.json';history=read_json(history_path)
    files=[PROTOCOL,B_PROTOCOL,SOURCE,AUDIT/'report.json',FEATURE_JOB/'launch.json',FEATURE_JOB/'report.json',history_path]
    protected={relative(p):sha256(p) for p in files}
    protected['models/emotion_model.pth']=b['legacy_model_sha256']
    protected['models/next_va/mixed-20261003-recovery/best.pth']=b['baseline_checkpoint_sha256']
    protected[relative(history['runs']['teacher']['checkpoint'])]=history['runs']['teacher']['checkpoint_sha256']
    verify_hashes(protected)
    contract=dict(schema_version=1,id='d-jamendo-training-inputs-v1',created_at=now(),protocol=relative(PROTOCOL),protocol_sha256=sha256(PROTOCOL),
        source_B_protocol=relative(B_PROTOCOL),source_B_protocol_sha256=sha256(B_PROTOCOL),source_candidates=relative(SOURCE),
        protected_hashes=protected,builder_code_hashes=training_code_hashes(),main_songs=main_membership(songs),auxiliary=admitted,
        feature_dimensions=531,scaler_digest=array_digest(mean,std),scaler_fit_songs=3070,
        supervision=dict(main='continuous_VA_independent_masks',auxiliary='whole_track_observed_tag_distribution',continuous_auxiliary_VA=False,pseudo_labels=False),
        training_admitted=True,calibration_validation_test_targets_read=0,
        protocol_activation='Activate immutable specification only after complete feature/code/resume checks; preserve original pending protocol.',limitations=protocol['limitations'])
    atomic_json(CONTRACT,contract)
    return contract


def load_inputs(contract, device, progress=None):
    verify_hashes(contract['protected_hashes']);verify_hashes(contract['builder_code_hashes'])
    if (not contract['training_admitted'] or contract['supervision']['continuous_auxiliary_VA']
            or contract['supervision']['pseudo_labels'] or contract['supervision']['auxiliary']!='whole_track_observed_tag_distribution'):
        raise ValueError('Wrong Jamendo supervision contract')
    songs=load_internal(read_json(ROOT/contract['source_B_protocol']),device,progress)
    if main_membership(songs)!=contract['main_songs']: raise ValueError('Main roles/labels/native clocks changed')
    mean,std=fit_scaler(songs['fit'])
    if array_digest(mean,std)!=contract['scaler_digest']: raise ValueError('Fit-only normalizer changed')
    rows=read_json(SOURCE);protocol=read_json(PROTOCOL)
    manifest=Path(read_json(B_PROTOCOL)['source_manifest'])
    validate_pool(rows,protocol,{json.loads(line)['audio_sha256'] for line in manifest.read_text(encoding='utf-8').splitlines() if line})
    by_id={r['song_id']:r for r in rows}
    if len(contract['auxiliary'])!=len(by_id) or {e['song_id'] for e in contract['auxiliary']}!=set(by_id):
        raise ValueError('Admitted auxiliary pool changed')
    auxiliary=[read_auxiliary_feature(e,by_id[e['song_id']],protocol['feature_key']) for e in contract['auxiliary']]
    return songs,auxiliary
