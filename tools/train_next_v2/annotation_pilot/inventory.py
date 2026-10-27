"""Prepare an honest, still-unassigned annotation pilot from local target audio."""
from collections import defaultdict
from pathlib import Path
import json
import sys

import soundfile as sf

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, lock_path
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes

JOB='e-target-annotation-inventory-v1'
DIRECTORY=BASE/'annotation_pilot'/JOB


def main():
    with run_lease(JOB):
        output=DIRECTORY/'inventory.json'
        if output.exists():raise ValueError('Preserve this inventory snapshot; later human work needs a new snapshot')
        DIRECTORY.mkdir(parents=True,exist_ok=True)
        manifest=ROOT/'data/datasets/combined/prepared/v1/manifest.jsonl'
        protocol=read_json(BASE/'protocols/b-v1.json')
        verify_hashes({manifest.relative_to(ROOT).as_posix():protocol['manifest_sha256']})
        launch=dict(job_id=JOB,registered_at=now(),original_argv=[sys.executable,'-u','-m','tools.train_next_v2.annotation_pilot.inventory'],
            cwd=str(ROOT),environment={'PYTHONUTF8':'1'},code={Path(__file__).relative_to(ROOT).as_posix():sha256(__file__)},
            os_lock=str(lock_path(JOB)),source_audio_root='assets/audio',source_manifest_sha256=sha256(manifest),
            labels_read='Existing human completion metadata and take counts only; no benchmark label targets.',
            log=str(DIRECTORY/'console.log'),recovery='Read-only inventory; preserve completed snapshot. No annotation or audio writes.')
        atomic_json(DIRECTORY/'launch.json',launch)
        record_event(dict(event='E_target_annotation_inventory_registered',job_id=JOB,launch=str((DIRECTORY/'launch.json').relative_to(ROOT))))
        existing=defaultdict(list)
        for path in sorted((ROOT/'data/annotations/continuous').glob('*/*.json')):
            doc=read_json(path)
            # History revisions are deliberately excluded and are not extra reviewers.
            existing[doc.get('source',{}).get('audio_sha256')].append(dict(path=path.relative_to(ROOT).as_posix(),
                sha256=sha256(path),reviewer=doc.get('reviewer'),protocol=doc.get('protocol'),
                completed_flags=doc.get('completed',{}),takes=len(doc.get('takes',[])),
                completion_status='unverified_human_work_in_progress',label_quality_audited=False))
        identities=defaultdict(list)
        for row in map(json.loads,manifest.read_text(encoding='utf-8').splitlines()):
            identities[row['audio_sha256']].append(dict(song_id=row['song_id'],dataset=row['source_dataset'],
                original_split=row['split'],internal_role=protocol['split']['assignments'].get(row['song_id'])))
        files=sorted(p for p in (ROOT/'assets/audio').rglob('*') if p.is_file() and p.suffix.lower() in ('.wav','.flac','.mp3','.m4a','.aac','.ogg'))
        groups=defaultdict(list);metadata_errors=[]
        for path in files:
            digest=sha256(path)
            item=dict(path=path.relative_to(ROOT).as_posix(),sha256=digest,bytes=path.stat().st_size,
                provisional_collection=path.parent.relative_to(ROOT/'assets/audio').as_posix())
            try:
                info=sf.info(path)
                item.update(header_duration_seconds=info.frames/info.samplerate,sample_rate=info.samplerate,channels=info.channels,
                    duration_scope='Header estimate only; full native decode and clock validation still required before annotation import/training.')
            except Exception as exc:
                item['metadata_error']=f'{type(exc).__name__}: {exc}';metadata_errors.append(item['path'])
            groups[digest].append(item)
        candidates=[]
        for digest,rows in sorted(groups.items()):
            matches=identities.get(digest,[])
            candidates.append(dict(candidate_id='target-'+digest[:16],audio_sha256=digest,files=rows,
                exact_main_dataset_matches=matches,annotations=existing.get(digest,[]),
                status='exclude_known_main_dataset_recording' if matches else 'inventory_only_pending_fingerprint_artist_and_clock_audit',
                intended_annotation='Music-expressed continuous valence/arousal, independent native-time masks and separately marked transitions.',
                reviewer_slots=[dict(slot=i,status='unassigned') for i in (1,2,3)],
                training_role='not_assigned',human_review_complete=False))
        available=sum(not c['exact_main_dataset_matches'] for c in candidates)
        result=dict(job_id=JOB,created_at=now(),status='pilot_prepared_waiting_for_dataset_audit_and_humans',
            audio_files=len(files),unique_recordings=len(candidates),nonmatching_recording_candidates=available,
            minimum_pilot_songs=50,preferred_maximum=100,minimum_additional_candidate_songs=max(0,50-available),
            required_distinct_human_reviewers_per_song=[2,3],
            existing_current_annotation_files=sum(len(v) for v in existing.values()),
            existing_distinct_reviewer_ids=sorted({r['reviewer'] for v in existing.values() for r in v if r['reviewer']}),
            existing_both_axes_marked_complete=sum(all(r['completed_flags'].get(a) is True for a in ('valence','arousal')) for v in existing.values() for r in v),
            verified_completed_human_reviews=0,metadata_errors=metadata_errors,candidates=candidates,
            annotation_protocol=dict(target='perceived musical expression, not induced listener mood',
                clock='Original whole-song audio currentTime; no stretching or inferred reaction-delay shift.',
                dimensions='Valence [-1,1] negative-positive; arousal [-1,1] calm-excited, with separate missing masks.',
                instructions=['Listen once to understand the whole song, then record V/A independently on the native timeline.',
                    'Keep uncertain or unannotated intervals missing; store obvious transition events separately.',
                    'Reviewers should work independently without viewing model curves or each other\'s labels.',
                    'Preserve raw takes and disagreements; completion flags alone are not quality verification.',
                    'Freeze fit/development/evaluation song groups before importing new labels. Reserve a fixed target evaluation group.']),
            blockers=['At least 50 suitable target songs must remain after exact, near-duplicate and artist/recording grouping checks.',
                'Need 2-3 actual people per song; no invitations or messages have been sent.',
                'Existing files are incomplete; historical revisions are not independent reviewers.',
                'Calibration screen failed; these songs are not admitted to pseudo-label training.'],
            untouched=['All source audio','All current human annotation files and history','Original applications and checkpoints'],
            next_independent_work='Audit native decoding, fingerprints/artist groups, and prepare an isolated reference inference adapter/performance benchmark without consuming final-test labels.')
        atomic_json(output,result)
        summary={k:result[k] for k in ('status','audio_files','unique_recordings','nonmatching_recording_candidates','minimum_additional_candidate_songs','existing_current_annotation_files','existing_both_axes_marked_complete')}
        (DIRECTORY/'console.log').write_text(json.dumps(summary,ensure_ascii=False)+'\n',encoding='utf-8')
        stages=read_json(PLAN)['stages']
        stages['E'].update(phase='calibration_failed_human_pilot_prepared',annotation_inventory=output.relative_to(ROOT).as_posix(),
            annotation_summary=summary,human_review_status='waiting_for_real_annotation',pseudo_training_enabled=False,
            blockers=result['blockers'])
        record_event(dict(event='E_target_annotation_inventory_completed',**summary,report=output.relative_to(ROOT).as_posix()),
            updates=dict(stages=stages,current_stage='E',next_action=result['next_independent_work']))
        print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
