"""Freeze the paired Jamendo hypothesis before extracting features or fitting."""
import json
from pathlib import Path
import sys

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, is_running, lock_path
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes
from .contract import TAG_ORDER, validate_candidates

JOB = 'd-jamendo-observed-tags-protocol-v1'
DIRECTORY = BASE/'analyses'/JOB
PROTOCOL = BASE/'protocols/d-jamendo-observed-tags-v1.json'
AUDIT = BASE/'analyses/d-jamendo-native-audit-v2'
SOURCE = ROOT/'data/datasets/mtg_jamendo/prepared/native-audit-v2/eligible-category-candidates.json'
B_PROTOCOL = BASE/'protocols/b-v1.json'


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def freeze():
    with run_lease(JOB):
        if DIRECTORY.exists() or PROTOCOL.exists():
            raise FileExistsError('Preserve the frozen protocol and preparation evidence')
        if is_running('d-jamendo-native-audit-v2'):
            raise RuntimeError('Audio audit must finish before protocol admission')
        audit, source_launch = read_json(AUDIT/'report.json'), read_json(AUDIT/'launch.json')
        verify_hashes(source_launch['protected_hashes']); verify_hashes(audit['outputs'])
        if audit['status'] != 'completed_native_audio_gate':
            raise ValueError('Audio audit did not complete')
        b, rows = read_json(B_PROTOCOL), read_json(SOURCE)
        summary = validate_candidates(rows)
        if summary['songs'] != audit['eligible_categories'] or summary['tag_counts'] != audit['selected_tag_counts']:
            raise ValueError('Eligible pool differs from the audited report')
        if set(summary['tag_counts']) != set(TAG_ORDER) or min(summary['tag_counts'].values()) < 10:
            raise ValueError('Preselected twelve-tag pilot lacks its minimum observed support')
        files = [Path(__file__), Path(__file__).parent/'contract.py', SOURCE, B_PROTOCOL, AUDIT/'report.json']
        protected = dict(source_launch['protected_hashes'])
        protected.update({rel(p):sha256(p) for p in files})
        argv = [str(Path(sys.executable).resolve()), '-m', 'tools.train_next_v2.jamendo_data.protocol']
        launch = dict(job_id=JOB, original_argv=argv, cwd=str(ROOT), python=argv[0],
            environment=source_launch['environment'], registered_at=now(), protected_hashes=protected,
            os_lock=str(lock_path(JOB)), output=rel(PROTOCOL), recovery='Preserve any partial output; diagnose, never overwrite a frozen protocol.')
        DIRECTORY.mkdir(parents=True)
        atomic_json(DIRECTORY/'launch.json',launch)
        record_event(dict(event='D_Jamendo_observed_tag_protocol_registered',launch=rel(DIRECTORY/'launch.json')))
        protocol = dict(id='d-jamendo-observed-tags-v1',created_at=now(),
            hypothesis='A low-weight whole-track observed-tag task improves continuous VA representations under unchanged main supervision and matched optimization exposure.',
            source_B_protocol=rel(B_PROTOCOL),source_B_protocol_sha256=sha256(B_PROTOCOL),
            auxiliary_candidates=rel(SOURCE),auxiliary_candidates_sha256=sha256(SOURCE),auxiliary_songs=len(rows),
            candidate_summary=summary,tag_order=list(TAG_ORDER),audio_audit=rel(AUDIT/'report.json'),audio_audit_sha256=sha256(AUDIT/'report.json'),
            feature_key=b['feature_key'],feature_spec=b['feature_spec'],
            main_split='Unchanged B internal fit3070/development383/calibration383; original validation584/test622 remain sealed.',
            fit_labels=dict(main='Real continuous VA with separate masks and native timestamps',auxiliary='One observed-tag distribution per full native track',
                continuous_auxiliary_VA=False,pseudo_labels=False,missing_tags='Unknown presence; no binary absent labels are created'),
            label_objective=dict(name='observed-tag conditional distribution',target='Uniform mass over the recorded selected positive tags of this song.',
                loss='-mean_song(sum_tag(target * log_softmax(whole_track_logits)))',
                meaning='Predict which recorded uploader tag would be drawn from this track. Zero target mass means unobserved in this annotation, not confirmed emotional absence.',
                limitations='Softmax makes observed tags compete and can inherit uploader omissions. This is a weak auxiliary representation task; outputs are neither exclusive emotion classes nor calibrated probabilities of emotion presence.'),
            model=dict(base='B00 concatenate + BiLSTM128x1 + unchanged VA head',configuration=b['defaults'],
                auxiliary_head='Masked mean of full-track shared temporal states -> linear12; no per-timepoint tag targets'),
            variants=[dict(id='dj00-control',tag_loss_weight=0.),dict(id='dj01-observed-tags-010',tag_loss_weight=.1)],
            optimization=dict(updates=20000,main_batch_songs=4,aux_batch_songs=4,main_exposures=80000,auxiliary_forward_exposures=80000,
                main_sampling_alpha=0.,aux_sampling='Uniform connected artist group then uniform track within group, independently keyed by seed and step.',
                seeds=[20261003,20261004,20261005],planned_runs=6,
                initialization='Fresh paired random backbone and auxiliary head; no previously trained checkpoints.',
                normalization='Main internal-fit songs only; identical scaler for paired arms.',
                control='Both arms initialize the same heads and perform identical main/auxiliary forward calls and random draws; only tag loss multiplier differs.',
                no_score_based_early_stop=True),
            selection=dict(primary=b['selection']['primary'],checkpoint=b['selection']['checkpoint'],evaluate_every=1000,
                minimum_mean_CCC_gain=.02,maximum_dataset_axis_CCC_drop=.01,maximum_dataset_axis_MAE_increase=.01,
                paired_song_group_bootstrap_replicates=2000,confidence=.95,minimum_dataset_axis_CCC_CI_lower=-.02,
                require_positive_gain_in_every_seed=True,original_validation_confirmation='Only after internal gates pass. Test labels never select candidates.'),
            required_report=['CCC/MAE/RMSE per dataset and axis','song mean, centered error, amplitude, derivative and constant-output comparison',
                'paired three-seed/group-bootstrap, exposure/RNG parity','fresh checkpoint reload, exact data/code/cache hashes','runtime and peak GPU memory'],
            training_enabled=False,implementation_status='Feature preparation admitted; paired training implementation and input/recovery checks pending.',
            limitations=audit['limitations']+['Whole-track uploader tags provide no continuous VA truth.',
                'Small curated noncommercial research pilot; selection does not establish commercial audio rights or broad genre/language coverage.'])
        atomic_json(PROTOCOL,protocol)
        verify_hashes(protected)
        report=dict(status='completed_protocol_frozen',completed_at=now(),output=rel(PROTOCOL),sha256=sha256(PROTOCOL),
            songs=len(rows),groups=summary['groups'],labels_used_for_model_fitting=False,training_started=False)
        atomic_json(DIRECTORY/'report.json',report)
        with run_lease('next-va-v2-d-ledger'):
            stages=read_json(PLAN)['stages']
            stages['D'].update(phase='Jamendo_features_pending',jamendo_protocol=dict(path=rel(PROTOCOL),sha256=sha256(PROTOCOL),status='frozen',songs=len(rows)))
            record_event(dict(event='D_Jamendo_observed_tag_protocol_frozen',**report),updates=dict(stages=stages))
        print(json.dumps(report))


if __name__ == '__main__':
    freeze()
