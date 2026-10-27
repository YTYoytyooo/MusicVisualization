"""Read true D-job activity from OS leases, process identity and bounded artifacts."""
import json
from pathlib import Path

from tools.train_next_v2.common import BASE,ROOT,GPU_LEASE,process_identity,read_json
from tools.train_next.runtime import is_running


def snapshot():
    jobs=[]
    for launch_path in sorted(list((BASE/'analyses').glob('d-*/launch.json'))+list((BASE/'feature_jobs').glob('d-*/launch.json'))):
        launch=read_json(launch_path);folder=launch_path.parent;job=launch['job_id']
        lease=is_running(job)
        process=read_json(folder/'process.json') if (folder/'process.json').exists() else {}
        expected=process.get('process_identity')
        identity_matches=bool(expected and expected==process_identity(expected['pid']))
        progress=read_json(folder/'status.json') if (folder/'status.json').exists() else {}
        report=read_json(folder/'report.json') if (folder/'report.json').exists() else None
        if lease:
            actual='active_identity_confirmed' if identity_matches else 'active_lease_identity_unconfirmed'
        elif report is not None:
            actual='finished_result_present'
        elif process:
            actual='exited_without_completion_report'
        else:
            actual='registered_not_started'
        row=dict(job_id=job,actual_state=actual,os_lease_active=lease,process_identity_matches=identity_matches,
                 progress={k:progress[k] for k in ('stage','completed','total','bytes','expected_bytes','updated_at','error') if k in progress},
                 result_status=report.get('status') if report else None)
        if launch.get('cache'):
            cache=ROOT/launch['cache']
            row['feature_shards']=len(list(cache.glob('*.npz')))
            row['feature_receipts']=len(list(cache.glob('*.json')))
            row['expected_songs']=launch['songs']
        jobs.append(row)
    return dict(gpu_lease_active=is_running(GPU_LEASE),jobs=jobs)


if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False,indent=2))
