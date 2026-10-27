"""Fetch bounded identity-source indexes and pinned audio checksum metadata only."""
import json
import sys
from pathlib import Path

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.jamendo_data.metadata import fetch_bytes, write_once, validate_blob

JOB = 'd-jamendo-source-details-v1'
DIRECTORY = BASE/'resources/20261004/jobs'/JOB
DEST = ROOT/'data/datasets/mtg_jamendo/raw/source-details-v1'


def main():
    with run_lease(JOB):
        launch = read_json(DIRECTORY/'launch.json');verify_hashes(launch['protected_hashes'])
        if (DIRECTORY/'report.json').exists():raise FileExistsError('Preserve completed source details')
        if Path(sys.executable).resolve()!=Path(launch['python']):raise ValueError('Wrong Python')
        atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=process_identity()))
        records=[]
        def fetch(name,url,limit=2*1024*1024,spec=None):
            path=DEST/name
            if path.exists():raise FileExistsError('Inspect partial evidence before a new version')
            data=fetch_bytes(url,limit)
            if spec:validate_blob(data,spec)
            write_once(path,data)
            row=dict(file=str(path.relative_to(ROOT)),url=url,bytes=len(data),sha256=sha256(path))
            records.append(row)
            with (DIRECTORY/'events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(time=now(),**row))+'\n')
            return data
        try:
            tree=read_json(ROOT/'data/datasets/mtg_jamendo/raw/metadata-v2/tree.json')
            bypath={r['path']:r for r in tree['tree']}
            for path in launch['pinned_jamendo_paths']:
                row=bypath[path]
                if row['type']!='blob' or row['mode']!='100644':raise ValueError('Expected regular metadata')
                spec=dict(path=path,size=row['size'],git_blob_sha1=row['sha'])
                fetch('jamendo/'+path,launch['jamendo_raw_prefix']+path,spec=spec)
            many_api='https://api.github.com/repos/manymusic/manymusic-mtg-jamendo-stim-scripts'
            commit=json.loads(fetch('manymusic-commit.json',many_api+'/commits/HEAD'))
            tree_id=commit['commit']['tree']['sha']
            many_tree=json.loads(fetch('manymusic-tree.json',many_api+'/git/trees/'+tree_id+'?recursive=1',4*1024*1024))
            if many_tree['truncated'] or many_tree['sha']!=tree_id:raise ValueError('Incomplete or different ManyMusic tree')
            index=[r for r in many_tree['tree'] if r['type']=='blob' and r.get('size',0)<=2*1024*1024
                   and r['path'].lower().endswith(('.csv','.tsv','.json','.txt','.md','.songids'))]
            atomic_json(DEST/'manymusic-identity-source-index.json',dict(revision=commit['sha'],files=index))
            verify_hashes(launch['protected_hashes'])
            result=dict(status='completed_metadata_only',completed_at=now(),job_id=JOB,files=records,
                manymusic_revision=commit['sha'],manymusic_candidate_files=index,
                audio_downloaded=False,remote_code_executed=False,gpu_used=False,training_admitted=False,
                musav_note='Official MusAV uses Spotify preview identities, not Jamendo IDs. Audio/title/artist metadata is request-only. Do not claim a verified Jamendo/MusAV exclusion without correspondence.',
                musav_source='https://github.com/MTG/musav-dataset',
                pending=['Read ManyMusic identity metadata before freezing any Jamendo training selection',
                         'Check bounded individual/tar-range audio access against official track SHA256; do not bulk download all archives'])
            atomic_json(DIRECTORY/'report.json',result)
            with run_lease('next-va-v2-d-ledger'):
                stages=read_json(PLAN)['stages'];stages['D']['jamendo_source_details']=str((DIRECTORY/'report.json').relative_to(ROOT))
                record_event(dict(event='D_Jamendo_source_details_downloaded',job_id=JOB),updates=dict(stages=stages))
            print(json.dumps(dict(status=result['status'],files=len(records),manymusic_revision=commit['sha'],manymusic_files=index)))
        except BaseException as exc:
            atomic_json(DIRECTORY/'failure.json',dict(failed_at=now(),error=f'{type(exc).__name__}: {exc}'))
            record_event(dict(event='D_Jamendo_source_details_failed',error=f'{type(exc).__name__}: {exc}'));raise


if __name__=='__main__':main()
