"""List a public ZIP via exact byte ranges without fetching its audio payload."""
import json
from pathlib import Path
import zipfile

from tools.train_next.core import atomic_json,sha256
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import BASE,ROOT,PLAN,read_json,verify_hashes,process_identity,record_event,now
from tools.train_next_v2.jamendo_data.metadata import fetch_bytes,write_once
from tools.train_next_v2.auxiliary_data.merge_audit import BoundedArchive

JOB='d-manymusic-archive-index-v1'
DIRECTORY=BASE/'resources/20261004/jobs'/JOB
DEST=ROOT/'data/datasets/manymusic/raw/v1/archive-index'


def main():
    with run_lease(JOB):
        launch=read_json(DIRECTORY/'launch.json');verify_hashes(launch['protected_hashes'])
        if DEST.exists():raise FileExistsError('Preserve previous network evidence')
        atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=process_identity()))
        try:
            data=fetch_bytes('https://zenodo.org/api/records/17425915',2*1024*1024)
            record=json.loads(data)
            if str(record['id'])!='17425915':raise ValueError('Unexpected record')
            source=next(row for row in record['files'] if row['key']=='manymusic-stim-zenodo.zip')
            if not 0<source['size']<64*1024**3:raise ValueError('Unexpected archive size')
            item=dict(name=source['key'],size=source['size'],md5=source['checksum'],url=source['links']['self'])
            write_once(DEST/'record.json',data)
            (DEST/'ranges').mkdir()
            atomic_json(DIRECTORY/'pinned-source.json',dict(record_sha256=sha256(DEST/'record.json'),item=item,pinned_at=now()))
            def progress(row):
                atomic_json(DIRECTORY/'status.json',dict(updated_at=now(),**row))
                with (DIRECTORY/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(time=now(),**row))+'\n')
            reader=BoundedArchive(item,DEST,progress)
            with zipfile.ZipFile(reader) as archive:
                rows=[dict(name=r.filename,bytes=r.file_size,compressed_bytes=r.compress_size,crc32=f'{r.CRC:08x}',
                           header_offset=r.header_offset,compression=r.compress_type,directory=r.is_dir()) for r in archive.infolist()]
            if len(rows)!=len({r['name'] for r in rows}):raise ValueError('Duplicate member names')
            atomic_json(DEST/'archive-index.json',rows)
            candidates=[r for r in rows if not r['directory'] and Path(r['name']).suffix.lower() in ('.csv','.tsv','.txt','.json','.md','.songids')]
            report=dict(status='completed_index_only',job_id=JOB,completed_at=now(),pinned_source=item,
                archive_index=str((DEST/'archive-index.json').relative_to(ROOT)),archive_index_sha256=sha256(DEST/'archive-index.json'),
                members=len(rows),metadata_candidates=candidates,transferred_bytes=reader.transferred,
                full_archive_checksum_verified=False,audio_downloaded=False,training_admitted=False,
                next_action='Select explicit identity metadata members and verify ZIP CRC. Do not read human ratings as training labels or download30GB audio archive.')
            atomic_json(DIRECTORY/'report.json',report)
            with run_lease('next-va-v2-d-ledger'):
                stages=read_json(PLAN)['stages'];stages['D']['manymusic_index']=str((DIRECTORY/'report.json').relative_to(ROOT))
                record_event(dict(event='D_ManyMusic_archive_indexed',job_id=JOB,bytes=reader.transferred),updates=dict(stages=stages))
            print(json.dumps(report))
        except BaseException as exc:
            atomic_json(DIRECTORY/'failure.json',dict(failed_at=now(),error=f'{type(exc).__name__}: {exc}'))
            record_event(dict(event='D_ManyMusic_index_failed',error=f'{type(exc).__name__}: {exc}'));raise


if __name__=='__main__':main()
