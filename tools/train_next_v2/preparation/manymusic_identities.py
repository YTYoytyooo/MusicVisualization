"""Retrieve an explicit identity-only allowlist from the pinned ManyMusic ZIP."""
import json
from pathlib import Path
import zipfile

from tools.train_next.core import atomic_json,sha256
from tools.train_next.runtime import run_lease
from tools.train_next_v2.common import BASE,ROOT,PLAN,read_json,verify_hashes,process_identity,record_event,now
from tools.train_next_v2.auxiliary_data.merge_audit import BoundedArchive

JOB='d-manymusic-identities-v1'
DIRECTORY=BASE/'resources/20261004/jobs'/JOB
DEST=ROOT/'data/datasets/manymusic/raw/v1/identity-members'


def main():
    with run_lease(JOB):
        launch=read_json(DIRECTORY/'launch.json');verify_hashes(launch['protected_hashes'])
        if DEST.exists():raise FileExistsError('Preserve previous evidence')
        atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=process_identity()))
        DEST.mkdir(parents=True);(DEST/'ranges').mkdir()
        try:
            def progress(row):atomic_json(DIRECTORY/'status.json',dict(updated_at=now(),**row))
            source=BoundedArchive(launch['item'],DEST,progress);source.etag=launch['archive_etag']
            rows=[]
            with zipfile.ZipFile(source) as archive:
                for expected in launch['members']:
                    info=archive.getinfo(expected['name'])
                    if (info.file_size!=expected['bytes'] or info.compress_size!=expected['compressed_bytes']
                            or f'{info.CRC:08x}'!=expected['crc32'] or info.header_offset!=expected['header_offset']
                            or info.file_size>1024*1024):raise ValueError('Member changed or exceeds identity bound')
                    data=archive.read(info)
                    path=DEST/Path(info.filename).name
                    if path.exists():raise ValueError('Duplicate identity output basename')
                    path.write_bytes(data)
                    rows.append(dict(archive_name=info.filename,path=str(path.relative_to(ROOT)),sha256=sha256(path),bytes=len(data),crc32=expected['crc32'],zip_crc32_verified=True))
            verify_hashes(launch['protected_hashes'])
            report=dict(status='completed_identity_members_verified',job_id=JOB,completed_at=now(),files=rows,
                transferred_bytes=source.transferred,full_archive_md5_verified=False,audio_downloaded=False,human_ratings_read=False,
                training_admitted=False,archive_source=launch['item'])
            atomic_json(DIRECTORY/'report.json',report)
            with run_lease('next-va-v2-d-ledger'):
                stages=read_json(PLAN)['stages'];stages['D']['manymusic_identity_members']=str((DIRECTORY/'report.json').relative_to(ROOT))
                record_event(dict(event='D_ManyMusic_identity_members_saved',job_id=JOB),updates=dict(stages=stages))
            print(json.dumps(report))
        except BaseException as exc:
            atomic_json(DIRECTORY/'failure.json',dict(failed_at=now(),error=f'{type(exc).__name__}: {exc}'))
            record_event(dict(event='D_ManyMusic_identity_fetch_failed',error=f'{type(exc).__name__}: {exc}'));raise


if __name__=='__main__':main()
