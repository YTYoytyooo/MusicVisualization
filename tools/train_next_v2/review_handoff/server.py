"""Compose the existing annotation UI with a pinned pool and native EOF clock."""
import argparse
import copy
import hashlib
from pathlib import Path

from tools.annotate.server import make_server, Conflict
from tools.train_next_v2.common import ROOT, read_json, verify_hashes


def create_server(config, output, port=8767):
    verify_hashes(config['protected_hashes'])
    pool=read_json(ROOT/config['pool'])['candidates']
    by_key={}
    audio_root=ROOT/config['audio_root']
    for row in pool:
        path=(ROOT/row['audio_path']).resolve()
        relative=path.relative_to(audio_root).as_posix()
        ids=row['raw_annotation_song_ids']
        if len(ids)>1: raise ValueError('Resolve multiple historical song IDs before handoff')
        key=ids[0] if ids else hashlib.sha256(relative.encode()).hexdigest()[:24]
        if key in by_key: raise ValueError('Duplicate song key')
        by_key[key]=(row,path,relative)
    if len(by_key)!=58: raise ValueError('The fixed pool must contain exactly 58 recordings')
    server=make_server(audio_root,output,port)
    store=server.store
    # Adapt this new server instance only. Existing module/default behavior is untouched.
    store.files={key:(path,relative) for key,(_,path,relative) in by_key.items()}
    original_source=store.source

    def native_source(key):
        source=original_source(key)
        row=by_key[key][0]
        if source['audio_sha256']!=row['audio_sha256']: raise Conflict('音频与固定试点清单不一致')
        native=row['native_audio']
        source['duration']=native['frames']/native['sample_rate']
        source['duration_basis']='verified_native_eof_samples'
        return source

    def reject_import(*args,**kwargs):
        raise ValueError('此入口固定为已核验的58首；新音频请使用原标注工具')

    store.source=native_source
    store.import_audio=reject_import
    server.pilot_keys=by_key
    return server


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8767);args=parser.parse_args()
    config=read_json(args.config)
    output=(ROOT/config['annotation_output']).resolve()
    if not output.is_relative_to(ROOT/'data/annotations'): raise ValueError('Wrong annotation output directory')
    server=create_server(config,output,args.port)
    print(f'58-song pilot: http://127.0.0.1:{server.server_port}',flush=True)
    print(f'Annotations: {output}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close();server.store.close()


if __name__=='__main__': main()
