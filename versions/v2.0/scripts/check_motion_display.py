"""Serve existing acceptance assets read-only for the small visual regression."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from studio.server import make_server

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--projects',type=Path,required=True)
    parser.add_argument('--project',required=True)
    args=parser.parse_args()
    server=make_server(args.projects,0,start_queue=False)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        env={**os.environ,'STUDIO_URL':f'http://127.0.0.1:{server.server_port}','STUDIO_PROJECT':args.project}
        result=subprocess.run(['node','scripts/check_motion_display.cjs'],cwd=ROOT,env=env,timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        raise SystemExit(result.returncode)
    finally:
        server.shutdown();server.server_close()
