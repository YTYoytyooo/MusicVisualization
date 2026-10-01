"""Run browser checks with an in-process server; no orphan service or user labels."""
from pathlib import Path
import math
import os
import struct
import subprocess
import sys
import tempfile
import threading
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import make_server, ROOT

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory)
    audio=folder/'audio';audio.mkdir()
    with wave.open(str(audio/'情绪标注演示.wav'),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(8000)
        f.writeframes(b''.join(struct.pack('<h',int(1000*math.sin(i*2*math.pi*220/8000))) for i in range(80000)))
    server=make_server(audio,folder/'labels',0)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    output=ROOT/'data/v2.0/validation-output/annotation-tool'
    output.mkdir(parents=True,exist_ok=True)
    try:
        env={**os.environ,'ANNOTATION_URL':f'http://127.0.0.1:{server.server_port}',
             'ANNOTATION_SCREENSHOT':str(output/'annotation-desktop.png')}
        result=subprocess.run(['node',str(Path(__file__).with_suffix('.cjs'))],env=env,timeout=45)
        if result.returncode:raise SystemExit(result.returncode)
    finally:
        server.shutdown();server.server_close();server.store.close();worker.join()
