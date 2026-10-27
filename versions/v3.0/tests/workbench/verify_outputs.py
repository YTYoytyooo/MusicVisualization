"""Verify actual muxed acceptance outputs and protected source hashes."""
import json
import subprocess
from pathlib import Path
import cv2
from tools.train_next_v2.v3_workbench.studio import store
from tools.train_next_v2.v3_workbench.studio.paths import WORKSPACE

reports = WORKSPACE / 'data/v3.0/reports/workbench'
integration = store.read_json(reports/'integration.json')
assert integration['status'] == 'passed'
project = WORKSPACE/'data/v3.0/works'/integration['project_id']
ffmpeg = WORKSPACE/'versions/v2.0/.runtime/bin/ffmpeg.exe'
results=[]
for check in integration['checks']:
    if 'render_id' not in check:
        continue
    folder=project/'renders'/check['render_id']
    video=folder/'output.mp4'
    if not video.exists():
        folder=project/'previews'/check['render_id']
        video=folder/'output.mp4'
    metadata=store.read_json(folder/'render.json')
    capture=cv2.VideoCapture(str(video))
    assert capture.isOpened(), str(video)
    width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=capture.get(cv2.CAP_PROP_FPS)
    frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_MSEC, min(5000, max(0, frames/fps/2*1000)))
    ok,frame=capture.read()
    assert ok and frame.std()>1
    if check['check']=='full_song_720p30_with_audio':
        cv2.imwrite(str(reports/'render-example.jpg'),frame)
    capture.release()
    assert frames==metadata['frames'], (frames,metadata['frames'])
    assert width==metadata['settings']['width'] and height==metadata['settings']['height']
    decoded=subprocess.run([str(ffmpeg),'-hide_banner','-v','error','-i',str(video),
        '-map','0:v:0','-map','0:a:0','-f','null','-'],capture_output=True,text=True)
    assert decoded.returncode==0, decoded.stderr
    results.append(dict(check=check['check'],video=str(video),width=width,height=height,
        fps=fps,frames=frames,video_seconds=frames/fps,
        full_video_and_audio_decode='passed',sha256=store.digest(video)))

snapshots={}
for name,path in [('workbench',reports/'protected-before.json'),
                  ('historical_v1_v2',WORKSPACE/'data/v3.0/reports/protected-before.json')]:
    before=store.read_json(path)
    hashes=before.get('hashes',before)
    mismatches=[key for key,sha in hashes.items() if store.digest(WORKSPACE/key)!=sha]
    snapshots[name]=dict(files=len(hashes),mismatches=mismatches)
    assert not mismatches, mismatches
store.atomic_json(reports/'media-and-protection.json',dict(status='passed',checked_at=store.now(),
    outputs=results,protected=snapshots,original_model_sha256=store.digest(WORKSPACE/'models/emotion_model.pth')))
print(json.dumps(dict(status='passed',outputs=len(results),protected=snapshots),ensure_ascii=False))
