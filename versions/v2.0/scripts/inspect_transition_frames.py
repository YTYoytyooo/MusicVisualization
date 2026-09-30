"""Inspect actual decoded transition videos at matching times; no inferred frames."""
import json
from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[3] / 'data/v2.0/validation-output/transitions'
report = json.loads((root / 'report.json').read_text(encoding='utf-8'))
sheet = np.zeros((4 * 205, 3 * 320, 3), dtype=np.uint8)
for row, video in enumerate(report['videos']):
    cap = cv2.VideoCapture(video['video'])
    for col, second in enumerate((.7, 1.5, 2.3)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(second * 30))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Cannot decode acceptance video frame')
        image = cv2.resize(frame, (320, 180))
        top = row * 205
        sheet[top+25:top+205, col*320:(col+1)*320] = image
        cv2.putText(sheet, f"{video['mode']} / {second:.1f}s", (col*320+8, top+18),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (240,240,240), 1, cv2.LINE_AA)
    cap.release()
if not cv2.imwrite(str(root / 'comparison.png'), sheet):
    raise RuntimeError('Could not save visual acceptance sheet')
print(root / 'comparison.png')
