"""Synthetic fixtures for UI and encoding verification; never music predictions."""
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
from .store import create_project


def create_demo(root, duration=8.):
    if not np.isfinite(duration) or not .5 <= duration <= 60:
        raise ValueError('合成演示时长需要在 0.5～60 秒')
    with tempfile.TemporaryDirectory() as temporary:
        audio = Path(temporary) / 'SYNTHETIC.wav'
        samples = np.arange(round(duration * 22050)) / 22050
        sf.write(audio, .05 * np.sin(2 * np.pi * 220 * samples), 22050)
        t = np.arange(int(np.ceil(duration * 10))) / 10
        raw = np.zeros((len(t), 5))
        raw[:, 0] = .16 + .04 * np.sin(t * 1.8)
        raw[:, 1] = .34 + .04 * np.cos(t * 1.2)
        return create_project(root, '合成演示 / SYNTHETIC — 非真实预测', audio, None, raw,
                              dict(duration=duration, tempo=120., beat_times=np.arange(0, duration, .5)),
                              provenance={'kind': 'synthetic', 'note': 'Synthetic tone and hand-designed values'})
