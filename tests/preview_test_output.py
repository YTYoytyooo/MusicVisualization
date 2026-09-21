"""Create a short HUD smoke preview with explicitly synthetic predictions.

Run from the repository root:
python tests/preview_test_output.py --output-dir ../test-output-preview
This does NOT run CLAP or validate emotion recognition.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import soundfile as sf

from main import _merge_audio_video
from mcts import VisualState
from prediction_overlay import PredictionOverlay
from renderer import VideoRenderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / 'synthetic-hud-preview'
    if any(args.output_dir.glob(stem.name + '*')):
        raise FileExistsError('Preview outputs exist; choose a new output directory.')
    duration, fps, sr = 4, 10, 22050
    times = np.arange(40) * 0.1
    states = [dict(valence=float(np.sin(t * 1.8) * .8),
                   arousal=float(np.cos(t * 1.2) * .7)) for t in times]
    hud = PredictionOverlay('显示测试 / SYNTHETIC DATA (not a song prediction)',
                            duration, states)
    sample_times = np.arange(duration * sr) / sr
    audio = (.08 * np.sin(2 * np.pi * 220 * sample_times)).astype(np.float32)
    silent_video = str(stem) + '_noaudio.avi'
    audio_path = str(stem) + '_audio.wav'
    renderer = VideoRenderer(silent_video, fps=fps, prediction_overlay=hud)
    try:
        for index in range(duration * fps):
            t = index / fps
            start = round(t * sr)
            frame = renderer.render_frame(
                VisualState(particle_count=30, trail_length=5, particle_speed=2),
                audio[start:start + sr // fps], 0, t * .05, t % .5 / .5)
            renderer.write_frame(frame)
            if index == 30:
                path = str(stem) + '.png'
                if not cv2.imwrite(path, frame):
                    raise RuntimeError('Could not save preview image')
    finally:
        renderer.release()
    sf.write(audio_path, audio, sr, subtype='PCM_16')
    _merge_audio_video(silent_video, audio_path, str(stem) + '.mp4')
    print(f'Preview: {stem} (synthetic data only)')


if __name__ == '__main__':
    main()
