"""Finite release gate. Writes only inside Studio; never changes the existing venv.

Run with --prepare-ffmpeg to download an isolated video encoder, then validate
real model inference and original/edited/restored H.264 + AAC outputs.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import DATA_ROOT, DEFAULT_PROJECTS, DEFAULT_MODEL
# Keep any new Hugging Face downloads inside this upgraded project.
# Existing cached model weights may be read without copying or modifying them.
existing_hub = Path.home() / '.cache/huggingface/hub'
if not (existing_hub / 'models--laion--clap-htsat-fused').is_dir():
    os.environ.setdefault('HF_HOME', str(ROOT / '.runtime/huggingface'))
os.environ.setdefault('PYTHONUNBUFFERED', '1')

from studio.store import atomic_json, digest, load_project, load_raw, now, save_revision
from studio.pipeline import analyze, doctor, ffmpeg_path, render


def run(argv, **kwargs):
    print('RUN:', ' '.join(map(str, argv)), flush=True)
    return subprocess.run(list(map(str, argv)), cwd=ROOT, check=True, **kwargs)


def inspect_video(project, result):
    import cv2
    folder = project / 'renders' / result['id']
    video = folder / result['video_file']
    with open(folder / 'frame_values.csv', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    reader = cv2.VideoCapture(str(video))
    fps = reader.get(cv2.CAP_PROP_FPS)
    frames = 0
    decoded_hash = __import__('hashlib').sha256()
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        frames += 1
        decoded_hash.update(frame.tobytes())
    reader.release()
    assert frames == len(rows) == result['frames'], (frames, len(rows), result)
    assert abs(fps - result['settings']['fps']) < .01
    decoded = run([ffmpeg_path(), '-v', 'error', '-i', video, '-map', '0:a:0',
                   '-ar', '16000', '-ac', '1', '-f', 's16le', '-'], capture_output=True, timeout=60)
    audio_seconds = len(decoded.stdout) / 32000
    assert abs(audio_seconds - frames / fps) < max(.08, 1 / fps), audio_seconds
    assert any(decoded.stdout), 'audio stream contains only zero samples'
    return {'folder': str(folder), 'video': str(video), 'frames': frames, 'fps': fps,
            'audio_seconds': audio_seconds, 'decoded_frames_sha256': decoded_hash.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-ffmpeg', action='store_true')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--audio', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    output = DATA_ROOT / 'validation-output'
    output.mkdir(exist_ok=True)
    report_path = output / 'release-report.json'
    report = {'status': 'running', 'started_at': now(), 'steps': [], 'projects': []}

    def record(label, details):
        report['steps'].append({'step': label, 'completed_at': now(), 'details': details})
        atomic_json(report_path, report)
        print(label, json.dumps(details, ensure_ascii=True), flush=True)

    atomic_json(report_path, report)
    originals = {str(p): digest(p) for p in [args.model, *args.audio]}
    try:
        if not ffmpeg_path() and args.prepare_ffmpeg:
            run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                 '--no-deps', '--target', ROOT / '.runtime', 'imageio-ffmpeg==0.6.0'], timeout=300)
        env = doctor()
        assert env['h264'] and env['aac'], 'H.264/AAC encoder unavailable'
        record('environment', env)
        run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], timeout=120)
        run(['node', '--check', 'web/app.js'], timeout=30)
        record('automated_tests', {'passed': True})

        from studio.demo import create_demo
        import soundfile as sf
        from unittest.mock import patch
        import numpy as np
        projects_root = DEFAULT_PROJECTS
        demo = create_demo(projects_root, 2.13)
        demo_result = render(demo, None, dict(fps=12, width=640, height=360), iterations=20)
        record('synthetic_mp4_audio_and_non_integer_duration', inspect_video(demo, demo_result))

        # Exercise the real subprocess scheduler in its own project collection;
        # never start a second manager against the running editor's projects.
        from studio.jobs import JobManager
        queue_root = output / ('queue-' + uuid.uuid4().hex[:8])
        queued_project = create_demo(queue_root, .6)
        manager = JobManager(queue_root)
        try:
            task = manager.submit('render', {'project_id': queued_project.name,
                       'revision': 'r000000', 'options': dict(fps=10, width=320, height=180)})
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                latest = next(j for j in manager.list() if j['id'] == task['id'])
                if latest['status'] not in ('running', 'queued', 'cancelling'):
                    break
                time.sleep(.25)
            else:
                manager.cancel(task['id'])
                raise TimeoutError('Subprocess queue did not finish in 180s')
            assert latest['status'] == 'succeeded', latest
            time.sleep(1)
            assert next(j for j in manager.list() if j['id'] == task['id'])['status'] == 'succeeded'
            record('subprocess_queue_and_terminal_state', inspect_video(queued_project, latest['result']))
        finally:
            manager.close()

        for source in args.audio:
            # Real decoded excerpt; the source song and checkpoint remain read-only.
            audio, rate = sf.read(str(source), frames=6 * 48000, always_2d=True, dtype='float32')
            audio = audio[:6 * rate]
            excerpt = output / (source.stem + '-validation-6s.wav')
            sf.write(str(excerpt), audio, rate)
            project = analyze(excerpt, args.model, projects_root,
                              name=source.stem + ' / VALIDATION 6s',
                              progress=lambda stage, ratio: print(stage, ratio, flush=True))
            before = digest(project / 'analysis/predictions_raw.npz')
            p = load_project(project)
            assert p['provenance']['kind'] == 'model'
            options = dict(fps=30, width=1280, height=720, mode='analysis')
            base = render(project, 'r000000', options)
            base_check = inspect_video(project, base)
            raw = load_raw(project)
            old_a = float(raw[min(20, len(raw)-1), 1])
            target = .7 if old_a < .5 else -.4
            edit = dict(id='acceptance-arousal', enabled=True, layer='emotion', field='arousal',
                        operation='point_target', time_us=2000000, value=target,
                        transition_in_us=500000, transition_out_us=500000,
                        note='Acceptance fixture; manual, not model output')
            revision = save_revision(project, 'r000000', [edit])
            # Prove rerender has no dependency on model/CLAP inference.
            with (patch('emotion_model.extract_clap_embeddings', side_effect=AssertionError('unexpected CLAP')),
                  patch('emotion_model.EmotionInterface.load', side_effect=AssertionError('unexpected model'))):
                changed = render(project, revision['id'], options)
            changed_check = inspect_video(project, changed)
            with np.load(project / 'renders' / changed['id'] / 'predictions_effective.npz') as data:
                exact = np.flatnonzero(data['times_us'] == 2000000)
                assert len(exact) == 1 and abs(float(data['values'][exact[0], 1]) - target) < 1e-9
            restored_revision = save_revision(project, revision['id'], [])
            restored = render(project, restored_revision['id'], {**options, 'mode': 'presentation'})
            restored_check = inspect_video(project, restored)
            assert base['plan_key'] == restored['plan_key']
            assert base['plan_key'] != changed['plan_key']
            with (np.load(project / 'renders' / base['id'] / 'predictions_effective.npz') as a,
                  np.load(project / 'renders' / restored['id'] / 'predictions_effective.npz') as b):
                assert np.array_equal(a['values'], b['values'])
            assert digest(project / 'analysis/predictions_raw.npz') == before
            item = {'project': str(project), 'original': base_check, 'edited': changed_check,
                    'restored': restored_check, 'raw_preserved': True, 'edit_target_verified': target,
                    'restored_numerical_plan_equal': True, 'no_model_calls_on_rerender': True}
            report['projects'].append(item)
            record('real_song_revision_roundtrip', item)
        assert all(digest(Path(p)) == value for p, value in originals.items())
        record('original_audio_and_checkpoint_unchanged', originals)
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['completed_at'] = now()
        atomic_json(report_path, report)


if __name__ == '__main__':
    main()
