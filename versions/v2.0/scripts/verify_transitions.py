"""Small actual MP4 comparison; synthetic source, no ML or downloads."""
from pathlib import Path
import csv
import sys
from unittest.mock import patch
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.paths import DATA_ROOT, DEFAULT_PROJECTS, DEFAULT_MODEL
from studio.demo import create_demo
from studio.pipeline import render
from studio.store import atomic_json, digest, load_project, save_revision
from validate_release import inspect_video


def main():
    output = DATA_ROOT / 'validation-output/transitions'
    output.mkdir(parents=True, exist_ok=True)
    project = create_demo(output, 3.)
    original_hash = digest(project / 'analysis/predictions_raw.npz')
    options = dict(fps=30, width=640, height=360, mode='analysis')
    report = {'project': str(project), 'source': 'synthetic', 'status': 'running', 'videos': []}
    atomic_json(output / 'report.json', report)
    # WAV is already mono 22050Hz. Real decode avoids irrelevant first-use JIT.
    with patch('feature_extraction.load_audio', side_effect=lambda path: sf.read(path, dtype='float32')):
        baseline = render(project, options=options)
        report['videos'].append({'mode':'original', **inspect_video(project, baseline)})
        for mode in ('linear', 'smoothstep', 'smootherstep'):
            edit = dict(id='speed',layer='visual',field='particle_speed',operation='point_target',
                        time_us=1500000,value=7.,transition_in_us=1000000,transition_out_us=1000000,
                        interpolation=mode,enabled=True)
            revision = save_revision(project, load_project(project)['current_revision'], [edit])
            result = render(project, revision['id'], options)
            checked = inspect_video(project, result)
            report['videos'].append({'mode':mode, **checked})
            with open(project/'renders'/result['id']/'frame_values.csv',encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            with open(project/'renders'/baseline['id']/'frame_values.csv',encoding='utf-8-sig') as f:
                original=list(csv.DictReader(f))
            for row, base in zip(rows, original):
                t=int(row['time_us'])
                speed=float(row['visual_particle_speed'])
                if t <= 500000 or t >= 2500000:
                    assert speed == float(base['visual_particle_speed'])
                if t == 1500000:
                    assert speed == 7.
            assert result['plan_key'] == baseline['plan_key']
            # The exact renderer inputs at u=0.2 have independent expected weights.
            point=next(r for r in rows if int(r['time_us']) == 700000)
            base=next(r for r in original if int(r['time_us']) == 700000)
            w={'linear':.2,'smoothstep':.104,'smootherstep':.05792}[mode]
            expected=float(base['visual_particle_speed'])*(1-w)+7*w
            assert abs(float(point['visual_particle_speed'])-expected)<1e-9
            atomic_json(output / 'report.json', report)
    assert digest(project / 'analysis/predictions_raw.npz') == original_hash
    report.update(status='passed',raw_preserved=True,frames_per_video=90,
                  target_and_boundary_values_verified=True)
    atomic_json(output / 'report.json', report)
    print('Transition MP4 comparison passed:', output / 'report.json', flush=True)


if __name__ == '__main__':
    main()
