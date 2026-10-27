"""Publish the bounded V3 workbench verification record; never train or import data."""
import csv
from collections import Counter
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2

from tools.train_next_v2.v3_workbench.studio import store
from tools.train_next_v2.v3_workbench.studio.jobs import process_identity
from tools.train_next_v2.v3_workbench.studio.paths import WORKSPACE


def main():
    reports = WORKSPACE / 'data/v3.0/reports/workbench'
    code = sorted((WORKSPACE / 'tools/train_next_v2/v3_workbench').rglob('*.py'))
    code += sorted((WORKSPACE / 'versions/v3.0/workbench-web').glob('*'))
    code += [WORKSPACE / 'versions/v3.0/main.py', WORKSPACE / 'Start V3.cmd']
    store.atomic_json(reports / 'finalization-launch.json', {
        'registered_at': store.now(),
        'argv': list(sys.orig_argv),
        'cwd': str(WORKSPACE), 'environment': {k: os.environ.get(k) for k in
            ('PYTHONPATH', 'PYTHONUTF8', 'PYTHONDONTWRITEBYTECODE', 'TEMP', 'TMP')},
        'script_sha256': store.digest(__file__),
        'code_hashes': {str(p.relative_to(WORKSPACE)): store.digest(p) for p in code if p.is_file()},
        'log': str(reports / 'finalization.log'),
        'recovery': 'Read-only checks and atomic report replacement; rerun only after this invocation exits.',
        'training': False,
    })
    python_log = (reports / 'unit-tests.log').read_text(encoding='utf-8-sig')
    js_log = (reports / 'ui-math-tests.log').read_text(encoding='utf-8-sig')
    old_log = (WORKSPACE / 'data/v3.0/reports/unit-tests.log').read_text(encoding='utf-8-sig')
    assert 'Ran 43 tests' in python_log and '\nOK' in python_log
    assert '"checks":17' in js_log and '"status":"passed"' in js_log
    assert 'Ran 56 tests' in old_log and '\nOK' in old_log
    evidence = {name: store.read_json(reports / name) for name in
        ('integration.json', 'media-and-protection.json', 'revision-motion-acceptance.json')}
    assert all(item['status'] == 'passed' for item in evidence.values())

    # Decode the additional manual-motion result and check actual per-frame controls.
    extra = evidence['revision-motion-acceptance.json']
    folder = WORKSPACE / 'data/v3.0/works' / extra['project_id'] / 'renders' / extra['render']['id']
    video = folder / 'output.mp4'
    assert store.digest(video) == extra['render']['video_sha256']
    result = subprocess.run([str(WORKSPACE / 'versions/v2.0/.runtime/bin/ffmpeg.exe'),
        '-hide_banner', '-v', 'error', '-i', str(video), '-map', '0:v:0', '-map', '0:a:0',
        '-f', 'null', '-'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    capture = cv2.VideoCapture(str(video))
    geometry = [int(capture.get(key)) for key in
        (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT)]
    capture.release()
    assert geometry == [640, 360, 30], geometry
    with (folder / 'frame_values.csv').open(encoding='utf-8-sig', newline='') as stream:
        frames = list(csv.DictReader(stream))
    assert len(frames) == 30
    assert all(row['motion_mode'] == 'orbit' and row['motion_source'] == 'manual'
        and row['revision'] == 'r000005' and float(row['motion_speed']) == .2
        and float(row['motion_radius']) == .25 and float(row['motion_rotation']) == -1
        for row in frames)
    store.atomic_json(reports / 'manual-motion-media.json', {
        'status': 'passed', 'checked_at': store.now(), 'video': str(video),
        'geometry_width_height_frames': geometry, 'complete_video_audio_decode': True,
        'actual_frame_controls_checked': len(frames), 'sha256': store.digest(video)})

    protected = {}
    for name, path in [('workbench', reports / 'protected-before.json'),
                       ('historical_v1_v2', WORKSPACE / 'data/v3.0/reports/protected-before.json')]:
        snapshot = store.read_json(path)
        hashes = snapshot.get('hashes', snapshot)
        bad = [key for key, value in hashes.items() if store.digest(WORKSPACE / key) != value]
        assert not bad, bad
        protected[name] = {'files': len(hashes), 'mismatches': bad}

    conn = http.client.HTTPConnection('127.0.0.1', 8768, timeout=10)
    conn.request('GET', '/api/bootstrap')
    response = conn.getresponse()
    bootstrap = json.loads(response.read())
    assert response.status == 200
    conn.close()  # Do not persist the session token.
    jobs = []
    for path in sorted((WORKSPACE / 'data/v3.0/works/.jobs').glob('j-*.json')):
        job = store.read_json(path)
        alive = bool(job.get('process_identity') and
            process_identity(job.get('pid', 0)) == job['process_identity'])
        assert not alive and job['status'] not in ('running', 'queued', 'cancelling'), job['id']
        jobs.append({'id': job['id'], 'kind': job['kind'], 'status': job['status'], 'process_alive': alive})
    service = store.read_json(reports / 'service-launch.json')
    service.update({'checked_at': store.now(), 'status': 'serving',
        'verified_bootstrap_project_count': len(bootstrap['projects']),
        'launcher_identity': process_identity(94904), 'worker_identity': process_identity(37308),
        'exec_session_id': 38003, 'log_tail': 'Local workbench service; no active worker jobs.'})
    assert service['launcher_identity'] and service['worker_identity']
    store.atomic_json(reports / 'service-status.json', service)

    browser = {
        'recorded_at': store.now(), 'status': 'core_interactions_verified_partial_coverage',
        'url': 'http://127.0.0.1:8768/',
        'method': 'Actual browser UI interactions through CUA; DOM observations are read-only.',
        'passed': [
            'Create project from existing 614.mp3; verified cached analysis reused, no training.',
            'Valence point target at 5 seconds = 0.4; undo, redo, save r000001.',
            'Retain edits while switching to B00 (r000002); reset to B01 (r000003).',
            'Historical revision inspection, comparison curves, zoom and video selection.',
            'Same video retained: r000002 -> r000001 raw CSV changes v2 -> b01; switching back restores v2.',
            'Reference comparison plays absolute song time 20-21 seconds: main 20.099727, reference 0.155972.',
            'Both videos pause at overlap end (main 21.005552, reference 1.0); only one is audible.',
            'Sound source switch changes main/reference muted flags from false/true to true/false.',
            'A/B 0.2-0.8 second loop remained playing at 0.759882 after intervening checks, then explicitly paused.',
            'Late 20-21 second video initializes curve/playhead at absolute song time 20 seconds.',
        ],
        'not_claimed_as_browser_passed': [
            'Historical restore and manual-motion save were verified by real API/worker, not UI click.',
            'Complete dirty-draft modal and browser restart recovery scenario.',
            'Every keyframe drag, every motion mode, and every video output size over a full song.',
        ],
        'human_listening': 'not_performed', 'console_errors_in_final_page': [],
        'screenshots': ['data/v3.0/reports/workbench/workbench-ui.png',
                        'data/v3.0/reports/workbench/workbench-preview.png'],
        'automation_issue': 'Earlier focus/control timeouts limited initial coverage; later continued interaction succeeded.'
    }
    store.atomic_json(reports / 'browser-acceptance.json', browser)
    report = {
        'status': 'implemented_core_acceptance_passed', 'completed_at': store.now(),
        'scope': 'Independent V3 creation workbench feature restoration and optimization.',
        'entrypoint': 'Start V3.cmd', 'url': 'http://127.0.0.1:8768/',
        'code': 'tools/train_next_v2/v3_workbench', 'web': 'versions/v3.0/workbench-web',
        'works': 'data/v3.0/works',
        'tests': {'workbench_python': 43, 'ui_javascript': 17, 'frozen_v3_regression': 56},
        'integration': 'data/v3.0/reports/workbench/integration.json',
        'media': 'data/v3.0/reports/workbench/media-and-protection.json',
        'manual_motion_media': 'data/v3.0/reports/workbench/manual-motion-media.json',
        'browser': 'data/v3.0/reports/workbench/browser-acceptance.json',
        'performance': store.read_json(reports / 'timeline-performance.json'),
        'full_video': evidence['integration.json']['full_video'],
        'full_video_mode': 'analysis',
        'protected': protected,
        'original_model_sha256': store.digest(WORKSPACE / 'models/emotion_model.pth'),
        'jobs': jobs, 'job_status_counts': dict(Counter(j['status'] for j in jobs)),
        'active_worker_jobs': 0, 'service_status': 'data/v3.0/reports/workbench/service-status.json',
        'new_data_used': False, 'training_performed': False, 'model_promoted': False,
        'final_test_evaluation': False, 'automation_status': 'PAUSED',
        'limitations': [
            'Human listening and subjective visual/emotion judgement remain unperformed.',
            'Browser coverage is bounded; see browser-acceptance.json for unexercised scenarios.',
            'Full render verified on 28.029-second audio at 720p30; 1080p verified on a late one-second excerpt.',
            'Performance timing is one real render sample and pure timeline calculations, not full UI or long-song guarantees.',
            'Feature restoration is not evidence that frozen V3 model accuracy surpasses V2.'
        ],
        'next_step': 'Use the workbench on existing authorized audio and perform human listening; no autonomous new-data training.'
    }
    report['artifacts_sha256'] = {str(p.relative_to(WORKSPACE)): store.digest(p) for p in
        [reports / name for name in ('integration.json', 'media-and-protection.json',
            'manual-motion-media.json', 'browser-acceptance.json', 'ui-math-tests.log', 'unit-tests.log')]
        + [WORKSPACE / 'versions/v3.0/README.md', WORKSPACE / 'versions/v3.0/docs/WORKBENCH_GUIDE.md',
           WORKSPACE / 'versions/v3.0/docs/WORKBENCH_ACCEPTANCE.md']}
    store.atomic_json(reports / 'delivery.json', report)
    plan_path = WORKSPACE / 'data/training/next_va/optimization/plan.json'
    plan = store.read_json(plan_path)
    assert plan['automation_status'] == 'PAUSED'
    assert plan['user_data_restriction']['additional_training_on_new_data_paused']
    wb = plan['v3_workbench']
    wb.update({'status': report['status'], 'completed_at': report['completed_at'],
        'active_stage': 'human_experience_acceptance_pending',
        'unit_tests': 43, 'browser_math_checks': 17, 'frozen_v3_regression_tests': 56,
        'report': 'data/v3.0/reports/workbench/delivery.json',
        'browser_report': report['browser'], 'protected_final': protected,
        'milestone_status': {stage: 'implemented_and_core_verified' for stage in wb['stages']},
        'limitations': report['limitations'], 'active_worker_jobs': 0, 'service_url': report['url'],
        'human_listening': 'pending', 'new_data_training': False, 'training_enabled': False})
    plan['next_action'] = plan['v3_next_action'] = report['next_step']
    plan['updated_at'] = store.now()
    store.atomic_json(plan_path, plan)
    event = {'time': store.now(), 'event': 'v3_workbench_core_delivery_verified',
        'status': report['status'], 'report': wb['report'], 'tests': report['tests'],
        'actions': ['restored_editor_visual_motion_render_and_migration', 'verified_actual_media',
                    'verified_browser_core_flow_and_reference_loop', 'published_guides_and_evidence'],
        'evidence': [report['integration'], report['media'], report['browser']],
        'stage_judgement': 'Core engineering implemented and verified; subjective acceptance and bounded browser gaps remain.',
        'next_step': report['next_step'], 'blockers': ['Human listening and subjective acceptance need a person.'],
        'training': False, 'model_promoted': False, 'protected': protected, 'automation_status': 'PAUSED'}
    with (plan_path.parent / 'events.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'tests': report['tests'], 'protected': protected,
                      'jobs': report['job_status_counts'], 'report': wb['report']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
