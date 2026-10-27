"""Read a bounded snapshot of the active two-stage training pipeline."""
from datetime import datetime, timezone
import json
from pathlib import Path

from .core import ROOT, output_paths
from .runtime import is_running


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def recent_progress(path):
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and ('done' in value or 'epoch' in value):
            return value
    return None


def run_snapshot(run_id, log):
    run, models = output_paths(run_id)
    active = is_running(run_id)
    state = read_json(run / 'status.json')
    report = read_json(run / 'report.json')
    completed = bool(report and report.get('status') == 'completed'
                     and report.get('legacy_model_unchanged')
                     and (models / 'best.pth').is_file())
    result = dict(run_id=run_id, running=active,
                  outcome='running' if active else 'completed' if completed else
                          'stopped_without_completion' if run.exists() else 'not_started',
                  status=state, recent_progress=recent_progress(Path(log)),
                  report=str(run / 'report.json') if report else None,
                  last_checkpoint_exists=(models / 'last.pth').is_file())
    if report:
        result['best_epoch'] = report.get('best_epoch')
        metrics = report.get('best_validation', {})
        result['best_validation'] = metrics.get('macro') if 'macro' in metrics else {
            name: value.get('macro') for name, value in metrics.items()}
    return result


def main():
    active_path = ROOT / 'data/training/next_va/active-training.json'
    active = read_json(active_path)
    if not active:
        raise ValueError(f'No active training configuration: {active_path}')
    teacher = run_snapshot(active['teacher'], active['teacher_log'])
    mixed = run_snapshot(active['mixed'], active['mixed_log'])
    queue = dict(running=is_running('queue-' + active['mixed']),
                 status=read_json(ROOT / 'data/training/next_va/queues' / active['mixed'] / 'status.json'))
    print(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),
                          teacher=teacher, mixed=mixed, queue=queue,
                          pipeline_completed=teacher['outcome'] == mixed['outcome'] == 'completed'),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
