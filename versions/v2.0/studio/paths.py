"""Keep application code and persistent workspace data separate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
DATA_ROOT = WORKSPACE / 'data' / 'v2.0'
DEFAULT_PROJECTS = DATA_ROOT / 'projects'
VALIDATION_OUTPUT = DATA_ROOT / 'validation-output'
DEFAULT_MODEL = WORKSPACE / 'models' / 'emotion_model.pth'


def migrated_path(value):
    """Resolve explicitly recorded moves without rewriting historical metadata."""
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        return path
    import json
    manifest = WORKSPACE / 'archive/migration-2026-09-30/moves.json'
    if not manifest.is_file():
        return path
    # Original paths were absolute on this workspace's D: drive.
    original = Path('D:/2_datas/Vm')
    for entry in json.loads(manifest.read_text(encoding='utf-8-sig')):
        try:
            suffix = path.relative_to(original / entry['source'])
        except ValueError:
            continue
        candidate = WORKSPACE / entry['destination'] / suffix
        if candidate.exists():
            return candidate
    return path
