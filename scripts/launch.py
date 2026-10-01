"""Select an existing usable Python; never install or modify environments."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('numpy', 'torch', 'cv2', 'librosa', 'transformers', 'soundfile', 'PIL')


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('v1', 'v2'):
        print('Usage: python scripts/launch.py v1|v2 [arguments]')
        return 2
    version = 'v1.0' if sys.argv[1] == 'v1' else 'v2.0'
    args = sys.argv[2:]
    if version == 'v1.0' and not args:
        print('Drag music files onto Start V1.cmd, or run Start V1.cmd "path\\song.mp3"')
        return 0
    probe = 'import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(n) for n in '+repr(REQUIRED)+') else 1)'
    candidates = [os.environ.get('MUSIC_PYTHON')] if os.environ.get('MUSIC_PYTHON') else [str(ROOT/'venv/Scripts/python.exe'), sys.executable]
    for candidate in dict.fromkeys(candidates):
        try:
            if subprocess.run([candidate, '-c', probe], capture_output=True, timeout=15).returncode:
                continue
        except (OSError, subprocess.TimeoutExpired):
            continue
        if version == 'v2.0' and not args:
            args = ['serve']
            print('Studio: http://127.0.0.1:8765', flush=True)
        child_env = os.environ.copy()
        paths = [str(Path(candidate).resolve().parent), child_env.get('PATH', '')]
        encoder = ROOT/'versions/v2.0/.runtime/bin/ffmpeg.exe'
        if not shutil.which('ffmpeg') and encoder.is_file():
            paths.insert(0, str(encoder.parent))
        child_env['PATH'] = os.pathsep.join(paths)
        # Both versions reuse the prepared CLAP cache; honor user overrides.
        clap_cache = ROOT/'versions/v2.0/.runtime/huggingface/hub'
        if clap_cache.is_dir() and not child_env.get('HF_HOME'):
            child_env.setdefault('HF_HUB_CACHE', str(clap_cache))
        return subprocess.call([candidate, str(ROOT/'versions'/version/'main.py'), *args], env=child_env)
    print('No usable Python environment with the music dependencies was found.')
    print('Check venv/Scripts/python.exe, or set MUSIC_PYTHON to a prepared Python executable.')
    print('See README.md for environment setup and version entry points.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
