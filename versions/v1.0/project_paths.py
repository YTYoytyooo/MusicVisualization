"""Workspace paths, independent of the caller's working directory."""
from pathlib import Path
import hashlib

WORKSPACE = Path(__file__).resolve().parents[2]
AUDIO_ROOT = WORKSPACE / 'assets' / 'audio'
DEFAULT_MODEL = WORKSPACE / 'models' / 'emotion_model.pth'
DEFAULT_OUTPUT = WORKSPACE / 'data' / 'v1.0' / 'outputs'
DEFAULT_CACHE = WORKSPACE / 'data' / 'v1.0' / 'cache'


def audio_key(audio):
    path = Path(audio).resolve()
    try:
        return path.relative_to(AUDIO_ROOT).with_suffix('')
    except ValueError:
        # Different external songs with the same filename must not share cache.
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return Path('external') / (path.stem + '-' + digest.hexdigest()[:16])


def cache_file(audio, directory=DEFAULT_CACHE):
    base = Path(directory) / audio_key(audio)
    result = base.with_name(base.name + '_clap_cache.npy')
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def output_file(audio, directory=DEFAULT_OUTPUT, test_output=False):
    base = Path(directory) / audio_key(audio)
    if test_output:
        base = base.with_name(base.name + '_test')
    base.parent.mkdir(parents=True, exist_ok=True)
    candidate = base.with_name(base.name + '.mp4')
    index = 1
    while candidate.exists() or candidate.with_name(candidate.stem + '_video.avi').exists() or candidate.with_name(candidate.stem + '_audio.wav').exists():
        candidate = base.with_name(base.name + f'_{index}.mp4')
        index += 1
    return candidate
