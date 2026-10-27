"""Isolated synthetic engineering fixtures; never model-quality/training data."""
from __future__ import annotations
from contextlib import ExitStack
from pathlib import Path
import tempfile
import wave
from unittest.mock import patch

from tools.train_next_v2.v3_studio import store


class StudioFixture:
    def __init__(self):
        scratch = store.ROOT / 'data/v3.0/test-scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='contract-', dir=scratch)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(store, 'DATA', self.root))
        self.manifest = self.root / 'fixture-manifest.json'
        store.atomic(self.manifest, {'schema_version': 3, 'id': 'synthetic-contract-only'})
        self.stack.enter_context(patch.object(store, 'MANIFEST', self.manifest))
        self.audio = self.root / 'fixture.wav'
        with wave.open(str(self.audio), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(8000)
            f.writeframes(b'\x00\x00' * 16000)
        self.row = {'id': 'fixture-audio', 'name': 'synthetic fixture.wav',
                    'path': self.audio.relative_to(store.ROOT).as_posix(),
                    'sha256': store.digest(self.audio), 'duration_seconds': 2.0,
                    'sample_rate': 8000, 'native_frames': 16000}
        store.atomic(self.root / 'library.json', {'songs': [self.row]})
        self.project_id = 'p-contract-fixture'
        self.project_dir = store.project_path(self.project_id)
        self.manifest_sha = store.digest(self.manifest)
        self.project = {'schema_version': 3, 'id': self.project_id, 'project_id': self.project_id,
                        'audio': dict(self.row), 'manifest_sha256': self.manifest_sha}
        self.result = {'schema_version': 3, 'project_id': self.project_id,
                       'audio': dict(self.row), 'manifest_sha256': self.manifest_sha,
                       'systems': {'b01': {'times_seconds': [0., .5, 1., 1.5, 2.],
                                           'raw_va': [[-.5, 0.], [0., .5], [.5, 1.], [0., .5], [-.5, 0.]],
                                           'seed_disagreement_std': [[.1, .2]] * 5}},
                       'acoustic': {'times_seconds': [0., .5, 1., 1.5, 2.],
                                    'rms_dbfs': [-160.] * 5,
                                    'rms_amplitude': [0.] * 5,
                                    'spectral_centroid_hz': [0.] * 5},
                       'calibration': {'status': 'unavailable', 'intervals': None},
                       'tension': {'status': 'unavailable', 'values': None}}
        store.atomic(self.project_dir / 'project.json', self.project)
        self.write_result()

    def write_result(self):
        path = self.project_dir / 'analysis/result.json'
        store.atomic(path, self.result)
        store.atomic(path.with_name('receipt.json'), {
            'schema_version': 3, 'status': 'completed', 'project_id': self.project_id,
            'audio_sha256': self.row['sha256'], 'manifest_sha256': self.manifest_sha,
            'project_sha256': store.digest(self.project_dir / 'project.json'),
            'result_sha256': store.digest(path),
        })

    def close(self):
        self.stack.close()
        self.temp.cleanup()
