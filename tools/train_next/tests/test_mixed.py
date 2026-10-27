import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.train_next.core import SequenceVA
from tools.train_next import mixed


class CachedFeatures:
    spec = {'test_cache': 1}

    def __init__(self, *args, **kwargs):
        pass

    def get(self, row, root):
        with np.load(Path(root) / row['feature_path']) as source:
            return source['times'], source['x'], True

    def unload(self):
        pass


class MixedTrainingTests(unittest.TestCase):
    def test_balanced_sampling_preserves_all_domains(self):
        songs = [{'source_dataset': 'deam'}] * 8 + [{'source_dataset': 'pmemo'}] * 2
        order = mixed.balanced_order(songs, 1, 12)
        self.assertEqual(sum(i < 8 for i in order), 8)
        self.assertEqual(sum(i >= 8 for i in order), 8)
        self.assertEqual(order, mixed.balanced_order(songs, 1, 12))

    def test_warm_start_train_save_reload_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / 'prepared'
            prepared.mkdir()
            (prepared / 'config.json').write_text(json.dumps(dict(schema_version=1, label_order=['valence', 'arousal'], dataset_root='..')))
            rng = np.random.default_rng(54)
            rows = []
            for name in ('deam', 'pmemo', 'memo2496'):
                for split in ('train', 'validation'):
                    key = f'{name}-{split}'
                    times = np.arange(12) * .5
                    x = rng.normal(size=(12, 531)).astype(np.float32)
                    y = np.stack([np.sin(times), np.cos(times)], axis=1).astype(np.float32) * .4
                    mask = np.ones_like(y, dtype=bool)
                    mask[:2, 0] = False
                    # An unobserved annotation beyond EOF must not become supervision.
                    np.savez(root / f'{key}.npz', times_seconds=np.append(times, 20),
                             targets=np.concatenate([y, [[0, 0]]]).astype(np.float32),
                             mask=np.concatenate([mask, [[False, False]]]))
                    np.savez(root / f'{key}-features.npz', times=times, x=x)
                    rows.append(dict(song_id=key, source_dataset=name, split=split, group_id=key,
                                     audio_sha256=key, labels_path=f'{key}.npz', feature_path=f'{key}-features.npz',
                                     duration_seconds=6, sample_rate=48000))
            (prepared / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            torch.manual_seed(50)
            model = SequenceVA(531, 128)
            initial = {k: v.clone() for k, v in model.state_dict().items()}
            initializer = root / 'initial.pth'
            torch.save(dict(state_dict=initial, mean=torch.zeros(531), std=torch.ones(531), feature_spec=CachedFeatures.spec), initializer)
            args = argparse.Namespace(device='cpu', run_id='fixture', resume=False, initialize=initializer, prepared=prepared, epochs=2)
            with patch.object(mixed, 'FeatureStore', CachedFeatures), patch.object(mixed, 'output_paths', return_value=(root / 'run', root / 'model')), redirect_stdout(io.StringIO()):
                mixed.main(args)
                final = torch.load(root / 'model/last.pth', weights_only=True)
                self.assertEqual(final['epoch'], 2)
                self.assertTrue(any(not torch.equal(initial[k], final['state_dict'][k]) for k in initial))
                self.assertEqual(set(final['validation_by_dataset']), {'deam', 'pmemo', 'memo2496'})
                (root / 'run/metrics.json').unlink()
                args.resume = True
                mixed.main(args)
                resumed = torch.load(root / 'model/last.pth', weights_only=True)
                self.assertTrue(all(torch.equal(v, resumed['state_dict'][k]) for k, v in final['state_dict'].items()))
                self.assertEqual(len(json.loads((root / 'run/metrics.json').read_text())), 2)
            report = json.loads((root / 'run/report.json').read_text())
            self.assertEqual(report['test_songs_used'], 0)
            self.assertEqual(report['counts']['train'], {'deam': 1, 'pmemo': 1, 'memo2496': 1})


if __name__ == '__main__':
    unittest.main()
