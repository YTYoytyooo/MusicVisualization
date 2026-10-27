"""Export validation contracts; real media encoding is a separate root-run check."""
import unittest
from unittest.mock import patch

from tools.train_next_v2.v3_studio import render, store
from support_v3 import StudioFixture


class ExportContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = StudioFixture()
        self.addCleanup(self.fixture.close)

    def test_invalid_export_range_rejected_before_any_encoder_process(self):
        f = self.fixture
        original = store.digest(f.project_dir / 'analysis/result.json')
        for start, end in ((-1., 1.), (0., 3.), (1., 1.), (True, 1.), (0., float('nan'))):
            with self.subTest(start=start, end=end), patch.object(render.subprocess, 'Popen') as spawn:
                with self.assertRaises(ValueError):
                    render.export(f.project_id, 0., start, end)
                spawn.assert_not_called()
        self.assertEqual(store.digest(f.project_dir / 'analysis/result.json'), original)
        self.assertFalse((f.project_dir / 'exports/latest.json').exists())

    def test_changed_source_audio_cannot_be_exported_against_stale_predictions(self):
        f = self.fixture
        f.audio.write_bytes(b'same path, different audio')
        with patch.object(render.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                render.export(f.project_id, 0., 0., 1.)
            spawn.assert_not_called()
        self.assertFalse((f.project_dir / 'exports/latest.json').exists())

    def test_failed_encoder_never_publishes_a_completed_export(self):
        f = self.fixture
        original = store.digest(f.project_dir / 'analysis/result.json')
        with patch.object(render.subprocess, 'Popen', side_effect=OSError('synthetic encoder launch failure')):
            with self.assertRaises(OSError):
                render.export(f.project_id, 0., 0., 1.)
        self.assertFalse((f.project_dir / 'exports/latest.json').exists())
        self.assertEqual(list((f.project_dir / 'exports').glob('e-*/export.json')), [])
        self.assertEqual(store.digest(f.project_dir / 'analysis/result.json'), original)


if __name__ == '__main__':
    unittest.main()
