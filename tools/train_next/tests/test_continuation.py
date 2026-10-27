"""Queue recovery and upstream failure must not dispatch the wrong training run."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.train_next import continue_training as queue
from tools.train_next import core


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prepared = self.root / 'data/datasets/combined/prepared/v1'
        self.prepared.mkdir(parents=True)
        (self.prepared / 'manifest.jsonl').write_text('fixture\n', encoding='utf-8')
        self.teacher = self.root / 'data/training/next_va/runs/teacher'
        self.teacher.mkdir(parents=True)
        self.queue_dir = self.root / 'data/training/next_va/queues/mixed'
        self.args = argparse.Namespace(teacher='teacher', run_id='mixed',
                                       prepared=self.prepared, resume=False)
        for module in (queue, core):
            patched = patch.object(module, 'ROOT', self.root)
            patched.start()
            self.addCleanup(patched.stop)

    def teacher_state(self, stage):
        (self.teacher / 'status.json').write_text(json.dumps({'stage': stage}), encoding='utf-8')

    def test_upstream_failure_does_not_dispatch_mixed_training(self):
        with patch.object(queue.subprocess, 'run') as dispatch, patch.object(queue.time, 'sleep') as sleep, patch.object(queue, 'is_running', return_value=False):
            for stage in ('failed', 'interrupted'):
                with self.subTest(stage=stage):
                    self.teacher_state(stage)
                    with self.assertRaisesRegex(RuntimeError, 'Teacher stopped'):
                        queue.main(self.args)
                    status = json.loads((self.queue_dir / 'status.json').read_text())
                    self.assertEqual(status['stage'], 'failed')
                    self.assertIn(stage, status['error'])
                    self.args.resume = True
            dispatch.assert_not_called()
            sleep.assert_not_called()

    def test_resume_after_interrupted_wait_starts_new_mixed_run_normally(self):
        self.teacher_state('training')
        with patch.object(queue, 'is_running', side_effect=[True, False]), patch.object(queue.subprocess, 'run') as dispatch, patch.object(
                queue.time, 'sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                queue.main(self.args)
            dispatch.assert_not_called()
            self.assertEqual(json.loads((self.queue_dir / 'status.json').read_text())['stage'], 'interrupted')
            mixed_run, mixed_models = core.output_paths('mixed')
            self.assertFalse(mixed_run.exists())
            self.assertFalse(mixed_models.exists())

            self.teacher_state('completed')
            checkpoint = self.root / 'models/next_va/teacher/best.pth'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'test checkpoint')
            # The stub publishes its report only when actually dispatched.
            def complete(*args, **kwargs):
                mixed_run.mkdir(parents=True)
                (mixed_run / 'report.json').write_text(json.dumps({'status': 'completed', 'legacy_model_unchanged': True}))
            dispatch.side_effect = complete
            (self.teacher / 'report.json').write_text(json.dumps({
                'status': 'completed', 'legacy_model_unchanged': True,
            }), encoding='utf-8')
            self.args.resume = True
            queue.main(self.args)
            dispatch.assert_called_once()
            command = dispatch.call_args.args[0]
            self.assertNotIn('--resume', command)
            self.assertEqual(command[command.index('--initialize') + 1],
                             str(self.root / 'models/next_va/teacher/best.pth'))
            self.assertEqual(command[command.index('--prepared') + 1], str(self.prepared))
            self.assertEqual(dispatch.call_args.kwargs['cwd'], self.root)
            self.assertTrue(dispatch.call_args.kwargs['check'])
            saved_command = json.loads((self.queue_dir / 'command.json').read_text())
            self.assertEqual(saved_command['argv'], command)
            self.assertEqual(json.loads((self.queue_dir / 'status.json').read_text())['stage'], 'completed')

    def test_stale_active_status_without_worker_is_failure(self):
        self.teacher_state('extracting_features')
        with patch.object(queue, 'is_running', return_value=False), patch.object(queue.subprocess, 'run') as dispatch:
            with self.assertRaisesRegex(RuntimeError, 'no active run lock'):
                queue.main(self.args)
            dispatch.assert_not_called()
            self.assertEqual(json.loads((self.queue_dir / 'status.json').read_text())['stage'], 'failed')


if __name__ == '__main__':
    unittest.main()
