"""PMEmo preparation contracts: explicit clocks, masks and selective extraction."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
import soundfile as sf

from tools.datasets.deam_dataset import DeamSequences
from tools.datasets.prepare_pmemo import (
    ANNOTATIONS, align_dynamic, audit_audio, dynamic_csv, extract_selected,
    prepare, static_csv,
)


class PmemoPreparationTests(unittest.TestCase):
    def test_dynamic_clock_and_axis_order_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.csv'
            path.write_text('musicId,frameTime,Arousal(mean),Valence(mean)\n1,15.5,0,1\n1,16,,0.5\n')
            labels = dynamic_csv(path)
        times, targets, mask, std, std_mask, outside = align_dynamic(labels[1], None, 15.5)
        np.testing.assert_array_equal(times, [15.5, 16])
        np.testing.assert_array_equal(targets, [[1, -1], [0, 0]])
        np.testing.assert_array_equal(mask, [[True, True], [False, False]])
        self.assertEqual(outside, 1)
        self.assertFalse(std_mask.any())
        self.assertFalse(std.any())

    def test_missing_axes_and_standard_deviations_stay_independent(self):
        means = (np.array([15.5, 16]), np.array([[.5, np.nan], [np.nan, .75]]))
        deviations = (np.array([15.5, 16]), np.array([[.2, np.nan], [.1, .3]]))
        _, targets, mask, std, std_mask, _ = align_dynamic(means, deviations, 20)
        np.testing.assert_array_equal(mask, [[True, False], [False, True]])
        np.testing.assert_allclose(targets, [[0, 0], [0, .5]])
        np.testing.assert_allclose(std, [[.4, 0], [.2, .6]])
        np.testing.assert_array_equal(std_mask, [[True, False], [True, True]])

    def test_dynamic_invalid_ranges_and_times_are_rejected(self):
        invalid = ['1,15.5,1.01,.5', '1,15.5,inf,.5', '1,nan,.5,.5',
                   '1,-1,.5,.5', '1,16,.5,.5\n1,15.5,.5,.5',
                   '1,15.5,.5,.5\n1,15.5,.5,.5']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.csv'
            for rows in invalid:
                with self.subTest(rows=rows):
                    path.write_text('musicId,frameTime,Arousal(mean),Valence(mean)\n' + rows + '\n')
                    with self.assertRaises(ValueError):
                        dynamic_csv(path)

    def test_static_values_are_read_without_dynamic_tiling(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'static.csv'
            path.write_text('musicId,Arousal(mean),Valence(mean)\n1,1,0\n')
            labels = static_csv(path)
        np.testing.assert_array_equal(labels[1], [0, 1])
        times, targets, mask, *_ = align_dynamic(None, None, 30)
        self.assertEqual(times.shape, (0,))
        self.assertEqual(targets.shape, (0, 2))
        self.assertFalse(mask.any())

    def test_extraction_keeps_only_allowlisted_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'source.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for name in ANNOTATIONS:
                    z.writestr('PMEmo2019/annotations/' + name, 'test')
                z.writestr('PMEmo2019/metadata.csv', 'test')
                z.writestr('PMEmo2019/chorus/1.mp3', b'audio')
                z.writestr('PMEmo2019/lyrics/1.txt', 'private lyrics')
                z.writestr('PMEmo2019/EDA/1.csv', 'physiology')
                z.writestr('__MACOSX/PMEmo2019/._metadata.csv', b'ignored')
            extract_selected(archive, root / 'source')
            names = {p.relative_to(root / 'source').as_posix() for p in (root / 'source').rglob('*') if p.is_file()}
            self.assertEqual(names, {'metadata.csv', 'chorus/1.mp3'} | {'annotations/' + n for n in ANNOTATIONS})
            self.assertTrue(archive.is_file())

    def test_zip_traversal_and_symlinks_fail_before_extraction(self):
        cases = ['PMEmo2019/chorus/../../escape.mp3', '../escape', 'PMEmo2019\\chorus\\1.mp3']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i, name in enumerate(cases):
                archive = root / f'{i}.zip'
                with zipfile.ZipFile(archive, 'w') as z:
                    z.writestr(name, b'bad')
                with self.assertRaises(ValueError):
                    extract_selected(archive, root / f'extract{i}')
            archive = root / 'link.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                info = zipfile.ZipInfo('PMEmo2019/chorus/1.mp3')
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                z.writestr(info, '../outside')
            with self.assertRaises(ValueError):
                extract_selected(archive, root / 'link')
            self.assertFalse((root / 'escape').exists())

    def test_audio_audit_counts_real_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audio.wav'
            sf.write(path, np.zeros((1003, 2)), 1000, subtype='FLOAT')
            info = audit_audio(path)
        self.assertEqual(info['decoded_frames'], 1003)
        self.assertEqual(info['duration_seconds'], 1.003)
        self.assertEqual(info['channels'], 2)

    def test_short_decode_ignores_unused_block_tail(self):
        stream = MagicMock()
        stream.__len__.return_value = 100
        stream.samplerate = 10
        stream.channels = 1
        stream.tell.side_effect = [0, 60, 90, 90]
        stream.blocks.return_value = iter([np.zeros((60, 1)),
                                          np.concatenate([np.zeros((30, 1)), np.full((30, 1), np.nan)]),
                                          np.full((40, 1), np.nan)])
        handle = MagicMock()
        handle.__enter__.return_value = stream
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audio.mp3'
            path.write_bytes(b'test fixture')
            with patch('tools.datasets.prepare_pmemo.sf.SoundFile', return_value=handle):
                info = audit_audio(path)
        self.assertEqual(info['decoded_frames'], 90)
        self.assertEqual(info['duration_seconds'], 9)
        self.assertEqual(info['reported_header_frames'], 100)

    def test_nonfinite_decoded_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audio.wav'
            sf.write(path, np.array([0, np.nan]), 1000, subtype='FLOAT')
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                audit_audio(path)

    def test_prepare_publishes_loadable_candidates_and_static_only_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'raw').mkdir()
            archive = root / 'raw/PMEmo2019.zip'
            audio = io.BytesIO()
            sf.write(audio, np.zeros(17000), 1000, format='WAV')
            with zipfile.ZipFile(archive, 'w') as z:
                for sid in (1, 2):
                    z.writestr(f'PMEmo2019/chorus/{sid}.mp3', audio.getvalue())
                z.writestr('PMEmo2019/metadata.csv', 'musicId,fileName,title,artist,album,duration,chorus_start_time,chorus_end_time\n1,1.mp3,One,Same-Band,Album,17,02:00,02:17\n2,2.mp3,Two,same band,Album,17,03:00,03:17\n')
                z.writestr('PMEmo2019/annotations/dynamic_annotations.csv', 'musicId,frameTime,Arousal(mean),Valence(mean)\n1,15.5,.25,.75\n1,18,.5,.5\n')
                z.writestr('PMEmo2019/annotations/dynamic_annotations_std.csv', 'musicId,frameTime,Arousal(std),Valence(std)\n1,15.5,.1,.2\n')
                for kind in ('mean', 'std'):
                    name = 'static_annotations.csv' if kind == 'mean' else 'static_annotations_std.csv'
                    z.writestr('PMEmo2019/annotations/' + name, f'musicId,Arousal({kind}),Valence({kind})\n1,.5,.5\n2,.25,.75\n')
            output = root / 'prepared/v1'
            prepare(root, output, expected_sha256=None, expected_bytes=None, workers=1)
            dataset = DeamSequences(output, split=None)
            self.assertEqual(len(dataset), 2)
            first, second = dataset[0], dataset[1]
            np.testing.assert_array_equal(first['times_seconds'], [15.5, 18])
            self.assertEqual(int(first['mask'].sum()), 2)
            self.assertEqual(first['metadata']['split'], 'candidate')
            self.assertFalse(first['metadata']['training_enabled'])
            self.assertEqual(first['metadata']['group_id'], second['metadata']['group_id'])
            self.assertFalse(second['metadata']['eligible_dynamic'])
            self.assertEqual(second['targets'].shape, (0, 2))
            static = [json.loads(line) for line in (output / 'static.jsonl').read_text().splitlines()]
            self.assertEqual(static[1]['mean_minus1_1'], [.5, -.5])
            with self.assertRaisesRegex(ValueError, 'new output directory'):
                prepare(root, output)
            with self.assertRaisesRegex(ValueError, 'new output directory'):
                prepare(root, root / 'outside')
            (root / 'prepared/v2.pending').mkdir()
            with self.assertRaises(FileExistsError):
                prepare(root, root / 'prepared/v2', expected_sha256=None, expected_bytes=None)


if __name__ == '__main__':
    unittest.main()
