"""Memo2496 source clocks, independent masks, archive safety and loader contract."""
import csv
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
import soundfile as sf

from tools.datasets.deam_dataset import DeamSequences
from tools.datasets.prepare_memo2496 import (
    WORKSPACE, align_labels, audit_audio, dynamic_csv, index_audio,
    prepare, verify_raw_ratings,
)


class MemoPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='memo2496-test-', dir=WORKSPACE)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def csv(self, text):
        path = self.root / 'labels.csv'
        path.write_text(text, encoding='utf-8')
        return path

    def test_native_clock_and_ragged_missing_values_are_preserved(self):
        table = dynamic_csv(self.csv('song_id,sample_0ms,sample_1000ms,sample_2000ms\n1,0,,nan\n2,.5\n'))
        np.testing.assert_array_equal(table[1][0], [0, 1, 2])
        np.testing.assert_array_equal(table[2][0], [0, 1, 2])
        self.assertEqual(table[1][1][0], 0)
        self.assertTrue(np.all(np.isnan(table[2][1][1:])))

    def test_wrong_or_reordered_clocks_are_rejected(self):
        for header in ('sample_0ms,sample_500ms', 'sample_1000ms,sample_2000ms',
                       'sample_0ms,sample_2000ms,sample_1000ms', 'sample_0ms,sample_0ms'):
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, '1 Hz'):
                dynamic_csv(self.csv('song_id,' + header + '\n1,0\n'))

    def test_wrong_scale_and_infinite_values_are_rejected(self):
        for value in ('5', '-1.0001', 'inf', '-inf'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Out-of-range'):
                dynamic_csv(self.csv('song_id,sample_0ms\n1,' + value + '\n'))
        with self.assertRaisesRegex(ValueError, 'Out-of-range'):
            dynamic_csv(self.csv('song_id,sample_0ms\n1,-.1\n'), standard_deviation=True)

    def test_axis_masks_and_std_masks_remain_independent_at_audio_boundary(self):
        v = (np.array([0., 1., 2.]), np.array([0., np.nan, .7]))
        a = (np.array([0., 1.]), np.array([np.nan, -.2]))
        vs = (np.array([0., 1., 2.]), np.array([.1, .2, .3]))
        ars = (np.array([0., 1.]), np.array([np.nan, np.nan]))
        result = align_labels(v, a, vs, ars, duration=1.)
        times, targets, mask, std, std_mask, outside = result
        np.testing.assert_array_equal(times, [0, 1, 2])
        np.testing.assert_array_equal(mask, [[True, False], [False, True], [False, False]])
        np.testing.assert_array_equal(std_mask, [[True, False], [False, False], [False, False]])
        np.testing.assert_array_equal(targets[~mask], 0)
        np.testing.assert_array_equal(std[~std_mask], 0)
        self.assertEqual(outside, [1, 0])

    def archive(self, names):
        path = self.root / 'audio.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            for name in names:
                archive.writestr(name, b'fixture')
        return path

    def test_archive_suffix_lookup_uses_exact_metadata_basename(self):
        path = self.archive(['MusicRawData/nested/a.mp3', 'MusicRawData/not-a.mp3', 'run.exe'])
        with zipfile.ZipFile(path) as archive:
            result = index_audio(archive, {1: {'file_name': 'a.mp3'}})
        self.assertEqual(result[1].filename, 'MusicRawData/nested/a.mp3')

    def test_ambiguous_duplicate_audio_basenames_are_rejected(self):
        path = self.archive(['one/a.mp3', 'two/a.mp3'])
        with zipfile.ZipFile(path) as archive, self.assertRaisesRegex(ValueError, 'Ambiguous'):
            index_audio(archive, {1: {'file_name': 'a.mp3'}})

    def test_zip_path_traversal_and_windows_paths_are_rejected(self):
        for name in ('../a.mp3', '/a.mp3', 'C:/a.mp3', 'folder/../a.mp3',
                     'folder\\a.mp3', 'folder/a.mp3:payload'):
            with self.subTest(name=name):
                path = self.archive([name])
                if '\\' in name:
                    # ZipFile normalizes Windows separators when writing; put
                    # the malicious path back into both raw ZIP headers.
                    path.write_bytes(path.read_bytes().replace(name.replace('\\', '/').encode(), name.encode()))
                with zipfile.ZipFile(path) as archive, self.assertRaisesRegex(ValueError, 'Unsafe'):
                    index_audio(archive, {1: {'file_name': 'a.mp3'}})

    def test_archive_symlink_and_case_colliding_names_are_rejected(self):
        info = zipfile.ZipInfo('a.mp3')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        path = self.archive([info])
        with zipfile.ZipFile(path) as archive, self.assertRaisesRegex(ValueError, 'Unsafe'):
            index_audio(archive, {1: {'file_name': 'a.mp3'}})
        path = self.archive(['one/a.mp3', 'ONE/A.MP3'])
        with zipfile.ZipFile(path) as archive, self.assertRaisesRegex(ValueError, 'Duplicate'):
            index_audio(archive, {1: {'file_name': 'a.mp3'}})

    def test_decoder_uses_actual_cursor_progress_and_rejects_nonfinite_audio(self):
        path = self.root / 'a.mp3'
        path.write_bytes(b'fixture')
        stream = MagicMock()
        stream.__len__.return_value = 100
        stream.samplerate = 10
        stream.channels = 1
        stream.read.side_effect = [np.zeros((100, 1)), np.empty((0, 1))]
        stream.tell.side_effect = [0, 90, 90, 90]
        handle = MagicMock()
        handle.__enter__.return_value = stream
        with patch('tools.datasets.prepare_memo2496.sf.SoundFile', return_value=handle):
            result = audit_audio(path)
        self.assertEqual(result['decoded_frames'], 90)
        self.assertEqual(result['duration_seconds'], 9.)
        self.assertEqual(result['reported_header_frames'], 100)
        with path.open('wb') as output:
            sf.write(output, np.array([0., np.nan]), 10, format='WAV', subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            audit_audio(path)

    def fixture(self):
        raw = self.root / 'raw'
        raw.mkdir()
        header = ['song_id', 'sample_0ms', 'sample_1000ms', 'sample_2000ms']
        for axis, values in [('valence', [0., .4, .6]), ('arousal', [.2, -.2, .8])]:
            for kind in ('average', 'std'):
                with (raw / f'{axis}_all_{kind}.csv').open('w', newline='') as output:
                    writer = csv.writer(output)
                    writer.writerow(header)
                    writer.writerow([1] + (values if kind == 'average' else [0., 0., 0.]))
        with (raw / 'songs_info_all.csv').open('w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['song_id', 'file_name', 'Song title', 'duration (min.sec)', 'emotion', 'distance', 'platform'])
            writer.writerow([1, 'a.mp3', 'Fixture', '99.59', 'Happy', '.4', 'fixture'])
        (raw / 'Label_pseudonymous.json').write_text(json.dumps([
            {'MID': 1, 'V': [0., .4, .6], 'A': [.2, -.2, .8]},
        ]))
        audio = io.BytesIO()
        sf.write(audio, np.zeros(150), 100, format='WAV')
        with zipfile.ZipFile(raw / 'MusicRawData.zip', 'w') as archive:
            archive.writestr('nested/a.mp3', audio.getvalue())
            archive.writestr('do-not-extract.exe', b'not audio')
        return raw

    def test_raw_ratings_detect_a_plausible_but_wrong_supplied_mean(self):
        raw = self.fixture()
        means = {axis: dynamic_csv(raw / f'{axis}_all_average.csv') for axis in ('valence', 'arousal')}
        stds = {axis: dynamic_csv(raw / f'{axis}_all_std.csv', True) for axis in means}
        result = verify_raw_ratings(raw / 'Label_pseudonymous.json', means, stds)
        self.assertEqual(result['mean_values_compared'], 6)
        means['valence'][1][1][0] = .1
        with self.assertRaisesRegex(ValueError, 'Raw-rating'):
            verify_raw_ratings(raw / 'Label_pseudonymous.json', means, stds)

    def test_prepare_is_loader_compatible_and_does_not_trust_metadata_duration(self):
        raw = self.fixture()
        archive_before = (raw / 'MusicRawData.zip').read_bytes()
        output = self.root / 'prepared/v1'
        report = prepare(self.root, output, expected_songs=1, workers=1)
        self.assertFalse(report['training_enabled'])
        self.assertEqual((raw / 'MusicRawData.zip').read_bytes(), archive_before)
        dataset = DeamSequences(output, split=None, include_audio=True)
        item = dataset[0]
        self.assertEqual(item['song_id'], 'memo2496:1')
        self.assertEqual(item['metadata']['duration_seconds'], 1.5)
        self.assertEqual(item['metadata']['split'], 'candidate')
        self.assertFalse(item['metadata']['training_enabled'])
        self.assertTrue(item['metadata']['eligible_dynamic'])
        np.testing.assert_array_equal(item['times_seconds'], [0, 1, 2])
        np.testing.assert_array_equal(item['mask'], [[True, True], [True, True], [False, False]])
        self.assertEqual(item['audio'].shape, (150, 1))
        self.assertFalse((output / 'audio/do-not-extract.exe').exists())
        with self.assertRaisesRegex(ValueError, 'new output directory'):
            prepare(self.root, output, expected_songs=1)

    def test_invalid_input_leaves_failure_report_without_publishing(self):
        raw = self.fixture()
        (raw / 'valence_all_average.csv').write_text('song_id,sample_0ms\n1,5\n')
        output = self.root / 'prepared/v1'
        with self.assertRaisesRegex(ValueError, 'Out-of-range'):
            prepare(self.root, output, expected_songs=1)
        self.assertFalse(output.exists())
        reports = list((self.root / 'prepared').glob('v1.pending-*/report.json'))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
