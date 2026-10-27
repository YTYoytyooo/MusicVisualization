"""Regression cases for DEAM's uneven axes, time origin and split isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from tools.datasets.deam_dataset import DeamSequences, contained_path
from tools.datasets.prepare_deam import align_labels, audit_audio, audit_features, dynamic_csv, prepare, split_by_artist, static_csvs


class DeamPreparationTests(unittest.TestCase):
    def test_ragged_rows_keep_15_second_time_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'valence.csv'
            path.write_text('song_id,sample_15000ms,sample_15500ms,sample_16000ms\n2,0.2,,0.4\n3,-0.1\n')
            songs = dynamic_csv(path)
        np.testing.assert_array_equal(songs[3][0], [15])
        self.assertTrue(np.isnan(songs[2][1][1]))
        np.testing.assert_array_equal(songs[2][0], [15, 15.5, 16])

    def test_union_preserves_independent_missing_axes_and_duration_boundary(self):
        valence = (np.array([15, 15.5, 16]), np.array([0, .4, .8]))
        arousal = (np.array([15, 15.5]), np.array([np.nan, -.2]))
        times, targets, mask, outside = align_labels(valence, arousal, duration=15.5)
        np.testing.assert_array_equal(times, [15, 15.5, 16])
        np.testing.assert_array_equal(mask, [[True, False], [True, True], [False, False]])
        self.assertEqual(outside, 1)
        self.assertEqual(targets[0, 0], 0)  # A real zero remains supervised.
        np.testing.assert_array_equal(targets[~mask], 0)

    def test_invalid_dynamic_range_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.csv'
            path.write_text('song_id,sample_15000ms\n2,1.1\n')
            with self.assertRaisesRegex(ValueError, 'Out-of-range'):
                dynamic_csv(path)

    def test_feature_clock_drift_is_recorded_without_inventing_timestamps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'features.csv'
            header = ';'.join(['frameTime'] + [f'f{i}' for i in range(260)])
            path.write_text(header + '\n' + '\n'.join(str(t) + ';0' * 260 for t in [0, .501134, 1.002268]) + '\n')
            report = audit_features(path, duration=45)
        self.assertEqual(report['non_half_second_steps'], 2)
        self.assertEqual(report['last_seconds'], 1.002268)
        self.assertFalse(report['training_enabled'])

    def test_duration_uses_decoded_frames_when_mp3_header_overstates_length(self):
        stream = MagicMock()
        stream.__len__.return_value = 100
        stream.samplerate = 10
        stream.channels = 1
        stream.read.side_effect = [np.zeros((60, 1)), np.zeros((30, 1)), np.zeros((0, 1))]
        handle = MagicMock()
        handle.__enter__.return_value = stream
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'example.mp3'
            path.write_bytes(b'test fixture')
            with patch('tools.datasets.prepare_deam.sf.SoundFile', return_value=handle), patch('tools.datasets.prepare_deam.subprocess.run') as run:
                run.return_value.returncode = 0
                run.return_value.stderr = b''
                result = audit_audio(path, 'ffmpeg')
        self.assertEqual(result['decoded_frames'], 90)
        self.assertEqual(result['duration_seconds'], 9)
        self.assertEqual(result['reported_header_frames'], 100)

    def test_static_scale_is_separate_from_dynamic(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / 'static.csv').write_text('song_id, valence_mean, valence_std, arousal_mean, arousal_std\n2, 1, 0.3, 9, 0.5\n')
            result = static_csvs(Path(folder))
        self.assertEqual(result[2]['mean_minus1_1'], [-1, 1])
        self.assertEqual(result[2]['mean_1_9'], [1, 9])

    def test_artist_and_duplicate_audio_form_transitive_groups(self):
        records = [dict(song_id=i, artist=f'Artist {i}', audio_sha256=f'hash{i}') for i in range(30)]
        records[0]['artist'] = 'Same-Band'
        records[1]['artist'] = 'same band'
        records[2]['audio_sha256'] = records[1]['audio_sha256']
        other = [dict(r) for r in records]
        counts, _ = split_by_artist(records, 7)
        split_by_artist(other, 7)
        self.assertEqual(records, other)
        self.assertTrue(all(counts.values()))
        self.assertEqual(len({r['group_id'] for r in records[:3]}), 1)
        self.assertEqual(len({r['split'] for r in records[:3]}), 1)
        self.assertEqual(sum(counts.values()), 30)

    def test_preparation_refuses_overwrite_and_outside_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'dataset'
            existing = root / 'prepared/v1'
            existing.mkdir(parents=True)
            marker = existing / 'keep.txt'
            marker.write_text('keep')
            for target in (existing, Path(folder) / 'outside'):
                with self.assertRaisesRegex(ValueError, 'new output directory'):
                    prepare(root, target)
            self.assertEqual(marker.read_text(), 'keep')

    def test_loader_keeps_per_axis_mask_and_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared = root / 'prepared/v1'
            prepared.mkdir(parents=True)
            (prepared / 'config.json').write_text(json.dumps(dict(schema_version=1, label_order=['valence', 'arousal'], dataset_root='../../')))
            np.savez_compressed(prepared / '2.npz', times_seconds=np.array([15, 15.5]),
                                targets=np.array([[0, .2], [.5, 0]], dtype=np.float32),
                                mask=np.array([[True, True], [True, False]]))
            record = dict(song_id=2, split='train', labels_path='prepared/v1/2.npz', duration_seconds=45, sample_rate=44100)
            (prepared / 'manifest.jsonl').write_text(json.dumps(record)+'\n')
            dataset = DeamSequences(prepared)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(DeamSequences(prepared, 'test')), 0)
            item = dataset[0]
            self.assertEqual(item['targets'].shape, (2, 2))
            self.assertEqual(int(item['mask'].sum()), 3)
            self.assertNotIn('audio', item)

    def test_dataset_paths_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'escapes root'):
                contained_path(folder, '../outside.npz')


if __name__ == '__main__':
    unittest.main()
