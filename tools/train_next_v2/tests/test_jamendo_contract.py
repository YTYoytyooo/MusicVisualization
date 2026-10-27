"""Admission, weak-label and whole-track feature boundary checks."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.train_next_v2.common import ROOT, BASE, read_json
from tools.train_next.core import atomic_json, sha256
from tools.train_next_v2.jamendo_data.contract import (
    TAG_ORDER, observed_tag_distribution, validate_candidates, feature_record,
    validate_feature_arrays,
)
from tools.train_next_v2.jamendo_data.features import read_cached


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_json(ROOT/'data/datasets/mtg_jamendo/prepared/native-audit-v2/eligible-category-candidates.json')

    def test_audited_real_pool_has_255_songs_and_161_groups(self):
        result = validate_candidates(self.rows)
        self.assertEqual(result['songs'], 255)
        self.assertEqual(result['groups'], 161)
        self.assertEqual(result['tag_counts']['mood/theme---energetic'], 54)

    def test_observed_tag_distribution_is_one_whole_track_vector(self):
        tags = ['mood/theme---calm', 'mood/theme---happy']
        values = observed_tag_distribution(tags)
        self.assertEqual(values.shape, (12,))
        self.assertEqual(float(values.sum()), 1.)
        self.assertEqual(float(values[TAG_ORDER.index(tags[0])]), .5)
        self.assertEqual(float(values[TAG_ORDER.index(tags[1])]), .5)
        self.assertEqual(int(np.count_nonzero(values)), 2)
        for invalid in ([], [tags[0], tags[0]], ['arousal'], ['mood/theme---unknown']):
            with self.subTest(tags=invalid), self.assertRaises(ValueError):
                observed_tag_distribution(invalid)

    def test_heldout_overlap_and_invented_va_are_rejected(self):
        for field, value in [('official_role', 'test'), ('heldout_identity_ids', ['protected']),
                             ('possible_overlap_ids', ['unknown']), ('timepoint_labels', [[0., .2]])]:
            rows = copy.deepcopy(self.rows[:1]); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_candidates(rows)

    def test_native_clock_identity_and_decoder_failures_are_rejected(self):
        for field, value in [('bulk_stream_sample_parity', False), ('source_song_offset_seconds', 2),
                             ('timeline_stretched', True), ('duration_seconds', 1.),
                             ('native_stderr_empty', False)]:
            rows = copy.deepcopy(self.rows[:1]); rows[0]['audio']['native_audio'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_candidates(rows)
        rows = copy.deepcopy(self.rows[:1]); rows[0]['audio']['audio_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            validate_candidates(rows)

    def test_duplicate_recording_and_unresolved_recording_groups_rejected(self):
        with self.assertRaises(ValueError):
            validate_candidates([self.rows[0], self.rows[0]])
        rows = copy.deepcopy(self.rows[:1]); rows[0]['recording_group'].append('protected:other')
        with self.assertRaises(ValueError):
            validate_candidates(rows)

    def test_feature_record_uses_native_time_without_supervision(self):
        row = self.rows[0]; record = feature_record(row)
        self.assertAlmostEqual(record['duration_seconds'], 3901353/44100)
        self.assertEqual(record['decoded_frames'], 3901353)
        self.assertFalse(set(record) & {'y', 'mask', 'targets', 'tags'})

    def test_feature_arrays_reject_lost_clock_nonfinite_and_wrong_width(self):
        record = dict(duration_seconds=1.25)
        times = np.array([0., .5, 1.], dtype=np.float64)
        values = np.zeros((3, 531), dtype=np.float32)
        validate_feature_arrays(times, values, record)
        for t, x in [(times+.1, values), (times, values[:, :-1]),
                     (times, np.full((3, 531), np.nan, dtype=np.float32))]:
            with self.assertRaises(ValueError):
                validate_feature_arrays(t, x, record)


class FeatureReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_json(ROOT/'data/datasets/mtg_jamendo/prepared/native-audit-v2/eligible-category-candidates.json')

    def setUp(self):
        (BASE/'tmp').mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='jamendo-cache-test-', dir=BASE/'tmp')
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name)
        self.row = self.rows[0]
        self.digest = self.row['expected_audio_sha256']
        self.npz = self.cache/(self.digest+'.npz')
        self.receipt = self.npz.with_suffix('.json')
        self.key = 'test-isolated-feature-key'
        self.times = np.arange(177, dtype=np.float64)*.5
        self.values = np.zeros((177, 531), dtype=np.float32)

    def create_pair(self, extra=None):
        np.savez_compressed(self.npz, times=self.times, features=self.values, feature_key=self.key, **(extra or {}))
        self.entry = dict(song_id=self.row['song_id'],audio_sha256=self.digest,feature_key=self.key,
            path=self.npz.relative_to(ROOT).as_posix(),sha256=sha256(self.npz),frames=177,dimensions=531,
            native_duration_seconds=self.row['audio']['native_audio']['duration_seconds'],
            native_float32_sha256=self.row['audio']['native_audio']['float32_sample_sha256'])
        atomic_json(self.receipt,self.entry)

    def test_valid_pair_reuses_without_mutation(self):
        self.create_pair(); before=[sha256(self.npz),sha256(self.receipt)]
        entry=read_cached(self.row,self.cache,self.key)
        self.assertEqual(entry['frames'],177)
        self.assertEqual(before,[sha256(self.npz),sha256(self.receipt)])

    def test_missing_cache_and_orphaned_shards_are_distinct(self):
        self.assertIsNone(read_cached(self.row,self.cache,self.key))
        np.savez_compressed(self.npz,times=self.times,features=self.values,feature_key=self.key)
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            read_cached(self.row,self.cache,self.key)

    def test_modified_shard_or_identity_stops_recovery(self):
        self.create_pair()
        with self.npz.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'hash'):
            read_cached(self.row,self.cache,self.key)
        self.create_pair()
        self.entry['song_id']='wrong-song'
        atomic_json(self.receipt,self.entry)
        with self.assertRaisesRegex(ValueError,'identity'):
            read_cached(self.row,self.cache,self.key)

    def test_timepoint_labels_cannot_enter_feature_file(self):
        self.create_pair(extra={'y':np.zeros((177,2),dtype=np.float32)})
        with self.assertRaisesRegex(ValueError,'fields/targets'):
            read_cached(self.row,self.cache,self.key)


if __name__ == '__main__':
    unittest.main()
