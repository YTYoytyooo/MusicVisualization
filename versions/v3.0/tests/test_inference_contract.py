"""Pure frozen-output invariants; no model loading, feature extraction or GPU use."""
import copy
import unittest

from tools.train_next_v2.v3_inference.pipeline import validate_output_system, verify_result
from tools.train_next_v2.v3_studio import store
from support_v3 import StudioFixture


def known_three_member_system():
    # This small fixture has analytically known mean 0 and sample std 0.5.
    return {'times_seconds': [0., .5, 1., 1.5, 2.], 'raw_va': [[0., 0.]] * 5,
            'member_raw_va': [[[-.5, -.5]] * 5, [[0., 0.]] * 5, [[.5, .5]] * 5],
            'seed_disagreement_std': [[.5, .5]] * 5,
            'seed_disagreement_is_confidence': False}


class NativeOutputContractTests(unittest.TestCase):
    def test_exact_whole_song_clock_and_known_three_member_aggregation(self):
        validate_output_system(known_three_member_system(), 2.1)
        validate_output_system(known_three_member_system(), 2.)

    def test_shift_stretch_truncate_duplicate_or_nonfinite_sequence_clock_fails(self):
        clocks = ([.1, .6, 1.1, 1.6, 2.1], [0., .4, .8, 1.2, 1.6], [0., .5, 1., 1.5],
                  [0., .5, 1., 1., 2.], [0., .5, 1., 1.5, float('nan')])
        for times in clocks:
            system = known_three_member_system(); system['times_seconds'] = times
            with self.subTest(times=times), self.assertRaises(ValueError):
                validate_output_system(system, 2.1)

    def test_nonfinite_duration_or_out_of_duration_endpoint_fails(self):
        for duration in (-1., 0., float('nan'), float('inf'), 1.9):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                validate_output_system(known_three_member_system(), duration)

    def test_wrong_mean_disagreement_member_count_or_range_fails(self):
        changes = [lambda s: s['raw_va'].__setitem__(0, [.01, 0.]),
                   lambda s: s['seed_disagreement_std'].__setitem__(0, [.1, .5]),
                   lambda s: s['member_raw_va'].pop(),
                   lambda s: s['member_raw_va'][0].__setitem__(0, [-1.01, -.5]),
                   lambda s: s['raw_va'].__setitem__(0, [float('nan'), 0.])]
        for change in changes:
            system = known_three_member_system(); change(system)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_output_system(system, 2.1)

    def test_legacy_keeps_its_own_native_point_one_second_clock(self):
        legacy = {'times_seconds': [0., .1, .2, .3], 'raw_va': [[0., 0.]] * 4}
        validate_output_system(legacy, .4, legacy=True)
        legacy['times_seconds'] = [0., .5, 1., 1.5]
        with self.assertRaises(ValueError):
            validate_output_system(legacy, 2., legacy=True)
        truncated = {'times_seconds': [0.], 'raw_va': [[0., 0.]]}
        with self.assertRaises(ValueError):
            validate_output_system(truncated, .4, legacy=True)


class CommittedResultContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = StudioFixture()
        self.addCleanup(self.fixture.close)
        self.fixture.result['systems'] = {'b01': known_three_member_system(), 'b00': known_three_member_system()}
        self.fixture.write_result()

    def verify(self):
        f = self.fixture
        return verify_result(f.project_dir, f.manifest, f.row['sha256'], False)

    def test_valid_committed_output_verifies_read_only(self):
        f = self.fixture
        before = (f.project_dir / 'analysis/result.json').read_bytes()
        self.assertEqual(self.verify(), f.result)
        self.assertEqual((f.project_dir / 'analysis/result.json').read_bytes(), before)

    def test_audio_identity_and_v2_option_cannot_be_reused_for_other_request(self):
        f = self.fixture
        with self.assertRaises(ValueError): verify_result(f.project_dir, f.manifest, 'f' * 64, False)
        with self.assertRaises(ValueError): verify_result(f.project_dir, f.manifest, f.row['sha256'], True)

    def test_project_metadata_tamper_is_rejected_even_when_raw_is_unchanged(self):
        f = self.fixture
        project = copy.deepcopy(f.project); project['audio']['duration_seconds'] = 20.
        store.atomic(f.project_dir / 'project.json', project)
        with self.assertRaises(ValueError): self.verify()

    def test_changed_manifest_cannot_reuse_committed_raw(self):
        self.fixture.manifest.write_text('{"changed":true}', encoding='utf-8')
        with self.assertRaises(ValueError): self.verify()

    def test_rehashed_semantically_wrong_acoustic_clock_or_missing_system_fails(self):
        f = self.fixture
        original = copy.deepcopy(f.result)
        for change in (lambda r: r['acoustic'].__setitem__('times_seconds', [0., .6, 1., 1.5, 2.]),
                       lambda r: r['systems'].pop('b00')):
            f.result = copy.deepcopy(original); change(f.result); f.write_result()
            with self.subTest(change=change), self.assertRaises(ValueError): self.verify()

    def test_unavailable_outputs_cannot_contain_fabricated_values_or_confidence_claims(self):
        f = self.fixture
        original = copy.deepcopy(f.result)
        changes = (lambda r: r['calibration'].__setitem__('values', [[0., 0.]] * 5),
                   lambda r: r['calibration'].__setitem__('intervals', [[[-1., 1.], [-1., 1.]]] * 5),
                   lambda r: r['tension'].__setitem__('values', [0.] * 5),
                   lambda r: r['systems']['b01'].__setitem__('seed_disagreement_is_confidence', True),
                   lambda r: r['acoustic'].__setitem__('rms_amplitude', [-.1] * 5),
                   lambda r: r['acoustic'].__setitem__('spectral_centroid_hz', [-1.] * 5),
                   lambda r: r['acoustic'].__setitem__('rms_dbfs', [-12.] * 5))
        for change in changes:
            f.result = copy.deepcopy(original); change(f.result); f.write_result()
            with self.subTest(change=change), self.assertRaises(ValueError): self.verify()


if __name__ == '__main__':
    unittest.main()
