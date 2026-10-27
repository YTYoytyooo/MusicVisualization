import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tools.train_next.core import atomic_json, sha256
from tools.train_next_v2.common import BASE, ROOT, read_json, verify_hashes
from tools.train_next_v2.jamendo_audit_v2.native import decode_exact, inspect
from tools.train_next_v2.jamendo_audit_v2.run import reference_values, screen_recordings
from tools.train_next_v2.jamendo_audit.native import decode_exact as previous_decode


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=BASE/'tmp', prefix='jamendo-native-test-')
        self.root = Path(self.tmp.name)
        self.audio = self.root/'native.wav'
        values = np.sin(np.arange(16000)*0.18).astype('float32')*0.2
        sf.write(self.audio, values, 16000, subtype='FLOAT')

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_decode_preserves_all_samples_and_original_file(self):
        before = sha256(self.audio)
        result = inspect(self.audio, 1.0, self.root/'logs')
        independent, _ = sf.read(self.audio, dtype='float32', always_2d=True)
        self.assertEqual(result['frames'], 16000)
        self.assertEqual(result['float32_sample_sha256'], hashlib.sha256(independent.tobytes()).hexdigest())
        self.assertTrue(result['bulk_stream_sample_parity'])
        self.assertTrue(result['native_stderr_empty'])
        self.assertEqual(sha256(self.audio), before)

    def test_wrong_duration_is_rejected_without_stretching(self):
        with self.assertRaisesRegex(ValueError, 'duration differs'):
            decode_exact(self.audio, 1.251)
        self.assertEqual(sf.info(self.audio).frames, 16000)

    def test_silence_is_quarantined(self):
        silent = self.root/'silent.wav'
        sf.write(silent, np.zeros(16000, dtype='float32'), 16000)
        with self.assertRaisesRegex(ValueError, 'Silent'):
            decode_exact(silent, 1.0)

    def test_equal_length_but_different_stream_samples_rejected(self):
        values = np.full((16000, 1), .1, dtype=np.float32)
        class Stream:
            samplerate, channels, frames = 16000, 1, 16000
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def blocks(self, **kwargs): yield values*.5
        with patch('tools.train_next_v2.jamendo_audit_v2.native.sf.read', return_value=(values, 16000)), \
             patch('tools.train_next_v2.jamendo_audit_v2.native.sf.SoundFile', return_value=Stream()), \
             patch('tools.train_next_v2.jamendo_audit_v2.native.read_unseeked_block',
                   side_effect=[values*.5, np.empty((0,1),dtype=np.float32)]):
            with self.assertRaisesRegex(ValueError, 'native samples differ'):
                decode_exact(self.audio, 1.0)

    def test_corrupt_audio_fails_strict_decoder(self):
        path = self.root/'bad.mp3'
        path.write_bytes(b'This is not MPEG audio')
        with self.assertRaisesRegex(ValueError, 'ffmpeg decoder rejected'):
            inspect(path, 1.0, self.root/'bad-logs')
        self.assertTrue((self.root/'bad-logs/ffmpeg.log').stat().st_size)

    def test_reference_fingerprint_requires_recorded_hash(self):
        path = self.root/'fingerprint.json'
        atomic_json(path, {'fingerprint': list(range(100))})
        receipt = {'fingerprint': {'path':path.relative_to(ROOT).as_posix(), 'sha256':sha256(path), 'frames':100}}
        self.assertEqual(len(reference_values(receipt)), 100)
        path.write_text('{"fingerprint": [1]}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            reference_values(receipt)

    def test_protected_overlap_propagates_and_tags_are_not_va_labels(self):
        selected = [dict(song_id=sid, positive_tags=['calm'], training_admitted=False) for sid in ('a','b','c','d','e')]
        checks = [dict(song_id=sid, passed=True) for sid in ('a','b','c','d','e')]
        graph = {
            'reserved':dict(identity_keys=[('recording','one')]),
            'a':dict(identity_keys=[('recording','one'),('recording','two')]),
            'b':dict(identity_keys=[('recording','two')]),
            'c':dict(identity_keys=[('recording','three')]),
            'd':dict(identity_keys=[('recording','four')]),
            'e':dict(identity_keys=[('recording','four')]),
        }
        eligible, excluded = screen_recordings(selected, checks, graph)
        self.assertEqual([r['song_id'] for r in eligible], ['c','d'])
        self.assertEqual({r['song_id']:r['reason'] for r in excluded}, {
            'a':'recording_link_to_protected_or_rejected_audio',
            'b':'recording_link_to_protected_or_rejected_audio',
            'e':'duplicate_candidate_recording'})
        for row in eligible:
            self.assertFalse(row['training_admitted'])
            self.assertFalse(row['fitting_enabled'])
            self.assertEqual(row['positive_tags'], ['calm'])
            self.assertNotIn('valence', row)
            self.assertNotIn('arousal', row)

    def test_failed_native_member_blocks_linked_duplicate(self):
        selected = [dict(song_id=sid) for sid in ('a','b')]
        checks = [dict(song_id='a', passed=False), dict(song_id='b', passed=True)]
        graph = {sid:dict(identity_keys=[('recording','same')]) for sid in ('a','b')}
        eligible, excluded = screen_recordings(selected, checks, graph)
        self.assertEqual(eligible, [])
        self.assertEqual(len(excluded), 2)

    def test_three_real_mp3_regressions_fail_old_reader_and_pass_exact_new_reader(self):
        launch = read_json(BASE/'analyses/d-jamendo-native-smoke-v1/launch.json')
        verify_hashes(launch['protected_hashes'])
        for row in launch['tracks']:
            with self.subTest(track=row['track_id']):
                path = ROOT/row['audio_path']
                with self.assertRaisesRegex(ValueError, 'native samples differ'):
                    previous_decode(path, row['metadata_duration_seconds'])
                result = inspect(path, row['metadata_duration_seconds'], self.root/row['track_id'])
                independent, _ = sf.read(path, dtype='float32', always_2d=True)
                self.assertEqual(result['float32_sample_sha256'], hashlib.sha256(independent.tobytes()).hexdigest())
                self.assertEqual(result['frames'], len(independent))
                self.assertTrue(result['bulk_stream_sample_parity'])
                self.assertFalse(result['timeline_stretched'])
                self.assertEqual(sha256(path), row['audio_sha256'])


if __name__ == '__main__':
    unittest.main()
