from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from tools.train_next.core import sha256
from tools.train_next_v2.common import BASE
from tools.train_next_v2.annotation_pilot.audit import artist_tokens,local_native,screen


def row(sid,artist='',pcm=None,passed=True):
    return dict(song_id=sid,audio_sha256=sid,audio_path=sid+'.flac',passed=passed,
        metadata=dict(artist=artist,title=sid),native_audio=dict(sample_rate=16000,channels=1,float32_sample_sha256=pcm or sid))


class PilotAuditTests(unittest.TestCase):
    def test_transitive_overlap_excludes_every_connected_candidate(self):
        checks=[row('a'),row('b'),row('c')]
        references=[dict(song_id='reserved',audio_sha256='reserved')]
        links=[dict(song_id='a',matched_song_id='reserved'),dict(song_id='b',matched_song_id='a')]
        result={r['song_id']:r for r in screen(checks,references,links,[])}
        self.assertFalse(result['a']['annotation_candidate']);self.assertFalse(result['b']['annotation_candidate'])
        self.assertTrue(result['c']['annotation_candidate'])

    def test_pcm_duplicates_are_deduped_but_same_artist_songs_are_kept(self):
        checks=[row('a','Band','pcm'),row('b','Band','pcm'),row('c','Band','different')]
        result=screen(checks,[],[],[])
        self.assertEqual([r['song_id'] for r in result if r['annotation_candidate']],['a','c'])
        self.assertEqual(result[1]['disposition'],'duplicate_target_recording')
        self.assertEqual(result[0]['provisional_artist_recording_group'],['a','b','c'])

    def test_unknown_artists_are_grouped_together_without_fabrication(self):
        result=screen([row('a'),row('b'),row('c','Known')],[],[],[])
        self.assertEqual(result[0]['provisional_artist_recording_group'],['a','b'])
        self.assertEqual(result[0]['normalized_artist_tokens'],[])
        self.assertFalse(result[0]['final_split_assigned'])

    def test_native_failure_and_artist_overlap_are_not_admitted(self):
        result=screen([row('a',passed=False),row('b','Artist A; B')],[],[],[dict(song_id='test-id',artist='Artist A')])
        self.assertFalse(any(r['annotation_candidate'] for r in result))
        self.assertEqual(result[1]['known_main_artist_ids'],['test-id'])

    def test_invalid_fingerprint_link_is_rejected(self):
        with self.assertRaises(ValueError):screen([row('a')],[],[dict(song_id='a',matched_song_id='missing')],[])

    def test_collaboration_tags_link_conservatively(self):
        self.assertEqual(artist_tokens('Band, Singer feat. Guest'),['band','guest','singer'])

    def test_native_clock_wrapper_does_not_invent_original_song_origin(self):
        with tempfile.TemporaryDirectory(dir=BASE/'tmp') as temp:
            path=Path(temp)/'native.wav';sf.write(path,np.sin(np.arange(16000)*.17).astype('float32')*.2,16000,subtype='FLOAT')
            digest=sha256(path)
            details,tags=local_native(path,1.,Path(temp)/'logs')
            self.assertEqual(details['frames'],16000);self.assertTrue(details['bulk_stream_sample_parity'])
            self.assertIsNone(details['source_song_offset_seconds']);self.assertFalse(details['original_song_completeness_verified'])
            self.assertEqual(tags['artist'],'');self.assertEqual(digest,sha256(path))


if __name__=='__main__':unittest.main()
