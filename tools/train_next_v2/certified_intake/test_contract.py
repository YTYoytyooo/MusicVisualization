import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from tools.train_next.core import ROOT, sha256
from .contract import adapt_raw_takes, verify_native


class ContractTests(unittest.TestCase):
    def setUp(self):
        temp_root = ROOT / 'data/training/next_va/optimization/tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.tmp.cleanup)
        self.audio = Path(self.tmp.name) / 'fixture.wav'
        values = np.array([[0., .125], [.25, -.25], [.5, -.5], [.75, -.75]], dtype=np.float32)
        sf.write(self.audio, values, 2, subtype='FLOAT')
        pcm = hashlib.sha256(values.tobytes()).hexdigest()
        self.c = dict(audio_sha256=sha256(self.audio), annotation_song_id='fixture',
                      bulk_stream_exact=True, ffmpeg_stderr_empty=True, ffmpeg_returncode=0,
                      actual_frames=4, actual_sample_rate=2, actual_channels=2,
                      stream_frames=4, ffmpeg_frames=4, bulk_float32_sha256=pcm, stream_float32_sha256=pcm)
        self.song = dict(source=dict(song_id='fixture', audio_sha256=self.c['audio_sha256']), reviewer='human',
                         completed=dict(valence=False, arousal=False), takes=[dict(id='one', axes=['valence','arousal'], points=[
                             dict(time=0., status='annotated', valence=.1, arousal=None),
                             dict(time=.2, status='annotated', valence=.2, arousal=.3),
                             dict(time=.45, status='missing', valence=None, arousal=None),
                             dict(time=.5, status='annotated', valence=.4, arousal=.6),
                             dict(time=1.5, status='annotated', valence=.5, arousal=.7)])])

    def test_real_native_pcm(self):
        self.assertEqual(verify_native(self.audio,self.c)['frames'],4)

    def test_source_change_rejected(self):
        with self.audio.open('ab') as f: f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'source bytes'): verify_native(self.audio,self.c)

    def test_truncation_rejected_even_with_rebound_file_hash(self):
        sf.write(self.audio,np.zeros((2,2),dtype=np.float32),2,subtype='FLOAT')
        self.c['audio_sha256']=sha256(self.audio)
        with self.assertRaisesRegex(ValueError,'EOF frame count'): verify_native(self.audio,self.c)

    def test_incorrect_pcm_certificate_rejected(self):
        self.c['bulk_float32_sha256']=self.c['stream_float32_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'PCM hash'): verify_native(self.audio,self.c)

    def test_independent_decoder_disagreement_rejected(self):
        self.c['ffmpeg_frames']=3
        with self.assertRaisesRegex(ValueError,'counts disagree'): verify_native(self.audio,self.c)

    def test_axis_masks_real_intervals_gaps_and_take_boundaries(self):
        self.song['takes'].append(copy.deepcopy(self.song['takes'][0]))
        original=copy.deepcopy(self.song)
        result=adapt_raw_takes(self.song,self.c,.5)
        for take in result['takes']:
            self.assertEqual(take['valid_masks']['valence'],[True,True,False,True,True])
            self.assertEqual(take['valid_masks']['arousal'],[False,True,False,True,True])
            self.assertEqual(take['rate_edge_masks']['valence'],[True,False,False,False])
            self.assertEqual(take['rate_edge_masks']['arousal'],[False]*4)
            self.assertEqual(take['delta_time_seconds'][:2],[.2,.25])
        self.assertEqual(result['raw_song'],original)
        self.assertEqual(self.song,original)
        self.assertFalse(result['training_admitted'])

    def test_timestamp_outside_native_not_stretched(self):
        self.song['takes'][0]['points'][-1]['time']=2.1
        with self.assertRaisesRegex(ValueError,'outside'): adapt_raw_takes(self.song,self.c,.5)

    def test_duplicate_timestamp_has_no_rate_edge(self):
        self.song['takes'][0]['points'][1]['time']=0.
        r=adapt_raw_takes(self.song,self.c,.5)
        self.assertFalse(r['takes'][0]['rate_edge_masks']['valence'][0])

    def test_identity_mismatch_rejected(self):
        self.song['source']['song_id']='other'
        with self.assertRaisesRegex(ValueError,'identity'): adapt_raw_takes(self.song,self.c,.5)


if __name__ == '__main__': unittest.main()
