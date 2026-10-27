import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf

from tools.train_next_v2.auxiliary_data.merge_audio_audit import decode, safe_audio_id
from tools.train_next_v2.common import BASE


class AudioAuditTests(unittest.TestCase):
    def test_archive_member_paths_cannot_escape(self):
        self.assertEqual(safe_audio_id('MERGE_Audio_Complete/Q4/A081-67.mp3'), 'A081-67')
        for value in ['../a.mp3','MERGE_Audio_Complete/Q4/../a.mp3',
                      'MERGE_Audio_Complete/Q4/a\\b.mp3','MERGE_Audio_Complete/Q4/a:foo.mp3']:
            with self.assertRaises(ValueError):
                safe_audio_id(value)

    def test_native_decode_does_not_stretch_time(self):
        with tempfile.TemporaryDirectory(dir=BASE/'tmp') as directory:
            path = Path(directory)/'tone.wav'
            signal = np.sin(np.arange(36000)*2*np.pi*440/24000)*.25
            sf.write(path,signal,24000,subtype='FLOAT')
            result = decode(path)
            self.assertEqual(result['frames'],36000)
            self.assertEqual(result['duration_seconds'],1.5)
            self.assertIsNone(result['source_song_offset_seconds'])
            sf.write(path,np.zeros(36000),24000,subtype='FLOAT')
            with self.assertRaisesRegex(ValueError,'Silent'):
                decode(path)


if __name__ == '__main__':
    unittest.main()
