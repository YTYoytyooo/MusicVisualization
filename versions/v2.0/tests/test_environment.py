import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studio.pipeline import ffmpeg_path


class EncoderDiscoveryTests(unittest.TestCase):
    def test_winget_encoder_found_without_path_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            encoder = root / 'Microsoft/WinGet/Packages/Gyan.FFmpeg_test/ffmpeg-8/bin/ffmpeg.exe'
            encoder.parent.mkdir(parents=True)
            encoder.touch()
            with (patch.dict(os.environ, {'LOCALAPPDATA': str(root), 'STUDIO_FFMPEG': ''}),
                  patch('studio.pipeline.ROOT', root),
                  patch('studio.pipeline.shutil.which', return_value=None),
                  patch.object(Path, 'is_file', lambda path: path == encoder)):
                self.assertEqual(ffmpeg_path(), str(encoder.resolve()))

    def test_explicit_encoder_remains_highest_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            encoder = Path(folder) / 'custom-ffmpeg.exe'
            encoder.touch()
            with patch.dict(os.environ, {'STUDIO_FFMPEG': str(encoder)}):
                self.assertEqual(ffmpeg_path(), str(encoder.resolve()))

    def test_inaccessible_system_encoder_does_not_hide_local_one(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            encoder = root / '.runtime/bin/ffmpeg.exe'
            encoder.parent.mkdir(parents=True)
            encoder.touch()
            def check(path):
                if path == Path('inaccessible.exe'):
                    raise PermissionError('system install access denied')
                return path == encoder
            with (patch.dict(os.environ, {'STUDIO_FFMPEG': 'inaccessible.exe'}),
                  patch('studio.pipeline.ROOT', root),
                  patch('studio.pipeline.shutil.which', return_value=None),
                  patch.object(Path, 'is_file', check)):
                self.assertEqual(ffmpeg_path(), str(encoder.resolve()))


if __name__ == '__main__':
    unittest.main()
