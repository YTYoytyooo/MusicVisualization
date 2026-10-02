"""Fresh-checkout startup and explicit audio-directory handling."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


class StartupTests(unittest.TestCase):
    def test_fresh_checkout_creates_default_audio_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(server, 'ROOT', root):
                httpd = server.make_server(output=root/'labels', port=0)
            try:
                self.assertTrue((root/'assets/audio').is_dir())
                self.assertEqual(httpd.store.audio_root, root/'assets/audio')
                self.assertEqual(httpd.store.catalog('me'), [])
            finally:
                httpd.server_close()
                httpd.store.close()

    def test_explicit_missing_directory_reports_path_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root/'wrong-music'
            with self.assertRaises(ValueError) as error:
                server.make_server(audio_root=missing, output=root/'labels', port=0)
            self.assertIn(str(missing), str(error.exception))
            self.assertFalse(missing.exists())


if __name__ == '__main__':
    unittest.main()
