import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch

from tools.train_next_v2.jamendo_data.metadata import select_files, validate_blob, blob_sha
from tools.train_next_v2.jamendo_data import metadata
from tools.train_next_v2.jamendo_data.metadata_v2 import tree_request


class MetadataAdmissionTests(unittest.TestCase):
    def tree(self):
        paths = ['README.md', 'LICENSE', 'audio_licenses.txt', 'data/raw.meta.tsv',
                 'data/autotagging_moodtheme.tsv', 'data/tag_map.json',
                 'data/splits/split-0/autotagging_moodtheme-train.tsv', 'audio/all.zip', 'script.py']
        return dict(truncated=False, tree=[dict(path=p, mode='100644', type='blob', size=3,
                                               sha=blob_sha(b'abc')) for p in paths])

    def test_only_metadata_selected(self):
        rows = select_files(self.tree())
        self.assertEqual(len(rows), 7)
        self.assertFalse(any(r['path'].endswith(('.py', '.zip')) for r in rows))

    def test_truncated_missing_or_duplicate_fails_closed(self):
        tree = self.tree(); tree['truncated'] = True
        with self.assertRaises(ValueError): select_files(tree)
        tree = self.tree(); tree['tree'].pop(0)
        with self.assertRaises(ValueError): select_files(tree)
        tree = self.tree(); tree['tree'].append(deepcopy(tree['tree'][0]))
        with self.assertRaises(ValueError): select_files(tree)

    def test_links_and_unsafe_split_paths_rejected(self):
        tree = self.tree(); tree['tree'][0]['mode'] = '120000'
        with self.assertRaises(ValueError): select_files(tree)
        tree = self.tree(); tree['tree'][6]['path'] = 'data/splits/split-0/../../moodtheme.tsv'
        with self.assertRaises(ValueError): select_files(tree)

    def test_oversized_and_corrupt_blobs_rejected(self):
        tree = self.tree(); tree['tree'][0]['size'] = 21 * 1024 * 1024
        with self.assertRaises(ValueError): select_files(tree)
        spec = select_files(self.tree())[0]
        validate_blob(b'abc', spec)
        for bad in (b'xyz', b'abcd'):
            with self.assertRaises(ValueError): validate_blob(bad, spec)

    def test_registration_creates_parents_and_preserves_existing_config(self):
        with tempfile.TemporaryDirectory(dir=metadata.BASE / 'tmp') as temporary:
            destination = Path(temporary) / 'dataset'
            directory = Path(temporary) / 'nested' / 'job'
            with patch.object(metadata, 'DIRECTORY', directory), patch.object(metadata, 'DESTINATION', destination), \
                 patch.object(metadata, 'update_ledger'):
                metadata.register()
                content = (directory / 'launch.json').read_bytes()
                with self.assertRaises(FileExistsError): metadata.register()
                self.assertEqual(content, (directory / 'launch.json').read_bytes())

    def test_git_tree_uses_explicit_tree_object_not_commit_alias(self):
        commit = dict(sha='a' * 40, commit=dict(tree=dict(sha='b' * 40)))
        revision, tree_id, url = tree_request(commit)
        self.assertEqual(revision, 'a' * 40)
        self.assertEqual(tree_id, 'b' * 40)
        self.assertIn('/git/trees/' + 'b' * 40 + '?', url)
        commit['commit']['tree']['sha'] = 'master'
        with self.assertRaises(ValueError): tree_request(commit)


if __name__ == '__main__':
    unittest.main()
