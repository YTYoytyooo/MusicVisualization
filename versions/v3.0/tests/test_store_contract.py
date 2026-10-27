"""Project/audio identity and path containment through persistence boundaries."""
import unittest
from unittest.mock import patch

from tools.train_next_v2.v3_studio import store
from support_v3 import StudioFixture


class StoreContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = StudioFixture()
        self.addCleanup(self.fixture.close)

    def test_valid_project_and_audio_are_read_without_modification(self):
        f = self.fixture
        original = (f.project_dir / 'analysis/result.json').read_bytes()
        loaded = store.load_project(f.project_id)
        self.assertEqual(loaded['result'], f.result)
        self.assertEqual(store.audio_entry('fixture-audio')['sha256'], f.row['sha256'])
        self.assertEqual((f.project_dir / 'analysis/result.json').read_bytes(), original)

    def test_project_ids_cannot_escape_or_select_path_syntax(self):
        for value in ('..', '../x', '..\\x', 'a/b', 'C:\\Windows', '', None, 'x' * 97, '%2e%2e'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                store.project_path(value)

    def test_inside_resolves_parent_segments_before_containment(self):
        root = self.fixture.root
        with self.assertRaises(ValueError):
            store.inside(root / '..' / 'outside.txt', root)

    def test_unknown_or_same_name_modified_audio_is_rejected(self):
        with self.assertRaises(ValueError):
            store.audio_entry('unknown')
        self.fixture.audio.write_bytes(b'changed content under the same filename')
        with self.assertRaises(ValueError):
            store.audio_entry('fixture-audio')

    def test_modified_prediction_without_receipt_update_is_rejected(self):
        f = self.fixture
        path = f.project_dir / 'analysis/result.json'
        value = store.read(path)
        value['systems']['b01']['raw_va'][0][0] = .99
        store.atomic(path, value)
        with self.assertRaises(ValueError):
            store.load_project(f.project_id)

    def test_project_geometry_cannot_change_without_invalidating_commit(self):
        f = self.fixture
        project = store.read(f.project_dir / 'project.json')
        project['audio']['duration_seconds'] = 200.
        store.atomic(f.project_dir / 'project.json', project)
        with self.assertRaises(ValueError):
            store.load_project(f.project_id)

    def test_rehashed_prediction_cannot_change_project_audio_or_manifest(self):
        f = self.fixture
        for field in ('project_id', 'audio_sha256', 'manifest_sha256'):
            with self.subTest(field=field):
                if field == 'audio_sha256':
                    old = f.result['audio']['sha256']; f.result['audio']['sha256'] = 'b' * 64
                else:
                    old = f.result[field]; f.result[field] = 'b' * 64
                f.write_result()
                with self.assertRaises(ValueError):
                    store.load_project(f.project_id)
                if field == 'audio_sha256': f.result['audio']['sha256'] = old
                else: f.result[field] = old
                f.write_result()

    def test_missing_raw_or_missing_receipt_never_appears_ready(self):
        f = self.fixture
        for missing in ('result.json', 'receipt.json'):
            with self.subTest(missing=missing):
                f.write_result()
                (f.project_dir / 'analysis' / missing).unlink()
                entry = next(p for p in store.projects() if p['id'] == f.project_id)
                self.assertNotEqual(entry['status'], 'ready')
                self.assertFalse(entry.get('has_result', False))

    def test_hash_tamper_is_not_advertised_as_ready(self):
        f = self.fixture
        (f.project_dir / 'analysis/result.json').write_text('{}', encoding='utf-8')
        entry = next(p for p in store.projects() if p['id'] == f.project_id)
        self.assertNotEqual(entry['status'], 'ready')
        self.assertFalse(entry.get('has_result', False))

    def test_atomic_retries_transient_windows_sharing_and_publishes_exact_bytes(self):
        original_replace = store.Path.replace
        for winerror in (5, 32, 33):
            with self.subTest(winerror=winerror):
                target = self.fixture.root / f'atomic-{winerror}.json'
                store.atomic(target, {'status': 'old'})
                old_bytes = target.read_bytes()
                expected = {'status': 'completed', 'message': '保留精确内容', 'progress': 1.0}
                attempts = []
                def sharing_then_success(temporary, destination):
                    attempts.append((str(temporary), temporary.read_bytes()))
                    self.assertEqual(target.read_bytes(), old_bytes)
                    if len(attempts) <= 2:
                        error = PermissionError('synthetic transient Windows sharing conflict')
                        error.winerror = winerror
                        raise error
                    return original_replace(temporary, destination)
                with patch.object(store.Path, 'replace', new=sharing_then_success), patch.object(store.time, 'sleep') as sleep:
                    store.atomic(target, expected)
                self.assertEqual(len(attempts), 3)
                self.assertEqual(len({name for name, _ in attempts}), 1)
                self.assertTrue(all(data == attempts[0][1] for _, data in attempts))
                self.assertEqual(target.read_bytes(), attempts[0][1])
                self.assertEqual(store.read(target), expected)
                self.assertEqual([call.args[0] for call in sleep.call_args_list], [.01, .02])
                self.assertEqual(list(target.parent.glob(target.name + '.*.tmp')), [])

    def test_atomic_permanent_windows_conflict_raises_after_eight_attempts_and_preserves_old(self):
        target = self.fixture.root / 'atomic-permanent.json'
        store.atomic(target, {'status': 'old'})
        original = target.read_bytes()
        pending = {'status': 'new', 'message': '不可吞掉失败'}
        attempts = []
        error = PermissionError('synthetic permanent Windows sharing conflict')
        error.winerror = 5
        def permanent(temporary, destination):
            attempts.append(temporary.read_bytes())
            self.assertEqual(target.read_bytes(), original)
            raise error
        with patch.object(store.Path, 'replace', new=permanent), patch.object(store.time, 'sleep') as sleep:
            with self.assertRaises(PermissionError) as caught:
                store.atomic(target, pending)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(attempts), 8)
        self.assertEqual(sleep.call_count, 7)
        self.assertTrue(all(0 < call.args[0] <= .2 for call in sleep.call_args_list))
        self.assertEqual(target.read_bytes(), original)
        temporary = list(target.parent.glob(target.name + '.*.tmp'))
        self.assertEqual(len(temporary), 1)
        self.assertEqual(store.read(temporary[0]), pending)
        self.assertTrue(all(data == attempts[0] for data in attempts))


if __name__ == '__main__':
    unittest.main()
