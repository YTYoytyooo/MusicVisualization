import tempfile
from pathlib import Path
import unittest

from tools.train_next_v2.common import BASE
from tools.train_next_v2.preparation.jamendo_candidates import deterministic_selection,licenses,connected_groups
from tools.train_next_v2.preparation.jamendo_schema import literal_tsv


class CandidateTests(unittest.TestCase):
    def test_literal_quotes_do_not_swallow_following_identity(self):
        with tempfile.TemporaryDirectory(dir=BASE/'tmp') as folder:
            path=Path(folder)/'identities.tsv';path.write_text('a\t"unfinished title\tx\nb\tnext\ty\n',encoding='utf-8')
            self.assertEqual(len(literal_tsv(path)),2)
            self.assertEqual(literal_tsv(path)[1][0],'b')

    def test_group_cap_and_order_independence(self):
        rows=[dict(track_id=str(i),group_id=str(i//3),selected_positive_tags=['happy' if i%2 else 'sad']) for i in range(18)]
        first=deterministic_selection(rows,count=18,cap=2)
        self.assertEqual(first,deterministic_selection(rows[::-1],count=18,cap=2))
        self.assertEqual(len(first),12)
        for group in {r['group_id'] for r in first}:self.assertEqual(sum(r['group_id']==group for r in first),2)

    def test_reserved_links_propagate_through_artist_aliases(self):
        identities={
            'candidate':dict(identity_keys=[('id','1'),('name','one')]),
            'alias':dict(identity_keys=[('id','1'),('name','two')]),
            'heldout':dict(identity_keys=[('name','two')]),
            'unrelated':dict(identity_keys=[('name','three')])}
        groups=connected_groups(identities)
        self.assertIn('heldout',groups['candidate']);self.assertNotIn('unrelated',groups['candidate'])

    def test_license_parser_retains_restrictive_code_and_rejects_duplicates(self):
        block='01/1.mp3\nA by B\nAvailable: http://creativecommons.org/licenses/by-nc-nd/3.0/\n\n'
        with tempfile.TemporaryDirectory(dir=BASE/'tmp') as folder:
            path=Path(folder)/'licenses.txt';path.write_text(block,encoding='utf-8')
            self.assertEqual(licenses(path)['01/1.mp3']['codes'],['by-nc-nd'])
            path.write_text(block*2,encoding='utf-8')
            with self.assertRaises(ValueError):licenses(path)


if __name__=='__main__':unittest.main()
