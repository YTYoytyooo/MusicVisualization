"""Guard independent V/A ordering and transitive metadata exclusion groups."""
import unittest

from tools.train_next_v2.auxiliary_data.merge_candidates import connected_groups, normalized, provisional_va


class CandidateTests(unittest.TestCase):
    def test_named_va_order_and_provisional_scale(self):
        raw, mapped = provisional_va({'Arousal': '.225', 'Valence': '.875'}, 'Q4')
        self.assertEqual(raw, [.875, .225])
        self.assertEqual(mapped, [.75, -.55])
        for value in ['nan', 'inf', '-.2', '1.2', '.5']:
            with self.assertRaises(ValueError):
                provisional_va({'Arousal': value, 'Valence': '.875'}, 'Q4')
        with self.assertRaisesRegex(ValueError, 'quadrant'):
            provisional_va({'Arousal': '.225', 'Valence': '.875'}, 'Q1')

    def test_artist_and_recording_links_propagate_reserved_identity(self):
        rows = {
            'merge:a': {'identity_keys': [('artist','x')]},
            'merge:b': {'identity_keys': [('artist','x'),('allmusic_track','MT1')]},
            'merge:c': {'identity_keys': [('allmusic_track','MT1')]},
            'main:heldout': {'identity_keys': [('artist','x')]},
            'merge:unrelated': {'identity_keys': [('artist','z')]}}
        groups = connected_groups(rows)
        self.assertEqual(set(groups['merge:c']), {'merge:a','merge:b','merge:c','main:heldout'})
        self.assertEqual(groups['merge:unrelated'], ['merge:unrelated'])
        self.assertEqual(normalized(' Guns N’ Roses '), normalized('Guns N Roses'))


if __name__ == '__main__':
    unittest.main()
