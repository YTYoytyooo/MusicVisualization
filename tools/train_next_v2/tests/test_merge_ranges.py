"""An ignored or misaligned HTTP range must never become a full audio download."""
import unittest
from tools.train_next_v2.auxiliary_data.merge_audit import validate_range_response


class MergeRangeTests(unittest.TestCase):
    def test_exact_response_accepted(self):
        validate_range_response(206, 'bytes 10-12/100', 10, 12, 100, b'abc')

    def test_full_or_wrong_range_and_wrong_length_rejected(self):
        for status, header, body in [(200, None, b'abc'), (206, 'bytes 0-2/100', b'abc'),
                                     (206, 'bytes 10-12/101', b'abc'),
                                     (206, 'bytes 10-12/100', b'ab'), (206, 'bytes 10-12/100', b'abcd')]:
            with self.subTest(status=status, header=header, length=len(body)):
                with self.assertRaises(ValueError):
                    validate_range_response(status, header, 10, 12, 100, body)


if __name__ == '__main__':
    unittest.main()
