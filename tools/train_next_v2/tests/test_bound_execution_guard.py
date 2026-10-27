import unittest

from tools.train_next_v2.bound_execution_guard.job import directory_for, validate_binding, LEGACY_JOB


class BoundGuardTests(unittest.TestCase):
    def test_registration_cannot_target_legacy_or_escape_directory(self):
        for job in (LEGACY_JOB, '../old', 'f-validation-ac-guard-../old', 'other-job',
                    'f-validation-ac-guard-'+'x'*57):
            with self.subTest(job=job), self.assertRaises(ValueError):
                directory_for(job)
        self.assertEqual(directory_for('f-validation-ac-guard-resume-v1').name,
                         'f-validation-ac-guard-resume-v1')

    def test_replacement_pid_needs_its_own_live_registration(self):
        identity = dict(pid=1234, creation_filetime='134356900000000000')
        validate_binding(identity, dict(process_identity=identity), True, True)
        for changed in (dict(pid=1235, creation_filetime=identity['creation_filetime']),
                        dict(pid=1234, creation_filetime='134356900000000001')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_binding(identity, dict(process_identity=changed), True, True)

    def test_unknown_dead_unleased_or_incomplete_identity_cannot_hold(self):
        identity = dict(pid=1234, creation_filetime='134356900000000000')
        for alive, lease in ((False, True), (None, True), (True, False), (1, True)):
            with self.subTest(alive=alive, lease=lease), self.assertRaises(ValueError):
                validate_binding(identity, dict(process_identity=identity), alive, lease)
        for invalid in ({}, dict(pid=1234, creation_filetime=None),
                        dict(pid=-1, creation_filetime='1')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_binding(invalid, dict(process_identity=invalid), True, True)


if __name__ == '__main__':
    unittest.main()
