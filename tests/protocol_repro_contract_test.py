"""Protocol audit validates field-specific differences and bad budget input."""
import unittest
from p3.protocol_repro import differences, audit, hardware_fingerprint


class ProtocolAuditTests(unittest.TestCase):
    def test_field_diff(self):
        self.assertEqual(differences({"a": 1, "b": 2}, {"a": 1, "b": 3}), ["b"])
        self.assertEqual(differences({"a": 1}, {"a": 1}), [])

    def test_bad_budget_rejected_without_starting_training(self):
        with self.assertRaises(ValueError):
            audit(frames=79)
        with self.assertRaises(ValueError):
            audit(frames=81)
        with self.assertRaises(ValueError):
            audit(seed=-1)

    def test_cpu_fingerprint_has_nonempty_machine_data(self):
        d=hardware_fingerprint()
        self.assertTrue(d["python"])
        self.assertTrue(d["torch"])
        self.assertEqual(len(d["cpu_flags_sha256"]), 16)


if __name__ == "__main__":
    unittest.main()
