import unittest

from p3.seed_sequence_audit import audit


class SequenceAuditContracts(unittest.TestCase):
    def test_duplicate_earlier_seeds_rejected(self):
        with self.assertRaises(ValueError):
            audit(earlier_seeds=(1, 1))
        with self.assertRaises(ValueError):
            audit(seed=2028, earlier_seeds=(2027,2028))
        with self.assertRaises(ValueError):
            audit(frames=13)

    def test_short_sequence_replay(self):
        report=audit(frames=160, evaluation_episodes=4)
        self.assertTrue(report["equal"])
        self.assertEqual(report["difference_fields"], [])
        self.assertEqual([x["seed"] for x in report["preceding_results"]], [2026,2027])


if __name__ == "__main__":
    unittest.main()
