import unittest
from p3.full_path_audit import audit

class FullPathAuditTests(unittest.TestCase):
    def test_invalid_frame_budget(self):
        with self.assertRaises(ValueError): audit(frames=79)
        with self.assertRaises(ValueError): audit(frames=81)

    def test_full_call_order_short_budget(self):
        d=audit(frames=160,evaluation_episodes=4)
        self.assertTrue(d["same_initial_vs_joint"])
        self.assertTrue(d["same_initial_vs_after"])
        self.assertEqual(d["seeds"],[2026,2027,2028])
        self.assertEqual(d["frames_per_model_seed"],160)

if __name__ == "__main__":
    unittest.main()
