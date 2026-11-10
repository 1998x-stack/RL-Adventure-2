import unittest

from p3.compare_runner_reports import aggregate

class MultiHostReportTests(unittest.TestCase):
    def record(self,cpu,score):
        return {"machine":{"cpu_model":cpu,"cpu_flags_sha256":"a"*16,"torch":"2.5.1"},
                "runner_image":"ubuntu", "result":{
                    "seed":2028,"evaluation":{"trained_greedy":{"mean_reward":score}}}}
    def test_identical_scores_and_reports(self):
        a=aggregate([self.record("A",40),self.record("B",40)])
        self.assertTrue(a["exact_match_across_runners"])
        self.assertEqual(len(a["records"]),2)
    def test_detect_difference_without_cherry_picking(self):
        a=aggregate([self.record("A",40),self.record("B",50)])
        self.assertFalse(a["exact_match_across_runners"])
        self.assertEqual(a["records"][1]["differences_from_first"],["evaluation"])
    def test_single_run_insufficient(self):
        with self.assertRaises(ValueError):
            aggregate([self.record("A",40)])

if __name__=="__main__":
    unittest.main()
