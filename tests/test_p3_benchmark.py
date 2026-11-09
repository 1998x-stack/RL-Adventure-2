"""P3 multiple independent-seed results without cherry-picking."""
import unittest

from p3.benchmark import summarize, benchmark


def observation(seed, trained, random):
    return {
        "seed": seed,
        "evaluation": {
            "trained_greedy": {"mean_reward": trained},
            "random": {"mean_reward": random},
            "initial_greedy": {"mean_reward": 12},
        }
    }


class SeedSummaryTests(unittest.TestCase):
    def test_summary_keeps_every_result_and_reports_spread(self):
        metrics = summarize([observation(1, 90, 10), observation(2, 110, 30),
                             observation(3, 70, 20)])
        self.assertEqual(metrics["trained_greedy"]["individual_seed_means"], [90,110,70])
        self.assertEqual(metrics["trained_greedy"]["mean_of_seed_means"], 90)
        self.assertEqual(metrics["trained_greedy"]["sample_sd_of_seed_means"], 20)
        self.assertEqual(metrics["trained_minus_random"]["individual_seed_gaps"],[80,80,50])
        self.assertEqual(metrics["trained_minus_random"]["mean_gap"],70)

    def test_invalid_seed_groups_rejected(self):
        with self.assertRaises(ValueError): summarize([])
        with self.assertRaises(ValueError): benchmark(seeds=(1,),train_episodes=1,eval_episodes=1)
        with self.assertRaises(ValueError): benchmark(seeds=(1,1),train_episodes=1,eval_episodes=1)


if __name__ == "__main__":
    unittest.main()
