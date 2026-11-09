"""A hard frame budget is stronger than an episode-count equality assertion."""
import unittest

import gymnasium as gym

from p3.equal_frame_comparison import FrameBudgetEnv, run_episodic, compare


class ExactFrameBudgetTests(unittest.TestCase):
    def test_hard_cutoff_does_not_overshoot_and_marks_partial(self):
        original = gym.make("CartPole-v1")
        env = FrameBudgetEnv(original, 1)
        try:
            env.reset(seed=13)
            _, _, _, truncated, _ = env.step(0)
            self.assertTrue(truncated)
            self.assertEqual(env.used, 1)
            self.assertEqual(env.budget_cutoff_episodes, 1)
            with self.assertRaisesRegex(RuntimeError, "exceed"):
                env.step(0)
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                env.reset(seed=14)
        finally:
            original.close()

    def test_budget_is_exact_for_both_episodic_learners(self):
        for neural in (False, True):
            with self.subTest(neural=neural):
                a = run_episodic(21, 160, 4, neural=neural, checkpoint_every=80)
                b = run_episodic(21, 160, 4, neural=neural, checkpoint_every=80)
                self.assertEqual(a, b)
                self.assertEqual(a["training_environment_steps"], 160)
                self.assertGreater(a["training_actor_weight_l2_change"], 0)
                self.assertGreater(a["training_critic_weight_l2_change"], 0)
                self.assertEqual(a["evaluation"]["trained_greedy"]["episodes"], 4)

    def test_short_equal_frames_all_three_algorithms(self):
        report, cost = compare(seeds=(42,), frames=160, eval_episodes=4)
        self.assertEqual(report["training_frames_per_seed_and_algorithm"], 160)
        self.assertEqual(set(report["runs"]), {
            "numpy_linear", "pytorch_small", "pytorch_nstep"})
        for name, rows in report["runs"].items():
            self.assertEqual(rows[0]["training_environment_steps"], 160)
        self.assertEqual(len(cost["timings"]), 1)

    def test_incorrect_frames_and_duplicate_seeds_rejected(self):
        with self.assertRaises(ValueError):
            compare(seeds=(1,), frames=17, eval_episodes=4)
        with self.assertRaises(ValueError):
            compare(seeds=(1, 1), frames=160, eval_episodes=4)
        with self.assertRaises(ValueError):
            run_episodic(1, 0, 4)


if __name__ == "__main__":
    unittest.main()
