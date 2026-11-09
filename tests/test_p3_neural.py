"""Full-gradient, held-out and deterministic checks for a small neural actor critic."""
import unittest

import numpy as np

from p3.cartpole_actor_critic import Step, features, run as run_linear
from p3.torch_actor_critic import TorchActorCritic, run_torch
from p3.compare_models import compare


class NeuralContractTests(unittest.TestCase):
    def test_real_network_gradient_updates_both_networks(self):
        learner = TorchActorCritic(31)
        s1 = features([0.1, 0.2, -0.1, 0.3])
        s2 = features([-0.2, 0.1, 0.03, -0.1])
        p1 = learner.probabilities(s1)
        p2 = learner.probabilities(s2)
        learner.update([[Step(s1, 1, p1, 1.0), Step(s2, 0, p2, 1.0)]])
        self.assertGreater(learner.actor_change(), 0)
        self.assertGreater(learner.critic_change(), 0)
        self.assertGreater(learner.parameter_count(), 15)

    def test_replay_under_fixed_seed(self):
        first = run_torch(seed=47, train_episodes=16, evaluation_episodes=6)
        again = run_torch(seed=47, train_episodes=16, evaluation_episodes=6)
        self.assertEqual(first, again)
        self.assertEqual(first["evaluation"]["trained_greedy"]["episodes"], 6)

    def test_same_random_baseline_and_seed_separation(self):
        original, neural = run_linear(28, 16, 6), run_torch(28, 16, 6)
        self.assertEqual(original["evaluation"]["random"],
                         neural["evaluation"]["random"])
        report, timings = compare((31, 32), train_episodes=8, eval_episodes=4)
        self.assertEqual(report["seeds"], [31, 32])
        self.assertEqual(len(timings["timings"]), 2)
        self.assertEqual(report["linear_model"]["parameters"], 15)
        for model in ("numpy_linear", "pytorch_neural"):
            self.assertEqual(len(report["runs"][model]), 2)


if __name__ == "__main__":
    unittest.main()
