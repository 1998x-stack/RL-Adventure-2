"""P3: gradient direction, policy parameter updates and reproducible CPU run."""
import unittest
import numpy as np

from p3.cartpole_actor_critic import LinearActorCritic, Step, features, run


class ActorCriticContracts(unittest.TestCase):
    def test_input_validation(self):
        self.assertEqual(features([0, 0, 0, 0]).shape, (5,))
        self.assertEqual(features([0, 0, 0, 0])[-1], 1)
        with self.assertRaises(ValueError):
            features([0, 0, np.nan, 0])
        with self.assertRaises(ValueError):
            LinearActorCritic(0, discount=1)
        with self.assertRaises(ValueError):
            run(-1, 2, 4)

    def test_actor_and_critic_both_update_without_external_rng(self):
        np.random.seed(9)
        expected = np.random.random()
        np.random.seed(9)
        agent = LinearActorCritic(2026)
        s = features([0.0, 0.0, 0.02, 0.01])
        p = agent.probabilities(s)
        self.assertAlmostEqual(float(p.sum()), 1.0)
        before_actor = agent.actor.copy()
        before_critic = agent.critic.copy()
        episode = [
            Step(s, 1, p.copy(), 1.0),
            Step(features([0.02, 0.1, -0.01, -0.2]), 0,
                 agent.probabilities(features([0.02, 0.1, -0.01, -0.2])), 1.0),
        ]
        info = agent.update([episode])
        self.assertEqual(info["steps"], 2)
        self.assertGreater(info["weight_delta"], 0)
        self.assertFalse(np.array_equal(before_actor, agent.actor))
        self.assertFalse(np.array_equal(before_critic, agent.critic))
        self.assertEqual(np.random.random(), expected)

    def test_short_cartpole_run_reproducible_and_learns(self):
        # Runs actual modern Gymnasium reset()/step() episodes.
        a = run(seed=9, train_episodes=40, evaluation_episodes=8)
        b = run(seed=9, train_episodes=40, evaluation_episodes=8)
        self.assertEqual(a, b)
        self.assertGreater(a["policy_weight_l2_change"], 0)
        self.assertEqual(a["evaluation"]["trained_greedy"]["episodes"], 8)
        self.assertEqual(a["checkpoint_validation_trace"][-1]["episode"], 40)


if __name__ == "__main__":
    unittest.main()
