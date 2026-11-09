"""Test terminal masks, real Gymnasium vector API, gradients and RNG replay."""
import unittest

import numpy as np
import torch

from p3.vector_nstep_a2c import (
    NotebookStyleA2C, evaluate, learn_rollout, make_envs, n_step_returns, run,
)


class VectorA2CContract(unittest.TestCase):
    def test_known_two_step_bootstrapped_returns(self):
        rewards = torch.tensor([[1., 1.], [2., 3.]])
        continues = torch.tensor([[1., 1.], [1., 0.]])
        result = n_step_returns(rewards, continues, torch.tensor([10., 10.]),
                                gamma=0.5)
        self.assertTrue(torch.allclose(result, torch.tensor([
            [4.5, 2.5], [7.0, 3.0]])))
        with self.assertRaises(ValueError):
            n_step_returns(rewards, continues[:, :1], torch.ones(2))
        with self.assertRaises(ValueError):
            n_step_returns(rewards, continues, torch.ones(2), gamma=1)

    def test_actual_same_step_vector_resets_are_all_real_transitions(self):
        envs, state = make_envs(31, count=2, max_episode_steps=1)
        try:
            for _ in range(3):
                state, rewards, ended, truncated, info = envs.step(
                    np.array([0, 1], dtype=np.int64))
                self.assertEqual(rewards.tolist(), [1., 1.])
                self.assertEqual(truncated.tolist(), [True, True])
                self.assertTrue(np.all(info["_final_obs"]))
                self.assertTrue(np.all([
                    np.asarray(x).shape == (4,) for x in info["final_obs"]]))
        finally:
            envs.close()

    def test_vector_rollout_updates_both_networks(self):
        torch.manual_seed(42)
        model = NotebookStyleA2C(hidden_size=16)
        actor_before = [p.detach().clone() for p in model.actor.parameters()]
        critic_before = [p.detach().clone() for p in model.critic.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        envs, state = make_envs(42, count=3)
        try:
            state, result = learn_rollout(
                model, optimizer, envs, state, steps=3)
            self.assertEqual(result["environment_steps"], 9)
            self.assertTrue(np.isfinite(result["loss"]))
            self.assertTrue(any(
                not torch.equal(a, b) for a, b in
                zip(actor_before, model.actor.parameters())))
            self.assertTrue(any(
                not torch.equal(a, b) for a, b in
                zip(critic_before, model.critic.parameters())))
        finally:
            envs.close()

    def test_small_train_run_is_repeatable_with_held_out_evaluation(self):
        left = run(seed=7, updates=3, evaluation_episodes=4,
                   num_envs=2, n_steps=2, hidden_size=16)
        right = run(seed=7, updates=3, evaluation_episodes=4,
                    num_envs=2, n_steps=2, hidden_size=16)
        self.assertEqual(left, right)
        self.assertEqual(left["training_environment_steps"], 12)
        self.assertEqual(left["evaluation"]["trained_greedy"]["episodes"], 4)
        self.assertGreater(left["actor_weight_l2_change"], 0)
        self.assertGreater(left["critic_weight_l2_change"], 0)
        self.assertGreater(left["model_parameters"], 0)


if __name__ == "__main__":
    unittest.main()
