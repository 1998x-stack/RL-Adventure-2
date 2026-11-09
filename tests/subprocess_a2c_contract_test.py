"""Real worker identity, autoreset and end-to-end equal-budget replay."""
import os
import unittest
import numpy as np

from p3.vector_nstep_a2c import make_envs
from p3.compare_vector_backends import compare


class SubprocessA2CTests(unittest.TestCase):
    def test_real_processes_start_and_close(self):
        env, _ = make_envs(100, count=2, backend="async")
        processes = env.processes
        try:
            pids = [p.pid for p in processes]
            self.assertEqual(len(set(pids)), 2)
            self.assertNotIn(os.getpid(), pids)
            self.assertTrue(all(p.is_alive() for p in processes))
        finally:
            env.close()
        self.assertTrue(all(not p.is_alive() for p in processes))

    def test_same_step_final_observation_and_truncation_equivalence(self):
        sync, a = make_envs(30, count=2, max_episode_steps=1, backend="sync")
        async_env, b = make_envs(30, count=2, max_episode_steps=1, backend="async")
        try:
            np.testing.assert_array_equal(a, b)
            for _ in range(3):
                actions = np.array([0, 1], dtype=np.int64)
                expected = sync.step(actions)
                actual = async_env.step(actions)
                for lhs, rhs in zip(expected[:4], actual[:4]):
                    np.testing.assert_array_equal(lhs, rhs)
                for i in range(2):
                    np.testing.assert_array_equal(
                        expected[4]["final_obs"][i],
                        actual[4]["final_obs"][i])
                self.assertTrue(actual[3].all())
                self.assertTrue(actual[4]["_final_obs"].all())
        finally:
            sync.close()
            async_env.close()

    def test_short_training_replays_equal_and_updates_weights(self):
        report, _ = compare(seed=17, updates=3, evaluation_episodes=4)
        self.assertTrue(report["exact_training_and_evaluation_parity"])
        self.assertEqual(report["transitions_per_backend"], 240)
        self.assertGreater(report["results"]["async"]["actor_weight_l2_change"], 0)
        self.assertGreater(report["results"]["async"]["critic_weight_l2_change"], 0)

    def test_invalid_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            make_envs(17, count=2, backend="fake")


if __name__ == "__main__":
    unittest.main()
