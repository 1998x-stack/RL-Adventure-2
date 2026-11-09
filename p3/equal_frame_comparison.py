"""Frame-matched comparison: three *different* Actor-Critic implementations.

The earlier NumPy/PyTorch baselines trained for 480 completed episodes while
n-step A2C trained for 320,000 environment transitions. This experiment gives
ALL three exactly the SAME number of CartPole-v1 training transitions, held-out
evaluation reset seeds and 8,000-frame checkpoint cadence.

Matching frames does NOT match trajectories, optimizer, architecture, updates
or compute. Do not interpret a difference as a causal algorithm ranking.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import numpy as np

from p3.cartpole_actor_critic import LinearActorCritic, episode, evaluate, features
from p3.torch_actor_critic import TorchActorCritic
from p3.vector_nstep_a2c import run as run_nstep


SEEDS = (2026, 2027, 2028)
FRAME_BUDGET = 320_000
NSTEP_FRAMES_PER_UPDATE = 16 * 5
CHECKPOINT_EVERY_FRAMES = 8_000


class FrameBudgetEnv:
    """Exact frame-budget cutoff, including the last partial episode."""
    def __init__(self, env, budget):
        if budget < 1:
            raise ValueError("frame budget must be positive")
        self.env = env
        self.budget = budget
        self.used = 0
        self.budget_cutoff_episodes = 0
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, seed=None):
        if self.used >= self.budget:
            raise RuntimeError("training frame budget is exhausted")
        return self.env.reset(seed=seed)

    def step(self, action):
        if self.used >= self.budget:
            raise RuntimeError("training would exceed frame budget")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.used += 1
        if self.used == self.budget and not (terminated or truncated):
            # Episodic Monte Carlo baselines use a zero return beyond
            # this artificial budget boundary; it is explicitly counted.
            truncated = True
            self.budget_cutoff_episodes += 1
        return observation, reward, terminated, truncated, info


def _checkpoint(agent, neural):
    if neural:
        return (copy.deepcopy(agent.actor.state_dict()),
                copy.deepcopy(agent.critic.state_dict()))
    return (agent.actor.copy(), agent.critic.copy())


def _restore(agent, state, neural):
    if neural:
        agent.actor.load_state_dict(state[0])
        agent.critic.load_state_dict(state[1])
    else:
        agent.actor = state[0].copy()
        agent.critic = state[1].copy()


def run_episodic(seed, frames, evaluation_episodes=64, *, neural=False,
                 checkpoint_every=CHECKPOINT_EVERY_FRAMES):
    if seed < 0 or frames < 1 or evaluation_episodes < 1 or checkpoint_every < 1:
        raise ValueError("invalid frame/evaluation budget or seed")
    import gymnasium as gym
    if neural:
        import torch
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        learner = TorchActorCritic(seed)
    else:
        learner = LinearActorCritic(seed)
    env = gym.make("CartPole-v1")
    capped = FrameBudgetEnv(env, frames)
    rng = np.random.default_rng(seed + 11)
    try:
        initial = evaluate(env, learner, seed + 200_000, evaluation_episodes)
        random = evaluate(env, None, seed + 200_000, evaluation_episodes,
                          random_policy=True)
        best_validation = float("-inf")
        best = _checkpoint(learner, neural)
        trace = []
        episode_count = 0
        completed_naturally = 0
        updates = 0
        batch = []
        next_validation = checkpoint_every

        while capped.used < frames:
            used_before = capped.used
            transitions, _ = episode(capped, learner, seed + episode_count, rng)
            batch.append(transitions)
            episode_count += 1
            if capped.budget_cutoff_episodes == 0:
                completed_naturally += 1
            # A training step is counted only in capped.step(), never in
            # held-out or checkpoint evaluation.
            if len(batch) == 8 or capped.used == frames:
                learner.update(batch)
                updates += 1
                batch.clear()
            if capped.used >= next_validation or capped.used == frames:
                val = evaluate(env, learner, seed + 100_000, 12)
                trace.append({"training_frames": capped.used,
                              "validation_mean_reward": val["mean_reward"]})
                if val["mean_reward"] > best_validation:
                    best_validation = val["mean_reward"]
                    best = _checkpoint(learner, neural)
                while next_validation <= capped.used:
                    next_validation += checkpoint_every
            if capped.used <= used_before:
                raise AssertionError("episode did not consume frames")

        # Check actual training movement BEFORE validation-based checkpoint
        # selection; the selected checkpoint can legitimately be initial.
        if neural:
            training_actor_delta = learner.actor_change()
            training_critic_delta = learner.critic_change()
        else:
            training_actor_delta = float(np.linalg.norm(
                learner.actor - learner.initial_actor))
            training_critic_delta = float(np.linalg.norm(learner.critic))
        _restore(learner, best, neural)
        final = evaluate(env, learner, seed + 200_000, evaluation_episodes)
        if capped.used != frames:
            raise AssertionError("training used incorrect number of frames")
        if neural:
            actor_delta = learner.actor_change()
            critic_delta = learner.critic_change()
            parameters = learner.parameter_count()
        else:
            actor_delta = float(np.linalg.norm(
                learner.actor - learner.initial_actor))
            critic_delta = None
            parameters = 15
        return {
            "algorithm": "PyTorch 32-Tanh neural Monte Carlo actor-critic"
                         if neural else "NumPy linear Monte Carlo actor-critic",
            "seed": seed,
            "environment": "CartPole-v1",
            "training_environment_steps": capped.used,
            "training_episodes_started": episode_count,
            "natural_episodes_completed": completed_naturally,
            "budget_forced_truncations": capped.budget_cutoff_episodes,
            "optimizer_updates": updates,
            "model_parameters": parameters,
            "training_actor_weight_l2_change": round(training_actor_delta, 6),
            "training_critic_weight_l2_change": round(training_critic_delta, 6),
            "actor_weight_l2_change": round(actor_delta, 6),
            "critic_weight_l2_change": (
                round(critic_delta, 6) if critic_delta is not None else None),
            "evaluation": {
                "initial_greedy": initial,
                "trained_greedy": final,
                "random": random,
            },
            "checkpoint_validation_trace": trace,
            "checkpoint_rule": "best 12-episode validation mean at the first episode boundary passing each 8000-frame threshold; disjoint held-out seeds",
            "artificial_episode_boundary": "last partial episode is forcibly truncated at exact budget; episodic Monte Carlo return has no post-budget bootstrap",
        }
    finally:
        env.close()


def compare(seeds=SEEDS, frames=FRAME_BUDGET, eval_episodes=64):
    seeds = tuple(seeds)
    if len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    if frames < NSTEP_FRAMES_PER_UPDATE or frames % NSTEP_FRAMES_PER_UPDATE:
        raise ValueError("frame budget must be a positive multiple of 80")
    if eval_episodes < 1:
        raise ValueError("held-out evaluation must contain episodes")

    reports = {"numpy_linear": [], "pytorch_small": [], "pytorch_nstep": []}
    timings = []
    for seed in seeds:
        row_time = {"seed": seed}
        for key in ("numpy_linear", "pytorch_small", "pytorch_nstep"):
            started = time.perf_counter()
            if key == "numpy_linear":
                result = run_episodic(seed, frames, eval_episodes)
            elif key == "pytorch_small":
                result = run_episodic(seed, frames, eval_episodes, neural=True)
            else:
                result = run_nstep(
                    seed, updates=frames // NSTEP_FRAMES_PER_UPDATE,
                    evaluation_episodes=eval_episodes,
                    num_envs=16, n_steps=5, backend="sync")
            if result["training_environment_steps"] != frames:
                raise AssertionError(key + " violates exact transition budget")
            reports[key].append(result)
            row_time[key + "_seconds"] = round(time.perf_counter() - started, 3)
        # Both episodic algorithms use exactly the same held-out random
        # policy as n-step; no benchmark-specific action sampling advantage.
        references = [reports[k][-1]["evaluation"]["random"]
                      for k in reports]
        if not all(ref == references[0] for ref in references):
            raise AssertionError("random held-out reference differs")
        timings.append(row_time)
    summary = {}
    for name, rows in reports.items():
        vals = [r["evaluation"]["trained_greedy"]["mean_reward"] for r in rows]
        summary[name] = {
            "individual_seed_means": vals,
            "mean_of_seed_means": round(statistics.mean(vals), 3),
            "sample_sd_of_seed_means": (
                round(statistics.stdev(vals), 3) if len(vals) > 1 else None),
        }
    return {
        "environment": "CartPole-v1",
        "training_frames_per_seed_and_algorithm": frames,
        "held_out_eval_episodes_per_seed": eval_episodes,
        "training_seeds": list(seeds),
        "shared_held_out_reset_seed_offset": 200_000,
        "shared_validation_reset_seed_offset": 100_000,
        "comparison_type": "equal observed environment frames, not equal optimizer, gradient updates, training trajectories or wall-clock cost",
        "summary": summary,
        "runs": reports,
        "limitations": "episodic baselines end partial final episodes at exact frame budget with MC zero bootstrap; n-step bootstraps within rollouts; checkpoint boundaries differ by final episode length, model sizes and optimizers differ",
    }, {
        "method": "sequential single-runner CPU wall clock; install excluded, timings non-deterministic",
        "timings": timings,
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--frames", type=int, default=FRAME_BUDGET)
    cli.add_argument("--eval-episodes", type=int, default=64)
    cli.add_argument("--output", type=Path,
                     default=Path("build/p3-equal-frames.json"))
    cli.add_argument("--cost", type=Path,
                     default=Path("build/p3-equal-frames-cost.json"))
    args = cli.parse_args()
    report, cost = compare(frames=args.frames, eval_episodes=args.eval_episodes)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.cost.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.cost.write_text(json.dumps(cost, indent=2, sort_keys=True) + "\n")
    print("P3_EQUAL_FRAMES", json.dumps(report["summary"], sort_keys=True))
    print("P3_EQUAL_FRAMES_TIMINGS", json.dumps(cost["timings"], sort_keys=True))


if __name__ == "__main__":
    main()
