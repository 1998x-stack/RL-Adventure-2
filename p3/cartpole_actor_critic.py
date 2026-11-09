"""CPU-only, seeded *linear* actor-critic companion to 1.actor-critic.ipynb.

The archived notebook uses old Gym CartPole-v0 and PyTorch vector environments.
This compact learning example avoids upgrading or editing the original notebooks.
It performs real policy/value updates, but is not a deep neural-network run.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def features(observation):
    """Scale CartPole's four continuous observations, preserving a bias term."""
    x = np.asarray(observation, dtype=np.float64)
    if x.shape != (4,) or not np.all(np.isfinite(x)):
        raise ValueError("CartPole expects four finite features")
    return np.r_[np.clip(x / [2.4, 3.0, 0.2095, 3.5], -5.0, 5.0), 1.0]


@dataclass
class Step:
    state: np.ndarray
    action: int
    probabilities: np.ndarray
    reward: float


class LinearActorCritic:
    """Two-action softmax policy and linear state-value baseline."""

    def __init__(self, seed: int, policy_lr: float = 0.22, value_lr: float = 0.04,
                 discount: float = 0.99):
        if policy_lr <= 0 or value_lr <= 0 or not 0 < discount < 1:
            raise ValueError("invalid hyperparameters")
        rng = np.random.default_rng(seed)
        self.actor = rng.normal(0, 0.01, size=(5, 2))
        self.critic = np.zeros(5, dtype=np.float64)
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.discount = discount
        self.initial_actor = self.actor.copy()

    def probabilities(self, state):
        logits = np.asarray(state) @ self.actor
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        return weights / weights.sum()

    def update(self, episodes: list[list[Step]]):
        """Monte Carlo advantage actor gradient and squared-error critic gradient.

        Both gradients use the pre-update parameters (no target leakage).
        Each batch is normalized once to avoid long episodes dominating.
        """
        if not episodes or not any(episodes):
            raise ValueError("empty rollout batch")
        states, actions, probabilities, returns = [], [], [], []
        for episode in episodes:
            future = 0.0
            rewards_to_go = []
            for step in reversed(episode):
                future = step.reward + self.discount * future
                rewards_to_go.append(future)
            for step, ret in zip(episode, reversed(rewards_to_go)):
                states.append(step.state)
                actions.append(step.action)
                probabilities.append(step.probabilities)
                returns.append(ret)
        X = np.vstack(states)
        old_p = np.vstack(probabilities)
        actions = np.asarray(actions, dtype=np.int64)
        targets = np.asarray(returns, dtype=np.float64)
        advantage = targets - X @ self.critic
        # Keep the critic's prediction scale stable on long CartPole episodes.
        critic_gradient = np.mean(X * advantage[:, None], axis=0)
        self.critic += self.value_lr * np.clip(critic_gradient, -10.0, 10.0)
        normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        residual = -old_p
        residual[np.arange(len(actions)), actions] += 1.0
        actor_gradient = X.T @ (residual * normalized[:, None]) / len(X)
        norm = np.linalg.norm(actor_gradient)
        if norm > 2.0:
            actor_gradient *= 2.0 / norm
        self.actor += self.policy_lr * actor_gradient
        return {"steps": int(len(X)), "weight_delta": float(np.linalg.norm(
            self.actor - self.initial_actor))}


def episode(env, policy: LinearActorCritic | None, seed: int,
            rng: np.random.Generator, *, greedy: bool = False):
    obs, _ = env.reset(seed=seed)
    steps, total = [], 0.0
    while True:
        state = features(obs)
        if policy is None:
            action = int(rng.integers(env.action_space.n))
            probs = np.ones(env.action_space.n) / env.action_space.n
        else:
            probs = policy.probabilities(state)
            action = int(np.argmax(probs)) if greedy else int(rng.choice(
                env.action_space.n, p=probs))
        obs, reward, terminated, truncated, _ = env.step(action)
        steps.append(Step(state, action, probs.copy(), float(reward)))
        total += reward
        if terminated or truncated:
            return steps, float(total)


def evaluate(env, policy, seed: int, count: int, *, random_policy=False):
    rng = np.random.default_rng(seed + 993)
    scores = [
        episode(env, None if random_policy else policy, seed + i, rng,
                greedy=not random_policy)[1]
        for i in range(count)
    ]
    return {
        "episodes": count,
        "mean_reward": round(float(np.mean(scores)), 3),
        "std_reward": round(float(np.std(scores)), 3),
        "min_reward": round(float(np.min(scores)), 3),
        "max_reward": round(float(np.max(scores)), 3),
    }


def run(seed: int = 2026, train_episodes: int = 480,
        evaluation_episodes: int = 64):
    if seed < 0 or train_episodes <= 0 or evaluation_episodes <= 0:
        raise ValueError("seed and episode budgets must be positive")
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed + 11)
    agent = LinearActorCritic(seed)
    first = evaluate(env, agent, seed + 200_000, evaluation_episodes)
    baseline = evaluate(env, None, seed + 200_000, evaluation_episodes,
                        random_policy=True)
    best_weights = agent.actor.copy()
    best_validation = -math.inf
    traces = []
    batch = []
    try:
        for i in range(train_episodes):
            transitions, score = episode(env, agent, seed + i, rng)
            batch.append(transitions)
            if len(batch) == 8:
                info = agent.update(batch)
                batch.clear()
            if (i + 1) % 40 == 0 or i + 1 == train_episodes:
                validation = evaluate(env, agent, seed + 100_000, 12)
                traces.append({"episode": i + 1,
                               "validation_mean_reward": validation["mean_reward"]})
                if validation["mean_reward"] > best_validation:
                    best_validation = validation["mean_reward"]
                    best_weights = agent.actor.copy()
        if batch:
            agent.update(batch)
        # A held-out seed set is never consulted for checkpoint selection.
        agent.actor = best_weights
        final = evaluate(env, agent, seed + 200_000, evaluation_episodes)
        change = float(np.linalg.norm(agent.actor - agent.initial_actor))
        return {
            "algorithm": "linear actor-critic: Monte Carlo advantage policy gradient",
            "environment": "CartPole-v1",
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
            "seed": seed,
            "training_episodes": train_episodes,
            "evaluation_episodes": evaluation_episodes,
            "policy_weight_l2_change": round(change, 6),
            "evaluation": {
                "initial_greedy": first,
                "trained_greedy": final,
                "random": baseline,
            },
            "checkpoint_validation_trace": traces,
            "checkpoint_selection": "best 12-episode validation mean; disjoint 64-episode held-out test",
        }
    finally:
        env.close()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--seed", type=int, default=2026)
    cli.add_argument("--train-episodes", type=int, default=480)
    cli.add_argument("--eval-episodes", type=int, default=64)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = run(args.seed, args.train_episodes, args.eval_episodes)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
