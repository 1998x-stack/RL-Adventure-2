"""Modern CPU-only PyTorch *neural* actor-critic CartPole-v1 companion.

Uses the SAME episode seeds, training budget, checkpoint validation and held-out
test protocol as p3/cartpole_actor_critic.py. This is not a verbatim conversion
of the historical A2C notebook's vectorized n-step / bootstrap algorithm.
"""
from __future__ import annotations

import copy
import math
import numpy as np

from p3.cartpole_actor_critic import episode, evaluate


class TorchActorCritic:
    def __init__(self, seed: int, policy_lr: float = 0.003,
                 value_lr: float = 0.003, discount: float = 0.99):
        import torch
        from torch import nn
        if policy_lr <= 0 or value_lr <= 0 or not 0 < discount < 1:
            raise ValueError("invalid hyperparameters")
        torch.manual_seed(seed)
        self.torch = torch
        self.actor = nn.Sequential(
            nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(5, 32), nn.Tanh(), nn.Linear(32, 1)
        )
        self.optimizer = torch.optim.Adam([
            {"params": self.actor.parameters(), "lr": policy_lr},
            {"params": self.critic.parameters(), "lr": value_lr},
        ])
        self.discount = discount
        self.initial_actor = torch.cat([
            p.detach().flatten().clone() for p in self.actor.parameters()
        ])
        self.initial_critic = torch.cat([
            p.detach().flatten().clone() for p in self.critic.parameters()
        ])

    def probabilities(self, state):
        torch = self.torch
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32)
            logits = self.actor(state_tensor)
            return torch.softmax(logits, -1).cpu().numpy().astype(np.float64)

    def update(self, episodes):
        if not episodes or not any(episodes):
            raise ValueError("empty rollout batch")
        torch = self.torch
        states, actions, discounted = [], [], []
        for rollout in episodes:
            future, returns = 0.0, []
            for step in reversed(rollout):
                future = step.reward + self.discount * future
                returns.append(future)
            for step, ret in zip(rollout, reversed(returns)):
                states.append(step.state)
                actions.append(step.action)
                discounted.append(ret)
        X = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        chosen = torch.as_tensor(actions, dtype=torch.int64)
        targets = torch.as_tensor(discounted, dtype=torch.float32) / 100.0

        logits = self.actor(X)
        values = self.critic(X).squeeze(-1)
        advantage = targets - values
        normalized = (advantage.detach() - advantage.detach().mean()) / (
            advantage.detach().std(unbiased=False) + 1e-7)
        log_prob = torch.log_softmax(logits, dim=-1).gather(
            1, chosen[:, None]).squeeze(1)
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        loss = -(log_prob * normalized).mean() + 0.5 * torch.nn.functional.mse_loss(
            values, targets) - 0.003 * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()), 5.0)
        self.optimizer.step()
        return {"batch_steps": len(states), "loss": round(float(loss.detach()), 5)}

    def actor_change(self):
        current = self.torch.cat([
            p.detach().flatten() for p in self.actor.parameters()])
        return float(self.torch.linalg.vector_norm(current - self.initial_actor))

    def critic_change(self):
        current = self.torch.cat([
            p.detach().flatten() for p in self.critic.parameters()])
        return float(self.torch.linalg.vector_norm(current - self.initial_critic))

    def parameter_count(self):
        return sum(p.numel() for p in self.actor.parameters()) + sum(
            p.numel() for p in self.critic.parameters())


def run_torch(seed: int = 2026, train_episodes: int = 480,
              evaluation_episodes: int = 64):
    if seed < 0 or train_episodes <= 0 or evaluation_episodes <= 0:
        raise ValueError("seed and episode budgets must be positive")
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed + 11)
    agent = TorchActorCritic(seed)
    first = evaluate(env, agent, seed + 200_000, evaluation_episodes)
    random_baseline = evaluate(env, None, seed + 200_000,
                               evaluation_episodes, random_policy=True)
    best = (copy.deepcopy(agent.actor.state_dict()),
            copy.deepcopy(agent.critic.state_dict()))
    best_validation = -math.inf
    trace, batch = [], []
    try:
        for i in range(train_episodes):
            transitions, _ = episode(env, agent, seed + i, rng)
            batch.append(transitions)
            if len(batch) == 8:
                agent.update(batch)
                batch.clear()
            if (i + 1) % 40 == 0 or i + 1 == train_episodes:
                validation = evaluate(env, agent, seed + 100_000, 12)
                trace.append({
                    "episode": i + 1,
                    "validation_mean_reward": validation["mean_reward"],
                })
                if validation["mean_reward"] > best_validation:
                    best_validation = validation["mean_reward"]
                    best = (copy.deepcopy(agent.actor.state_dict()),
                            copy.deepcopy(agent.critic.state_dict()))
        if batch:
            agent.update(batch)
        agent.actor.load_state_dict(best[0])
        agent.critic.load_state_dict(best[1])
        return {
            "algorithm": "two-layer PyTorch neural Monte Carlo actor-critic",
            "environment": "CartPole-v1",
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "seed": seed,
            "training_episodes": train_episodes,
            "evaluation_episodes": evaluation_episodes,
            "model_parameters": agent.parameter_count(),
            "actor_weight_l2_change": round(agent.actor_change(), 6),
            "critic_weight_l2_change": round(agent.critic_change(), 6),
            "evaluation": {
                "initial_greedy": first,
                "trained_greedy": evaluate(
                    env, agent, seed + 200_000, evaluation_episodes),
                "random": random_baseline,
            },
            "checkpoint_validation_trace": trace,
            "checkpoint_selection": "best 12-episode validation mean; disjoint held-out evaluation",
        }
    finally:
        env.close()
