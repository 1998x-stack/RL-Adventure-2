"""Notebook-style n-step A2C on modern Gymnasium CPU vector environments.

Faithful core: separate 256-ReLU actor/critic, 16 environments, 5-step
bootstrapped targets, entropy bonus and shared Adam optimizer.
Deliberate modernizations: SyncVectorEnv or real multiprocessing AsyncVectorEnv, SAME_STEP autoreset,
correct TimeLimit final_observation bootstrap, explicit seeding and a held-out
checkpoint protocol. Not a byte-level reproduction of the 2018 notebook.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


DEFAULT_SEEDS = (2026, 2027, 2028)
NUM_ENVS = 16
NUM_STEPS = 5


class NotebookStyleA2C(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(4, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(4, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations):
        dist = Categorical(logits=self.actor(observations))
        values = self.critic(observations).squeeze(-1)
        return dist, values

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def n_step_returns(rewards, continues, bootstrap, gamma=0.99):
    """Bellman returns for [time, env] tensors with separated terminal masks."""
    if rewards.shape != continues.shape or rewards.ndim != 2:
        raise ValueError("Expected matching [time, environment] reward/mask tensors")
    if bootstrap.shape != rewards.shape[1:]:
        raise ValueError("Expected one bootstrap value per environment")
    if not 0 < gamma < 1:
        raise ValueError("discount must be in (0,1)")
    future = bootstrap
    backwards = []
    for t in range(len(rewards) - 1, -1, -1):
        future = rewards[t] + gamma * continues[t] * future
        backwards.append(future)
    return torch.stack(list(reversed(backwards)))


def make_envs(seed, count=NUM_ENVS, max_episode_steps=None, *,
              backend="sync"):
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, AutoresetMode

    if count < 1:
        raise ValueError("count must be positive")
    if backend not in ("sync", "async"):
        raise ValueError("backend must be sync or async")
    kwargs = {} if max_episode_steps is None else {
        "max_episode_steps": max_episode_steps}
    factories = [lambda options=kwargs: gym.make("CartPole-v1", **options)
                 for _ in range(count)]
    if backend == "sync":
        envs = SyncVectorEnv(factories, autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        # Gymnasium's real worker processes replace historical old-Gym
        # SubprocVecEnv; no fake or in-process concurrency.
        envs = AsyncVectorEnv(
            factories, autoreset_mode=AutoresetMode.SAME_STEP,
            shared_memory=False, context="spawn")
    try:
        state, _ = envs.reset(seed=[seed + i for i in range(count)])
        return envs, state
    except BaseException:
        envs.close()
        raise


def learn_rollout(model, optimizer, envs, state, *, steps=NUM_STEPS, gamma=0.99):
    """Execute exactly num_envs*steps real transitions, then update both nets.

    SAME_STEP returns the RESET state in next_state for terminated sub-envs.
    TimeLimit bootstrap uses infos['final_obs'], never the reset state.
    """
    logps, values, entropies, rewards, continues = [], [], [], [], []
    completed = 0
    truncated_count = 0
    for _ in range(steps):
        batch = torch.as_tensor(state, dtype=torch.float32)
        distribution, value = model(batch)
        action = distribution.sample()
        next_state, reward, terminated, truncated, infos = envs.step(
            action.detach().cpu().numpy())
        done = np.logical_or(terminated, truncated)
        reward_vector = torch.as_tensor(reward, dtype=torch.float32).clone()

        # Bootstrap at TimeLimit only: a truncated episode was not a true
        # environment failure, while the reset observation is not a successor.
        time_limit = np.logical_and(truncated, np.logical_not(terminated))
        if np.any(time_limit):
            assert "final_obs" in infos and "_final_obs" in infos
            assert np.all(infos["_final_obs"][time_limit])
            final_states = np.stack([
                infos["final_obs"][i] for i in np.flatnonzero(time_limit)
            ])
            with torch.no_grad():
                _, final_values = model(torch.as_tensor(
                    final_states, dtype=torch.float32))
            reward_vector[torch.as_tensor(time_limit)] += (
                gamma * final_values.detach())

        logps.append(distribution.log_prob(action))
        values.append(value)
        entropies.append(distribution.entropy())
        rewards.append(reward_vector)
        continues.append(torch.as_tensor(
            np.logical_not(done).astype(np.float32)))
        completed += int(np.sum(done))
        truncated_count += int(np.sum(truncated))
        state = next_state

    with torch.no_grad():
        _, final_value = model(torch.as_tensor(state, dtype=torch.float32))
    targets = n_step_returns(
        torch.stack(rewards), torch.stack(continues),
        final_value.detach(), gamma)
    predictions = torch.stack(values)
    advantage = targets.detach() - predictions
    actor_loss = -(torch.stack(logps) * advantage.detach()).mean()
    critic_loss = advantage.square().mean()
    entropy = torch.stack(entropies).mean()
    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite loss in n-step policy update")
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return state, {
        "completed_episodes": completed,
        "truncated_episodes": truncated_count,
        "environment_steps": steps * envs.num_envs,
        "loss": float(loss.detach()),
        "gradient_norm": float(grad_norm),
    }


def evaluate(model, seed, episodes=64, *, random_actions=False):
    """Greedy deterministic held-out evaluation, same seeds as P3 baselines."""
    import gymnasium as gym
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed + 993)
    rewards = []
    try:
        for i in range(episodes):
            observation, _ = env.reset(seed=seed + i)
            total = 0.0
            while True:
                if random_actions:
                    action = int(rng.integers(env.action_space.n))
                else:
                    with torch.no_grad():
                        dist, _ = model(torch.as_tensor(
                            observation, dtype=torch.float32))
                        action = int(torch.argmax(dist.probs).item())
                observation, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    rewards.append(total)
                    break
        return {
            "episodes": episodes,
            "mean_reward": round(statistics.mean(rewards), 3),
            "std_reward": round(statistics.pstdev(rewards), 3),
            "min_reward": round(min(rewards), 3),
            "max_reward": round(max(rewards), 3),
        }
    finally:
        env.close()


def run(seed=2026, updates=4000, evaluation_episodes=64,
        *, num_envs=NUM_ENVS, n_steps=NUM_STEPS, hidden_size=256,
        backend="sync"):
    if seed < 0 or updates < 1 or evaluation_episodes < 1:
        raise ValueError("Seed, updates and evaluation budget must be positive")
    if num_envs < 1 or n_steps < 1:
        raise ValueError("Invalid vectorization parameters")
    import gymnasium as gym

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = NotebookStyleA2C(hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    initial_actor = torch.cat([
        p.detach().flatten().clone() for p in model.actor.parameters()])
    initial_critic = torch.cat([
        p.detach().flatten().clone() for p in model.critic.parameters()])
    random_baseline = evaluate(model, seed + 200_000,
                               evaluation_episodes, random_actions=True)
    initial = evaluate(model, seed + 200_000, evaluation_episodes)
    envs, state = make_envs(seed, num_envs, backend=backend)
    best = copy.deepcopy(model.state_dict())
    best_validation = float("-inf")
    best_update = 0
    total_episodes = 0
    truncations = 0
    trace = []
    try:
        for u in range(1, updates + 1):
            state, result = learn_rollout(
                model, optimizer, envs, state, steps=n_steps)
            total_episodes += result["completed_episodes"]
            truncations += result["truncated_episodes"]
            if u % 100 == 0 or u == updates:
                validation = evaluate(model, seed + 100_000, 12)
                trace.append({"update": u,
                              "validation_mean_reward": validation["mean_reward"]})
                if validation["mean_reward"] > best_validation:
                    best_validation = validation["mean_reward"]
                    best = copy.deepcopy(model.state_dict())
                    best_update = u
    finally:
        envs.close()

    model.load_state_dict(best)
    actor_now = torch.cat([
        p.detach().flatten() for p in model.actor.parameters()])
    critic_now = torch.cat([
        p.detach().flatten() for p in model.critic.parameters()])
    actor_delta = float(torch.linalg.vector_norm(actor_now - initial_actor))
    critic_delta = float(torch.linalg.vector_norm(critic_now - initial_critic))
    return {
        "algorithm": "vectorized 5-step bootstrapped advantage actor-critic",
        "scope": "notebook-style n-step structure, modern Gymnasium SAME_STEP; not archived legacy SubprocVecEnv",
        "vector_backend": backend,
        "seed": seed,
        "gymnasium": gym.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "environment": "CartPole-v1",
        "num_envs": num_envs,
        "n_steps": n_steps,
        "updates": updates,
        "training_environment_steps": num_envs * n_steps * updates,
        "completed_train_episodes": total_episodes,
        "truncated_train_episodes": truncations,
        "evaluation_episodes": evaluation_episodes,
        "model_parameters": model.parameter_count(),
        "actor_weight_l2_change": round(actor_delta, 6),
        "critic_weight_l2_change": round(critic_delta, 6),
        "selected_checkpoint_update": best_update,
        "validation_trace": trace,
        "evaluation": {
            "initial_greedy": initial,
            "trained_greedy": evaluate(
                model, seed + 200_000, evaluation_episodes),
            "random": random_baseline,
        },
    }


def multi_seed(seeds=DEFAULT_SEEDS, updates=4000, evaluation_episodes=64,
               *, num_envs=NUM_ENVS, n_steps=NUM_STEPS, backend="sync"):
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct seeds are required")
    reports = []
    timings = []
    for seed in seeds:
        started = time.perf_counter()
        report = run(seed, updates, evaluation_episodes,
                     num_envs=num_envs, n_steps=n_steps, backend=backend)
        timings.append({"seed": seed, "seconds": round(
            time.perf_counter() - started, 3)})
        reports.append(report)
    rewards = [
        r["evaluation"]["trained_greedy"]["mean_reward"] for r in reports]
    return {
        "seeds": list(seeds),
        "summary": {
            "per_seed_mean_rewards": rewards,
            "mean_seed_reward": round(statistics.mean(rewards), 3),
            "sample_sd_seed_rewards": round(statistics.stdev(rewards), 3),
        },
        "runs": reports,
        "comparison_caveat": "training budget here is environment transitions, not the earlier 480 completed episodes; no equal-sample head-to-head claim",
    }, {
        "method": "wall-clock full run per seed, CPU sequential; excludes package installation; not byte-deterministic",
        "timings": timings,
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--updates", type=int, default=4000)
    cli.add_argument("--eval-episodes", type=int, default=64)
    cli.add_argument("--backend", choices=("sync", "async"), default="sync")
    cli.add_argument("--output", type=Path,
                     default=Path("build/p3-vector-a2c-results.json"))
    cli.add_argument("--timing-output", type=Path,
                     default=Path("build/p3-vector-a2c-cost.json"))
    args = cli.parse_args()
    report, cost = multi_seed(updates=args.updates,
                              evaluation_episodes=args.eval_episodes, backend=args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.timing_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    args.timing_output.write_text(
        json.dumps(cost, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    print("P3_VECTOR_A2C", json.dumps(report["summary"], sort_keys=True))
    print("P3_VECTOR_A2C_COST", json.dumps(cost["timings"], sort_keys=True))


if __name__ == "__main__":
    main()
