"""Side-by-side reproducible CartPole-v1 comparison, with honest CPU costs.

Fixed environment reset seeds, episode budgets and checkpoint-validation/test
protocol. Model capacities and optimizers DIFFER, so this is not an isolation
of 'framework overhead' or a reproduction of historical vectorized A2C.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

from p3.benchmark import DEFAULT_SEEDS
from p3.cartpole_actor_critic import run as run_linear
from p3.torch_actor_critic import run_torch


def _stats(rows):
    vals = [float(r["evaluation"]["trained_greedy"]["mean_reward"]) for r in rows]
    return {
        "per_seed_mean_reward": vals,
        "mean_reward": round(statistics.mean(vals), 3),
        "sample_sd_reward": round(statistics.stdev(vals), 3),
    }


def compare(seeds=DEFAULT_SEEDS, train_episodes=480, eval_episodes=64):
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("requires at least two distinct training seeds")
    if train_episodes <= 0 or eval_episodes <= 0:
        raise ValueError("invalid episode budgets")
    linear, neural, timings = [], [], []
    for seed in seeds:
        t = time.perf_counter()
        linear_run = run_linear(seed, train_episodes, eval_episodes)
        linear_elapsed = time.perf_counter() - t
        t = time.perf_counter()
        neural_run = run_torch(seed, train_episodes, eval_episodes)
        neural_elapsed = time.perf_counter() - t
        assert linear_run["environment"] == neural_run["environment"]
        assert linear_run["evaluation"]["random"] == neural_run["evaluation"]["random"]
        linear.append(linear_run)
        neural.append(neural_run)
        timings.append({
            "seed": seed,
            "numpy_linear_seconds": round(linear_elapsed, 3),
            "pytorch_neural_seconds": round(neural_elapsed, 3),
        })

    # Timings are kept OUTSIDE the deterministic result, since wall clock
    # varies with CI machine load and must never be byte-compared as a KPI.
    results = {
        "environment": "CartPole-v1",
        "seeds": list(seeds),
        "training_episodes_per_seed": train_episodes,
        "held_out_episodes_per_seed": eval_episodes,
        "validation_episodes": 12,
        "checkpoint_interval": 40,
        "torch_model": {"architecture": "separate 5-32-2 actor and 5-32-1 critic",
                        "parameters": neural[0]["model_parameters"]},
        "linear_model": {"architecture": "5x2 linear actor and 5x1 linear critic",
                         "parameters": 15},
        "summary": {
            "numpy_linear": _stats(linear),
            "pytorch_neural": _stats(neural),
            "neural_minus_linear_by_seed": [
                round(n["evaluation"]["trained_greedy"]["mean_reward"] -
                      l["evaluation"]["trained_greedy"]["mean_reward"], 3)
                for l, n in zip(linear, neural)
            ],
        },
        "runs": {"numpy_linear": linear, "pytorch_neural": neural},
        "scope": "same Gymnasium env, seeds, budgets and held-out evaluation; different architectures/hyperparameters, no claim of causality or original deep A2C replication",
    }
    cost = {
        "environment": "GitHub Actions ubuntu-latest, single Python 3.11 CPU process",
        "python": platform.python_version(),
        "method": "time.perf_counter around full per-seed run including validation and held-out evaluation; non-reproducible timing, NumPy run always executed first; excludes dependency installation",
        "timings": timings,
    }
    return results, cost


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, default=Path("build/p3-model-comparison.json"))
    cli.add_argument("--timing-output", type=Path,
                     default=Path("build/p3-model-timing.json"))
    cli.add_argument("--train-episodes", type=int, default=480)
    cli.add_argument("--eval-episodes", type=int, default=64)
    args = cli.parse_args()
    result, cost = compare(train_episodes=args.train_episodes,
                           eval_episodes=args.eval_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.timing_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    args.timing_output.write_text(json.dumps(cost, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print("P3_MODEL_COMPARE", json.dumps(result["summary"], sort_keys=True))
    print("P3_MODEL_TIMING", json.dumps(cost["timings"], sort_keys=True))


if __name__ == "__main__":
    main()
