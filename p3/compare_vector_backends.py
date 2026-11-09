"""Exactly matched frame-budget test of the same n-step A2C in two vector backends.

This compares SyncVectorEnv with real spawning Gymnasium AsyncVectorEnv,
not two different algorithms or the unchanged historical 2018 SubprocVecEnv.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from p3.vector_nstep_a2c import run


def compare(seed=2026, updates=4000, evaluation_episodes=64):
    if seed < 0 or updates < 1 or evaluation_episodes < 1:
        raise ValueError("invalid parity budgets")
    results, elapsed = {}, {}
    for backend in ("sync", "async"):
        started = time.perf_counter()
        results[backend] = run(
            seed, updates, evaluation_episodes,
            num_envs=16, n_steps=5, backend=backend)
        elapsed[backend] = round(time.perf_counter() - started, 3)
    left, right = results["sync"].copy(), results["async"].copy()
    if left.pop("vector_backend") != "sync" or right.pop("vector_backend") != "async":
        raise AssertionError("unexpected vector-backend identity")
    if left != right:
        fields = [field for field in left if left[field] != right[field]]
        raise AssertionError("sync/async model training mismatch: " + repr(fields))
    frames = 16 * 5 * updates
    if results["sync"]["training_environment_steps"] != frames:
        raise AssertionError("inconsistent transition budget")
    return {
        "environment": "CartPole-v1",
        "seed": seed,
        "updates": updates,
        "transitions_per_backend": frames,
        "exact_training_and_evaluation_parity": True,
        "results": results,
        "scope": "same n-step algorithm and same environment transitions, not separate-algorithm ranking or original 2018 SubprocVecEnv execution",
    }, {
        "runner": "GitHub ubuntu-latest single CPU; sync first, async second; installation excluded",
        "cpu_count": os.cpu_count(),
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("build/p3-backend-parity.json"))
    parser.add_argument("--cost", type=Path, default=Path("build/p3-backend-timing.json"))
    args = parser.parse_args()
    result, cost = compare(args.seed, args.updates, args.eval_episodes)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.cost.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    args.cost.write_text(json.dumps(cost, sort_keys=True, indent=2) + "\n")
    print("P3_REAL_SUBPROCESS_PARITY", json.dumps({
        "seed": result["seed"],
        "transitions": result["transitions_per_backend"],
        "heldout_mean": result["results"]["async"]["evaluation"]["trained_greedy"]["mean_reward"],
        "equal": result["exact_training_and_evaluation_parity"],
        "times": cost["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
