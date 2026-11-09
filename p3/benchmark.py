"""Three precommitted independent seeds; descriptive, not significance testing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from p3.cartpole_actor_critic import run


DEFAULT_SEEDS = (2026, 2027, 2028)


def summarize(results: list[dict]) -> dict:
    if not results:
        raise ValueError("expected at least one trained run")
    keys = ("trained_greedy", "initial_greedy", "random")
    metrics = {}
    for key in keys:
        values = [float(r["evaluation"][key]["mean_reward"]) for r in results]
        metrics[key] = {
            "mean_of_seed_means": round(statistics.mean(values), 3),
            "sample_sd_of_seed_means": round(statistics.stdev(values), 3)
            if len(values) > 1 else None,
            "individual_seed_means": values,
        }
    metrics["trained_minus_random"] = {
        "individual_seed_gaps": [
            round(r["evaluation"]["trained_greedy"]["mean_reward"]
                  - r["evaluation"]["random"]["mean_reward"], 3)
            for r in results
        ]
    }
    metrics["trained_minus_random"]["mean_gap"] = round(statistics.mean(
        metrics["trained_minus_random"]["individual_seed_gaps"]), 3)
    return metrics


def benchmark(seeds=DEFAULT_SEEDS, train_episodes=480, eval_episodes=64):
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least two unique predefined training seeds")
    runs = [run(seed, train_episodes, eval_episodes) for seed in seeds]
    return {
        "method": "independent fixed seeds with held-out test episodes; descriptive statistics",
        "seeds": list(seeds),
        "training_episodes_per_seed": train_episodes,
        "held_out_episodes_per_seed": eval_episodes,
        "summary": summarize(runs),
        "runs": runs,
        "scope": "NumPy linear actor-critic, not full deep A2C; three seeds cannot establish general statistical significance",
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    result = benchmark()
    blob = json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(blob, encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
