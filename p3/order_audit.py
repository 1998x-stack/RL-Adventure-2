"""Audit whether earlier training of another algorithm changes n-step results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from p3.equal_frame_comparison import run_episodic
from p3.vector_nstep_a2c import run as run_nstep


def audit(seed=2028, frames=320000, eval_episodes=64):
    if frames % 80:
        raise ValueError("frames must align to n-step updates")
    baseline = run_nstep(seed=seed, updates=frames // 80,
                         evaluation_episodes=eval_episodes, backend="sync")
    # Explicitly introduce a DIFFERENT PyTorch model training call between
    # two n-step runs. Both n-step invocations independently set torch seed.
    unrelated = run_episodic(seed=seed, frames=frames,
                             evaluation_episodes=eval_episodes, neural=True)
    after = run_nstep(seed=seed, updates=frames // 80,
                      evaluation_episodes=eval_episodes, backend="sync")
    return {
        "seed": seed,
        "frames": frames,
        "nstep_before": baseline,
        "other_model_train_frames": unrelated["training_environment_steps"],
        "nstep_after": after,
        "identical_nstep_json": baseline == after,
        "differences": [k for k in baseline if baseline[k] != after[k]],
        "scope": "cross-model in-process determinism; same Gymnasium and PyTorch CPU versions",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("build/p3-rng-order-audit.json"))
    args = p.parse_args()
    report = audit()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("P3_ORDER_AUDIT", json.dumps({
        "seed": report["seed"],
        "before": report["nstep_before"]["evaluation"]["trained_greedy"]["mean_reward"],
        "after": report["nstep_after"]["evaluation"]["trained_greedy"]["mean_reward"],
        "equal": report["identical_nstep_json"],
        "differences": report["differences"],
    }, sort_keys=True))
    if not report["identical_nstep_json"]:
        raise AssertionError("Other model execution changed seeded n-step results")


if __name__ == "__main__":
    main()
