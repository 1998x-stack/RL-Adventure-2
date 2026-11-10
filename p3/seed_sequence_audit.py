"""Check if prior n-step training seeds pollute a later fixed seed.

Runs seed 2028 before and after training seeds 2026 and 2027 in the
same Python process and on the same CPU. The selected fixed-seed reports
are compared deeply, including validation trace and final evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from p3.protocol_repro import differences, hardware_fingerprint
from p3.vector_nstep_a2c import run


def audit(*, seed=2028, earlier_seeds=(2026, 2027),
          frames=320000, evaluation_episodes=64):
    if frames < 80 or frames % 80:
        raise ValueError("frame budget must be divisible by 80")
    earlier_seeds = tuple(earlier_seeds)
    if seed in earlier_seeds or len(set(earlier_seeds)) != len(earlier_seeds):
        raise ValueError("earlier seeds must be unique and distinct")
    def train(s):
        return run(s, updates=frames // 80, evaluation_episodes=evaluation_episodes,
                   num_envs=16, n_steps=5, backend="sync")

    before = train(seed)
    preceding = [{
        "seed": s,
        "heldout_mean": train(s)["evaluation"]["trained_greedy"]["mean_reward"]
    } for s in earlier_seeds]
    after = train(seed)
    return {
        "seed": seed,
        "preceding_seeds": list(earlier_seeds),
        "frames_per_seed": frames,
        "before": before,
        "after": after,
        "preceding_results": preceding,
        "equal": before == after,
        "difference_fields": differences(before, after),
        "hardware": hardware_fingerprint(),
        "scope": "same-runner prior n-step seed sequence, not cross-runner ISA comparison or combined preceding three-algorithm seed sequence",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=320000)
    p.add_argument("--output", type=Path, default=Path("build/p3-seed-sequence-audit.json"))
    args=p.parse_args()
    report=audit(frames=args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print("P3_SEED_SEQUENCE", json.dumps({
        "baseline": report["before"]["evaluation"]["trained_greedy"]["mean_reward"],
        "after_earlier_seeds": report["after"]["evaluation"]["trained_greedy"]["mean_reward"],
        "preceding": report["preceding_results"],
        "equal": report["equal"],
        "different_fields": report["difference_fields"],
        "cpu": report["hardware"]["cpu_model"],
    }, sort_keys=True), flush=True)
    if not report["equal"]:
        raise AssertionError("Previous n-step training seeds changed 2028; see artifact")


if __name__ == "__main__":
    main()
