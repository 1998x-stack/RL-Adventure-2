"""Replay the exact historical 3-seed, 3-algorithm call order on ONE runner.

The earlier isolated n-step CI gave seed 2028 = 466.641; combined full
3-seed equal-frame CI gave 482.297. Previous audits only tested one joint
seed or a sequence of n-step seeds. This tests the actual complete call order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from p3.vector_nstep_a2c import run as run_nstep
from p3.equal_frame_comparison import compare
from p3.protocol_repro import hardware_fingerprint, differences


SEEDS=(2026,2027,2028)


def audit(frames=320000, evaluation_episodes=64):
    if frames < 80 or frames % 80 or evaluation_episodes < 1:
        raise ValueError("invalid frame or evaluation budget")

    def standalone():
        return run_nstep(seed=2028, updates=frames // 80,
                         evaluation_episodes=evaluation_episodes,
                         num_envs=16, n_steps=5, backend="sync")

    before=standalone()
    joint,_=compare(seeds=SEEDS, frames=frames, eval_episodes=evaluation_episodes)
    embedded=joint["runs"]["pytorch_nstep"][2]
    after=standalone()
    return {
        "seeds":list(SEEDS),
        "frames_per_model_seed":frames,
        "before":before,
        "embedded":embedded,
        "after":after,
        "joint_summary":joint["summary"],
        "same_initial_vs_joint":before==embedded,
        "same_initial_vs_after":before==after,
        "different_initial_vs_joint":differences(before, embedded),
        "different_initial_vs_after":differences(before, after),
        "hardware":hardware_fingerprint(),
        "scope":"the exact full joint 3-algorithm × 3-seed sequence on a single Runner, compared to standalone seed 2028 before and after",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--frames",type=int,default=320000)
    p.add_argument("--eval-episodes",type=int,default=64)
    p.add_argument("--output",type=Path,default=Path("build/p3-full-order-audit.json"))
    args=p.parse_args()
    d=audit(args.frames,args.eval_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
    print("P3_FULL_ORDER_AUDIT",json.dumps({
        "standalone_before":d["before"]["evaluation"]["trained_greedy"]["mean_reward"],
        "embedded_joint":d["embedded"]["evaluation"]["trained_greedy"]["mean_reward"],
        "standalone_after":d["after"]["evaluation"]["trained_greedy"]["mean_reward"],
        "same_initial_vs_joint":d["same_initial_vs_joint"],
        "same_initial_vs_after":d["same_initial_vs_after"],
        "diff_joint":d["different_initial_vs_joint"],
        "cpu":d["hardware"]["cpu_model"],
    },sort_keys=True),flush=True)
    if not d["same_initial_vs_joint"] or not d["same_initial_vs_after"]:
        raise AssertionError("Full joint sequence changed n-step seed 2028 on one runner")


if __name__=="__main__":
    main()
