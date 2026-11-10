"""Cross-protocol n-step reproducibility audit, same runner and exact model.

Earlier standalone three-seed CI logged seed 2028 mean=466.641; the frame-
matched joint CI logged 482.297. Test both invocation paths within ONE job
before attributing the discrepancy to RNG, hardware, or code revisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch

from p3.vector_nstep_a2c import run as nstep_run
from p3.equal_frame_comparison import compare as joint_compare


SEED = 2028


def hardware_fingerprint():
    cpu_model = "unknown"
    cpu_flags = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
            if line.startswith("flags"):
                cpu_flags = line.split(":", 1)[1].strip()
            if cpu_model != "unknown" and cpu_flags:
                break
    except OSError:
        pass
    return {
        "cpu_model": cpu_model,
        "cpu_flags_sha256": hashlib.sha256(cpu_flags.encode()).hexdigest()[:16],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "runner_os": os.environ.get("RUNNER_OS", "local"),
    }


def differences(left, right):
    return [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]


def audit(seed=SEED, frames=320000, evaluation_episodes=64):
    if seed < 0 or frames < 80 or frames % 80 or evaluation_episodes < 1:
        raise ValueError("exact frame budget requires a positive multiple of 80")
    identity = hardware_fingerprint()
    standalone = nstep_run(
        seed=seed, updates=frames // 80,
        evaluation_episodes=evaluation_episodes,
        num_envs=16, n_steps=5, backend="sync",
    )
    # Exactly the joint three-algorithm code path; only the precommitted
    # seed list is reduced to 2028 so the invocation difference is isolated.
    joint, _ = joint_compare(
        seeds=(seed,), frames=frames, eval_episodes=evaluation_episodes)
    included = joint["runs"]["pytorch_nstep"][0]
    return {
        "seed": seed,
        "frames": frames,
        "standalone_nstep": standalone,
        "joint_nstep": included,
        "equal_exact_json": standalone == included,
        "different_fields": differences(standalone, included),
        "hardware": identity,
        "scope": "same GitHub runner, exact training and evaluation protocol, independent vs joint invocation; does not establish cross-machine reproducibility",
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--frames", type=int, default=320000)
    cli.add_argument("--eval-episodes", type=int, default=64)
    cli.add_argument("--output", type=Path, default=Path("build/p3-protocol-repro.json"))
    args = cli.parse_args()
    report = audit(frames=args.frames, evaluation_episodes=args.eval_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("P3_PROTOCOL_REPRO", json.dumps({
        "seed": report["seed"],
        "frames": report["frames"],
        "standalone_reward": report["standalone_nstep"]["evaluation"]["trained_greedy"]["mean_reward"],
        "joint_reward": report["joint_nstep"]["evaluation"]["trained_greedy"]["mean_reward"],
        "equal": report["equal_exact_json"],
        "difference": report["different_fields"],
        "cpu": report["hardware"]["cpu_model"],
    }, sort_keys=True), flush=True)
    if not report["equal_exact_json"]:
        raise AssertionError(
            "Same runner, different invocation path changed n-step result; "
            "raw diagnostics preserved as CI artifact")


if __name__ == "__main__":
    main()
