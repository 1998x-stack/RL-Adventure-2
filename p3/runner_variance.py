"""Independent-host seed-2028 n-step replay with machine fingerprints."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from p3.vector_nstep_a2c import run
from p3.protocol_repro import hardware_fingerprint


def execute(seed=2028,frames=320000,episodes=64):
    if frames < 80 or frames % 80:
        raise ValueError("frames must divide into complete n-step updates")
    result=run(seed=seed,updates=frames//80,evaluation_episodes=episodes,
               num_envs=16,n_steps=5,backend="sync")
    return {
        "result":result,
        "machine":hardware_fingerprint(),
        "runner_image":os.environ.get("ImageVersion","unknown"),
        "runner_name":os.environ.get("RUNNER_NAME","unknown"),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("build/host-result.json"))
    a=p.parse_args()
    result=execute()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("P3_INDEPENDENT_HOST",json.dumps({
        "mean":result["result"]["evaluation"]["trained_greedy"]["mean_reward"],
        "cpu":result["machine"]["cpu_model"],
        "runner_image":result["runner_image"],
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
