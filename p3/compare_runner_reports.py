"""Aggregate independent machine results; do not silently average mismatches."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from p3.protocol_repro import differences


def aggregate(items):
    items=list(items)
    if len(items)<2:
        raise ValueError("need at least two independent runs")
    base=items[0]["result"]
    records=[]
    for item in items:
        records.append({
            "cpu":item["machine"]["cpu_model"],
            "flags":item["machine"]["cpu_flags_sha256"],
            "torch":item["machine"]["torch"],
            "runner_image":item["runner_image"],
            "mean_reward":item["result"]["evaluation"]["trained_greedy"]["mean_reward"],
            "differences_from_first":differences(base,item["result"]),
        })
    return {
        "exact_match_across_runners":all(not r["differences_from_first"] for r in records),
        "records":records,
        "scope":"same Python/Gymnasium/Torch pinned versions, separate Ubuntu Actions runners; CPU details may differ",
    }


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("--folder",type=Path,required=True)
    cli.add_argument("--output",type=Path,default=Path("build/p3-cross-host-summary.json"))
    args=cli.parse_args()
    paths=sorted(args.folder.glob("**/result.json"))
    report=aggregate([json.loads(p.read_text()) for p in paths])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print("P3_HOST_COMPARISON",json.dumps(report,sort_keys=True),flush=True)
    if not report["exact_match_across_runners"]:
        raise AssertionError("Same seed differs across separate runners; see artifacts")


if __name__=="__main__":
    main()
