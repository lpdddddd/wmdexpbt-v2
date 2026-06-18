#!/usr/bin/env python3
"""Analyze Strategy Zoo v17 Online Elite-AWR training curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metrics = run_dir / "dexpbt_8gpu_ws" / "metrics"
    if not metrics.is_dir():
        print(f"missing metrics dir: {metrics}")
        return
    awr_keys = (
        "awr_loss_weighted",
        "awr_buffer_steps",
        "awr_sidecar_num_steps",
        "awr_truncated_prefix_frac",
        "awr_starts_at_reset_frac",
        "awr_loss_cap_hit_frac",
        "awr_action_convention_checked",
    )
    for path in sorted(metrics.glob("training_curve_policy_*.jsonl")):
        print(f"\n=== {path.name} ===")
        last = None
        for line in path.read_text().splitlines():
            if line.strip():
                last = json.loads(line)
        if not last:
            print("(empty)")
            continue
        for k in awr_keys:
            if k in last:
                print(f"  {k}: {last[k]}")


if __name__ == "__main__":
    main()
