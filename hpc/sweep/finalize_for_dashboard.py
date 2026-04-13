"""Finalize an evaluated experiment for dashboard.py ingestion.

Reads:
  - <baseline_csv>   : cached fixed-time baseline (per-intersection rewards in JSON column)
  - <rl_csv>         : RL-only eval result for one experiment (same format)
  - <experiment_dir> : path to results/experiments/<run_id>/

Writes:
  - <experiment_dir>/results.csv   (per-intersection: intersection, tls_id,
                                    baseline_reward, rl_reward, improvement_pct)
  - <experiment_dir>/meta.json     (adds baseline_total_reward, rl_total_reward,
                                    improvement_pct fields)

Usage:
  python hpc/sweep/finalize_for_dashboard.py \
      --baseline_csv results/baseline_morning_rush.csv \
      --rl_csv       results/experiments/<run_id>/eval_rl.csv \
      --experiment_dir results/experiments/<run_id>
"""
import argparse
import json
import os
import sys

import pandas as pd

# Add src/ to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from config import TS_IDS, TS_NAMES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_csv", required=True)
    p.add_argument("--rl_csv", required=True)
    p.add_argument("--experiment_dir", required=True)
    return p.parse_args()


def load_per_ts(csv_path, controller):
    """Load per-intersection rewards dict from one CSV row matching `controller`."""
    df = pd.read_csv(csv_path)
    matching = df[df["controller"] == controller]
    if matching.empty:
        raise ValueError(f"No '{controller}' row in {csv_path}")
    raw = matching.iloc[0]["per_ts_rewards_json"]
    return json.loads(raw)


def main():
    args = parse_args()

    # Load per-intersection data
    bl_per_ts = load_per_ts(args.baseline_csv, "fixed_time")
    rl_per_ts = load_per_ts(args.rl_csv, "rl")

    # Build dashboard-format rows (one per target TLS)
    rows = []
    for ts_id in TS_IDS:
        name = TS_NAMES.get(ts_id, ts_id)
        bl = float(bl_per_ts.get(ts_id, 0.0))
        rl = float(rl_per_ts.get(ts_id, 0.0))
        pct = ((rl - bl) / abs(bl) * 100) if bl != 0 else 0.0
        rows.append({
            "intersection": name,
            "tls_id": ts_id,
            "baseline_reward": bl,
            "rl_reward": rl,
            "improvement_pct": pct,
        })

    df_out = pd.DataFrame(rows)
    results_path = os.path.join(args.experiment_dir, "results.csv")
    df_out.to_csv(results_path, index=False)
    print(f"  Wrote {results_path}")

    # Update meta.json with totals (dashboard requires baseline_total_reward present)
    bl_total = sum(r["baseline_reward"] for r in rows)
    rl_total = sum(r["rl_reward"] for r in rows)
    total_pct = ((rl_total - bl_total) / abs(bl_total) * 100) if bl_total != 0 else 0.0

    meta_path = os.path.join(args.experiment_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["baseline_total_reward"] = bl_total
    meta["rl_total_reward"] = rl_total
    meta["improvement_pct"] = total_pct
    meta["eval_baseline_csv"] = args.baseline_csv  # provenance

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Updated {meta_path} (improvement: {total_pct:+.2f}%)")


if __name__ == "__main__":
    main()
