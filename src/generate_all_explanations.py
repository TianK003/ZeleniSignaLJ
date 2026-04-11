"""
Zeleni SignaLJ - Batch Explanation Generator
============================================
Scans results/experiments/ for experiments that have a trained model
but no explanations/ folder, then runs collect_states.py + explain.py
for each one.

Usage:
    # Dry run — see what would be processed
    python src/generate_all_explanations.py --dry_run

    # Run locally with 4 workers
    python src/generate_all_explanations.py --num_workers 4

    # Run on HPC with 16 parallel SUMO instances
    python src/generate_all_explanations.py --num_workers 16 --episodes 12

    # Force regeneration (even if explanations/ exists)
    python src/generate_all_explanations.py --force
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


EXPERIMENTS_DIR = "results/experiments"
MODEL_FILE = "ppo_shared_policy.zip"
META_FILE = "meta.json"
HARVEST_FILE = "harvested_data.pkl"
EXPLANATIONS_DIR = "explanations"


def discover_experiments(experiments_dir, force=False):
    """Scan experiments dir and return list of dicts for experiments needing explanations.

    Each dict has: run_id, run_dir, model_path, scenario, needs_harvest, needs_explain.
    """
    if not os.path.isdir(experiments_dir):
        print(f"ERROR: {experiments_dir} does not exist.")
        return []

    results = []
    for entry in sorted(os.listdir(experiments_dir)):
        run_dir = os.path.join(experiments_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        model_path = os.path.join(run_dir, MODEL_FILE)
        if not os.path.isfile(model_path):
            continue  # No trained model — skip

        has_harvest = os.path.isfile(os.path.join(run_dir, HARVEST_FILE))
        has_explain = os.path.isdir(os.path.join(run_dir, EXPLANATIONS_DIR))

        if has_explain and not force:
            continue  # Already done

        # Read scenario from meta.json
        meta_path = os.path.join(run_dir, META_FILE)
        scenario = "uniform"
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                scenario = meta.get("scenario", "uniform")
            except Exception:
                pass

        results.append({
            "run_id": entry,
            "run_dir": run_dir,
            "model_path": model_path,
            "scenario": scenario,
            "needs_harvest": not has_harvest or force,
            "needs_explain": True,
        })

    return results


def process_experiment(exp, episodes, python_exe):
    """Run collect_states + explain for one experiment. Returns (run_id, success, message)."""
    run_id = exp["run_id"]
    t0 = time.time()

    # Step 1: Harvest states (if needed)
    if exp["needs_harvest"]:
        cmd = [
            python_exe, "src/collect_states.py",
            "--model_path", exp["model_path"],
            "--scenario", exp["scenario"],
            "--episodes", str(episodes),
            "--output_dir", exp["run_dir"],
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            return (run_id, False,
                    f"collect_states failed: {result.stderr[-500:]}")

        pkl_path = os.path.join(exp["run_dir"], HARVEST_FILE)
        if not os.path.isfile(pkl_path):
            return (run_id, False,
                    f"collect_states produced no {HARVEST_FILE}")

    # Step 2: Generate explanations
    pkl_path = os.path.join(exp["run_dir"], HARVEST_FILE)
    cmd = [
        python_exe, "src/explain.py",
        "--data_path", pkl_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        return (run_id, False,
                f"explain.py failed: {result.stderr[-500:]}")

    elapsed = time.time() - t0
    return (run_id, True, f"done in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate explanations for all experiments missing them."
    )
    parser.add_argument("--experiments_dir", type=str, default=EXPERIMENTS_DIR,
                        help=f"Path to experiments directory (default: {EXPERIMENTS_DIR})")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel workers (each runs its own SUMO instance)")
    parser.add_argument("--episodes", type=int, default=12,
                        help="Episodes to harvest per experiment (default: 12)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if explanations/ already exists")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only list what would be processed, don't run anything")
    args = parser.parse_args()

    experiments = discover_experiments(args.experiments_dir, force=args.force)

    if not experiments:
        print("Nothing to process — all experiments with models already have explanations.")
        return

    needs_harvest = sum(1 for e in experiments if e["needs_harvest"])
    print(f"Found {len(experiments)} experiments to process "
          f"({needs_harvest} need harvesting, {len(experiments)} need explanations)")
    print()

    for i, exp in enumerate(experiments, 1):
        status = []
        if exp["needs_harvest"]:
            status.append("harvest")
        status.append("explain")
        print(f"  [{i:2d}] {exp['run_id']}  scenario={exp['scenario']}  "
              f"steps: {'+'.join(status)}")

    if args.dry_run:
        print(f"\nDry run — nothing executed. Use without --dry_run to process.")
        return

    print(f"\nProcessing with {min(args.num_workers, len(experiments))} workers, "
          f"{args.episodes} episodes each...")
    print("=" * 70)

    python_exe = sys.executable
    effective_workers = min(args.num_workers, len(experiments))
    completed = 0
    failed = 0
    t_start = time.time()

    if effective_workers <= 1:
        for exp in experiments:
            run_id, success, msg = process_experiment(exp, args.episodes, python_exe)
            completed += 1
            if success:
                print(f"  [{completed}/{len(experiments)}] OK  {run_id} — {msg}")
            else:
                failed += 1
                print(f"  [{completed}/{len(experiments)}] FAIL {run_id} — {msg}")
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(process_experiment, exp, args.episodes, python_exe): exp
                for exp in experiments
            }
            for future in as_completed(futures):
                run_id, success, msg = future.result()
                completed += 1
                if success:
                    print(f"  [{completed}/{len(experiments)}] OK  {run_id} — {msg}")
                else:
                    failed += 1
                    print(f"  [{completed}/{len(experiments)}] FAIL {run_id} — {msg}")

    elapsed = time.time() - t_start
    print("=" * 70)
    print(f"Done: {completed - failed}/{len(experiments)} succeeded, "
          f"{failed} failed, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
