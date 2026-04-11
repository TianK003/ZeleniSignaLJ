"""
Zeleni SignaLJ - Batch Explanation Generator
============================================
Scans results/experiments/ for experiments that have a trained model
but no explanations/ folder, then runs collect_states.py + explain.py
for each one.

Scenario detection: inferred from folder name first ("morningrush" ->
morning_rush, "eveningrush" -> evening_rush), falls back to meta.json.

Usage:
    # Dry run — see what would be processed
    python src/generate_all_explanations.py --dry_run

    # Run locally with 4 workers
    python src/generate_all_explanations.py --num_workers 4

    # Run on HPC with 16 parallel SUMO instances
    python src/generate_all_explanations.py --num_workers 16 --episodes 15

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

VALID_SCENARIOS = {"morning_rush", "evening_rush", "offpeak", "uniform"}


def detect_scenario(run_id, run_dir):
    """Detect scenario from folder name, falling back to meta.json.

    Folder name patterns:
        *_morningrush_*  ->  morning_rush
        *_eveningrush_*  ->  evening_rush
    """
    # Primary: folder name
    name_lower = run_id.lower()
    if "morningrush" in name_lower or "morning_rush" in name_lower:
        return "morning_rush"
    if "eveningrush" in name_lower or "evening_rush" in name_lower:
        return "evening_rush"
    if "offpeak" in name_lower:
        return "offpeak"

    # Fallback: meta.json
    meta_path = os.path.join(run_dir, META_FILE)
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            scenario = meta.get("scenario", "")
            if scenario in VALID_SCENARIOS:
                return scenario
        except Exception:
            pass

    return "uniform"


def discover_experiments(experiments_dir, force=False):
    """Scan experiments dir and return list of dicts for experiments needing explanations."""
    if not os.path.isdir(experiments_dir):
        print(f"ERROR: {experiments_dir} does not exist.")
        return []

    results = []
    skipped_no_model = 0
    skipped_done = 0

    for entry in sorted(os.listdir(experiments_dir)):
        run_dir = os.path.join(experiments_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        model_path = os.path.join(run_dir, MODEL_FILE)
        if not os.path.isfile(model_path):
            skipped_no_model += 1
            continue

        has_harvest = os.path.isfile(os.path.join(run_dir, HARVEST_FILE))
        has_explain = os.path.isdir(os.path.join(run_dir, EXPLANATIONS_DIR))

        if has_explain and not force:
            skipped_done += 1
            continue

        scenario = detect_scenario(entry, run_dir)

        results.append({
            "run_id": entry,
            "run_dir": run_dir,
            "model_path": model_path,
            "scenario": scenario,
            "needs_harvest": not has_harvest or force,
            "needs_explain": True,
        })

    if skipped_no_model:
        print(f"Skipped {skipped_no_model} folder(s) without {MODEL_FILE}")
    if skipped_done:
        print(f"Skipped {skipped_done} folder(s) that already have {EXPLANATIONS_DIR}/")

    return results


def process_experiment(exp, episodes, python_exe, log_dir):
    """Run collect_states + explain for one experiment.

    Subprocess output is written to per-experiment log files for debugging.
    Returns (run_id, success, message).
    """
    run_id = exp["run_id"]
    t0 = time.time()

    log_file = os.path.join(log_dir, f"{run_id}.log")

    with open(log_file, "w") as lf:
        lf.write(f"=== {run_id} ===\n")
        lf.write(f"scenario: {exp['scenario']}\n")
        lf.write(f"model: {exp['model_path']}\n")
        lf.write(f"needs_harvest: {exp['needs_harvest']}\n\n")

        # Step 1: Harvest states
        if exp["needs_harvest"]:
            cmd = [
                python_exe, "src/collect_states.py",
                "--model_path", exp["model_path"],
                "--scenario", exp["scenario"],
                "--episodes", str(episodes),
                "--output_dir", exp["run_dir"],
            ]
            lf.write(f"[HARVEST] {' '.join(cmd)}\n")
            lf.flush()

            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    text=True, timeout=7200)

            if result.returncode != 0:
                msg = f"collect_states FAILED (exit code {result.returncode}), see {log_file}"
                lf.write(f"\n[HARVEST] {msg}\n")
                return (run_id, False, msg)

            pkl_path = os.path.join(exp["run_dir"], HARVEST_FILE)
            if not os.path.isfile(pkl_path):
                msg = f"collect_states ran but no {HARVEST_FILE} produced, see {log_file}"
                lf.write(f"\n[HARVEST] {msg}\n")
                return (run_id, False, msg)

            lf.write(f"\n[HARVEST] OK — {HARVEST_FILE} created\n\n")

        # Step 2: Generate explanations
        pkl_path = os.path.join(exp["run_dir"], HARVEST_FILE)
        if not os.path.isfile(pkl_path):
            msg = f"no {HARVEST_FILE} to explain (harvest was skipped but pkl missing)"
            lf.write(f"[EXPLAIN] {msg}\n")
            return (run_id, False, msg)

        cmd = [
            python_exe, "src/explain.py",
            "--data_path", pkl_path,
        ]
        lf.write(f"[EXPLAIN] {' '.join(cmd)}\n")
        lf.flush()

        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                text=True, timeout=3600)

        if result.returncode != 0:
            msg = f"explain.py FAILED (exit code {result.returncode}), see {log_file}"
            lf.write(f"\n[EXPLAIN] {msg}\n")
            return (run_id, False, msg)

        elapsed = time.time() - t0
        lf.write(f"\n[DONE] {run_id} completed in {elapsed:.0f}s\n")
        return (run_id, True, f"done in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Generate explanations for all experiments missing them."
    )
    parser.add_argument("--experiments_dir", type=str, default=EXPERIMENTS_DIR,
                        help=f"Path to experiments directory (default: {EXPERIMENTS_DIR})")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel workers (each runs its own SUMO instance)")
    parser.add_argument("--episodes", type=int, default=15,
                        help="Episodes to harvest per experiment (default: 15)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if explanations/ already exists")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only list what would be processed, don't run anything")
    parser.add_argument("--log_dir", type=str, default="logs/explanations",
                        help="Directory for per-experiment log files")
    args = parser.parse_args()

    print(f"Scanning {args.experiments_dir}/ ...")
    experiments = discover_experiments(args.experiments_dir, force=args.force)

    if not experiments:
        print("Nothing to process — all experiments with models already have explanations.")
        return

    needs_harvest = sum(1 for e in experiments if e["needs_harvest"])
    print(f"\nFound {len(experiments)} experiments to process "
          f"({needs_harvest} need harvesting, {len(experiments)} need explain.py)")
    print()

    for i, exp in enumerate(experiments, 1):
        status = []
        if exp["needs_harvest"]:
            status.append("harvest")
        status.append("explain")
        print(f"  [{i:2d}] {exp['run_id']}")
        print(f"       scenario={exp['scenario']}  steps={'+'.join(status)}")

    if args.dry_run:
        print(f"\nDry run — nothing executed. Remove --dry_run to process.")
        return

    # Verify route files exist before starting
    needed_routes = set()
    route_map = {
        "morning_rush": "data/routes/routes_morning_rush.rou.xml",
        "evening_rush": "data/routes/routes_evening_rush.rou.xml",
        "offpeak": "data/routes/routes_offpeak.rou.xml",
        "uniform": "data/routes/routes.rou.xml",
    }
    for exp in experiments:
        route = route_map.get(exp["scenario"])
        if route:
            needed_routes.add(route)

    missing_routes = [r for r in needed_routes if not os.path.isfile(r)]
    if missing_routes:
        print(f"\nERROR: Required route files are missing:")
        for r in missing_routes:
            print(f"  {r}")
        print(f"\nGenerate them first:")
        print(f"  python src/generate_demand.py --scenario all")
        sys.exit(1)

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    effective_workers = min(args.num_workers, len(experiments))
    print(f"\nProcessing with {effective_workers} workers, "
          f"{args.episodes} episodes each...")
    print(f"Per-experiment logs: {args.log_dir}/")
    print("=" * 70)

    python_exe = sys.executable
    completed = 0
    failed = 0
    failed_list = []
    t_start = time.time()

    if effective_workers <= 1:
        for exp in experiments:
            run_id, success, msg = process_experiment(
                exp, args.episodes, python_exe, args.log_dir)
            completed += 1
            if success:
                print(f"  [{completed}/{len(experiments)}] OK   {run_id} — {msg}")
            else:
                failed += 1
                failed_list.append(run_id)
                print(f"  [{completed}/{len(experiments)}] FAIL {run_id} — {msg}")
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(process_experiment, exp, args.episodes,
                                python_exe, args.log_dir): exp
                for exp in experiments
            }
            for future in as_completed(futures):
                run_id, success, msg = future.result()
                completed += 1
                if success:
                    print(f"  [{completed}/{len(experiments)}] OK   {run_id} — {msg}")
                else:
                    failed += 1
                    failed_list.append(run_id)
                    print(f"  [{completed}/{len(experiments)}] FAIL {run_id} — {msg}")

    elapsed = time.time() - t_start
    print("=" * 70)
    print(f"Done: {completed - failed}/{len(experiments)} succeeded, "
          f"{failed} failed, {elapsed:.0f}s total")

    if failed_list:
        print(f"\nFailed experiments (check logs in {args.log_dir}/):")
        for rid in failed_list:
            print(f"  - {rid}")
            log_path = os.path.join(args.log_dir, f"{rid}.log")
            if os.path.isfile(log_path):
                # Print last 5 lines of the log for quick diagnosis
                with open(log_path) as f:
                    lines = f.readlines()
                tail = lines[-5:] if len(lines) > 5 else lines
                for line in tail:
                    print(f"    {line.rstrip()}")


if __name__ == "__main__":
    main()
