"""
Zeleni SignaLJ - Supplement Missing Evaluation Results
=======================================================
HPC experiments completed training (ppo_shared_policy.zip exists) but
crashed before evaluation finished (no results.csv). This script runs
baseline + RL evaluation locally for all incomplete experiments and
fills in the missing results.csv and meta.json fields so the dashboard
can display them.

Baselines are cached per (route_file, num_seconds) tuple so experiments
sharing the same scenario only run the baseline once.

Usage:
    python src/supplement_missing_results.py
    python src/supplement_missing_results.py --num-workers 6
    python src/supplement_missing_results.py --dry-run
    python src/supplement_missing_results.py --filter morningrush
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from stable_baselines3 import PPO

from config import TS_IDS, TS_NAMES, WARMUP_SECONDS
import experiment
from experiment import (
    run_baseline,
    run_evaluation,
    SCENARIO_PRESETS,
    TimeEncodedObservationFunction,
)

EXPERIMENTS_DIR = "results/experiments"


def find_incomplete_experiments(name_filter=None):
    """Find experiments that have a trained model but no evaluation results."""
    incomplete = []
    for run_id in sorted(os.listdir(EXPERIMENTS_DIR)):
        run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
        meta_path = os.path.join(run_dir, "meta.json")
        model_path = os.path.join(run_dir, "ppo_shared_policy.zip")
        results_path = os.path.join(run_dir, "results.csv")

        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Already complete
        if os.path.exists(results_path) and meta.get("baseline_total_reward") is not None:
            continue

        if name_filter and name_filter not in run_id:
            continue

        incomplete.append((run_id, run_dir, meta))

    return incomplete


def baseline_cache_key(meta):
    """Cache key for baseline results — same route + duration = same baseline."""
    return f"{meta.get('route_file')}|{meta.get('num_seconds')}"


def compute_baselines(experiments):
    """Run all unique baselines and return a serializable cache dict."""
    seen = {}
    for _run_id, _run_dir, meta in experiments:
        key = baseline_cache_key(meta)
        if key not in seen:
            seen[key] = meta

    cache = {}
    for key, meta in seen.items():
        net_file = meta.get("net_file", "data/networks/ljubljana.net.xml")
        route_file = meta.get("route_file")
        num_seconds = meta.get("num_seconds")
        scenario = meta.get("scenario", "uniform")
        preset = SCENARIO_PRESETS.get(scenario, {})
        start_hour = preset.get("start_hour", 0.0)

        if not os.path.exists(route_file):
            print(f"  SKIP baseline: route file missing: {route_file}")
            continue

        experiment.CURRENT_HOUR = start_hour

        print(f"  Running baseline for {scenario} "
              f"({num_seconds}s sim)...", end="", flush=True)
        t0 = time.time()
        rewards, steps = run_baseline(net_file, route_file, num_seconds)
        total = sum(rewards.values())
        print(f" reward={total:.0f} ({time.time() - t0:.0f}s)")

        cache[key] = {
            "start_hour": start_hour,
            "rewards": rewards,
        }

    return cache


def save_experiment_results(run_dir, meta, baseline_rewards, rl_rewards):
    """Write results.csv and update meta.json for one experiment."""
    bl_total = sum(baseline_rewards.values())
    rl_total = sum(rl_rewards.values())
    total_pct = ((rl_total - bl_total) / abs(bl_total) * 100) if bl_total != 0 else 0

    rows = []
    for ts_id in TS_IDS:
        name = TS_NAMES.get(ts_id, ts_id)
        bl = baseline_rewards.get(ts_id, 0)
        rl = rl_rewards.get(ts_id, 0)
        pct = ((rl - bl) / abs(bl) * 100) if bl != 0 else 0
        rows.append({
            "intersection": name,
            "tls_id": ts_id,
            "baseline_reward": bl,
            "rl_reward": rl,
            "improvement_pct": pct,
        })

    results_path = os.path.join(run_dir, "results.csv")
    pd.DataFrame(rows).to_csv(results_path, index=False)

    meta["baseline_total_reward"] = bl_total
    meta["rl_total_reward"] = rl_total
    meta["improvement_pct"] = total_pct
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return total_pct


# ── Sequential evaluation (single process, --num-workers 1) ──────────────


def _format_eta(elapsed, done, total):
    """Compute ETA string from elapsed time and progress."""
    if done == 0:
        return "estimating..."
    avg = elapsed / done
    remaining = avg * (total - done)
    if remaining < 60:
        return f"{remaining:.0f}s"
    return f"{remaining / 60:.0f}min"


def run_sequential(experiments, baseline_cache):
    """Evaluate all experiments in the current process."""
    success = 0
    failed = 0
    total = len(experiments)
    t_start = time.time()

    for i, (run_id, run_dir, meta) in enumerate(experiments):
        eta = _format_eta(time.time() - t_start, i, total)
        print(f"\n[{i + 1}/{total}] {run_id}  (ETA: {eta})")

        key = baseline_cache_key(meta)
        if key not in baseline_cache:
            print(f"    SKIP: no cached baseline for {meta.get('scenario')}")
            failed += 1
            continue

        baseline_rewards = baseline_cache[key]["rewards"]
        start_hour = baseline_cache[key]["start_hour"]
        experiment.CURRENT_HOUR = start_hour

        net_file = meta.get("net_file", "data/networks/ljubljana.net.xml")
        route_file = meta.get("route_file")
        num_seconds = meta.get("num_seconds")

        try:
            model = PPO.load(os.path.join(run_dir, "ppo_shared_policy.zip"))

            print(f"    Running RL evaluation...", end="", flush=True)
            t0 = time.time()
            rl_rewards, _ = run_evaluation(net_file, route_file, num_seconds, model)
            rl_total = sum(rl_rewards.values())
            elapsed = time.time() - t0
            print(f" reward={rl_total:.0f} ({elapsed:.0f}s)")

            pct = save_experiment_results(run_dir, meta, baseline_rewards, rl_rewards)
            print(f"    Result: {pct:+.1f}% improvement")
            success += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

    return success, failed


# ── Parallel evaluation (subprocess workers, --num-workers > 1) ──────────


def _worker_eval_single(run_dir, baseline_cache_path):
    """
    Called as a subprocess: evaluate one experiment and save results.
    Entry point: --_eval-single run_dir baseline_cache.json
    """
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    with open(baseline_cache_path) as f:
        baseline_cache = json.load(f)

    scenario = meta.get("scenario", "uniform")
    key = baseline_cache_key(meta)
    if key not in baseline_cache:
        print(f"ERROR: no cached baseline for key={key}", file=sys.stderr)
        sys.exit(1)

    entry = baseline_cache[key]
    baseline_rewards = entry["rewards"]
    start_hour = entry["start_hour"]
    experiment.CURRENT_HOUR = start_hour

    net_file = meta.get("net_file", "data/networks/ljubljana.net.xml")
    route_file = meta.get("route_file")
    num_seconds = meta.get("num_seconds")

    model = PPO.load(os.path.join(run_dir, "ppo_shared_policy.zip"))
    rl_rewards, _ = run_evaluation(net_file, route_file, num_seconds, model)

    pct = save_experiment_results(run_dir, meta, baseline_rewards, rl_rewards)
    rl_total = sum(rl_rewards.values())
    print(f"{pct:+.1f}%  reward={rl_total:.0f}")


def _launch_worker(run_dir, baseline_cache_path):
    """Launch a subprocess to evaluate one experiment. Blocks until done."""
    result = subprocess.run(
        [sys.executable, __file__, "--_eval-single", run_dir, baseline_cache_path],
        capture_output=True, text=True,
        env={**os.environ, "LIBSUMO_AS_TRACI": "1"},
    )
    run_id = os.path.basename(run_dir)
    stdout_last = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else ""
    stderr_last = result.stderr.strip().split("\n")[-1] if result.stderr.strip() else ""
    return run_id, result.returncode, stdout_last, stderr_last


def run_parallel(experiments, baseline_cache_path, num_workers):
    """Fan out RL evaluations as subprocesses via ThreadPoolExecutor."""
    success = 0
    failed = 0
    total = len(experiments)
    t_start = time.time()

    print(f"Launching {total} evaluations across {num_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for run_id, run_dir, _meta in experiments:
            f = executor.submit(_launch_worker, run_dir, baseline_cache_path)
            futures[f] = run_id

        for i, f in enumerate(as_completed(futures), 1):
            run_id = futures[f]
            done = success + failed
            eta = _format_eta(time.time() - t_start, done, total) if done > 0 else "estimating..."
            try:
                rid, retcode, stdout_msg, stderr_msg = f.result()
                if retcode == 0:
                    success += 1
                    print(f"  [{i}/{total}] OK   {rid}  {stdout_msg}  (ETA: {eta})")
                else:
                    failed += 1
                    print(f"  [{i}/{total}] FAIL {rid}  {stderr_msg}")
            except Exception as e:
                failed += 1
                print(f"  [{i}/{total}] FAIL {run_id}  {e}")

    return success, failed


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Supplement missing evaluation results for HPC experiments"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Only list incomplete experiments, don't evaluate.")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only process experiments whose run_id contains this string.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel evaluation workers. Each worker "
                             "runs SUMO in a separate process (bypasses libsumo "
                             "singleton limitation). Recommended: num_cores - 2.")
    # Hidden flag: used by subprocess workers
    parser.add_argument("--_eval-single", nargs=2, metavar=("RUN_DIR", "BASELINE_JSON"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Subprocess worker mode — evaluate a single experiment and exit
    if args._eval_single:
        run_dir, baseline_cache_path = args._eval_single
        _worker_eval_single(run_dir, baseline_cache_path)
        return

    if not os.path.exists(EXPERIMENTS_DIR):
        print("No experiments directory found.")
        return

    incomplete = find_incomplete_experiments(args.filter)

    if not incomplete:
        print("All experiments have results. Nothing to supplement.")
        return

    # Group by scenario for reporting
    by_scenario = {}
    for run_id, _, meta in incomplete:
        s = meta.get("scenario", "uniform")
        by_scenario.setdefault(s, []).append(run_id)

    print(f"Found {len(incomplete)} experiments missing evaluation results:\n")
    for scenario, ids in sorted(by_scenario.items()):
        print(f"  {scenario} ({len(ids)} experiments):")
        for rid in ids:
            print(f"    - {rid}")
    print()

    n_baselines = len(set(baseline_cache_key(m) for _, _, m in incomplete))
    n_evals = len(incomplete)
    mode = f"{args.num_workers} workers" if args.num_workers > 1 else "sequential"
    print(f"Will run: {n_baselines} baseline(s) + {n_evals} RL evaluation(s) [{mode}]\n")

    if args.dry_run:
        print("Dry run — exiting without evaluating.")
        return

    # Phase 1: Run all unique baselines (sequential, same process)
    print("Phase 1: Computing baselines...")
    baseline_cache = compute_baselines(incomplete)
    print()

    if not baseline_cache:
        print("No baselines could be computed. Check route files.")
        return

    t_total = time.time()

    if args.num_workers > 1:
        # Phase 2: Save baseline cache to temp file for subprocess workers
        cache_fd, cache_path = tempfile.mkstemp(suffix=".json", prefix="bl_cache_")
        try:
            with os.fdopen(cache_fd, "w") as f:
                json.dump(baseline_cache, f)

            # Phase 3: Fan out RL evaluations as parallel subprocesses
            success, failed = run_parallel(incomplete, cache_path, args.num_workers)
        finally:
            os.unlink(cache_path)
    else:
        # Sequential mode (no subprocess overhead)
        success, failed = run_sequential(incomplete, baseline_cache)

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed / 60:.1f} min: {success} supplemented, {failed} failed")
    print(f"{'=' * 60}")

    if success > 0:
        print(f"\nRun 'python src/dashboard.py' to regenerate the dashboard.")


if __name__ == "__main__":
    main()
