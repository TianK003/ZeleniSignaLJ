"""
Zeleni SignaLJ - Isolated Rush-Hour Statistical Test Runner
============================================================
Runs isolated rush-hour evaluations (morning or evening) with different route
files to test how well an RL model generalizes across random traffic patterns.

Each run uses a single scenario (e.g., 4h morning rush) with a different route
file from a directory of pre-generated variants. Both baseline (fixed-time) and
RL results are collected per seed for paired statistical comparison.

This tests generalization WITHOUT the 24h cascading-error problem — each run is
an independent 4h episode with fresh traffic state.

Usage:
    # Generate 50 morning rush route variants
    python src/generate_demand.py --scenario morning_rush --num_variants 50 \\
        --output_dir data/routes/rush-test-morning

    # Test a morning model on 50 variants
    python src/run_rush_test.py \\
        --model results/experiments/.../ppo_shared_policy.zip \\
        --scenario morning_rush \\
        --route_dir data/routes/rush-test-morning \\
        --num_runs 50 --num_workers 10 \\
        --output_dir results/rush-test/M1_morning

    # Baseline only
    python src/run_rush_test.py --baseline \\
        --scenario morning_rush \\
        --route_dir data/routes/rush-test-morning \\
        --num_runs 50 --num_workers 10 \\
        --output_dir results/rush-test/baseline_morning
"""

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TS_IDS, TS_NAMES, DELTA_TIME, YELLOW_TIME, MIN_GREEN, MAX_GREEN,
    REWARD_FN, WARMUP_SECONDS,
    MORNING_RUSH_START, MORNING_RUSH_END, MORNING_RUSH_SECONDS,
    EVENING_RUSH_START, EVENING_RUSH_END, EVENING_RUSH_SECONDS,
    OFFPEAK_SECONDS,
)
from tls_programs import parse_original_programs, restore_non_target_programs


# ── Scenario configs ─────────────────────────────────────────────────────

RUSH_SCENARIOS = {
    "morning_rush": {
        "rl_seconds": MORNING_RUSH_SECONDS,   # 14400 (4h)
        "start_hour": MORNING_RUSH_START,      # 6.0
        "label": "Morning rush (06:00-10:00)",
        "route_pattern": "routes_morning_rush",
    },
    "evening_rush": {
        "rl_seconds": EVENING_RUSH_SECONDS,    # 14400 (4h)
        "start_hour": EVENING_RUSH_START,      # 14.0
        "label": "Evening rush (14:00-18:00)",
        "route_pattern": "routes_evening_rush",
    },
    "offpeak": {
        "rl_seconds": OFFPEAK_SECONDS,         # 3600 (1h)
        "start_hour": 12.0,
        "label": "Off-peak (midday)",
        "route_pattern": "routes_offpeak",
    },
}


# ── Single-run evaluation ────────────────────────────────────────────────

def run_single_episode(net_file, route_file, num_seconds, model_path=None,
                       start_hour=0.0, fixed_ts=False):
    """Run one evaluation episode and return metrics.

    Args:
        model_path: Path to PPO model .zip. None for baseline.
        start_hour: CURRENT_HOUR for time encoding.
        fixed_ts: If True, run with SUMO's original programs (baseline).

    Returns dict with per-intersection rewards, avg_queue, avg_wait, etc.
    """
    import sumo_rl
    import experiment
    from experiment import TimeEncodedObservationFunction

    experiment.CURRENT_HOUR = start_hour

    model = None
    if model_path and not fixed_ts:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)

    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=num_seconds,
        reward_fn=REWARD_FN,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        single_agent=False,
        fixed_ts=fixed_ts,
        sumo_warnings=False,
        additional_sumo_cmd="--ignore-route-errors",
        observation_class=TimeEncodedObservationFunction,
    )

    observations = env.reset()
    target_set = set(TS_IDS)
    obs_size = model.observation_space.shape[0] if model else None

    # Restore non-target TLS programs (RL mode only)
    if not fixed_ts and model is not None:
        original_programs = parse_original_programs(net_file)
        restore_non_target_programs(env, TS_IDS, original_programs)

    # Metric accumulators
    per_intersection_reward = {ts_id: 0.0 for ts_id in TS_IDS}
    per_intersection_queue = defaultdict(list)
    per_intersection_wait = defaultdict(list)
    total_teleports = 0
    done = False

    while not done:
        actions = {}
        for ts_id in env.ts_ids:
            if model is not None and ts_id in target_set:
                obs = observations[ts_id]
                if len(obs) < obs_size:
                    padded = np.zeros(obs_size, dtype=np.float32)
                    padded[:len(obs)] = obs
                    obs = padded
                action, _ = model.predict(obs, deterministic=True)
                actual_n = env.traffic_signals[ts_id].action_space.n
                actions[ts_id] = int(action) % actual_n
            else:
                actions[ts_id] = 0

        observations, rewards, dones, info = env.step(actions)
        done = dones["__all__"]

        for ts_id in TS_IDS:
            if ts_id in rewards:
                per_intersection_reward[ts_id] += rewards[ts_id]

        try:
            sumo = env.sumo
            for ts_id in TS_IDS:
                if ts_id not in env.traffic_signals:
                    continue
                ts = env.traffic_signals[ts_id]
                stopped = sum(sumo.lane.getLastStepHaltingNumber(l) for l in ts.lanes)
                per_intersection_queue[ts_id].append(stopped)
                wait = sum(sumo.lane.getWaitingTime(l) for l in ts.lanes)
                per_intersection_wait[ts_id].append(wait / max(len(ts.lanes), 1))
            total_teleports += sumo.simulation.getStartingTeleportNumber()
        except Exception:
            pass

    env.close()

    # Build result
    total_reward = sum(per_intersection_reward.values())
    named_rewards = {}
    named_queues = {}
    named_waits = {}
    for ts_id in TS_IDS:
        name = TS_NAMES.get(ts_id, ts_id[:20])
        named_rewards[name] = per_intersection_reward[ts_id]
        q = per_intersection_queue[ts_id]
        w = per_intersection_wait[ts_id]
        named_queues[name] = float(np.mean(q)) if q else 0.0
        named_waits[name] = float(np.mean(w)) if w else 0.0

    all_queues = []
    all_waits = []
    for ts_id in TS_IDS:
        all_queues.extend(per_intersection_queue[ts_id])
        all_waits.extend(per_intersection_wait[ts_id])

    return {
        "total_reward": total_reward,
        "avg_queue": float(np.mean(all_queues)) if all_queues else 0.0,
        "avg_wait": float(np.mean(all_waits)) if all_waits else 0.0,
        "total_teleports": total_teleports,
        "per_intersection_reward": named_rewards,
        "per_intersection_avg_queue": named_queues,
        "per_intersection_avg_wait": named_waits,
    }


# ── Multiprocessing worker ───────────────────────────────────────────────

def _worker(args):
    """Worker for parallel runs."""
    (seed, net_file, route_file, num_seconds, model_path,
     start_hour, is_baseline) = args

    wall_start = time.time()
    try:
        result = run_single_episode(
            net_file, route_file, num_seconds,
            model_path=None if is_baseline else model_path,
            start_hour=start_hour,
            fixed_ts=is_baseline,
        )
    except Exception as e:
        print(f"  [FAIL] Seed {seed}: {e}", flush=True)
        result = {
            "total_reward": float("nan"), "avg_queue": float("nan"),
            "avg_wait": float("nan"), "total_teleports": 0,
            "per_intersection_reward": {}, "per_intersection_avg_queue": {},
            "per_intersection_avg_wait": {}, "error": str(e),
        }

    result["seed"] = seed
    result["route_file"] = route_file
    result["wall_time_s"] = round(time.time() - wall_start, 1)
    print(f"  [OK] Seed {seed:>2d} done in {result['wall_time_s']:.0f}s  "
          f"reward={result['total_reward']:.0f}", flush=True)
    return result


# ── Output ────────────────────────────────────────────────────────────────

def _write_summary_csv(results, output_dir):
    """Write summary.csv with one row per seed."""
    intersections = list(TS_NAMES.values())

    fieldnames = [
        "seed", "total_reward", "avg_queue", "avg_wait",
        "teleports", "wall_time_s",
    ]
    for name in intersections:
        fieldnames.append(f"reward_{name}")
    for name in intersections:
        fieldnames.append(f"queue_{name}")
    for name in intersections:
        fieldnames.append(f"wait_{name}")

    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in sorted(results, key=lambda x: x["seed"]):
            row = {
                "seed": r["seed"],
                "total_reward": round(r["total_reward"], 2),
                "avg_queue": round(r["avg_queue"], 3),
                "avg_wait": round(r["avg_wait"], 3),
                "teleports": r["total_teleports"],
                "wall_time_s": r["wall_time_s"],
            }
            for name in intersections:
                row[f"reward_{name}"] = round(
                    r["per_intersection_reward"].get(name, 0), 2)
            for name in intersections:
                row[f"queue_{name}"] = round(
                    r["per_intersection_avg_queue"].get(name, 0), 3)
            for name in intersections:
                row[f"wait_{name}"] = round(
                    r["per_intersection_avg_wait"].get(name, 0), 3)
            writer.writerow(row)

    return csv_path


def _print_summary(results, tag):
    """Print statistical summary."""
    rewards = [r["total_reward"] for r in results]
    queues = [r["avg_queue"] for r in results]
    n = len(rewards)
    mean_r = np.mean(rewards)
    std_r = np.std(rewards, ddof=1) if n > 1 else 0
    ci95 = 1.96 * std_r / np.sqrt(n) if n > 1 else 0

    print(f"\n{'='*60}")
    print(f"Summary: {tag} ({n} runs)")
    print(f"{'='*60}")
    print(f"  Total reward:  {mean_r:>12.1f} +/- {ci95:.1f}  (std={std_r:.1f})")
    print(f"  Avg queue:     {np.mean(queues):>12.2f} +/- "
          f"{1.96*np.std(queues,ddof=1)/np.sqrt(n):.2f}")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run isolated rush-hour statistical test (generalization test)"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to PPO model (.zip)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline (all fixed-time, no RL)")
    parser.add_argument("--scenario", type=str, required=True,
                        choices=list(RUSH_SCENARIOS.keys()),
                        help="Rush-hour scenario to evaluate")
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml")
    parser.add_argument("--route_dir", type=str, required=True,
                        help="Directory with per-seed route files "
                             "(routes_*_seed_NN.rou.xml)")
    parser.add_argument("--num_runs", type=int, default=50,
                        help="Number of replications (one per route file)")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Parallel workers")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output JSONs and summary CSV")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for this test run")

    args = parser.parse_args()

    if not args.baseline and not args.model:
        parser.error("Provide --model or use --baseline")

    cfg = RUSH_SCENARIOS[args.scenario]
    num_seconds = cfg["rl_seconds"] + WARMUP_SECONDS
    start_hour = cfg["start_hour"]
    route_pattern = cfg["route_pattern"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Write meta.json
    meta = {
        "tag": args.tag,
        "baseline": args.baseline,
        "model": args.model,
        "scenario": args.scenario,
        "label": cfg["label"],
        "start_hour": start_hour,
        "num_seconds": num_seconds,
        "rl_seconds": cfg["rl_seconds"],
        "warmup_seconds": WARMUP_SECONDS,
        "net_file": args.net_file,
        "route_dir": args.route_dir,
        "num_runs": args.num_runs,
        "num_workers": args.num_workers,
        "reward_fn": REWARD_FN,
        "ts_ids": TS_IDS,
        "ts_names": TS_NAMES,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Find route files
    route_files = sorted([
        f for f in os.listdir(args.route_dir)
        if f.startswith(route_pattern) and f.endswith(".rou.xml")
           and "seed_" in f
    ])

    if not route_files:
        print(f"ERROR: No route files matching '{route_pattern}_seed_*.rou.xml' "
              f"found in {args.route_dir}")
        print(f"Generate with: python src/generate_demand.py "
              f"--scenario {args.scenario} --num_variants {args.num_runs} "
              f"--output_dir {args.route_dir}")
        sys.exit(1)

    if len(route_files) < args.num_runs:
        print(f"WARNING: Only {len(route_files)} route files found, "
              f"but {args.num_runs} runs requested. Using {len(route_files)} runs.")
        args.num_runs = len(route_files)

    mode_str = "BASELINE" if args.baseline else f"RL ({args.tag or os.path.basename(args.model)})"
    print(f"\n{'='*60}")
    print(f"Rush-Hour Statistical Test: {mode_str}")
    print(f"  Scenario: {cfg['label']}")
    print(f"  Runs: {args.num_runs} | Workers: {args.num_workers}")
    print(f"  Routes: {args.route_dir}/ ({len(route_files)} files)")
    if not args.baseline:
        print(f"  Model: {args.model}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    # Build worker args
    worker_args = []
    for seed in range(1, args.num_runs + 1):
        route_idx = seed - 1
        route_file = os.path.join(args.route_dir, route_files[route_idx])
        worker_args.append(
            (seed, args.net_file, route_file, num_seconds,
             args.model, start_hour, args.baseline)
        )

    # Run in parallel
    t0 = time.time()
    if args.num_workers <= 1:
        results = [_worker(a) for a in worker_args]
    else:
        with multiprocessing.Pool(args.num_workers) as pool:
            results = pool.map(_worker, worker_args)

    total_wall = time.time() - t0
    print(f"\nAll {args.num_runs} runs completed in {total_wall:.0f}s")

    # Save individual JSONs
    for r in results:
        seed = r["seed"]
        json_path = os.path.join(args.output_dir, f"run_seed_{seed:02d}.json")
        with open(json_path, "w") as f:
            json.dump(r, f, indent=2, default=str)

    # Filter failed runs
    ok_results = [r for r in results if "error" not in r]
    failed = len(results) - len(ok_results)
    if failed:
        print(f"\nWARNING: {failed}/{len(results)} runs failed")

    if ok_results:
        csv_path = _write_summary_csv(ok_results, args.output_dir)
        print(f"Summary CSV: {csv_path}  ({len(ok_results)} successful runs)")
        _print_summary(ok_results, args.tag or ("baseline" if args.baseline else "rl"))
    else:
        print("ERROR: All runs failed.")


if __name__ == "__main__":
    main()
