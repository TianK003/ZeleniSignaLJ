"""
Zeleni SignaLJ - Multi-Scenario Evaluation Script
====================================================
Evaluate a trained PPO model across rush-hour and off-peak scenarios.

Runs evaluation scenarios x two controllers (RL vs fixed-time baseline):
  1. morning_rush  - routes_morning_rush.rou.xml  (4h RL window, 06:00-10:00)
  2. evening_rush  - routes_evening_rush.rou.xml  (4h RL window, 14:00-18:00)
  3. offpeak       - routes_offpeak.rou.xml        (1h RL window, reference)

Output CSV columns: scenario, controller, avg_queue, avg_wait, total_teleports

Usage:
    python src/evaluate.py
    python src/evaluate.py --gui
    python src/evaluate.py --model models/ppo_morning_rush_final.zip
    python src/evaluate.py --scenario morning_rush --model models/ppo_morning_rush_final.zip
"""

import argparse
import os
import numpy as np
import pandas as pd
import sumo_rl
from stable_baselines3 import PPO

from config import (
    TS_IDS, TS_NAMES,
    DELTA_TIME, YELLOW_TIME, MIN_GREEN, MAX_GREEN, REWARD_FN,
    WARMUP_SECONDS,
    MORNING_RUSH_SECONDS, EVENING_RUSH_SECONDS, OFFPEAK_SECONDS,
    MORNING_RUSH_START, EVENING_RUSH_START,
)
from tls_programs import parse_original_programs, restore_non_target_programs

# Import time encoding
import experiment
from experiment import TimeEncodedObservationFunction


# ── Scenario registry ──────────────────────────────────────────────────────
EVAL_SCENARIOS = {
    "morning_rush": {
        "route_file": "data/routes/routes_morning_rush.rou.xml",
        "rl_seconds": MORNING_RUSH_SECONDS,   # 14400
        "start_hour": MORNING_RUSH_START,      # 6.0
        "label": "Morning rush (06:00-10:00)",
    },
    "evening_rush": {
        "route_file": "data/routes/routes_evening_rush.rou.xml",
        "rl_seconds": EVENING_RUSH_SECONDS,    # 14400
        "start_hour": EVENING_RUSH_START,      # 14.0
        "label": "Evening rush (14:00-18:00)",
    },
    "offpeak": {
        "route_file": "data/routes/routes_offpeak.rou.xml",
        "rl_seconds": OFFPEAK_SECONDS,         # 3600
        "start_hour": 12.0,
        "label": "Off-peak reference (midday)",
    },
}


def run_episode(net_file, route_file, model=None, use_gui=False,
                num_seconds=None, fixed_ts=False):
    """
    Run one evaluation episode.

    Args:
        model: loaded PPO model. None + fixed_ts=True -> fixed-time baseline.
        fixed_ts: if True, SUMO runs original signal programs without RL.

    Returns dict with:
        rewards       - per-TLS cumulative reward
        avg_queue     - mean halted vehicles/step across target TLS lanes
        avg_wait      - mean lane waiting time/step across target TLS (seconds)
        total_teleports - total vehicle teleports during episode
    """
    if num_seconds is None:
        num_seconds = 3600 + WARMUP_SECONDS

    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        num_seconds=num_seconds,
        reward_fn=REWARD_FN,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        single_agent=False,
        fixed_ts=fixed_ts,
        sumo_warnings=False,
        observation_class=TimeEncodedObservationFunction,
    )

    observations = env.reset()
    rewards = {ts_id: 0.0 for ts_id in TS_IDS}
    done = False

    obs_size = model.observation_space.shape[0] if model else None
    target_set = set(TS_IDS)

    # Restore original SUMO programs for non-target TLS (RL mode only)
    if not fixed_ts and model is not None:
        original_programs = parse_original_programs(net_file)
        restore_non_target_programs(env, TS_IDS, original_programs)

    # Metric accumulators
    queue_steps = []
    wait_steps = []
    total_teleports = 0
    n_lanes = sum(len(env.traffic_signals[ts].lanes) for ts in TS_IDS
                  if ts in env.traffic_signals)

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

        observations, reward_dict, done_dict, info = env.step(actions)
        done = done_dict["__all__"]

        for ts_id in TS_IDS:
            if ts_id in reward_dict:
                rewards[ts_id] += reward_dict[ts_id]

        # Per-step system metrics via TraCI
        try:
            sumo = env.sumo

            stopped = sum(
                sumo.lane.getLastStepHaltingNumber(lane)
                for ts_id in TS_IDS if ts_id in env.traffic_signals
                for lane in env.traffic_signals[ts_id].lanes
            )
            queue_steps.append(stopped)

            wait = sum(
                sumo.lane.getWaitingTime(lane)
                for ts_id in TS_IDS if ts_id in env.traffic_signals
                for lane in env.traffic_signals[ts_id].lanes
            )
            wait_steps.append(wait / max(n_lanes, 1))

            total_teleports += sumo.simulation.getStartingTeleportNumber()

        except Exception:
            pass

    env.close()

    avg_queue = float(np.mean(queue_steps)) if queue_steps else float("nan")
    avg_wait = float(np.mean(wait_steps)) if wait_steps else float("nan")

    return {
        "rewards": rewards,
        "avg_queue": avg_queue,
        "avg_wait": avg_wait,
        "total_teleports": total_teleports,
    }


def run_scenario(scenario_name, net_file, model, use_gui=False, skip_baseline=False):
    """
    Run one scenario with both RL and fixed-time controllers.
    Returns list of result dicts (one per controller).
    If skip_baseline=True, only the RL controller runs (use when baseline is cached).
    """
    cfg = EVAL_SCENARIOS[scenario_name]
    route_file = cfg["route_file"]
    num_seconds = cfg["rl_seconds"] + WARMUP_SECONDS

    if not os.path.exists(route_file):
        print(f"  WARNING: Route file missing: {route_file}")
        print(f"           Run: python src/generate_demand.py --scenario {scenario_name}")
        return []

    # Set time-of-day for observation encoding
    experiment.CURRENT_HOUR = cfg.get("start_hour", 0.0)

    controllers = []
    if not skip_baseline:
        controllers.append(("fixed_time", (None, True)))
    if model is not None:
        controllers.append(("rl", (model, False)))

    results = []
    for controller, (use_model, fixed_ts) in controllers:
        if controller == "rl" and model is None:
            print(f"  Skipping RL controller (no model loaded).")
            continue

        print(f"  [{scenario_name}] {controller}...", end="", flush=True)
        ep = run_episode(
            net_file, route_file,
            model=use_model,
            use_gui=use_gui,
            num_seconds=num_seconds,
            fixed_ts=fixed_ts,
        )
        total_reward = sum(ep["rewards"].values())
        print(f" reward={total_reward:.0f}  queue={ep['avg_queue']:.1f}  "
              f"wait={ep['avg_wait']:.1f}s  teleports={ep['total_teleports']}")

        results.append({
            "scenario": scenario_name,
            "label": cfg["label"],
            "controller": controller,
            "total_reward": total_reward,
            "avg_queue": ep["avg_queue"],
            "avg_wait": ep["avg_wait"],
            "total_teleports": ep["total_teleports"],
        })

    return results


def print_comparison(df):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"{'Scenario':<30} {'Controller':<12} {'Avg Queue':>10} "
          f"{'Avg Wait(s)':>12} {'Teleports':>10} {'Reward':>10}")
    print(f"{'-'*80}")

    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario]
        for _, row in sub.iterrows():
            print(f"  {row['label']:<28} {row['controller']:<12} "
                  f"{row['avg_queue']:>10.1f} {row['avg_wait']:>12.1f} "
                  f"{row['total_teleports']:>10} {row['total_reward']:>10.0f}")

        if len(sub) == 2:
            bl = sub[sub["controller"] == "fixed_time"].iloc[0]
            rl = sub[sub["controller"] == "rl"].iloc[0]
            q_imp = (bl["avg_queue"] - rl["avg_queue"]) / max(abs(bl["avg_queue"]), 1) * 100
            w_imp = (bl["avg_wait"] - rl["avg_wait"]) / max(abs(bl["avg_wait"]), 1) * 100
            print(f"  {'  -> RL improvement':<28} {'':12} {q_imp:>+9.1f}% "
                  f"{w_imp:>+11.1f}%")
        print()

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Multi-scenario traffic signal evaluation")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained PPO model (.zip). If omitted, runs baseline-only.",
    )
    parser.add_argument(
        "--net_file", type=str, default="data/networks/ljubljana.net.xml"
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--scenario", type=str, default="all",
        choices=["morning_rush", "evening_rush", "offpeak", "all"],
        help="Which scenario(s) to evaluate. Default: all three.",
    )
    parser.add_argument(
        "--output", type=str, default="results/rush_hour_comparison.csv",
        help="Path for output CSV."
    )
    parser.add_argument(
        "--skip_baseline", action="store_true",
        help="Skip the fixed-time baseline run (use when baseline is already cached)."
    )
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Load model (optional -- baseline-only run works without it)
    model = None
    if args.model:
        if not os.path.exists(args.model):
            print(f"WARNING: Model file not found: {args.model}")
            print("Running fixed-time baseline only.\n")
        else:
            print(f"Loading model: {args.model}")
            model = PPO.load(args.model)

    scenarios = (
        list(EVAL_SCENARIOS.keys())
        if args.scenario == "all"
        else [args.scenario]
    )

    print(f"\nEvaluating {len(scenarios)} scenario(s)...\n")
    all_results = []

    for scenario in scenarios:
        rows = run_scenario(scenario, args.net_file, model, use_gui=args.gui,
                            skip_baseline=args.skip_baseline)
        all_results.extend(rows)

    if not all_results:
        print("No results collected (check route files exist).")
        return

    df = pd.DataFrame(all_results)

    print_comparison(df)

    # Save full comparison CSV
    csv_cols = ["scenario", "label", "controller",
                "avg_queue", "avg_wait", "total_teleports", "total_reward"]
    df[csv_cols].to_csv(args.output, index=False)
    print(f"Saved comparison to {args.output}")

    # Also save per-intersection breakdown for the best RL model
    if model is not None:
        _save_per_intersection(df, args, model, scenarios)


def _save_per_intersection(df, args, model, scenarios):
    """Save per-intersection reward breakdown for the first available RL scenario."""
    target_scenario = next((s for s in ["morning_rush", "evening_rush"] if s in scenarios), None)
    if target_scenario is None:
        return

    cfg = EVAL_SCENARIOS[target_scenario]
    if not os.path.exists(cfg["route_file"]):
        return

    print(f"\nCollecting per-intersection breakdown for {target_scenario}...")
    num_seconds = cfg["rl_seconds"] + WARMUP_SECONDS

    # Set time encoding for this scenario
    experiment.CURRENT_HOUR = cfg.get("start_hour", 0.0)

    bl_ep = run_episode(args.net_file, cfg["route_file"],
                        model=None, num_seconds=num_seconds, fixed_ts=True)
    rl_ep = run_episode(args.net_file, cfg["route_file"],
                        model=model, num_seconds=num_seconds, fixed_ts=False)

    rows = []
    for ts_id in TS_IDS:
        name = TS_NAMES.get(ts_id, ts_id)
        bl = bl_ep["rewards"].get(ts_id, 0)
        rl = rl_ep["rewards"].get(ts_id, 0)
        pct = ((rl - bl) / abs(bl) * 100) if bl != 0 else 0
        rows.append({
            "scenario": target_scenario,
            "intersection": name,
            "tls_id": ts_id,
            "baseline_reward": bl,
            "rl_reward": rl,
            "improvement_pct": pct,
        })

    detail_path = "results/comparison_summary.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False)
    print(f"Per-intersection detail saved to {detail_path}")

    print(f"\n{'Intersection':<38} {'Baseline':>10} {'RL':>10} {'Pct':>8}")
    print(f"{'-'*68}")
    for row in rows:
        print(f"  {row['intersection']:<36} {row['baseline_reward']:>10.1f} "
              f"{row['rl_reward']:>10.1f} {row['improvement_pct']:>+7.1f}%")

    bl_total = sum(r["baseline_reward"] for r in rows)
    rl_total = sum(r["rl_reward"] for r in rows)
    total_pct = ((rl_total - bl_total) / abs(bl_total) * 100) if bl_total != 0 else 0
    print(f"{'-'*68}")
    print(f"  {'TOTAL':<36} {bl_total:>10.1f} {rl_total:>10.1f} {total_pct:>+7.1f}%")


if __name__ == "__main__":
    main()
