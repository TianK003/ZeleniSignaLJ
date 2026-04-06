"""
Zeleni SignaLJ - Evaluation Script
====================================
Evaluate a trained PPO model against the fixed-time baseline.

Usage:
    python src/evaluate.py
    python src/evaluate.py --gui
    python src/evaluate.py --model results/experiments/XXXXX/ppo_shared_policy.zip
"""

import argparse
import os
import numpy as np
import pandas as pd
import sumo_rl
from stable_baselines3 import PPO

from config import TS_IDS, TS_NAMES
from tls_programs import parse_original_programs, restore_non_target_programs


def run_episode(net_file, route_file, model=None, use_gui=False,
                num_seconds=3600, fixed_ts=False):
    """Run one episode. model=None with fixed_ts=True → real baseline."""
    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        num_seconds=num_seconds,
        reward_fn="queue",
        delta_time=5,
        yellow_time=2,
        min_green=10,
        max_green=90,
        single_agent=False,
        fixed_ts=fixed_ts,
        sumo_warnings=False,
    )

    observations = env.reset()
    rewards = {ts_id: 0.0 for ts_id in TS_IDS}
    done = False

    # Get padded sizes from model if available
    obs_size = model.observation_space.shape[0] if model else None
    n_actions = model.action_space.n if model else None

    target_set = set(TS_IDS)

    # Restore original SUMO programs for non-target TLS (when not baseline)
    if not fixed_ts and model is not None:
        original_programs = parse_original_programs(net_file)
        restore_non_target_programs(env, TS_IDS, original_programs)

    while not done:
        actions = {}
        for ts_id in env.ts_ids:
            if model is not None and ts_id in target_set:
                # Target intersection: use RL model
                obs = observations[ts_id]
                # Pad observation to match training space
                if len(obs) < obs_size:
                    padded = np.zeros(obs_size, dtype=np.float32)
                    padded[:len(obs)] = obs
                    obs = padded
                action, _ = model.predict(obs, deterministic=True)
                # Clip action to valid range for this intersection
                actual_n = env.traffic_signals[ts_id].action_space.n
                actions[ts_id] = int(action) % actual_n
            else:
                # Non-target or baseline: action ignored by patched TLS
                # (SUMO runs original program); for fixed_ts baseline,
                # SUMO skips action processing entirely
                actions[ts_id] = 0

        observations, reward_dict, done_dict, info = env.step(actions)
        done = done_dict["__all__"]

        for ts_id in TS_IDS:
            if ts_id in reward_dict:
                rewards[ts_id] += reward_dict[ts_id]

    env.close()
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Evaluate traffic signal control")
    parser.add_argument("--model", type=str, default="models/ppo_traffic_final.zip")
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml")
    parser.add_argument("--route_file", type=str,
                        default="data/routes/routes.rou.xml")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--num_seconds", type=int, default=3600)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Baseline (real signal programs)
    print("Running fixed-time baseline (real signal programs)...")
    baseline_rewards = run_episode(
        args.net_file, args.route_file,
        model=None, num_seconds=args.num_seconds, fixed_ts=True,
    )
    print(f"  Baseline total reward: {sum(baseline_rewards.values()):.1f}")

    # RL model
    print(f"\nLoading model: {args.model}")
    model = PPO.load(args.model)

    print("Running RL agent...")
    rl_rewards = run_episode(
        args.net_file, args.route_file,
        model=model, use_gui=args.gui, num_seconds=args.num_seconds,
    )

    # Comparison
    print(f"\n{'='*70}")
    print(f"{'Intersection':<40} {'Baseline':>10} {'RL':>10} {'Δ%':>8}")
    print(f"{'-'*70}")

    rows = []
    for ts_id in TS_IDS:
        name = TS_NAMES.get(ts_id, ts_id)[:38]
        bl = baseline_rewards.get(ts_id, 0)
        rl = rl_rewards.get(ts_id, 0)
        pct = ((rl - bl) / abs(bl) * 100) if bl != 0 else 0
        print(f"  {name:<38} {bl:>10.1f} {rl:>10.1f} {pct:>+7.1f}%")
        rows.append({
            "intersection": name, "tls_id": ts_id,
            "baseline_reward": bl, "rl_reward": rl,
            "improvement_pct": pct,
        })

    bl_total = sum(baseline_rewards.values())
    rl_total = sum(rl_rewards.values())
    total_pct = ((rl_total - bl_total) / abs(bl_total) * 100) if bl_total != 0 else 0
    print(f"{'-'*70}")
    print(f"  {'TOTAL':<38} {bl_total:>10.1f} {rl_total:>10.1f} {total_pct:>+7.1f}%")
    print(f"{'='*70}")

    pd.DataFrame(rows).to_csv("results/comparison_summary.csv", index=False)
    print("\nSaved to results/comparison_summary.csv")


if __name__ == "__main__":
    main()
