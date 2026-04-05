"""
Zeleni SignaLJ - Evaluation Script
====================================
Evaluate a trained PPO model against the fixed-time baseline.
Outputs KPI comparison and optionally records a demo via sumo-gui.

Usage:
    python src/evaluate.py                          # headless eval
    python src/evaluate.py --gui                    # with SUMO GUI (for demo recording)
    python src/evaluate.py --model models/ppo_ljubljana_final.zip
"""

import argparse
import os
import numpy as np
import pandas as pd
import sumo_rl
from stable_baselines3 import PPO


def run_episode(net_file, route_file, model=None, use_gui=False,
                num_seconds=3600, out_csv=None):
    """Run one episode. If model=None, runs fixed-time baseline."""
    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        num_seconds=num_seconds,
        single_agent=True,
        reward_fn="queue",
        out_csv_name=out_csv,
    )

    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = 0  # Keep current phase = fixed-time baseline
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

    env.close()
    return total_reward, step_count


def main():
    parser = argparse.ArgumentParser(description="Evaluate traffic signal control")
    parser.add_argument("--model", type=str,
                        default="models/ppo_ljubljana_final.zip",
                        help="Path to trained model")
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml")
    parser.add_argument("--route_file", type=str,
                        default="data/routes/routes.rou.xml")
    parser.add_argument("--gui", action="store_true",
                        help="Run with SUMO GUI for visual demo")
    parser.add_argument("--num_seconds", type=int, default=3600)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # ── Run baseline (fixed-time) ──
    print("Running fixed-time baseline...")
    baseline_reward, baseline_steps = run_episode(
        args.net_file, args.route_file,
        model=None, use_gui=False,
        num_seconds=args.num_seconds,
        out_csv="results/baseline",
    )
    print(f"  Baseline: reward={baseline_reward:.1f}, steps={baseline_steps}")

    # ── Run trained RL agent ──
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)
    print("Running RL agent...")
    rl_reward, rl_steps = run_episode(
        args.net_file, args.route_file,
        model=model, use_gui=args.gui,
        num_seconds=args.num_seconds,
        out_csv="results/rl_agent",
    )
    print(f"  RL Agent: reward={rl_reward:.1f}, steps={rl_steps}")

    # ── Compare ──
    improvement = ((rl_reward - baseline_reward) / abs(baseline_reward)) * 100
    print(f"\n{'='*50}")
    print(f"  Baseline reward:  {baseline_reward:.1f}")
    print(f"  RL agent reward:  {rl_reward:.1f}")
    print(f"  Improvement:      {improvement:+.1f}%")
    print(f"{'='*50}")

    # Save summary
    summary = pd.DataFrame({
        "metric": ["total_reward", "steps", "improvement_pct"],
        "baseline": [baseline_reward, baseline_steps, 0],
        "rl_agent": [rl_reward, rl_steps, improvement],
    })
    summary.to_csv("results/comparison_summary.csv", index=False)
    print("Results saved to results/comparison_summary.csv")


if __name__ == "__main__":
    main()
