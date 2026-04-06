"""
Zeleni SignaLJ - Experiment Runner
====================================
Runs a complete experiment: baseline → train → evaluate → save.

Uses PettingZoo parallel API + SuperSuit to convert multi-agent env
into a vectorized SB3-compatible env. This enables proper PPO training
with rollout collection, advantage estimation, and policy updates.

Usage:
    python src/experiment.py --total_timesteps 10000 --tag smoke_test
    python src/experiment.py --max_hours 1.0 --tag 1h_local
    python src/experiment.py --compare_only
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from config import TS_IDS, TS_NAMES


RESULTS_DIR = "results"
EXPERIMENTS_DIR = "results/experiments"


def get_run_id(tag=""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{tag}" if tag else ts


# ── Callbacks ──

class TimeLimitCallback(BaseCallback):
    """Stop training after max_seconds wall time."""
    def __init__(self, max_seconds, verbose=0):
        super().__init__(verbose)
        self.max_seconds = max_seconds
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        if self.max_seconds and (time.time() - self.start_time) >= self.max_seconds:
            print(f"\n  Time limit reached ({self.max_seconds/3600:.1f}h). Stopping.")
            return False
        return True


class ProgressCallback(BaseCallback):
    """Print training progress periodically."""
    def __init__(self, print_freq=5000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        if self.num_timesteps % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            sps = self.num_timesteps / elapsed if elapsed > 0 else 0
            print(f"    Step {self.num_timesteps}: "
                  f"elapsed={elapsed/60:.1f}min, {sps:.0f} steps/s")
        return True


# ── Environment creation ──

def make_train_env(net_file, route_file, num_seconds):
    """
    Create a PettingZoo parallel env → SuperSuit vec env for SB3.
    This is the standard IPPO approach: parameter sharing across agents.
    Each agent (traffic signal) is treated as a separate sub-env in the
    vectorized environment. PPO trains a shared policy that works for all.
    """
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=num_seconds,
        reward_fn="queue",
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=50,
        sumo_warnings=False,
    )
    # PettingZoo parallel → SB3 VecEnv (each agent = sub-environment)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    return env


def make_eval_env(net_file, route_file, num_seconds, fixed_ts=False):
    """Create a raw SumoEnvironment for evaluation (parallel dict API)."""
    return sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=num_seconds,
        reward_fn="queue",
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=50,
        single_agent=False,
        fixed_ts=fixed_ts,
        sumo_warnings=False,
    )


# ── Evaluation functions ──

def run_baseline(net_file, route_file, num_seconds):
    """
    Run true fixed-time baseline: SUMO uses default signal programs from
    the network file (real OSM timing). No RL actions applied.
    """
    env = make_eval_env(net_file, route_file, num_seconds, fixed_ts=True)
    observations = env.reset()

    rewards = {ts_id: 0.0 for ts_id in TS_IDS}
    done = False
    steps = 0

    while not done:
        actions = {ts_id: 0 for ts_id in env.ts_ids}
        observations, reward_dict, done_dict, info = env.step(actions)
        done = done_dict["__all__"]
        for ts_id in TS_IDS:
            if ts_id in reward_dict:
                rewards[ts_id] += reward_dict[ts_id]
        steps += 1

    env.close()
    return rewards, steps


def run_evaluation(net_file, route_file, num_seconds, model):
    """
    Evaluate trained PPO model on all traffic signals.
    The shared policy is used for ALL intersections (parameter sharing).
    """
    env = make_eval_env(net_file, route_file, num_seconds, fixed_ts=False)
    observations = env.reset()

    rewards = {ts_id: 0.0 for ts_id in TS_IDS}
    done = False
    steps = 0

    while not done:
        actions = {}
        for ts_id in env.ts_ids:
            obs = observations[ts_id]
            action, _ = model.predict(obs, deterministic=True)
            actions[ts_id] = int(action)

        observations, reward_dict, done_dict, info = env.step(actions)
        done = done_dict["__all__"]
        for ts_id in TS_IDS:
            if ts_id in reward_dict:
                rewards[ts_id] += reward_dict[ts_id]
        steps += 1

    env.close()
    return rewards, steps


# ── Training ──

def train_ppo(net_file, route_file, num_seconds, total_timesteps, run_dir,
              max_seconds=None):
    """
    Train PPO with parameter sharing via SuperSuit.
    All traffic signals share one policy (standard IPPO approach).
    Returns the trained model.
    """
    print("  Creating vectorized training environment...")
    env = make_train_env(net_file, route_file, num_seconds)

    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=os.path.join(run_dir, "tb_logs"),
    )

    # Callbacks
    callbacks = [ProgressCallback(print_freq=5000)]
    if max_seconds:
        callbacks.append(TimeLimitCallback(max_seconds))

    stop_desc = f"{total_timesteps} steps"
    if max_seconds:
        stop_desc += f" or {max_seconds/3600:.1f}h"
    print(f"  Training PPO (stop: {stop_desc})...")

    t_start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
    )
    train_time = time.time() - t_start

    # Save model
    model_path = os.path.join(run_dir, "ppo_shared_policy")
    model.save(model_path)
    print(f"  Model saved: {model_path}.zip")
    print(f"  Training time: {train_time/60:.1f}min "
          f"({model.num_timesteps} actual steps)")

    env.close()
    return model, train_time


# ── Experiment comparison ──

def compare_experiments():
    if not os.path.exists(EXPERIMENTS_DIR):
        print("No experiments found.")
        return None

    rows = []
    for run_id in sorted(os.listdir(EXPERIMENTS_DIR)):
        meta_path = os.path.join(EXPERIMENTS_DIR, run_id, "meta.json")
        results_path = os.path.join(EXPERIMENTS_DIR, run_id, "results.csv")
        if not os.path.exists(meta_path) or not os.path.exists(results_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        df = pd.read_csv(results_path)

        bl_total = df["baseline_reward"].sum()
        rl_total = df["rl_reward"].sum()
        improvement = ((rl_total - bl_total) / abs(bl_total) * 100) if bl_total != 0 else 0

        rows.append({
            "run_id": run_id,
            "tag": meta.get("tag", ""),
            "timesteps": meta.get("actual_timesteps", meta.get("total_timesteps", 0)),
            "baseline_total": bl_total,
            "rl_total": rl_total,
            "improvement_pct": improvement,
            "train_time_s": meta.get("train_time_s", 0),
        })

    if not rows:
        print("No completed experiments found.")
        return None

    df_compare = pd.DataFrame(rows)
    df_compare.to_csv(os.path.join(RESULTS_DIR, "experiments_comparison.csv"), index=False)

    print("\n" + "=" * 90)
    print("EXPERIMENT COMPARISON")
    print("=" * 90)
    print(f"{'Run ID':<25s} {'Tag':<15s} {'Steps':>8s} "
          f"{'Baseline':>10s} {'RL':>10s} {'Δ%':>8s} {'Time':>8s}")
    print("-" * 90)
    for _, row in df_compare.iterrows():
        print(f"  {row['run_id']:<23s} {row['tag']:<15s} "
              f"{int(row['timesteps']):>8d} "
              f"{row['baseline_total']:>10.1f} {row['rl_total']:>10.1f} "
              f"{row['improvement_pct']:>+7.1f}% "
              f"{row['train_time_s']:>7.0f}s")
    print("=" * 90)
    return df_compare


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Run reproducible experiment")
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml")
    parser.add_argument("--route_file", type=str,
                        default="data/routes/routes.rou.xml")
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--max_hours", type=float, default=None)
    parser.add_argument("--num_seconds", type=int, default=3600)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--compare_only", action="store_true")
    args = parser.parse_args()

    if args.compare_only:
        compare_experiments()
        return

    max_seconds = args.max_hours * 3600 if args.max_hours else None
    run_id = get_run_id(args.tag)
    run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {run_id}")
    print(f"{'='*60}")

    meta = {
        "run_id": run_id,
        "tag": args.tag,
        "date": datetime.now().isoformat(),
        "net_file": args.net_file,
        "route_file": args.route_file,
        "total_timesteps": args.total_timesteps,
        "max_hours": args.max_hours,
        "num_seconds": args.num_seconds,
        "ts_ids": TS_IDS,
        "ts_names": TS_NAMES,
        "approach": "PPO with parameter sharing (SuperSuit vec env)",
        "hyperparams": {
            "lr": 3e-4, "n_steps": 2048, "batch_size": 128,
            "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
            "ent_coef": 0.01, "delta_time": 5, "yellow_time": 2,
            "min_green": 5, "max_green": 50,
        },
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Phase 1: Baseline
    print("\n[1/3] Running fixed-time baseline (real signal programs)...")
    t0 = time.time()
    baseline_rewards, bl_steps = run_baseline(
        args.net_file, args.route_file, args.num_seconds,
    )
    bl_time = time.time() - t0
    print(f"  Baseline total reward: {sum(baseline_rewards.values()):.1f} "
          f"({bl_time:.0f}s)")

    # Phase 2: Train
    print("\n[2/3] Training PPO agents...")
    model, train_time = train_ppo(
        args.net_file, args.route_file, args.num_seconds,
        args.total_timesteps, run_dir, max_seconds,
    )
    meta["train_time_s"] = train_time
    meta["actual_timesteps"] = model.num_timesteps

    # Phase 3: Evaluate
    print("\n[3/3] Evaluating trained model vs baseline...")
    rl_rewards, _ = run_evaluation(
        args.net_file, args.route_file, args.num_seconds, model,
    )

    # Results
    rows = []
    print(f"\n{'='*70}")
    print(f"{'Intersection':<30s} {'Baseline':>10s} {'RL':>10s} {'Δ%':>8s}")
    print(f"{'-'*70}")
    for ts_id in TS_IDS:
        name = TS_NAMES.get(ts_id, ts_id)
        bl = baseline_rewards.get(ts_id, 0)
        rl = rl_rewards.get(ts_id, 0)
        pct = ((rl - bl) / abs(bl) * 100) if bl != 0 else 0
        print(f"  {name:<28s} {bl:>10.1f} {rl:>10.1f} {pct:>+7.1f}%")
        rows.append({
            "intersection": name,
            "tls_id": ts_id,
            "baseline_reward": bl,
            "rl_reward": rl,
            "improvement_pct": pct,
        })

    bl_total = sum(baseline_rewards.values())
    rl_total = sum(rl_rewards.values())
    total_pct = ((rl_total - bl_total) / abs(bl_total) * 100) if bl_total != 0 else 0
    print(f"{'-'*70}")
    print(f"  {'TOTAL':<28s} {bl_total:>10.1f} {rl_total:>10.1f} {total_pct:>+7.1f}%")
    print(f"{'='*70}")

    df_results = pd.DataFrame(rows)
    df_results.to_csv(os.path.join(run_dir, "results.csv"), index=False)

    meta["baseline_total_reward"] = bl_total
    meta["rl_total_reward"] = rl_total
    meta["improvement_pct"] = total_pct
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_wall = time.time() - t0
    print(f"\nTotal wall time: {total_wall/60:.1f}min")
    print(f"Artifacts: {run_dir}/")
    print()
    compare_experiments()


if __name__ == "__main__":
    main()
