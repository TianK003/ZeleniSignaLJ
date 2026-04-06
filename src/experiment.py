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

from config import (
    TS_IDS, TS_NAMES, NUM_AGENTS, STEPS_PER_EPISODE,
    NUM_SECONDS, DELTA_TIME, YELLOW_TIME, MIN_GREEN, MAX_GREEN, REWARD_FN,
    TOTAL_DAILY_CARS, WARMUP_SECONDS,
    LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS,
    GAMMA, GAE_LAMBDA, ENT_COEF, CLIP_RANGE,
)
from agent_filter import AgentFilterWrapper
from tls_programs import parse_original_programs, restore_non_target_programs
from demand_math import get_vph
from generate_demand import write_demand_xml
import random

from sumo_rl.environment.observations import ObservationFunction
from gymnasium import spaces

CURRENT_HOUR = 0.0
CURRENT_VPH = 1000.0

class TimeEncodedObservationFunction(ObservationFunction):
    """
    Extends the default observation function with cyclically encoded time:
    sin(t) and cos(t), giving the AI a smooth perception of the time of day.
    """
    def __init__(self, ts):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        
        global CURRENT_HOUR
        time_seconds = CURRENT_HOUR * 3600.0 + self.ts.env.sim_step
        sin_time = float(np.sin(2 * np.pi * time_seconds / 86400.0))
        cos_time = float(np.cos(2 * np.pi * time_seconds / 86400.0))
        
        observation = np.array(phase_id + min_green + density + queue + [sin_time, cos_time], dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        base_size = self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes)
        low = np.concatenate([np.zeros(base_size, dtype=np.float32), [-1.0, -1.0]])
        high = np.concatenate([np.ones(base_size, dtype=np.float32), [1.0, 1.0]])
        return spaces.Box(low=low, high=high)

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


class TrainingLogCallback(BaseCallback):
    """Log training metrics to CSV and print progress.

    SuperSuit vec envs don't propagate SB3 Monitor info["episode"] stats,
    so we track rewards/dones directly from the rollout locals.
    Each "step" in the vec env = one agent's step. A done=True means that
    agent's episode ended. We accumulate per-agent episode rewards.

    Also tracks per-step reward mean for the training curve (useful even
    before episodes complete).
    """
    def __init__(self, log_path, print_freq=5000, steps_per_episode=3600,
                 verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.print_freq = print_freq
        self.steps_per_episode = steps_per_episode
        self.start_time = None
        self.episode_rewards = []
        self.log_rows = []
        # Running reward accumulator per sub-env in the vec env
        self._running_rewards = None
        # Per-step reward tracking (for training curves even without episodes)
        self._step_rewards = []
        self._step_log_rows = []

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        rewards = self.locals.get("rewards", np.array([]))
        dones = self.locals.get("dones", np.array([]))

        # Initialize running reward accumulators on first step
        if self._running_rewards is None:
            self._running_rewards = np.zeros(len(rewards), dtype=np.float64)

        self._running_rewards += rewards

        # Track per-step mean reward for training curves
        step_mean_reward = float(np.mean(rewards))
        self._step_rewards.append(step_mean_reward)

        # Log per-step data every 100 steps (for training curves)
        if self.num_timesteps % 100 == 0 and len(self._step_rewards) > 0:
            recent_mean = float(np.mean(self._step_rewards[-100:]))
            self._step_log_rows.append({
                "timestep": self.num_timesteps,
                "reward_step_mean": recent_mean,
                "reward_step_sum": float(np.sum(self._step_rewards[-100:])),
                "elapsed_s": time.time() - self.start_time,
            })

        # When a sub-env is done, record its episode reward and reset
        for i, d in enumerate(dones):
            if d:
                ep_reward = float(self._running_rewards[i])
                self.episode_rewards.append(ep_reward)
                self.log_rows.append({
                    "timestep": self.num_timesteps,
                    "episode": len(self.episode_rewards),
                    "reward": ep_reward,
                    "elapsed_s": time.time() - self.start_time,
                })
                self._running_rewards[i] = 0.0

        if self.num_timesteps % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            sps = self.num_timesteps / elapsed if elapsed > 0 else 0

            # Show step-level reward (SuperSuit VecEnv masks done signals,
            # so episode-level tracking doesn't work reliably)
            recent = self._step_rewards[-500:] if self._step_rewards else [0]
            avg_r = np.mean(recent)
            label = f"avg_step_reward(last500)={avg_r:.2f}"

            # Compute episodes from timesteps (reliable regardless of done signals)
            n_eps = self.num_timesteps // self.steps_per_episode
            print(f"    Step {self.num_timesteps}: "
                  f"{label}, ~{n_eps} episodes, "
                  f"{sps:.0f} steps/s, {elapsed/60:.1f}min")
        return True

    def _on_training_end(self):
        # Save episode-level log
        if self.log_rows:
            df = pd.DataFrame(self.log_rows)
            df.to_csv(self.log_path, index=False)
            print(f"  Training log saved: {self.log_path} "
                  f"({len(self.log_rows)} episodes)")

        # Save step-level log for training curves
        if self._step_log_rows:
            step_log_path = self.log_path.replace(
                "training_log.csv", "training_steps.csv"
            )
            df_steps = pd.DataFrame(self._step_log_rows)
            df_steps.to_csv(step_log_path, index=False)
            print(f"  Step-level training log saved: {step_log_path} "
                  f"({len(self._step_log_rows)} data points)")


# ── Environment creation ──

def make_train_env(net_file, route_file, num_seconds):
    """
    Create a PettingZoo parallel env → AgentFilter → SuperSuit vec env.

    Key insight: without filtering, SuperSuit + PPO trains one shared
    policy for ALL 37 traffic lights. This is terrible because:
    1. A single policy can't handle 37 structurally different intersections
    2. RL-controlling the 32 non-target intersections makes them worse
    3. Worse non-target traffic cascades into the 5 target intersections

    With AgentFilterWrapper, only the 5 target TLS are RL agents.
    The other 32 keep their real signal programs (default action=0).
    This means PPO only needs to learn a policy for 5 similar intersections.
    """
    env = sumo_rl.parallel_env(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=num_seconds,
        reward_fn=REWARD_FN,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        sumo_warnings=False,
        additional_sumo_cmd="--ignore-route-errors",
        observation_class=TimeEncodedObservationFunction,
    )
    # Patch: sumo-rl 1.4.5 doesn't set render_mode, but SuperSuit expects it
    if not hasattr(env.unwrapped, "render_mode"):
        env.unwrapped.render_mode = None
        
    env.unwrapped.warmup_seconds = WARMUP_SECONDS

    # CRITICAL: Filter to only target intersections BEFORE SuperSuit
    # Non-target TLS run their original SUMO signal programs (restored
    # from net.xml after sumo-rl replaces them during env creation)
    env = AgentFilterWrapper(
        env, target_agents=TS_IDS, net_file=net_file, default_action=0
    )

    # Pad observations so all 5 target agents have the same obs space size
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
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


def pad_obs(obs, target_size):
    """Zero-pad observation to match the padded training obs space."""
    if len(obs) >= target_size:
        return obs[:target_size]
    padded = np.zeros(target_size, dtype=np.float32)
    padded[:len(obs)] = obs
    return padded


def run_evaluation(net_file, route_file, num_seconds, model):
    """
    Evaluate trained PPO model on the 5 target traffic signals.
    Non-target TLS run their original SUMO signal programs, restored
    from the .net.xml file after sumo-rl replaces them.

    This ensures a fair comparison with the fixed_ts=True baseline:
    non-target traffic flows identically, and only the 5 target TLS
    differ (RL policy vs. original program).
    """
    env = make_eval_env(net_file, route_file, num_seconds, fixed_ts=False)
    observations = env.reset()

    # Restore original SUMO programs for non-target TLS
    original_programs = parse_original_programs(net_file)
    restore_non_target_programs(env, TS_IDS, original_programs)

    # Get the padded obs size from the trained model
    obs_size = model.observation_space.shape[0]

    target_set = set(TS_IDS)
    rewards = {ts_id: 0.0 for ts_id in TS_IDS}
    done = False
    steps = 0

    while not done:
        actions = {}
        for ts_id in env.ts_ids:
            if ts_id in target_set:
                # Target intersection: use RL model
                obs = pad_obs(observations[ts_id], obs_size)
                action, _ = model.predict(obs, deterministic=True)
                ts = env.traffic_signals[ts_id]
                actual_n_actions = ts.action_space.n
                actions[ts_id] = int(action) % actual_n_actions
            else:
                # Non-target: action is ignored (set_next_phase is patched)
                # SUMO runs the restored original program
                actions[ts_id] = 0

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
              max_seconds=None, run_curriculum=False, log_curriculum=False):
    """
    Train PPO with parameter sharing via SuperSuit.
    All traffic signals share one policy (standard IPPO approach).
    Returns the trained model.
    """
    if run_curriculum:
        train_route_file = route_file.replace(".rou.xml", "_train.rou.xml")
        if train_route_file == route_file:
            train_route_file += "_train.xml"
        tmp_trips = train_route_file.replace(".rou.xml", "_trips.xml")
        
        print("  [Setup] Generating initial training demand slice for the environment...")
        initial_hour = random.uniform(0, 24)
        initial_vph = get_vph(initial_hour, TOTAL_DAILY_CARS)
        write_demand_xml([(0, num_seconds, initial_vph/3600.0)], net_file, tmp_trips, train_route_file)
        
        print("  Creating vectorized training environment with curriculum routes...")
        env = make_train_env(net_file, train_route_file, num_seconds)
    else:
        print("  Creating vectorized training environment...")
        env = make_train_env(net_file, route_file, num_seconds)

    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # All hyperparameters from src/config.py — single source of truth
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        clip_range=CLIP_RANGE,
        verbose=0,
        tensorboard_log=os.path.join(run_dir, "tb_logs"),
    )

    # Callbacks
    log_path = os.path.join(run_dir, "training_log.csv")
    callbacks = [TrainingLogCallback(log_path, print_freq=5000,
                                     steps_per_episode=STEPS_PER_EPISODE)]
    if max_seconds:
        callbacks.append(TimeLimitCallback(max_seconds))

    stop_desc = f"{total_timesteps} steps"
    if max_seconds:
        stop_desc += f" or {max_seconds/3600:.1f}h"
    print(f"  Training PPO (stop: {stop_desc})...")

    t_start = time.time()
    
    if run_curriculum:
        print("\n  [Curriculum Mode ON] Generating random 1h slices of the day...")
        episodes = total_timesteps // STEPS_PER_EPISODE
        if episodes == 0: episodes = 1
            
        for ep in range(episodes):
            global CURRENT_HOUR, CURRENT_VPH
            
            if ep > 0:
                hour_of_day = random.uniform(0, 24)
                CURRENT_HOUR = hour_of_day
                vph = get_vph(hour_of_day, TOTAL_DAILY_CARS)
                CURRENT_VPH = vph
                vps = vph / 3600.0
                
                print(f"\n  -- Curriculum Episode {ep+1}/{episodes} --")
                print(f"     Random Time of day: {hour_of_day:.1f}h, Traffic Flow: {vph:.0f} cars/hour")
                
                write_demand_xml([(0, num_seconds, vps)], net_file, tmp_trips, train_route_file)
            else:
                CURRENT_HOUR = initial_hour
                CURRENT_VPH = initial_vph
                print(f"\n  -- Curriculum Episode {ep+1}/{episodes} --")
                print(f"     Random Time of day: {initial_hour:.1f}h, Traffic Flow: {initial_vph:.0f} cars/hour")
            
            # Force SB3 to reset the environment state to load the newly generated routes!
            model._last_obs = None

            model.learn(
                total_timesteps=STEPS_PER_EPISODE,
                callback=callbacks,
                reset_num_timesteps=False
            )
            
            if log_curriculum:
                # We do exact 1-to-1 comparison by running the baseline and deterministic RL
                # model on the exact same randomized routing file we just generated.
                import subprocess, json, sys, tempfile
                
                h_val = hour_of_day if ep > 0 else initial_hour
                v_val = vph if ep > 0 else initial_vph
                
                tmp_json = os.path.join(tempfile.gettempdir(), "eval_out.json")
                
                # 1. Baseline in Subprocess (Safe for Libsumo)
                subprocess.run([sys.executable, "src/eval_helper.py", "baseline", net_file, train_route_file, str(num_seconds), tmp_json])
                with open(tmp_json, "r") as f:
                    bl_r = json.load(f)
                bl_tot = sum(bl_r.values())
                
                # 2. RL Evaluation in Subprocess (Safe for Libsumo)
                tmp_model = os.path.join(run_dir, "tmp_eval_model")
                model.save(tmp_model)
                subprocess.run([sys.executable, "src/eval_helper.py", "evaluate", net_file, train_route_file, str(num_seconds), tmp_json, tmp_model+".zip", str(h_val)])
                with open(tmp_json, "r") as f:
                    rl_r = json.load(f)
                rl_tot = sum(rl_r.values())
                
                impr = ((rl_tot - bl_tot) / abs(bl_tot) * 100) if bl_tot != 0 else 0
                
                log_line = f"Ep {ep+1:03d} | Hour {h_val:04.1f}h | Traffic {v_val:4.0f} vph | Baseline: {bl_tot:7.1f} | RL: {rl_tot:7.1f} | Impr: {impr:+5.1f}%"
                print(f"     [LOG] {log_line}")
                with open(os.path.join(run_dir, "curriculum_progress.txt"), "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
            
            if max_seconds and (time.time() - t_start) >= max_seconds:
                print("  Time limit reached during curriculum learning.")
                break
    else:
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
    parser = argparse.ArgumentParser(
        description="Run reproducible experiment",
        epilog=(
            "Timestep math: 1 episode = num_seconds/delta_time * num_agents "
            "= 3600/5 * 5 = 3600 SB3 timesteps. "
            "Use --episode_count for human-readable episode counts."
        ),
    )
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml")
    parser.add_argument("--route_file", type=str,
                        default="data/routes/routes.rou.xml")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Total SB3 timesteps (1 episode = 3600 steps)")
    parser.add_argument("--episode_count", type=int, default=None,
                        help="Number of full episodes to train (converted to timesteps)")
    parser.add_argument("--max_hours", type=float, default=None)
    parser.add_argument("--num_seconds", type=int, default=3600)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--compare_only", action="store_true")
    parser.add_argument("--curriculum", action="store_true", 
                        help="Train the AI on random 1-hour chunks drawn from the 24h mathematical traffic baseline (Curriculum Learning).")
    parser.add_argument("--log_curriculum", action="store_true",
                        help="During curriculum learning, exact evaluate against baseline every episode and append to log file.")
    args = parser.parse_args()

    if args.compare_only:
        compare_experiments()
        return

    # Convert episode_count to timesteps if provided
    # 1 episode = NUM_SECONDS / DELTA_TIME * NUM_AGENTS = 3600 SB3 timesteps
    if args.episode_count is not None:
        args.total_timesteps = args.episode_count * STEPS_PER_EPISODE
        print(f"  Episode count: {args.episode_count} episodes "
              f"= {args.total_timesteps} timesteps "
              f"({STEPS_PER_EPISODE} per episode)")
    elif args.total_timesteps is None:
        args.total_timesteps = 100_000  # default

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
            "lr": LEARNING_RATE, "n_steps": N_STEPS, "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS, "gamma": GAMMA, "gae_lambda": GAE_LAMBDA,
            "ent_coef": ENT_COEF, "clip_range": CLIP_RANGE,
            "delta_time": DELTA_TIME, "yellow_time": YELLOW_TIME,
            "min_green": MIN_GREEN, "max_green": MAX_GREEN,
        },
        "agent_filter": "5 target TLS only (other 32 run fixed-time)",
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Ensure route file exists for evaluation
    if not os.path.exists(args.route_file):
        print(f"\n  [Setup] Route file '{args.route_file}' not found.")
        print("  Generating an average traffic demand for testing (Phase 1 & 3)...")
        eval_vph = get_vph(12.0, TOTAL_DAILY_CARS)
        tmp_trips = args.route_file.replace(".rou.xml", "_eval_trips.xml")
        if tmp_trips == args.route_file: tmp_trips += "_trips.xml"
        os.makedirs(os.path.dirname(args.route_file), exist_ok=True)
        write_demand_xml([(0, args.num_seconds, eval_vph/3600.0)], args.net_file, tmp_trips, args.route_file)

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
        args.total_timesteps, run_dir, max_seconds, args.curriculum, args.log_curriculum
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
