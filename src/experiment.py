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
import sys
import time

# ── SUMO_HOME Detection (HPC Compatibility) ────────────────────────────────
if "SUMO_HOME" not in os.environ:
    for path in [
        os.path.join(os.environ.get("HOME", ""), "sumo_src"),
        "/usr/share/sumo",
        "/usr/local/share/sumo",
        "C:\\Program Files (x86)\\Eclipse\\Sumo" if sys.platform == "win32" else ""
    ]:
        if path and os.path.exists(path):
            os.environ["SUMO_HOME"] = path
            break

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
    MORNING_RUSH_START, MORNING_RUSH_SECONDS,
    EVENING_RUSH_START, EVENING_RUSH_SECONDS,
    OFFPEAK_SECONDS,
    LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS,
    GAMMA, GAE_LAMBDA, ENT_COEF, CLIP_RANGE,
)
from agent_filter import AgentFilterWrapper
from tls_programs import parse_original_programs, restore_non_target_programs
from demand_math import get_vph
from generate_demand import write_demand_xml
import random
import uuid

from sumo_rl.environment.observations import ObservationFunction
import gymnasium as gym
from gymnasium import spaces

CURRENT_HOUR = 0.0
CURRENT_VPH = 1000.0
ACTIVE_REWARD_FN = REWARD_FN  # overridden by --reward_fn CLI flag
ACTIVE_LEARNING_RATE = LEARNING_RATE  # overridden by --learning_rate CLI flag
ACTIVE_ENT_COEF = ENT_COEF  # overridden by --ent_coef CLI flag

# ── Scenario presets (route file, duration, time-of-day encoding) ─────────
SCENARIO_PRESETS = {
    "morning_rush": {
        "route_file": "data/routes/routes_morning_rush.rou.xml",
        "rl_seconds": MORNING_RUSH_SECONDS,   # 14400 (4h)
        "start_hour": MORNING_RUSH_START,      # 6.0
    },
    "evening_rush": {
        "route_file": "data/routes/routes_evening_rush.rou.xml",
        "rl_seconds": EVENING_RUSH_SECONDS,    # 14400 (4h)
        "start_hour": EVENING_RUSH_START,      # 14.0
    },
    "offpeak": {
        "route_file": "data/routes/routes_offpeak.rou.xml",
        "rl_seconds": OFFPEAK_SECONDS,         # 3600 (1h)
        "start_hour": 12.0,
    },
    "uniform": {
        "route_file": "data/routes/routes.rou.xml",
        "rl_seconds": 3600,
        "start_hour": 0.0,
    },
}

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
        low = np.concatenate([np.zeros(base_size, dtype=np.float32), np.array([-1.0, -1.0], dtype=np.float32)])
        high = np.concatenate([np.ones(base_size, dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)])
        return spaces.Box(low=low, high=high)

RESULTS_DIR = "results"
EXPERIMENTS_DIR = "results/experiments"


def get_run_id(tag=""):
    """
    Generate a unique run ID using timestamp, tag, and SLURM_JOB_ID/UUID 
    to prevent folder collisions on HPC clusters.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Priority 1: SLURM Job ID (if on HPC)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if job_id:
        ts = f"{ts}_{job_id}"
    else:
        # Priority 2: Random suffix (for local parallel runs)
        ts = f"{ts}_{uuid.uuid4().hex[:4]}"
        
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


class EntropyAnnealingCallback(BaseCallback):
    """Linearly anneal entropy coefficient from start to end over training."""
    def __init__(self, ent_start, ent_end, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.ent_start = ent_start
        self.ent_end = ent_end
        self.total_timesteps = total_timesteps

    def _on_step(self):
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        new_ent = self.ent_start + (self.ent_end - self.ent_start) * progress
        self.model.ent_coef = new_ent
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
            sys.stdout.flush()
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


class ExplicitCheckpointCallback(BaseCallback):
    """Save model explicitly every N on_step calls with metadata JSON."""
    def __init__(self, save_path, save_freq, log_callback=None,
                 baseline_total_reward=None, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.log_callback = log_callback
        self.baseline_total_reward = baseline_total_reward
        self._save_count = 0

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self._save_count += 1
            name = f"ppo_policy_{self.num_timesteps}steps"
            path = os.path.join(self.save_path, name)
            self.model.save(path)
            # Save metadata
            avg_r = 0.0
            if self.log_callback and self.log_callback._step_rewards:
                recent = self.log_callback._step_rewards[-500:]
                avg_r = float(np.mean(recent))
            meta = {
                "timesteps": self.num_timesteps,
                "rollout": self._save_count,
                "avg_step_reward_last500": avg_r,
                "baseline_total_reward": self.baseline_total_reward,
            }
            with open(os.path.join(self.save_path, f"{name}.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  [Checkpoint] {name}.zip ({self.num_timesteps} steps, avg_r={avg_r:.2f})")
            sys.stdout.flush()
        return True


# ── Environment creation ──

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnvWrapper

class FlattenMultiAgentVecEnv(VecEnvWrapper):
    """
    SubprocVecEnv handles SB3 vectorization by sending commands to child processes.
    However, when PettingZoo is used underneath, each child process returns batches of shape (num_agents, obs_dim).
    SB3 natively assumes each child returns a single shape (obs_dim), so it stacks them into (num_cpus, num_agents, obs_dim).
    This wrapper effortlessly flattens the output natively to (num_cpus * num_agents, obs_dim) so PPO accepts it.
    """
    def __init__(self, venv, num_agents):
        self.num_agents = num_agents
        v_envs = getattr(venv, 'num_envs', 1)
        # IMPORTANT: call super().__init__ FIRST, then override num_envs.
        # VecEnvWrapper.__init__ -> VecEnv.__init__ sets self.num_envs = venv.num_envs,
        # which would overwrite our multiplied value if we set it before.
        super().__init__(venv)
        # Override num_envs AFTER parent init so PPO sees the flattened agent count.
        self.num_envs = v_envs * num_agents
        # DummyVecEnv.observation_space is now (num_agents, obs_dim) after the
        # GymnasiumSubEnv fix. PPO needs to see (obs_dim,) per agent, so override.
        batched = venv.observation_space
        obs_dim = batched.shape[-1]
        self.observation_space = gym.spaces.Box(
            low=batched.low.reshape(-1, obs_dim)[0],
            high=batched.high.reshape(-1, obs_dim)[0],
            dtype=np.float32,
        )

    def reset(self):
        res = self.venv.reset()
        if isinstance(res, tuple):
            obs, info = res
            return obs.reshape((self.num_envs, -1)), info
        else:
            return res.reshape((self.num_envs, -1))
        
    def step_async(self, actions):
        # actions shape: (num_cpus * num_agents,)
        acts = actions.reshape((self.venv.num_envs, self.num_agents))
        self.venv.step_async(acts)
        
    def step_wait(self):
        obs, _rew_scalar, done, info = self.venv.step_wait()
        obs = obs.reshape((self.num_envs, -1))

        flat_done = np.zeros(self.num_envs, dtype=bool)
        flat_rew = np.zeros(self.num_envs, dtype=np.float32)
        flat_info = []
        idx = 0
        for worker_info in info:
            agents_info = worker_info.get("agents_info", [])
            for item in agents_info:
                flat_done[idx] = bool(item.get('__real_tm', 0) or item.get('__real_tc', 0))
                flat_rew[idx] = float(item.get('__real_rew', 0.0))
                flat_info.append(item)
                idx += 1

        return obs, flat_rew, flat_done, flat_info



class GymnasiumSubEnv(gym.Env):
    """
    SubprocVecEnv forces all child node environments to formally subclass `gymnasium.Env`.
    Since PettingZoo outputs a VectorEnv, we wrap it cleanly here to bypass SB3's restrictive typings.
    """
    def __init__(self, venv):
        self.venv = venv
        # venv is a MarkovVectorEnv (num_agents sub-envs). Its observation_space
        # reports a single-agent shape (obs_dim,), but reset()/step() return
        # (num_agents, obs_dim). Advertise the true batched shape so DummyVecEnv
        # allocates a buffer large enough to hold the full batch.
        single_obs = venv.observation_space
        n = getattr(venv, 'num_envs', 1)
        self.observation_space = gym.spaces.Box(
            low=np.tile(single_obs.low, (n, 1)),
            high=np.tile(single_obs.high, (n, 1)),
            dtype=np.float32,
        )
        self.action_space = venv.action_space
        self.render_mode = None
        self.num_envs = n

    def step(self, action):
        obs, rews, tms, tcs, infs = self.venv.step(action)
        # Pack the real booleans and per-agent rewards inside the infs dictionaries
        # This blinds SubprocVecEnv from trying to evaluate array truth values
        for i in range(self.num_envs):
            infs[i]['__real_tm'] = float(tms[i])
            infs[i]['__real_tc'] = float(tcs[i])
            infs[i]['__real_rew'] = float(rews[i])  # per-agent reward extracted in step_wait
        # Return scalar reward so DummyVecEnv's buf_rews (scalar slot) accepts it.
        # We wrap 'infs' in a dictionary because SubprocVecEnv forces mutation on the info dict.
        return obs, float(rews.sum()), False, False, {"agents_info": infs}

    def reset(self, seed=None, options=None):
        return self.venv.reset(seed=seed, options=options)
    
    def close(self):
        self.venv.close()

def make_env_factory(net_file, route_file, num_seconds, route_dir=None):
    """
    Delays the SUMO instantiations so 'libsumo' only boots up INSIDE the child CPU nodes!
    If route_dir is provided, each subprocess picks its own random route file for diversity.
    """
    def _init():
        actual_route = route_file
        if route_dir:
            import glob as _g
            files = sorted(_g.glob(os.path.join(route_dir, "*.rou.xml")))
            if files:
                # Each subprocess reads a different original file directly (no copy).
                # Use PID for selection — guarantees diversity even with forked random state.
                actual_route = files[os.getpid() % len(files)]

        env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=actual_route,
            use_gui=False,
            num_seconds=num_seconds,
            reward_fn=ACTIVE_REWARD_FN,
            delta_time=DELTA_TIME,
            yellow_time=YELLOW_TIME,
            min_green=MIN_GREEN,
            max_green=MAX_GREEN,
            sumo_warnings=False,
            additional_sumo_cmd="--ignore-route-errors",
            observation_class=TimeEncodedObservationFunction,
        )
        if not hasattr(env.unwrapped, "render_mode"):
            env.unwrapped.render_mode = None
        env.unwrapped.warmup_seconds = WARMUP_SECONDS

        env = AgentFilterWrapper(
            env, target_agents=TS_IDS, net_file=net_file, default_action=0
        )
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        return GymnasiumSubEnv(env)
    return _init

def build_vectorized_env(net_file, route_file, num_seconds, num_cpus, route_dir=None):
    factories = [make_env_factory(net_file, route_file, num_seconds, route_dir) for _ in range(num_cpus)]
    if num_cpus > 1:
        # Multiprocessing across CPU nodes safely
        base_env = SubprocVecEnv(factories)
    else:
        # Single process (local testing)
        base_env = DummyVecEnv(factories)

    # Flatten the matrices so SB3 PPO accepts it cleanly
    flat_env = FlattenMultiAgentVecEnv(base_env, num_agents=NUM_AGENTS)
    return flat_env


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
              max_seconds=None, run_curriculum=False, log_curriculum=False, num_cpus=1, resume_model_path=None,
              entropy_annealing=False, episodes_per_save=10, route_files=None,
              baseline_total_reward=None):
    """
    Train PPO with parameter sharing via SuperSuit.
    All traffic signals share one policy (standard IPPO approach).

    Args:
        route_files: Optional list of route file paths for route randomization.
                     Each episode picks a random file from this list, copying it
                     to the training route path so the env picks up new OD pairs.
        baseline_total_reward: Total reward from the fixed-time baseline run.
                               Used to compute improvement % in checkpoint metadata.
    Returns the trained model.
    """
    use_route_rotation = route_files and len(route_files) > 1

    if run_curriculum or use_route_rotation:
        train_route_file = route_file.replace(".rou.xml", "_train.rou.xml")
        if train_route_file == route_file:
            train_route_file += "_train.xml"
        tmp_trips = train_route_file.replace(".rou.xml", "_trips.xml")

        if use_route_rotation:
            # Route randomization: copy first route file to seed the env
            import shutil
            initial_route = random.choice(route_files)
            shutil.copy(initial_route, train_route_file)
            route_dir = os.path.dirname(route_files[0])
            print(f"  [Route randomization] {len(route_files)} route files available")
            print(f"  Initial route: {os.path.basename(initial_route)}")
            if num_cpus > 1:
                print(f"  Per-subprocess diversity: each of {num_cpus} workers picks its own route file")
        else:
            route_dir = None

        if run_curriculum:
            print("  [Setup] Generating initial training demand slice for the environment...")
            initial_hour = random.uniform(0, 24)
            initial_vph = get_vph(initial_hour, TOTAL_DAILY_CARS)
            write_demand_xml([(0, num_seconds, initial_vph/3600.0)], net_file, tmp_trips, train_route_file)

        print(f"  Creating vectorized training environment on {num_cpus} CPUs...")
        # Pass route_dir so each subprocess can independently pick a route file
        env = build_vectorized_env(net_file, train_route_file, num_seconds, num_cpus,
                                   route_dir=route_dir if not run_curriculum else None)
    else:
        print(f"  Creating vectorized training environment on {num_cpus} CPUs...")
        env = build_vectorized_env(net_file, route_file, num_seconds, num_cpus)

    # Scale BATCH_SIZE to maintain ~20 mini-batches per epoch regardless of num_cpus.
    # With num_cpus=1: buffer = 720*5 = 3600, batch=180, 20 mini-batches/epoch.
    # With num_cpus=128: buffer = 720*640 = 460800, batch=180*128=23040, still 20 mini-batches/epoch.
    buffer_size = N_STEPS * env.num_envs
    actual_batch_size = min(BATCH_SIZE * num_cpus, buffer_size)
    # Ensure buffer_size is divisible by actual_batch_size (SB3 requirement)
    while buffer_size % actual_batch_size != 0 and actual_batch_size > BATCH_SIZE:
        actual_batch_size -= BATCH_SIZE
    if buffer_size % actual_batch_size != 0:
        actual_batch_size = BATCH_SIZE  # fallback to unscaled

    # Compute actual steps_per_episode from scenario duration (not hardcoded RL_SECONDS=3600)
    rl_seconds = num_seconds - WARMUP_SECONDS
    actual_steps_per_episode = (rl_seconds // DELTA_TIME) * NUM_AGENTS

    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Parallelization: {env.num_envs} active agents ({num_cpus} CPUs x {NUM_AGENTS} agents)")
    print(f"  Rollout buffer: {buffer_size:,} transitions ({N_STEPS} steps x {env.num_envs} envs)")
    print(f"  Batch size: {actual_batch_size} (scaled from {BATCH_SIZE} for {num_cpus} CPUs, "
          f"{buffer_size // actual_batch_size} mini-batches/epoch)")
    print(f"  Steps per SUMO episode: {actual_steps_per_episode} "
          f"({rl_seconds}s RL / {DELTA_TIME}s delta x {NUM_AGENTS} agents)")

    # All hyperparameters from src/config.py — single source of truth
    if resume_model_path:
        print(f"  Resuming model from {resume_model_path}...")
        model = PPO.load(
            resume_model_path,
            env=env,
            custom_objects={
                "learning_rate": ACTIVE_LEARNING_RATE,
                "n_steps": N_STEPS,
                "batch_size": actual_batch_size,
                "n_epochs": N_EPOCHS,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "ent_coef": ACTIVE_ENT_COEF,
                "clip_range": CLIP_RANGE
            }
        )
        model.tensorboard_log = os.path.join(run_dir, "tb_logs")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=ACTIVE_LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=actual_batch_size,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ent_coef=ACTIVE_ENT_COEF,
            clip_range=CLIP_RANGE,
            verbose=0,
            tensorboard_log=os.path.join(run_dir, "tb_logs"),
        )

    # Training Log callback (use actual scenario duration, not hardcoded STEPS_PER_EPISODE)
    log_path = os.path.join(run_dir, "training_log.csv")
    log_callback = TrainingLogCallback(log_path, print_freq=5000,
                                       steps_per_episode=actual_steps_per_episode)
    callbacks = [log_callback]

    # Checkpoint directory for explicit saves
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # For the non-curriculum path (single learn() call), add a callback that
    # saves explicitly. save_freq = n_steps so it saves after every rollout.
    ckpt_callback = ExplicitCheckpointCallback(
        save_path=checkpoint_dir, save_freq=N_STEPS,
        log_callback=log_callback, baseline_total_reward=baseline_total_reward)
    callbacks.append(ckpt_callback)

    print(f"  [Checkpoints] Saving every rollout ({N_STEPS} steps) + every {episodes_per_save} episodes to {checkpoint_dir}/")

    if max_seconds:
        callbacks.append(TimeLimitCallback(max_seconds))
    if entropy_annealing:
        ent_end = 0.01
        callbacks.append(EntropyAnnealingCallback(ACTIVE_ENT_COEF, ent_end, total_timesteps))
        print(f"  Entropy annealing: {ACTIVE_ENT_COEF} -> {ent_end} over {total_timesteps} steps")

    stop_desc = f"{total_timesteps} steps"
    if max_seconds:
        stop_desc += f" or {max_seconds/3600:.1f}h"
    print(f"  Training PPO (stop: {stop_desc})...")

    t_start = time.time()

    # One rollout = N_STEPS env steps across all parallel envs.
    # With num_cpus=128: 720 * 640 = 460,800 transitions per PPO update.
    rollout_timesteps = N_STEPS * env.num_envs

    # Wrap training in try/finally so the model is ALWAYS saved, even on crash/timeout
    try:
        if run_curriculum or use_route_rotation:
            import shutil as _shutil
            desc_parts = []
            if run_curriculum:
                desc_parts.append("Curriculum")
            if use_route_rotation:
                desc_parts.append(f"Route randomization ({len(route_files)} files)")
            print(f"\n  [{' + '.join(desc_parts)}] Per-episode training loop...")

            # Each PPO update collects experience from num_cpus parallel SUMO
            # instances. With --episode_count 300 and --num_cpus 128, we need
            # ceil(300/128) = 3 PPO updates, not 300.
            requested_episodes = total_timesteps // actual_steps_per_episode
            if requested_episodes == 0: requested_episodes = 1
            episodes_per_update = max(1, num_cpus)
            ppo_updates = max(1, -(-requested_episodes // episodes_per_update))  # ceil division

            print(f"  Requested episodes: {requested_episodes}")
            print(f"  Episodes per PPO update: {episodes_per_update} ({num_cpus} CPUs)")
            print(f"  PPO updates to perform: {ppo_updates}")

            for ep in range(ppo_updates):
                global CURRENT_HOUR, CURRENT_VPH

                if run_curriculum:
                    # Curriculum: generate new demand each iteration, force env reset
                    if ep > 0:
                        hour_of_day = random.uniform(0, 24)
                        CURRENT_HOUR = hour_of_day
                        vph = get_vph(hour_of_day, TOTAL_DAILY_CARS)
                        CURRENT_VPH = vph
                        vps = vph / 3600.0

                        print(f"\n  -- Curriculum PPO Update {ep+1}/{ppo_updates} "
                              f"(~episodes {ep*episodes_per_update+1}-{min((ep+1)*episodes_per_update, requested_episodes)}) --")
                        print(f"     Random Time of day: {hour_of_day:.1f}h, Traffic Flow: {vph:.0f} cars/hour")

                        write_demand_xml([(0, num_seconds, vps)], net_file, tmp_trips, train_route_file)
                    else:
                        CURRENT_HOUR = initial_hour
                        CURRENT_VPH = initial_vph
                        print(f"\n  -- Curriculum PPO Update {ep+1}/{ppo_updates} "
                              f"(~episodes 1-{min(episodes_per_update, requested_episodes)}) --")
                        print(f"     Random Time of day: {initial_hour:.1f}h, Traffic Flow: {initial_vph:.0f} cars/hour")

                    # Curriculum needs env reset to load new demand files
                    model._last_obs = None

                    if use_route_rotation and ep > 0:
                        chosen = random.choice(route_files)
                        _shutil.copy(chosen, train_route_file)
                        print(f"     Route: {os.path.basename(chosen)}")
                else:
                    # Pure route-rotation (no curriculum): DON'T force env reset.
                    # Let SUMO episodes play out fully across multiple rollouts so
                    # the agent sees the entire rush-hour period, not just the first 25%.
                    # Route diversity comes from per-subprocess selection (fix 5).
                    if ep % 25 == 0:
                        print(f"\n  -- Route-Randomized PPO Update {ep+1}/{ppo_updates} "
                              f"(~episodes {ep*episodes_per_update+1}-{min((ep+1)*episodes_per_update, requested_episodes)}) --")

                model.learn(
                    total_timesteps=rollout_timesteps,
                    callback=callbacks,
                    reset_num_timesteps=False
                )

                # Checkpoint save logic:
                # Convert episodes_per_save to PPO-update interval.
                # If there are few PPO updates (e.g., 3 with 128 CPUs), save EVERY update.
                equiv_episodes = min((ep + 1) * episodes_per_update, requested_episodes)
                # How many equivalent episodes pass per PPO update?
                # Save when accumulated equivalent episodes cross an episodes_per_save boundary.
                prev_equiv = min(ep * episodes_per_update, requested_episodes) if ep > 0 else 0
                crossed_boundary = (equiv_episodes // episodes_per_save) > (prev_equiv // episodes_per_save)
                is_last = (ep == ppo_updates - 1)
                # Always save if <= 10 total PPO updates (fast training, every checkpoint matters)
                few_updates = (ppo_updates <= 10)

                if crossed_boundary or is_last or few_updates:
                    ckpt_name = f"ppo_policy_{equiv_episodes}ep"
                    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
                    model.save(ckpt_path)

                    # Compute training stats for checkpoint metadata
                    elapsed = time.time() - t_start
                    recent_rewards = log_callback._step_rewards[-500:] if log_callback._step_rewards else [0]
                    avg_step_reward = float(np.mean(recent_rewards))
                    total_step_reward = float(np.sum(log_callback._step_rewards)) if log_callback._step_rewards else 0.0
                    improvement_pct = (
                        (avg_step_reward * actual_steps_per_episode - abs(baseline_total_reward or 0))
                        / abs(baseline_total_reward) * 100
                        if baseline_total_reward and baseline_total_reward != 0 else None
                    )

                    ckpt_meta = {
                        "ppo_update": ep + 1,
                        "total_ppo_updates": ppo_updates,
                        "equivalent_episodes": equiv_episodes,
                        "requested_episodes": requested_episodes,
                        "num_cpus": num_cpus,
                        "timesteps": model.num_timesteps,
                        "elapsed_s": elapsed,
                        "avg_step_reward_last500": avg_step_reward,
                        "total_step_reward_sum": total_step_reward,
                        "baseline_total_reward": baseline_total_reward,
                        "estimated_improvement_pct": improvement_pct,
                        "n_step_samples": len(log_callback._step_rewards),
                    }
                    meta_path = os.path.join(checkpoint_dir, f"{ckpt_name}.json")
                    with open(meta_path, "w") as f:
                        json.dump(ckpt_meta, f, indent=2)

                    msg = (f"  [Checkpoint] {ckpt_name}.zip (update {ep+1}/{ppo_updates}, "
                           f"~{equiv_episodes} episodes, "
                           f"avg_r={avg_step_reward:.2f}, {elapsed/60:.1f}min)")
                    print(msg)
                    sys.stdout.flush()  # Force flush even without PYTHONUNBUFFERED

                if log_curriculum and run_curriculum:
                    # We do exact 1-to-1 comparison by running the baseline and deterministic RL
                    # model on the exact same randomized routing file we just generated.
                    import subprocess, tempfile

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
                    print("  Time limit reached during per-episode training.")
                    break
        else:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                reset_num_timesteps=False if resume_model_path else True,
            )
    except (KeyboardInterrupt, Exception) as e:
        print(f"\n  [!] Training interrupted: {type(e).__name__}: {e}")
        print(f"  [!] Saving emergency checkpoint...")
        sys.stdout.flush()
    finally:
        # ALWAYS save the model, even on crash/timeout/interrupt
        emergency_path = os.path.join(checkpoint_dir, "ppo_model_latest")
        try:
            model.save(emergency_path)
            print(f"  [Checkpoint] Latest model saved: {emergency_path}.zip")
            sys.stdout.flush()
        except Exception as save_err:
            print(f"  [!] Failed to save emergency checkpoint: {save_err}")

    train_time = time.time() - t_start

    # Save final model
    model_path = os.path.join(run_dir, "ppo_shared_policy")
    model.save(model_path)
    print(f"  Model saved: {model_path}.zip")
    print(f"  Training time: {train_time/60:.1f}min "
          f"({model.num_timesteps} actual steps)")
    sys.stdout.flush()

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
    parser.add_argument("--route_file", type=str, default=None,
                        help="Override route file (ignores --scenario preset).")
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Total SB3 timesteps (1 episode = 3600 steps)")
    parser.add_argument("--episode_count", type=int, default=None,
                        help="Number of full episodes to train (converted to timesteps)")
    parser.add_argument("--max_hours", type=float, default=None)
    parser.add_argument("--num_seconds", type=int, default=None,
                        help="Total simulation duration (seconds). "
                             "Defaults to scenario rl_seconds + WARMUP_SECONDS.")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--compare_only", action="store_true")
    parser.add_argument("--curriculum", action="store_true", 
                        help="Train the AI on random 1-hour chunks drawn from the 24h mathematical traffic baseline (Curriculum Learning).")
    parser.add_argument("--log_curriculum", action="store_true",
                        help="During curriculum learning, exact evaluate against baseline every episode and append to log file.")
    parser.add_argument("--num_cpus", type=int, default=1,
                        help="Number of independent parallel SUMO CPU processes (useful for HPCs like Vega).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .zip file to resume training from.")
    parser.add_argument("--reward_fn", type=str, default=None,
                        choices=["queue", "pressure", "diff-waiting-time", "average-speed"],
                        help="Reward function (overrides config.py REWARD_FN).")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Override learning rate (default from config.py: 1e-3).")
    parser.add_argument("--ent_coef", type=float, default=None,
                        help="Override entropy coefficient (default from config.py: 0.05).")
    parser.add_argument("--entropy_annealing", action="store_true",
                        help="Linearly anneal entropy from ent_coef to 0.01 over training.")
    parser.add_argument("--episodes_per_save", type=int, default=50,
                        help="Save a model checkpoint every N episodes (default: 50). "
                             "With --num_cpus, this is measured in equivalent episodes "
                             "(PPO updates * num_cpus).")
    parser.add_argument("--scenario", type=str, default="uniform",
                        choices=list(SCENARIO_PRESETS.keys()),
                        help="Demand scenario. Sets route_file, num_seconds, and "
                             "CURRENT_HOUR from the scenario preset. "
                             "Use --route_file to override.")
    parser.add_argument("--route_dir", type=str, default=None,
                        help="Directory with multiple route files (*.rou.xml). "
                             "Each episode uses a random route file from this "
                             "directory for route-randomized training. "
                             "Generate with: python src/generate_demand.py "
                             "--scenario X --num_variants N --output_dir DIR")
    args = parser.parse_args()

    # Resolve scenario preset
    preset = SCENARIO_PRESETS[args.scenario]

    if args.route_file is None:
        args.route_file = preset["route_file"]

    if args.num_seconds is None:
        args.num_seconds = preset["rl_seconds"] + WARMUP_SECONDS

    if not args.curriculum:
        global CURRENT_HOUR
        CURRENT_HOUR = preset["start_hour"]

    # Override hyperparameters if specified via CLI
    if args.reward_fn is not None:
        global ACTIVE_REWARD_FN
        ACTIVE_REWARD_FN = args.reward_fn
    if args.learning_rate is not None:
        global ACTIVE_LEARNING_RATE
        ACTIVE_LEARNING_RATE = args.learning_rate
    if args.ent_coef is not None:
        global ACTIVE_ENT_COEF
        ACTIVE_ENT_COEF = args.ent_coef

    if args.compare_only:
        compare_experiments()
        return

    # Convert episode_count to timesteps if provided
    # Compute actual steps_per_episode from scenario duration (not hardcoded RL_SECONDS)
    rl_secs = args.num_seconds - WARMUP_SECONDS
    actual_spe = (rl_secs // DELTA_TIME) * NUM_AGENTS
    if args.episode_count is not None:
        args.total_timesteps = args.episode_count * actual_spe
        print(f"  Episode count: {args.episode_count} episodes "
              f"= {args.total_timesteps} timesteps "
              f"({actual_spe} per episode, {rl_secs}s RL)")
        if args.num_cpus > 1:
            ppo_upd = max(1, -(-args.episode_count // args.num_cpus))
            print(f"  With {args.num_cpus} CPUs: {ppo_upd} PPO updates "
                  f"({args.num_cpus} parallel episodes per update)")
    elif args.total_timesteps is None:
        args.total_timesteps = 100_000  # default

    # Scan route_dir for route files
    import glob as _glob
    route_files = None
    if args.route_dir:
        route_files = sorted(_glob.glob(os.path.join(args.route_dir, "*.rou.xml")))
        if not route_files:
            print(f"\n  ERROR: No .rou.xml files found in {args.route_dir}")
            print(f"  Generate with: python src/generate_demand.py "
                  f"--scenario {args.scenario} --num_variants 20 --output_dir {args.route_dir}")
            raise SystemExit(1)
        print(f"\n  Route randomization: {len(route_files)} route files in {args.route_dir}")

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
        "scenario": args.scenario,
        "date": datetime.now().isoformat(),
        "net_file": args.net_file,
        "route_file": args.route_file,
        "route_dir": args.route_dir,
        "route_randomization": len(route_files) if route_files else 0,
        "total_timesteps": args.total_timesteps,
        "max_hours": args.max_hours,
        "num_seconds": args.num_seconds,
        "ts_ids": TS_IDS,
        "ts_names": TS_NAMES,
        "reward_fn": ACTIVE_REWARD_FN,
        "approach": "PPO with parameter sharing (SuperSuit vec env)",
        "entropy_annealing": args.entropy_annealing,
        "num_cpus": args.num_cpus,
        "hyperparams": {
            "lr": ACTIVE_LEARNING_RATE, "n_steps": N_STEPS,
            "batch_size_base": BATCH_SIZE,
            "batch_size_note": f"Scaled to BATCH_SIZE*num_cpus at runtime (see training log)",
            "n_epochs": N_EPOCHS, "gamma": GAMMA, "gae_lambda": GAE_LAMBDA,
            "ent_coef": ACTIVE_ENT_COEF, "clip_range": CLIP_RANGE,
            "delta_time": DELTA_TIME, "yellow_time": YELLOW_TIME,
            "min_green": MIN_GREEN, "max_green": MAX_GREEN,
        },
        "agent_filter": "5 target TLS only (other 32 run fixed-time)",
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Ensure route file exists (skip check when --route_dir provides the actual files)
    if route_files:
        # With --route_dir, use the first route file as the base route for baseline eval
        args.route_file = route_files[0]
    elif not os.path.exists(args.route_file):
        if args.scenario in ("morning_rush", "evening_rush", "offpeak"):
            print(f"\n  ERROR: Route file not found: {args.route_file}")
            print(f"  Generate it first:")
            print(f"    python src/generate_demand.py --scenario {args.scenario}")
            raise SystemExit(1)
        else:
            print(f"\n  [Setup] Route file '{args.route_file}' not found.")
            print("  Generating an average traffic demand for testing...")
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
    bl_total = sum(baseline_rewards.values())
    model, train_time = train_ppo(
        args.net_file, args.route_file, args.num_seconds,
        args.total_timesteps, run_dir, max_seconds, args.curriculum, args.log_curriculum, args.num_cpus, args.resume,
        entropy_annealing=args.entropy_annealing, episodes_per_save=args.episodes_per_save,
        route_files=route_files, baseline_total_reward=bl_total,
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
