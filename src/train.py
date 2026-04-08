"""
Zeleni SignaLJ - PPO Training Script
======================================
Train PPO with parameter sharing for traffic signal control.
Uses PettingZoo + SuperSuit to vectorize the multi-agent env.

All traffic signals share one policy -- this is standard IPPO.
Each signal is treated as an independent sub-environment.

Usage:
    # Uniform demand (baseline, backwards-compatible)
    python src/train.py --total_timesteps 50000

    # Rush-hour scenarios (recommended for production training)
    python src/train.py --scenario morning_rush
    python src/train.py --scenario evening_rush

    # Vega HPC
    python src/train.py --scenario morning_rush --total_timesteps 2000000
"""

import argparse
import os
import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from config import (
    TS_IDS, DELTA_TIME, YELLOW_TIME, MIN_GREEN, MAX_GREEN,
    REWARD_FN, LEARNING_RATE, BATCH_SIZE, N_EPOCHS,
    GAMMA, GAE_LAMBDA, ENT_COEF, CLIP_RANGE,
    WARMUP_SECONDS, NUM_SECONDS,
    MORNING_RUSH_START, MORNING_RUSH_SECONDS,
    EVENING_RUSH_START, EVENING_RUSH_SECONDS,
    OFFPEAK_SECONDS,
)
from agent_filter import AgentFilterWrapper

# Import time encoding from experiment.py
import experiment
from experiment import TimeEncodedObservationFunction


# ── Scenario presets ───────────────────────────────────────────────────────
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
    "uniform": {
        "route_file": "data/routes/routes.rou.xml",
        "rl_seconds": 3600,
        "start_hour": 0.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for traffic signal control")
    parser.add_argument(
        "--scenario", type=str, default="uniform",
        choices=list(SCENARIO_PRESETS.keys()),
        help="Training scenario. Selects route file and episode length.",
    )
    # Allow explicit overrides (backwards-compatible)
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml")
    parser.add_argument("--route_file", type=str, default=None,
                        help="Override route file (ignores --scenario preset).")
    parser.add_argument("--num_seconds", type=int, default=None,
                        help="Override total simulation duration (seconds). "
                             "Defaults to scenario rl_seconds + WARMUP_SECONDS.")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--checkpoint_dir", type=str, default="models/")
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .zip to resume training from.")
    return parser.parse_args()


def resolve_config(args):
    """
    Resolve route_file and num_seconds from args + scenario presets.
    Returns (route_file, num_seconds, rl_seconds, n_steps, start_hour).
    """
    preset = SCENARIO_PRESETS[args.scenario]

    route_file = args.route_file if args.route_file is not None else preset["route_file"]

    rl_seconds = preset["rl_seconds"]
    if args.num_seconds is not None:
        num_seconds = args.num_seconds
        rl_seconds = max(num_seconds - WARMUP_SECONDS, num_seconds)
    else:
        num_seconds = rl_seconds + WARMUP_SECONDS

    # n_steps = one full RL episode worth of steps per agent
    n_steps = rl_seconds // DELTA_TIME  # e.g. 14400/5=2880, 3600/5=720

    start_hour = preset["start_hour"]

    return route_file, num_seconds, rl_seconds, n_steps, start_hour


def main():
    args = parse_args()
    route_file, num_seconds, rl_seconds, n_steps, start_hour = resolve_config(args)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"\nTraining configuration:")
    print(f"  Scenario     : {args.scenario}")
    print(f"  Route file   : {route_file}")
    print(f"  num_seconds  : {num_seconds}s (RL={rl_seconds}s + warmup={WARMUP_SECONDS}s)")
    print(f"  n_steps      : {n_steps}  (= {rl_seconds}s / {DELTA_TIME}s/step)")
    print(f"  start_hour   : {start_hour:.1f}h (time encoding)")
    print(f"  Total steps  : {args.total_timesteps}")

    if not os.path.exists(route_file):
        print(f"\nERROR: Route file not found: {route_file}")
        print("Generate it first:")
        print(f"  python src/generate_rush_demand.py --scenario {args.scenario}")
        raise SystemExit(1)

    # Set time-of-day for observation encoding
    experiment.CURRENT_HOUR = start_hour

    # Create PettingZoo parallel env -> SuperSuit vec env
    env = sumo_rl.parallel_env(
        net_file=args.net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=num_seconds,
        reward_fn=REWARD_FN,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        sumo_warnings=False,
        observation_class=TimeEncodedObservationFunction,
    )
    if not hasattr(env.unwrapped, "render_mode"):
        env.unwrapped.render_mode = None

    # Filter to 5 target intersections only -- others run original SUMO programs
    env = AgentFilterWrapper(
        env, target_agents=TS_IDS, net_file=args.net_file, default_action=0
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    print(f"\n  Observation space: {env.observation_space}")
    print(f"  Action space     : {env.action_space}")

    # Model name encodes scenario for easy identification
    model_prefix = f"ppo_{args.scenario}"

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000, 1),
        save_path=args.checkpoint_dir,
        name_prefix=model_prefix,
    )

    if args.resume:
        print(f"\n  Resuming from {args.resume}...")
        model = PPO.load(
            args.resume,
            env=env,
            custom_objects={
                "learning_rate": LEARNING_RATE,
                "n_steps": n_steps,
                "batch_size": BATCH_SIZE,
                "n_epochs": N_EPOCHS,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "ent_coef": ENT_COEF,
                "clip_range": CLIP_RANGE,
            }
        )
        model.tensorboard_log = args.log_dir
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            n_steps=n_steps,          # one full episode per PPO update
            batch_size=BATCH_SIZE,    # 180 divides evenly into all scenario step counts
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ent_coef=ENT_COEF,
            clip_range=CLIP_RANGE,
            verbose=1,
            tensorboard_log=args.log_dir,
        )

    print(f"\nTraining for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
        reset_num_timesteps=False if args.resume else True,
    )

    final_path = os.path.join(args.checkpoint_dir, f"{model_prefix}_final")
    model.save(final_path)
    print(f"\nTraining complete! Model saved to {final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
