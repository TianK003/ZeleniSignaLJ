"""
Zeleni SignaLJ - PPO Training Script
======================================
Train PPO with parameter sharing for traffic signal control.
Uses PettingZoo + SuperSuit to vectorize the multi-agent env.

All traffic signals share one policy — this is standard IPPO.
Each signal is treated as an independent sub-environment.

Usage:
    Local:  python src/train.py --total_timesteps 50000
    Vega:   python src/train.py --total_timesteps 2000000
"""

import argparse
import os
import sumo_rl
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from config import TS_IDS
from agent_filter import AgentFilterWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for traffic signal control")
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml")
    parser.add_argument("--route_file", type=str,
                        default="data/routes/routes.rou.xml")
    parser.add_argument("--total_timesteps", type=int, default=500_000)
    parser.add_argument("--num_seconds", type=int, default=3600)
    parser.add_argument("--checkpoint_dir", type=str, default="models/")
    parser.add_argument("--log_dir", type=str, default="logs/")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create PettingZoo parallel env → SuperSuit vec env
    env = sumo_rl.parallel_env(
        net_file=args.net_file,
        route_file=args.route_file,
        use_gui=False,
        num_seconds=args.num_seconds,
        reward_fn="queue",
        delta_time=5,
        yellow_time=2,
        min_green=10,
        max_green=90,
        sumo_warnings=False,
    )
    if not hasattr(env.unwrapped, "render_mode"):
        env.unwrapped.render_mode = None
    # Filter to 5 target intersections only — others run original SUMO programs
    env = AgentFilterWrapper(
        env, target_agents=TS_IDS, net_file=args.net_file, default_action=0
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ppo_traffic",
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        n_steps=720,        # 1 full episode per rollout (3600s / 5s)
        batch_size=180,     # divides evenly into 720*5=3600 transitions
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=args.log_dir,
    )

    print(f"\nTraining for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
    )

    final_path = os.path.join(args.checkpoint_dir, "ppo_traffic_final")
    model.save(final_path)
    print(f"\nTraining complete! Model saved to {final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
