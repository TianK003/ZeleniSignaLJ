"""
Zeleni SignaLJ - PPO Training Script
=====================================
Train Independent PPO agents for traffic signal control
in the Bleiweisova triangle, Ljubljana.

Usage:
    Local:  python src/train.py
    Vega:   python src/train.py --num_envs 32 --total_timesteps 2000000
"""

import argparse
import os
import sumo_rl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(net_file, route_file, use_gui=False, num_seconds=3600):
    """Factory function for creating SUMO environments."""
    def _init():
        env = sumo_rl.SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=num_seconds,
            single_agent=True,
            reward_fn="queue",
            delta_time=5,
            yellow_time=2,
            min_green=5,
            max_green=60,
        )
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for traffic signal control")
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml",
                        help="Path to SUMO network file")
    parser.add_argument("--route_file", type=str,
                        default="data/routes/routes.rou.xml",
                        help="Path to SUMO route file")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments (1=local, 32=Vega)")
    parser.add_argument("--total_timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--checkpoint_dir", type=str, default="models/",
                        help="Directory for saving checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs/",
                        help="Directory for tensorboard logs")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Choose vectorization strategy based on environment count
    # DummyVecEnv: sequential, low memory (good for local)
    # SubprocVecEnv: parallel processes, high memory (good for HPC)
    env_fns = [make_env(args.net_file, args.route_file)
               for _ in range(args.num_envs)]

    if args.num_envs > 1:
        env = SubprocVecEnv(env_fns)
        print(f"Using SubprocVecEnv with {args.num_envs} parallel environments")
    else:
        env = DummyVecEnv(env_fns)
        print("Using DummyVecEnv (single environment)")

    # Checkpoint callback: save every 50k steps
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.num_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ppo_ljubljana",
    )

    # PPO with tuned hyperparameters for traffic signal control
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
        verbose=1,
        tensorboard_log=args.log_dir,
    )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
    )

    final_path = os.path.join(args.checkpoint_dir, "ppo_ljubljana_final")
    model.save(final_path)
    print(f"Training complete! Model saved to {final_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
