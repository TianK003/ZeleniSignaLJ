"""
Zeleni SignaLJ - State & Latent Harvester
=========================================
Runs a trained PPO model and saves:
1. Observations (States)
2. Chosen Actions
3. Latent layer activations (64-dim hidden layer)
4. Metadata (Hour, VPH, Intersection ID)

This data is used by explain.py for SHAP, UMAP, and Decision Trees.
"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO

import experiment
from config import TS_IDS, NUM_AGENTS, DELTA_TIME, WARMUP_SECONDS, TOTAL_DAILY_CARS
from demand_math import get_vph

def harvest_data(model_path, num_episodes=5, run_dir=None):
    if run_dir is None:
        run_dir = os.path.dirname(model_path)
    
    output_path = os.path.join(run_dir, "harvested_data.pkl")
    print(f"  Harvesting data from {model_path}...")
    
    # Load model
    load_path = model_path
    if load_path.endswith(".zip"):
        load_path = load_path[:-4]
    model = PPO.load(load_path)
    
    # We need an environment to collect observations
    # We'll use the 'uniform' scenario for varied data collection
    net_file = "data/networks/ljubljana.net.xml"
    route_file = "data/routes/routes.rou.xml"
    num_seconds = 3600 + WARMUP_SECONDS
    
    # Use the make_eval_env logic from experiment.py
    env = experiment.make_eval_env(net_file, route_file, num_seconds, fixed_ts=False)
    
    # Access underlying sumo_rl environment for lane names etc
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    if hasattr(unwrapped_env, 'unwrapped'):
        unwrapped_env = unwrapped_env.unwrapped
        
    # Get feature mapping (indices to names)
    # The order is: Phase indicators, MinGreen, Density, Queue, SinTime, CosTime
    feature_maps = {}
    for ts_id in TS_IDS:
        ts = unwrapped_env.traffic_signals[ts_id]
        n_phases = ts.num_green_phases
        lanes = ts.lanes
        
        f_names = []
        for i in range(n_phases): f_names.append(f"Phase_{i}")
        f_names.append("MinGreenPassed")
        for l in lanes: f_names.append(f"Density_{l}")
        for l in lanes: f_names.append(f"Queue_{l}")
        f_names.extend(["SinTime", "CosTime"])
        feature_maps[ts_id] = f_names

    collected_data = {
        "observations": [],
        "actions": [],
        "latents": [],
        "metadata": [],
        "feature_maps": feature_maps,
        "ts_ids": TS_IDS
    }

    obs_size = model.observation_space.shape[0]

    for ep in range(num_episodes):
        # Randomize hour for variety
        hour_of_day = np.random.uniform(0, 24)
        experiment.CURRENT_HOUR = hour_of_day
        vph = get_vph(hour_of_day, TOTAL_DAILY_CARS)
        print(f"    Episode {ep+1}/{num_episodes}: Hour {hour_of_day:.1f}h, {vph:.0f} VPH")
        
        observations = env.reset()
        done = False
        step = 0
        
        while not done:
            # We only track the 5 target agents
            for ts_id in TS_IDS:
                if ts_id not in observations: continue
                
                raw_obs = observations[ts_id]
                padded_obs = experiment.pad_obs(raw_obs, obs_size)
                
                # Get action and latent
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(padded_obs).unsqueeze(0).to(model.device)
                    # SB3 PPO mlp_extractor returns (latent_pi, latent_vf)
                    latent_pi, _ = model.policy.mlp_extractor(obs_tensor)
                    action, _ = model.predict(padded_obs, deterministic=True)
                
                collected_data["observations"].append(padded_obs)
                collected_data["actions"].append(int(action))
                collected_data["latents"].append(latent_pi.cpu().numpy()[0])
                collected_data["metadata"].append({
                    "ts_id": ts_id,
                    "hour": hour_of_day,
                    "vph": vph,
                    "step": step
                })
            
            # Step the environment
            act_dict = {}
            for ts_id in env.ts_ids:
                if ts_id in TS_IDS:
                    o = experiment.pad_obs(observations[ts_id], obs_size)
                    a, _ = model.predict(o, deterministic=True)
                    # Stay within bounds for this specific signal
                    ts_obj = unwrapped_env.traffic_signals[ts_id]
                    act_dict[ts_id] = int(a) % ts_obj.action_space.n
                else:
                    act_dict[ts_id] = 0
            
            observations, _, done_dict, _ = env.step(act_dict)
            done = done_dict["__all__"]
            step += 1

    env.close()
    
    # Save to disk
    with open(output_path, "wb") as f:
        pickle.dump(collected_data, f)
    
    print(f"  Done! Harvested {len(collected_data['observations'])} data points to {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to ppo_shared_policy.zip")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to harvest")
    args = parser.parse_args()
    
    harvest_data(args.model_path, args.episodes)
