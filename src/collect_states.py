"""
Zeleni SignaLJ - State & Latent Harvester
=========================================
Runs a trained PPO model and saves:
1. Observations (States)
2. Chosen Actions
3. Latent layer activations (64-dim hidden layer)
4. Metadata (Hour, VPH, Intersection ID)

This data is used by explain.py for SHAP, UMAP, and Decision Trees.

Supports three modes:
  --scenario morning_rush   Use morning rush routes/hours (6-10h)
  --scenario evening_rush   Use evening rush routes/hours (14-18h)
  --megapolicy              Use schedule controller with two models (full 24h)
"""

import argparse
import os
import sys

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

import pickle

import numpy as np
import torch
from stable_baselines3 import PPO

import experiment
from config import TS_IDS, WARMUP_SECONDS, TOTAL_DAILY_CARS
from demand_math import get_vph


def harvest_data(model_path, num_episodes=5, run_dir=None, scenario="uniform"):
    """Harvest observations, actions, and latents from a single PPO model.

    Args:
        model_path: Path to ppo_shared_policy.zip.
        num_episodes: Number of episodes to run.
        run_dir: Output directory (defaults to model's parent dir).
        scenario: One of 'uniform', 'morning_rush', 'evening_rush', 'offpeak'.
                  Controls which route file and hour range are used.
    """
    if run_dir is None:
        run_dir = os.path.dirname(model_path)

    output_path = os.path.join(run_dir, "harvested_data.pkl")
    print(f"  Harvesting data from {model_path} (scenario={scenario})...")

    # Load model on CPU — inference on a small MLP is faster on CPU than
    # paying GPU kernel launch overhead (especially on AMD/ROCm).
    load_path = model_path
    if load_path.endswith(".zip"):
        load_path = load_path[:-4]
    model = PPO.load(load_path, device="cpu")

    # Resolve scenario preset for route file and hour range
    preset = experiment.SCENARIO_PRESETS[scenario]
    net_file = "data/networks/ljubljana.net.xml"
    route_file = preset["route_file"]

    # Hour range per scenario (for randomizing episodes)
    HOUR_RANGES = {
        "morning_rush": (6.0, 10.0),
        "evening_rush": (14.0, 18.0),
        "offpeak": (10.0, 14.0),
        "uniform": (0.0, 24.0),
    }
    hour_lo, hour_hi = HOUR_RANGES[scenario]

    # Cap episode duration at 1h of RL time (3600s). The full rush scenario
    # is 4h (14400s) which is unnecessarily long for interpretability — we get
    # variety from multiple episodes with different randomized hours instead.
    MAX_RL_SECONDS = 3600
    rl_seconds = min(preset["rl_seconds"], MAX_RL_SECONDS)
    num_seconds = rl_seconds + WARMUP_SECONDS

    env = experiment.make_eval_env(net_file, route_file, num_seconds, fixed_ts=False)

    # Access underlying sumo_rl environment for lane names etc
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    if hasattr(unwrapped_env, 'unwrapped'):
        unwrapped_env = unwrapped_env.unwrapped

    # Get feature mapping (indices to names)
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
        "phase_info": {},  # populated after first reset() when TraCI is live
        "ts_ids": TS_IDS,
        "scenario": scenario,
    }

    obs_size = model.observation_space.shape[0]
    phase_info_collected = False

    for ep in range(num_episodes):
        # Randomize hour within the scenario's window
        hour_of_day = np.random.uniform(hour_lo, hour_hi)
        experiment.CURRENT_HOUR = hour_of_day
        vph = get_vph(hour_of_day, TOTAL_DAILY_CARS)
        print(f"    Episode {ep+1}/{num_episodes}: Hour {hour_of_day:.1f}h, {vph:.0f} VPH")

        observations = env.reset()
        done = False
        step = 0

        # Collect phase info once after first reset (TraCI is now live)
        if not phase_info_collected:
            phase_info = {}
            for tid in TS_IDS:
                ts = unwrapped_env.traffic_signals[tid]
                green_states = [gp.state for gp in ts.green_phases]
                raw_links = unwrapped_env.sumo.trafficlight.getControlledLinks(tid)
                link_data = []
                for link_group in raw_links:
                    if link_group:
                        fl, tl, via = link_group[0]
                        link_data.append({"from": fl, "to": tl})
                    else:
                        link_data.append({"from": "", "to": ""})
                phase_info[tid] = {"green_states": green_states, "links": link_data}
            collected_data["phase_info"] = phase_info
            phase_info_collected = True

        while not done:
            # We only track the 5 target agents
            for ts_id in TS_IDS:
                if ts_id not in observations: continue

                raw_obs = observations[ts_id]
                padded_obs = experiment.pad_obs(raw_obs, obs_size)

                # Get action and latent
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(padded_obs).unsqueeze(0).to(model.device)
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


def harvest_megapolicy(model_morning_path, model_evening_path, num_episodes=5, run_dir=None):
    """Harvest data from a megapolicy (morning + evening models with schedule controller).

    Runs full 24h-range episodes. The schedule controller dispatches to the
    correct model based on the hour. Hours outside rush windows use the nearest
    rush model so we still collect RL decisions (the point is to see how each
    model behaves, not to run fixed-time).

    Args:
        model_morning_path: Path to morning rush PPO model (.zip).
        model_evening_path: Path to evening rush PPO model (.zip).
        num_episodes: Number of episodes to run.
        run_dir: Output directory.
    """
    if run_dir is None:
        run_dir = os.path.dirname(model_morning_path)

    output_path = os.path.join(run_dir, "harvested_data_megapolicy.pkl")
    print(f"  Harvesting megapolicy data...")
    print(f"    Morning model: {model_morning_path}")
    print(f"    Evening model: {model_evening_path}")

    # Load both models on CPU
    mp = model_morning_path
    if mp.endswith(".zip"): mp = mp[:-4]
    model_morning = PPO.load(mp, device="cpu")

    ep = model_evening_path
    if ep.endswith(".zip"): ep = ep[:-4]
    model_evening = PPO.load(ep, device="cpu")

    from schedule_controller import ScheduleController
    controller = ScheduleController(model_morning_path, model_evening_path)

    # Use uniform routes for broad coverage, or full_day if available
    route_file = "data/routes/routes_full_day.rou.xml"
    if not os.path.exists(route_file):
        print(f"  WARNING: {route_file} not found, falling back to uniform routes")
        route_file = "data/routes/routes.rou.xml"

    net_file = "data/networks/ljubljana.net.xml"
    num_seconds = 3600 + WARMUP_SECONDS

    env = experiment.make_eval_env(net_file, route_file, num_seconds, fixed_ts=False)

    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    if hasattr(unwrapped_env, 'unwrapped'):
        unwrapped_env = unwrapped_env.unwrapped

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
        "phase_info": {},  # populated after first reset()
        "ts_ids": TS_IDS,
        "scenario": "megapolicy",
    }

    # Use the morning model's obs size (should be same due to parameter sharing)
    obs_size = model_morning.observation_space.shape[0]
    phase_info_collected = False

    # Sample episodes across the full day, focusing on rush hours
    # 40% morning rush, 40% evening rush, 20% other hours
    rush_hours = []
    for _ in range(num_episodes):
        r = np.random.random()
        if r < 0.4:
            rush_hours.append(np.random.uniform(6.0, 10.0))
        elif r < 0.8:
            rush_hours.append(np.random.uniform(14.0, 18.0))
        else:
            rush_hours.append(np.random.uniform(0.0, 24.0))

    for ep_idx, hour_of_day in enumerate(rush_hours):
        mode = controller.get_mode(hour_of_day)
        model = controller.get_model(hour_of_day)

        # Outside rush hours, use the nearest rush model for data collection
        if model is None:
            if hour_of_day < 12.0:
                model = model_morning
                mode = "rl_morning (forced)"
            else:
                model = model_evening
                mode = "rl_evening (forced)"

        experiment.CURRENT_HOUR = hour_of_day
        vph = get_vph(hour_of_day, TOTAL_DAILY_CARS)
        print(f"    Episode {ep_idx+1}/{num_episodes}: Hour {hour_of_day:.1f}h, {vph:.0f} VPH, mode={mode}")

        observations = env.reset()
        done = False
        step = 0

        if not phase_info_collected:
            pi = {}
            for tid in TS_IDS:
                ts = unwrapped_env.traffic_signals[tid]
                green_states = [gp.state for gp in ts.green_phases]
                raw_links = unwrapped_env.sumo.trafficlight.getControlledLinks(tid)
                link_data = []
                for lg in raw_links:
                    if lg:
                        fl, tl, via = lg[0]
                        link_data.append({"from": fl, "to": tl})
                    else:
                        link_data.append({"from": "", "to": ""})
                pi[tid] = {"green_states": green_states, "links": link_data}
            collected_data["phase_info"] = pi
            phase_info_collected = True

        while not done:
            for ts_id in TS_IDS:
                if ts_id not in observations: continue

                raw_obs = observations[ts_id]
                padded_obs = experiment.pad_obs(raw_obs, obs_size)

                with torch.no_grad():
                    obs_tensor = torch.as_tensor(padded_obs).unsqueeze(0).to(model.device)
                    latent_pi, _ = model.policy.mlp_extractor(obs_tensor)
                    action, _ = model.predict(padded_obs, deterministic=True)

                collected_data["observations"].append(padded_obs)
                collected_data["actions"].append(int(action))
                collected_data["latents"].append(latent_pi.cpu().numpy()[0])
                collected_data["metadata"].append({
                    "ts_id": ts_id,
                    "hour": hour_of_day,
                    "vph": vph,
                    "step": step,
                    "mode": mode,
                })

            act_dict = {}
            for ts_id in env.ts_ids:
                if ts_id in TS_IDS:
                    o = experiment.pad_obs(observations[ts_id], obs_size)
                    a, _ = model.predict(o, deterministic=True)
                    ts_obj = unwrapped_env.traffic_signals[ts_id]
                    act_dict[ts_id] = int(a) % ts_obj.action_space.n
                else:
                    act_dict[ts_id] = 0

            observations, _, done_dict, _ = env.step(act_dict)
            done = done_dict["__all__"]
            step += 1

    env.close()

    with open(output_path, "wb") as f:
        pickle.dump(collected_data, f)

    print(f"  Done! Harvested {len(collected_data['observations'])} data points to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Harvest observations, actions, and latents from a trained PPO model."
    )
    parser.add_argument("--model_path", type=str,
                        help="Path to ppo_shared_policy.zip (single model mode)")
    parser.add_argument("--episodes", type=int, default=12,
                        help="Number of episodes to harvest (default: 12)")
    parser.add_argument("--scenario", type=str, default="uniform",
                        choices=["uniform", "morning_rush", "evening_rush", "offpeak"],
                        help="Demand scenario — sets route file and hour range")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to model's parent dir)")

    # Megapolicy mode
    parser.add_argument("--megapolicy", action="store_true",
                        help="Megapolicy mode: use schedule controller with two models")
    parser.add_argument("--model_morning", type=str,
                        help="Path to morning rush model (megapolicy mode)")
    parser.add_argument("--model_evening", type=str,
                        help="Path to evening rush model (megapolicy mode)")

    args = parser.parse_args()

    if args.megapolicy:
        if not args.model_morning or not args.model_evening:
            parser.error("--megapolicy requires --model_morning and --model_evening")
        harvest_megapolicy(
            args.model_morning,
            args.model_evening,
            num_episodes=args.episodes,
            run_dir=args.output_dir,
        )
    else:
        if not args.model_path:
            parser.error("--model_path is required (or use --megapolicy mode)")
        harvest_data(
            args.model_path,
            num_episodes=args.episodes,
            run_dir=args.output_dir,
            scenario=args.scenario,
        )
