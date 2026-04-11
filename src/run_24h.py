"""
Zeleni SignaLJ - 24h Mega-Policy Statistical Test Runner
=========================================================
Runs full 24-hour SUMO simulations with dynamic RL/fixed-time switching.

During rush hours (06:00-10:00, 14:00-18:00), the 5 target traffic signals
are controlled by pre-trained PPO models. Outside rush hours, all traffic
signals run their original SUMO fixed-time programs.

Supports running N replications in parallel with different SUMO seeds for
statistical significance testing.

Usage:
    # Mega-policy: RL during rush hours, fixed-time otherwise
    python src/run_24h.py \\
        --model_morning results/experiments/.../ppo_shared_policy.zip \\
        --model_evening results/experiments/.../ppo_shared_policy.zip \\
        --num_runs 50 --num_workers 50 \\
        --output_dir results/statistical-test/M1E1 --tag M1E1

    # Baseline: all fixed-time for 24h
    python src/run_24h.py --baseline \\
        --num_runs 50 --num_workers 50 \\
        --output_dir results/statistical-test/baseline --tag baseline
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TS_IDS, TS_NAMES, DELTA_TIME, YELLOW_TIME, MIN_GREEN, MAX_GREEN,
    REWARD_FN, FULL_DAY_SECONDS,
    MORNING_RUSH_START, MORNING_RUSH_END,
    EVENING_RUSH_START, EVENING_RUSH_END,
)
from tls_programs import parse_original_programs, restore_non_target_programs


# ── Schedule logic (inline to avoid importing experiment.py in workers) ────

def get_mode(hour):
    """Return control mode for given hour of day."""
    hour = hour % 24.0
    if MORNING_RUSH_START <= hour < MORNING_RUSH_END:
        return "rl_morning"
    if EVENING_RUSH_START <= hour < EVENING_RUSH_END:
        return "rl_evening"
    return "fixed_time"


def get_window_label(hour):
    """Return human-readable window label for given hour."""
    hour = hour % 24.0
    if hour < MORNING_RUSH_START:
        return "night_0006"
    if hour < MORNING_RUSH_END:
        return "morning_rush"
    if hour < EVENING_RUSH_START:
        return "shoulder_day"
    if hour < EVENING_RUSH_END:
        return "evening_rush"
    if hour < 21.0:
        return "shoulder_evening"
    return "night_2100"


# ── TLS mode switching ────────────────────────────────────────────────────

def _save_rl_programs(env, ts_ids):
    """Save sumo-rl's RL phase programs for target TLS (after env.reset)."""
    rl_programs = {}
    for ts_id in ts_ids:
        ts_obj = env.traffic_signals[ts_id]
        rl_programs[ts_id] = list(ts_obj.all_phases)
    return rl_programs


def _save_original_methods(env, ts_ids):
    """Save original set_next_phase and update methods before any patching."""
    methods = {}
    for ts_id in ts_ids:
        ts_obj = env.traffic_signals[ts_id]
        methods[ts_id] = (ts_obj.set_next_phase, ts_obj.update)
    return methods


def _switch_to_fixed_time(env, ts_id, original_programs):
    """Switch a target TLS to fixed-time (SUMO-native) control.

    Restores the original SUMO program and patches set_next_phase/update
    to not interfere, while keeping the timing machinery alive so
    sumo-rl's _run_steps loop doesn't hang.
    """
    ts_obj = env.traffic_signals[ts_id]
    sumo = env.sumo

    # Restore original SUMO program via TraCI
    prog = original_programs[ts_id]
    phases = [sumo.trafficlight.Phase(p["duration"], p["state"]) for p in prog["phases"]]
    programs = sumo.trafficlight.getAllProgramLogics(ts_id)
    logic = programs[0]
    logic.type = 0
    logic.phases = phases
    sumo.trafficlight.setProgramLogic(ts_id, logic)
    # Re-activate automatic cycling: setRedYellowGreenState (called by
    # sumo-rl's _build_phases) puts the TLS in manual mode. setProgramLogic
    # alone only updates phase definitions. setProgram re-activates the
    # automatic program so SUMO cycles through phases on its own.
    sumo.trafficlight.setProgram(ts_id, logic.programID)
    # Restore original offset (for signal coordination)
    offset = original_programs[ts_id].get("offset", 0)
    if offset != 0:
        try:
            sumo.trafficlight.setParameter(ts_id, "offset", str(offset))
        except Exception:
            pass

    # Passthrough patch: keeps timing alive but doesn't override SUMO's program
    def passthrough_set_next_phase(new_phase):
        ts_obj.next_action_time = env.sim_step + ts_obj.delta_time

    def passthrough_update():
        ts_obj.time_since_last_phase_change += 1

    ts_obj.set_next_phase = passthrough_set_next_phase
    ts_obj.update = passthrough_update


def _restore_all_programs(env, original_programs):
    """Restore original SUMO programs for ALL TLS (including target).

    Used by the baseline to ensure identical code path as the megapolicy.
    Patches all TLS with passthrough methods so sumo-rl doesn't interfere.
    """
    sumo = env.sumo
    for ts_id, ts_obj in env.traffic_signals.items():
        if ts_id not in original_programs:
            continue
        prog = original_programs[ts_id]
        try:
            phases = [sumo.trafficlight.Phase(p["duration"], p["state"])
                      for p in prog["phases"]]
            programs = sumo.trafficlight.getAllProgramLogics(ts_id)
            logic = programs[0]
            logic.type = 0
            logic.phases = phases
            sumo.trafficlight.setProgramLogic(ts_id, logic)
            sumo.trafficlight.setProgram(ts_id, logic.programID)
            offset = prog.get("offset", 0)
            if offset != 0:
                try:
                    sumo.trafficlight.setParameter(ts_id, "offset", str(offset))
                except Exception:
                    pass
        except Exception as e:
            print(f"  WARNING: Could not restore program for {ts_id}: {e}")
            continue

        # Closure factory to avoid loop-variable capture bug
        def _make_passthrough(ts_ref, env_ref):
            def passthrough_set_next_phase(new_phase):
                ts_ref.next_action_time = env_ref.sim_step + ts_ref.delta_time
            def passthrough_update():
                ts_ref.time_since_last_phase_change += 1
            return passthrough_set_next_phase, passthrough_update

        set_fn, update_fn = _make_passthrough(ts_obj, env)
        ts_obj.set_next_phase = set_fn
        ts_obj.update = update_fn


def _switch_to_rl(env, ts_id, rl_programs, original_methods):
    """Switch a target TLS back to RL control.

    Restores sumo-rl's RL phase program and original methods.
    """
    ts_obj = env.traffic_signals[ts_id]
    sumo = env.sumo

    # Restore sumo-rl's RL program
    programs = sumo.trafficlight.getAllProgramLogics(ts_id)
    logic = programs[0]
    logic.type = 0
    logic.phases = rl_programs[ts_id]
    sumo.trafficlight.setProgramLogic(ts_id, logic)

    # Restore original methods
    orig_set, orig_update = original_methods[ts_id]
    ts_obj.set_next_phase = orig_set
    ts_obj.update = orig_update

    # Re-sync: set to first green phase to start clean
    sumo.trafficlight.setRedYellowGreenState(ts_id, rl_programs[ts_id][0].state)
    ts_obj.green_phase = 0
    ts_obj.is_yellow = False
    ts_obj.time_since_last_phase_change = 0
    ts_obj.next_action_time = env.sim_step


# ── 24h simulation episode ────────────────────────────────────────────────

def run_24h_baseline(net_file, route_file, sumo_seed=42):
    """Run a 24h baseline simulation (all TLS on fixed-time).

    Uses fixed_ts=False with full program restoration to ensure identical
    code path as the megapolicy. The only difference: RL is never activated.
    """
    import sumo_rl

    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=FULL_DAY_SECONDS,
        reward_fn=REWARD_FN,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        single_agent=False,
        fixed_ts=False,
        sumo_warnings=False,
        sumo_seed=sumo_seed,
    )

    env.reset()

    # Restore ALL 37 TLS programs (same mechanism as megapolicy)
    original_programs = parse_original_programs(net_file)
    _restore_all_programs(env, original_programs)

    # Metrics accumulators
    window_metrics = defaultdict(lambda: {
        "rewards": [], "queues": [], "waits": [], "teleports": 0, "steps": 0,
    })
    per_intersection_reward = {ts_id: 0.0 for ts_id in TS_IDS}
    per_intersection_queue = defaultdict(list)
    per_intersection_wait = defaultdict(list)
    done = False

    while not done:
        current_hour = env.sim_step / 3600.0
        wlabel = get_window_label(current_hour)

        # All TLS are patched — action 0 is ignored by passthrough
        actions = {ts_id: 0 for ts_id in env.ts_ids}
        observations, rewards, dones, info = env.step(actions)
        done = dones["__all__"]

        # Accumulate rewards
        for ts_id in TS_IDS:
            if ts_id in rewards:
                per_intersection_reward[ts_id] += rewards[ts_id]
                window_metrics[wlabel]["rewards"].append(rewards[ts_id])

        # Per-step metrics via TraCI
        try:
            sumo = env.sumo
            for ts_id in TS_IDS:
                if ts_id not in env.traffic_signals:
                    continue
                ts = env.traffic_signals[ts_id]
                stopped = sum(sumo.lane.getLastStepHaltingNumber(l) for l in ts.lanes)
                per_intersection_queue[ts_id].append(stopped)
                wait = sum(sumo.lane.getWaitingTime(l) for l in ts.lanes)
                per_intersection_wait[ts_id].append(wait / max(len(ts.lanes), 1))

            total_stopped = sum(per_intersection_queue[ts_id][-1] for ts_id in TS_IDS
                                if per_intersection_queue[ts_id])
            total_wait = sum(per_intersection_wait[ts_id][-1] for ts_id in TS_IDS
                             if per_intersection_wait[ts_id])
            window_metrics[wlabel]["queues"].append(total_stopped)
            window_metrics[wlabel]["waits"].append(total_wait)
            window_metrics[wlabel]["teleports"] += sumo.simulation.getStartingTeleportNumber()
            window_metrics[wlabel]["steps"] += 1
        except Exception:
            pass

    # Collect final stats before closing
    try:
        departed = env.sumo.simulation.getDepartedNumber()
        arrived = env.sumo.simulation.getArrivedNumber()
    except Exception:
        departed = arrived = 0

    env.close()

    return _build_result(per_intersection_reward, per_intersection_queue,
                         per_intersection_wait, window_metrics, departed, arrived)


def run_24h_megapolicy(net_file, route_file, model_morning_path, model_evening_path,
                       sumo_seed=42):
    """Run a 24h simulation with RL during rush hours, fixed-time otherwise."""
    import sumo_rl
    from stable_baselines3 import PPO

    # Import for time encoding
    import experiment
    from experiment import TimeEncodedObservationFunction

    # Set CURRENT_HOUR=0 so time_seconds = sim_step (0→86400 covers full day)
    experiment.CURRENT_HOUR = 0.0

    # Load models
    model_morning = PPO.load(model_morning_path) if model_morning_path else None
    model_evening = PPO.load(model_evening_path) if model_evening_path else None

    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=FULL_DAY_SECONDS,
        reward_fn=REWARD_FN,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        single_agent=False,
        fixed_ts=False,
        sumo_warnings=False,
        sumo_seed=sumo_seed,
        observation_class=TimeEncodedObservationFunction,
    )

    observations = env.reset()
    target_set = set(TS_IDS)

    # Parse and restore non-target TLS programs (permanent for full sim)
    original_programs = parse_original_programs(net_file)
    restore_non_target_programs(env, TS_IDS, original_programs)

    # Save RL programs and original methods for target TLS
    rl_programs = _save_rl_programs(env, TS_IDS)
    original_methods = _save_original_methods(env, TS_IDS)

    # Determine obs_size from loaded models
    obs_size = 0
    if model_morning:
        obs_size = max(obs_size, model_morning.observation_space.shape[0])
    if model_evening:
        obs_size = max(obs_size, model_evening.observation_space.shape[0])

    # Start in fixed-time mode (midnight)
    current_mode = "fixed_time"
    for ts_id in TS_IDS:
        _switch_to_fixed_time(env, ts_id, original_programs)

    # Metrics accumulators
    window_metrics = defaultdict(lambda: {
        "rewards": [], "queues": [], "waits": [], "teleports": 0, "steps": 0,
    })
    per_intersection_reward = {ts_id: 0.0 for ts_id in TS_IDS}
    per_intersection_queue = defaultdict(list)
    per_intersection_wait = defaultdict(list)
    done = False

    while not done:
        current_hour = env.sim_step / 3600.0
        wlabel = get_window_label(current_hour)
        new_mode = get_mode(current_hour)

        # Handle mode transitions
        if new_mode != current_mode:
            if new_mode == "fixed_time":
                for ts_id in TS_IDS:
                    _switch_to_fixed_time(env, ts_id, original_programs)
            else:
                for ts_id in TS_IDS:
                    _switch_to_rl(env, ts_id, rl_programs, original_methods)
            current_mode = new_mode

        # Determine actions
        model = None
        if current_mode == "rl_morning":
            model = model_morning
        elif current_mode == "rl_evening":
            model = model_evening

        actions = {}
        for ts_id in env.ts_ids:
            if model is not None and ts_id in target_set:
                obs = observations[ts_id]
                if len(obs) < obs_size:
                    padded = np.zeros(obs_size, dtype=np.float32)
                    padded[:len(obs)] = obs
                    obs = padded
                action, _ = model.predict(obs, deterministic=True)
                actual_n = env.traffic_signals[ts_id].action_space.n
                actions[ts_id] = int(action) % actual_n
            else:
                actions[ts_id] = 0  # ignored by patched TLS

        observations, rewards, dones, info = env.step(actions)
        done = dones["__all__"]

        # Accumulate rewards
        for ts_id in TS_IDS:
            if ts_id in rewards:
                per_intersection_reward[ts_id] += rewards[ts_id]
                window_metrics[wlabel]["rewards"].append(rewards[ts_id])

        # Per-step metrics via TraCI
        try:
            sumo = env.sumo
            for ts_id in TS_IDS:
                if ts_id not in env.traffic_signals:
                    continue
                ts = env.traffic_signals[ts_id]
                stopped = sum(sumo.lane.getLastStepHaltingNumber(l) for l in ts.lanes)
                per_intersection_queue[ts_id].append(stopped)
                wait = sum(sumo.lane.getWaitingTime(l) for l in ts.lanes)
                per_intersection_wait[ts_id].append(wait / max(len(ts.lanes), 1))

            total_stopped = sum(per_intersection_queue[ts_id][-1] for ts_id in TS_IDS
                                if per_intersection_queue[ts_id])
            total_wait = sum(per_intersection_wait[ts_id][-1] for ts_id in TS_IDS
                             if per_intersection_wait[ts_id])
            window_metrics[wlabel]["queues"].append(total_stopped)
            window_metrics[wlabel]["waits"].append(total_wait)
            window_metrics[wlabel]["teleports"] += sumo.simulation.getStartingTeleportNumber()
            window_metrics[wlabel]["steps"] += 1
        except Exception:
            pass

    # Collect final stats before closing
    try:
        departed = env.sumo.simulation.getDepartedNumber()
        arrived = env.sumo.simulation.getArrivedNumber()
    except Exception:
        departed = arrived = 0

    env.close()

    return _build_result(per_intersection_reward, per_intersection_queue,
                         per_intersection_wait, window_metrics, departed, arrived)


def _build_result(per_intersection_reward, per_intersection_queue,
                  per_intersection_wait, window_metrics, departed, arrived):
    """Build the result dict from accumulated metrics."""
    total_reward = sum(per_intersection_reward.values())

    # Per-intersection averages
    per_intersection_avg_queue = {}
    per_intersection_avg_wait = {}
    for ts_id in TS_IDS:
        name = TS_NAMES.get(ts_id, ts_id[:20])
        q = per_intersection_queue[ts_id]
        w = per_intersection_wait[ts_id]
        per_intersection_avg_queue[name] = float(np.mean(q)) if q else 0.0
        per_intersection_avg_wait[name] = float(np.mean(w)) if w else 0.0

    # Per-window summaries
    per_window = {}
    all_queues = []
    all_waits = []
    total_teleports = 0
    for wlabel, m in window_metrics.items():
        per_window[wlabel] = {
            "total_reward": float(sum(m["rewards"])) if m["rewards"] else 0.0,
            "avg_queue": float(np.mean(m["queues"])) if m["queues"] else 0.0,
            "avg_wait": float(np.mean(m["waits"])) if m["waits"] else 0.0,
            "teleports": m["teleports"],
            "steps": m["steps"],
        }
        all_queues.extend(m["queues"])
        all_waits.extend(m["waits"])
        total_teleports += m["teleports"]

    # Named intersection rewards
    named_rewards = {}
    for ts_id in TS_IDS:
        name = TS_NAMES.get(ts_id, ts_id[:20])
        named_rewards[name] = per_intersection_reward[ts_id]

    return {
        "total_reward": total_reward,
        "avg_queue": float(np.mean(all_queues)) if all_queues else 0.0,
        "avg_wait": float(np.mean(all_waits)) if all_waits else 0.0,
        "total_teleports": total_teleports,
        "vehicles_departed": departed,
        "vehicles_arrived": arrived,
        "per_intersection_reward": named_rewards,
        "per_intersection_avg_queue": per_intersection_avg_queue,
        "per_intersection_avg_wait": per_intersection_avg_wait,
        "per_window": per_window,
    }


# ── Multiprocessing worker ────────────────────────────────────────────────

def _worker(args):
    """Worker function for parallel runs. Catches exceptions so one crash doesn't kill all 50."""
    (seed, net_file, route_file, model_morning_path,
     model_evening_path, baseline) = args

    wall_start = time.time()

    try:
        if baseline:
            result = run_24h_baseline(net_file, route_file, sumo_seed=seed)
        else:
            result = run_24h_megapolicy(
                net_file, route_file,
                model_morning_path, model_evening_path,
                sumo_seed=seed,
            )
    except Exception as e:
        print(f"  [FAIL] Seed {seed}: {e}", flush=True)
        result = {"total_reward": float("nan"), "avg_queue": float("nan"),
                  "avg_wait": float("nan"), "total_teleports": 0,
                  "vehicles_departed": 0, "vehicles_arrived": 0,
                  "per_intersection_reward": {}, "per_intersection_avg_queue": {},
                  "per_intersection_avg_wait": {}, "per_window": {},
                  "error": str(e)}

    result["seed"] = seed
    result["route_file"] = route_file
    result["wall_time_s"] = round(time.time() - wall_start, 1)
    print(f"  [OK] Seed {seed:>2d} done in {result['wall_time_s']:.0f}s  "
          f"reward={result['total_reward']:.0f}", flush=True)
    return result


# ── Output helpers ─────────────────────────────────────────────────────────

def _write_summary_csv(results, output_dir):
    """Write summary.csv with one row per seed."""
    import csv

    # Collect all window labels
    all_windows = set()
    for r in results:
        all_windows.update(r["per_window"].keys())
    all_windows = sorted(all_windows)

    # Intersection names
    intersections = list(TS_NAMES.values())

    fieldnames = [
        "seed", "total_reward", "avg_queue", "avg_wait",
        "teleports", "vehicles_departed", "vehicles_arrived", "wall_time_s",
    ]
    for name in intersections:
        fieldnames.append(f"reward_{name}")
    for name in intersections:
        fieldnames.append(f"queue_{name}")
    for name in intersections:
        fieldnames.append(f"wait_{name}")
    for w in all_windows:
        fieldnames.append(f"reward_{w}")
    for w in all_windows:
        fieldnames.append(f"queue_{w}")
    for w in all_windows:
        fieldnames.append(f"wait_{w}")

    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in sorted(results, key=lambda x: x["seed"]):
            row = {
                "seed": r["seed"],
                "total_reward": round(r["total_reward"], 2),
                "avg_queue": round(r["avg_queue"], 3),
                "avg_wait": round(r["avg_wait"], 3),
                "teleports": r["total_teleports"],
                "vehicles_departed": r["vehicles_departed"],
                "vehicles_arrived": r["vehicles_arrived"],
                "wall_time_s": r["wall_time_s"],
            }
            for name in intersections:
                row[f"reward_{name}"] = round(r["per_intersection_reward"].get(name, 0), 2)
            for name in intersections:
                row[f"queue_{name}"] = round(r["per_intersection_avg_queue"].get(name, 0), 3)
            for name in intersections:
                row[f"wait_{name}"] = round(r["per_intersection_avg_wait"].get(name, 0), 3)
            for w in all_windows:
                pw = r["per_window"].get(w, {})
                row[f"reward_{w}"] = round(pw.get("total_reward", 0), 2)
            for w in all_windows:
                pw = r["per_window"].get(w, {})
                row[f"queue_{w}"] = round(pw.get("avg_queue", 0), 3)
            for w in all_windows:
                pw = r["per_window"].get(w, {})
                row[f"wait_{w}"] = round(pw.get("avg_wait", 0), 3)
            writer.writerow(row)

    return csv_path


def _print_summary(results, tag):
    """Print statistical summary to stdout."""
    rewards = [r["total_reward"] for r in results]
    queues = [r["avg_queue"] for r in results]
    waits = [r["avg_wait"] for r in results]
    teleports = [r["total_teleports"] for r in results]
    wall_times = [r["wall_time_s"] for r in results]

    n = len(rewards)
    mean_r = np.mean(rewards)
    std_r = np.std(rewards, ddof=1) if n > 1 else 0
    ci95 = 1.96 * std_r / np.sqrt(n) if n > 1 else 0

    print(f"\n{'='*60}")
    print(f"Statistical Summary: {tag} ({n} runs)")
    print(f"{'='*60}")
    print(f"  Total reward:  {mean_r:>10.1f} +/- {ci95:.1f}  (std={std_r:.1f})")
    print(f"  Avg queue:     {np.mean(queues):>10.2f} +/- {1.96*np.std(queues,ddof=1)/np.sqrt(n):.2f}")
    print(f"  Avg wait:      {np.mean(waits):>10.2f} +/- {1.96*np.std(waits,ddof=1)/np.sqrt(n):.2f}")
    print(f"  Teleports:     {np.mean(teleports):>10.1f} +/- {1.96*np.std(teleports,ddof=1)/np.sqrt(n):.1f}")
    print(f"  Wall time:     {np.mean(wall_times):>10.1f}s (min={min(wall_times):.0f}s, max={max(wall_times):.0f}s)")
    print(f"{'='*60}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run 24h mega-policy statistical test"
    )
    parser.add_argument("--model_morning", type=str, default=None,
                        help="Path to morning rush PPO model (.zip)")
    parser.add_argument("--model_evening", type=str, default=None,
                        help="Path to evening rush PPO model (.zip)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline (all fixed-time, no RL)")
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml")
    parser.add_argument("--route_file", type=str,
                        default="data/routes/routes_full_day.rou.xml")
    parser.add_argument("--route_dir", type=str, default=None,
                        help="Directory with per-seed route files "
                             "(routes_full_day_seed_NN.rou.xml). "
                             "Overrides --route_file.")
    parser.add_argument("--num_runs", type=int, default=50,
                        help="Number of replications (different SUMO seeds)")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Parallel workers (each needs 1 CPU + ~200MB RAM)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output JSONs and summary CSV")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for this mega-policy (e.g. M1E1)")

    args = parser.parse_args()

    if not args.baseline and (not args.model_morning or not args.model_evening):
        parser.error("Provide --model_morning and --model_evening, or use --baseline")

    os.makedirs(args.output_dir, exist_ok=True)

    # Write meta.json
    meta = {
        "tag": args.tag,
        "baseline": args.baseline,
        "model_morning": args.model_morning,
        "model_evening": args.model_evening,
        "net_file": args.net_file,
        "route_file": args.route_file if not args.route_dir else None,
        "route_dir": args.route_dir,
        "per_seed_routes": args.route_dir is not None,
        "num_runs": args.num_runs,
        "num_workers": args.num_workers,
        "num_seconds": FULL_DAY_SECONDS,
        "delta_time": DELTA_TIME,
        "reward_fn": REWARD_FN,
        "ts_ids": TS_IDS,
        "ts_names": TS_NAMES,
        "schedule": {
            "morning_rush": f"{MORNING_RUSH_START:.0f}:00-{MORNING_RUSH_END:.0f}:00",
            "evening_rush": f"{EVENING_RUSH_START:.0f}:00-{EVENING_RUSH_END:.0f}:00",
            "other": "fixed-time",
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    mode_str = "BASELINE (all fixed-time)" if args.baseline else f"MEGA-POLICY {args.tag}"
    print(f"\n{'='*60}")
    print(f"24h Statistical Test: {mode_str}")
    print(f"  Runs: {args.num_runs} | Workers: {args.num_workers}")
    if args.route_dir:
        print(f"  Routes: {args.route_dir}/ (per-seed)")
    else:
        print(f"  Route: {args.route_file}")
    if not args.baseline:
        print(f"  Morning model: {args.model_morning}")
        print(f"  Evening model: {args.model_evening}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    # Build worker args — resolve per-seed route files if --route_dir is set
    worker_args = []
    for seed in range(1, args.num_runs + 1):
        if args.route_dir:
            route_idx = seed - 1  # seed 1 -> seed_00, seed 2 -> seed_01, etc.
            route_file = os.path.join(
                args.route_dir,
                f"routes_full_day_seed_{route_idx:02d}.rou.xml")
            if not os.path.exists(route_file):
                raise FileNotFoundError(
                    f"Route file not found: {route_file}\n"
                    f"Generate with: python src/generate_demand.py "
                    f"--statistical_routes {args.num_runs}")
        else:
            route_file = args.route_file
        worker_args.append(
            (seed, args.net_file, route_file,
             args.model_morning, args.model_evening, args.baseline)
        )

    # Run in parallel
    t0 = time.time()
    if args.num_workers <= 1:
        results = [_worker(a) for a in worker_args]
    else:
        with multiprocessing.Pool(args.num_workers) as pool:
            results = pool.map(_worker, worker_args)

    total_wall = time.time() - t0
    print(f"\nAll {args.num_runs} runs completed in {total_wall:.0f}s")

    # Save individual run JSONs
    for r in results:
        seed = r["seed"]
        json_path = os.path.join(args.output_dir, f"run_seed_{seed:02d}.json")
        with open(json_path, "w") as f:
            json.dump(r, f, indent=2, default=str)

    # Filter out failed runs for summary
    ok_results = [r for r in results if "error" not in r]
    failed = len(results) - len(ok_results)
    if failed:
        print(f"\nWARNING: {failed}/{len(results)} runs failed (see run_seed_*.json for errors)")

    if ok_results:
        # Write summary CSV
        csv_path = _write_summary_csv(ok_results, args.output_dir)
        print(f"Summary CSV: {csv_path}  ({len(ok_results)} successful runs)")

        # Print statistical summary
        _print_summary(ok_results, args.tag or ("baseline" if args.baseline else "mega"))
    else:
        print("ERROR: All runs failed. No summary produced.")


if __name__ == "__main__":
    main()
