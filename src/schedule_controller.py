"""
Zeleni SignaLJ - Time-of-Day Schedule Controller
==================================================
Decides which traffic signal control mode to activate based on the current
time of day. This implements the deployment strategy:

  Rush hour  (06:00-10:00, 14:00-18:00) -> RL agent (PPO model)
  Shoulder   (10:00-14:00, 18:00-21:00) -> Fixed-time (original SUMO programs)
  Off-peak   (21:00-06:00)              -> Fixed-time (original SUMO programs)

In production, the scheduler would sit between the traffic management system
and the individual intersection controllers. Here it wraps SumoEnvironment
so evaluation and demonstration can simulate the schedule logic.

Usage:
    from schedule_controller import ScheduleController
    ctrl = ScheduleController(model_morning="models/ppo_morning_rush_final.zip",
                              model_evening="models/ppo_evening_rush_final.zip")
    mode = ctrl.get_mode(hour=7.5)   # -> "rl_morning"
    mode = ctrl.get_mode(hour=14.0)  # -> "rl_evening"
    mode = ctrl.get_mode(hour=22.0)  # -> "fixed_time"

    # Run a full simulation episode under scheduled control:
    result = ctrl.run_episode(net_file, route_file, hour_start=7.0,
                              num_seconds=14400, use_gui=False)
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sumo_rl
from stable_baselines3 import PPO

from config import (
    TS_IDS, DELTA_TIME, YELLOW_TIME, MIN_GREEN, MAX_GREEN,
    REWARD_FN, WARMUP_SECONDS,
    MORNING_RUSH_START, MORNING_RUSH_END,
    EVENING_RUSH_START, EVENING_RUSH_END,
)
from tls_programs import parse_original_programs, restore_non_target_programs

# Import time encoding
import experiment
from experiment import TimeEncodedObservationFunction


# ── Schedule definition ────────────────────────────────────────────────────

@dataclass
class TimeWindow:
    """A contiguous time window and the controller mode it maps to."""
    start: float   # Hour of day (0.0-24.0), inclusive
    end: float     # Hour of day, exclusive
    mode: str      # "rl_morning", "rl_evening", "fixed_time"
    label: str     # Human-readable description


SCHEDULE: list[TimeWindow] = [
    TimeWindow(0.0,                MORNING_RUSH_START, "fixed_time",  "Off-peak night        (00:00-06:00)"),
    TimeWindow(MORNING_RUSH_START, MORNING_RUSH_END,   "rl_morning",  "Morning rush          (06:00-10:00)"),
    TimeWindow(MORNING_RUSH_END,   EVENING_RUSH_START, "fixed_time",  "Shoulder daytime      (10:00-14:00)"),
    TimeWindow(EVENING_RUSH_START, EVENING_RUSH_END,   "rl_evening",  "Evening rush          (14:00-18:00)"),
    TimeWindow(EVENING_RUSH_END,   21.0,               "fixed_time",  "Shoulder evening      (18:00-21:00)"),
    TimeWindow(21.0,               24.0,               "fixed_time",  "Off-peak night        (21:00-00:00)"),
]


class ScheduleController:
    """
    Time-aware controller that dispatches to the appropriate mode.

    Attributes:
        model_morning: PPO model trained on morning rush demand.
        model_evening: PPO model trained on evening rush demand.
    """

    def __init__(
        self,
        model_morning: Optional[str] = None,
        model_evening: Optional[str] = None,
    ):
        self._model_morning: Optional[PPO] = None
        self._model_evening: Optional[PPO] = None

        if model_morning and os.path.exists(model_morning):
            print(f"Loading morning model: {model_morning}")
            self._model_morning = PPO.load(model_morning)
        elif model_morning:
            print(f"WARNING: Morning model not found: {model_morning}")

        if model_evening and os.path.exists(model_evening):
            print(f"Loading evening model: {model_evening}")
            self._model_evening = PPO.load(model_evening)
        elif model_evening:
            print(f"WARNING: Evening model not found: {model_evening}")

    def get_mode(self, hour: float) -> str:
        """
        Return the controller mode for the given hour of day.

        Args:
            hour: Decimal hour (e.g., 7.5 = 07:30, 16.25 = 16:15).

        Returns:
            One of: "rl_morning", "rl_evening", "fixed_time"
        """
        hour = hour % 24.0
        for window in SCHEDULE:
            if window.start <= hour < window.end:
                return window.mode
        return "fixed_time"

    def get_window(self, hour: float) -> TimeWindow:
        """Return the full TimeWindow metadata for the given hour."""
        hour = hour % 24.0
        for window in SCHEDULE:
            if window.start <= hour < window.end:
                return window
        return SCHEDULE[-1]

    def get_model(self, hour: float) -> Optional[PPO]:
        """Return the PPO model appropriate for the given hour, or None."""
        mode = self.get_mode(hour)
        if mode == "rl_morning":
            return self._model_morning
        if mode == "rl_evening":
            return self._model_evening
        return None  # fixed_time: no model needed

    def run_episode(
        self,
        net_file: str,
        route_file: str,
        hour_start: float,
        num_seconds: int,
        use_gui: bool = False,
    ) -> dict:
        """
        Run one simulation episode under scheduled control.

        The controller mode is determined by hour_start and held constant for
        the whole episode (in practice, mode transitions happen between episodes
        at the traffic management system level, not mid-episode).

        Returns:
            dict with keys: mode, total_reward, avg_queue, avg_wait, total_teleports
        """
        mode = self.get_mode(hour_start)
        window = self.get_window(hour_start)
        model = self.get_model(hour_start)
        fixed_ts = (model is None)

        print(f"\nSchedule: {window.label}")
        print(f"  Mode: {mode}  |  RL model: {'loaded' if model else 'none (fixed-time)'}")

        # Set time-of-day for observation encoding
        experiment.CURRENT_HOUR = hour_start

        env = sumo_rl.SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=num_seconds,
            reward_fn=REWARD_FN,
            delta_time=DELTA_TIME,
            yellow_time=YELLOW_TIME,
            min_green=MIN_GREEN,
            max_green=MAX_GREEN,
            single_agent=False,
            fixed_ts=fixed_ts,
            sumo_warnings=False,
            observation_class=TimeEncodedObservationFunction,
        )

        observations = env.reset()
        target_set = set(TS_IDS)
        rewards = {ts_id: 0.0 for ts_id in TS_IDS}
        done = False

        obs_size = model.observation_space.shape[0] if model else None

        if model is not None:
            original_programs = parse_original_programs(net_file)
            restore_non_target_programs(env, TS_IDS, original_programs)

        queue_steps: list[float] = []
        wait_steps: list[float] = []
        total_teleports = 0
        n_lanes = sum(len(env.traffic_signals[ts].lanes) for ts in TS_IDS
                      if ts in env.traffic_signals)

        while not done:
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
                    actions[ts_id] = 0

            observations, reward_dict, done_dict, info = env.step(actions)
            done = done_dict["__all__"]

            for ts_id in TS_IDS:
                if ts_id in reward_dict:
                    rewards[ts_id] += reward_dict[ts_id]

            try:
                sumo = env.sumo
                stopped = sum(
                    sumo.lane.getLastStepHaltingNumber(lane)
                    for ts_id in TS_IDS if ts_id in env.traffic_signals
                    for lane in env.traffic_signals[ts_id].lanes
                )
                queue_steps.append(stopped)
                wait = sum(
                    sumo.lane.getWaitingTime(lane)
                    for ts_id in TS_IDS if ts_id in env.traffic_signals
                    for lane in env.traffic_signals[ts_id].lanes
                )
                wait_steps.append(wait / max(n_lanes, 1))
                total_teleports += sumo.simulation.getStartingTeleportNumber()
            except Exception:
                pass

        env.close()

        result = {
            "mode": mode,
            "window": window.label,
            "total_reward": sum(rewards.values()),
            "avg_queue": float(np.mean(queue_steps)) if queue_steps else float("nan"),
            "avg_wait": float(np.mean(wait_steps)) if wait_steps else float("nan"),
            "total_teleports": total_teleports,
        }
        print(f"  Result: reward={result['total_reward']:.0f}  "
              f"queue={result['avg_queue']:.1f}  "
              f"wait={result['avg_wait']:.1f}s  "
              f"teleports={result['total_teleports']}")
        return result

    def print_schedule(self):
        """Print the full daily schedule."""
        print("\nZeleni SignaLJ -- Daily Control Schedule")
        print("=" * 55)
        for w in SCHEDULE:
            model_info = ""
            if w.mode == "rl_morning":
                model_info = "<- PPO morning model"
                if self._model_morning is None:
                    model_info += " (NOT LOADED)"
            elif w.mode == "rl_evening":
                model_info = "<- PPO evening model"
                if self._model_evening is None:
                    model_info += " (NOT LOADED)"
            print(f"  {w.label:<45} {model_info}")
        print("=" * 55)


if __name__ == "__main__":
    # Quick demo: print the schedule and check which mode applies at various hours
    ctrl = ScheduleController()
    ctrl.print_schedule()

    print("\nMode lookup examples:")
    for h in [5.0, 6.0, 8.0, 10.5, 14.0, 16.0, 18.5, 22.0]:
        mode = ctrl.get_mode(h)
        window = ctrl.get_window(h)
        print(f"  {h:>5.1f}h  ->  {mode:<14}  ({window.label})")
