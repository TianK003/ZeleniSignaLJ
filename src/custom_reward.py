"""
Zeleni SignaLJ - Custom Reward Functions
==========================================
Drop-in reward functions for sumo-rl.

Usage in train.py:
    from custom_reward import queue_reward, multi_objective_reward
    env = sumo_rl.SumoEnvironment(..., reward_fn=queue_reward)
"""

import traci


def queue_reward(traffic_signal):
    """Primary reward: negative total queue length.

    Intuition: the agent is penalized for every stopped car.
    This drives the agent to flush queues and create green waves.
    """
    return -traffic_signal.get_total_queued()


def multi_objective_reward(traffic_signal):
    """Weighted multi-objective: queue + CO2 + waiting time.

    Use this for Day 8+ experiments if time permits.
    Requires TraCI (not libsumo) for emission data.
    """
    # Queue component (primary)
    queue = traffic_signal.get_total_queued()

    # Emission component
    co2_total = sum(
        traci.lane.getCO2Emission(lane)
        for lane in traffic_signal.lanes
    )

    # Waiting time component (proxy for noise)
    waiting = sum(
        traci.lane.getWaitingTime(lane)
        for lane in traffic_signal.lanes
    )

    # Weighted combination (tune weights empirically)
    w_queue, w_co2, w_wait = 0.5, 0.3, 0.2
    return -(w_queue * queue + w_co2 * co2_total / 1000 + w_wait * waiting / 100)


def stops_reward(traffic_signal):
    """Penalty per vehicle stop. Best proxy for noise reduction.

    Each stop = brake noise + acceleration noise.
    Minimizing stops directly minimizes noise events.
    """
    total_stops = sum(
        traci.lane.getLastStepHaltingNumber(lane)
        for lane in traffic_signal.lanes
    )
    return -total_stops
