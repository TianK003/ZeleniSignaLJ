"""
Zeleni SignaLJ - Agent Filter Wrapper
======================================
PettingZoo parallel env wrapper that exposes only selected agents.
Non-selected agents run their original SUMO signal programs, restored
after sumo-rl replaces them during env creation.

CRITICAL FIX (2026-04-06): Previous versions sent action=0 or cycled
phases for non-target agents. Neither approach works correctly:
- action=0 freezes the TLS on its initial green phase forever
- Round-robin cycling has unrealistic 5s phases (real cycles are 90s)

Current approach: Parse original programs from .net.xml, restore them
via TraCI after env.reset(), and monkey-patch non-target TrafficSignal
objects so sumo-rl doesn't override the restored programs.
"""

import functools

from pettingzoo.utils.env import ParallelEnv
from tls_programs import parse_original_programs, restore_non_target_programs


class AgentFilterWrapper(ParallelEnv):
    """
    Wraps a PettingZoo parallel env to only expose selected agents.
    Non-selected agents run their original SUMO signal programs.
    """

    metadata = {"render_modes": [], "name": "agent_filter_v0"}

    def __init__(self, env, target_agents, net_file=None, default_action=0):
        """
        Args:
            env: PettingZoo parallel environment (e.g. sumo_rl.parallel_env())
            target_agents: list of agent IDs to keep as RL-controlled
            net_file: path to .net.xml file (for original TLS programs)
            default_action: fallback action if program restoration fails
        """
        self.env = env
        self.target_agents = list(target_agents)
        self.default_action = default_action
        self.net_file = net_file

        # PettingZoo parallel API attributes
        self.possible_agents = list(target_agents)
        self.agents = list(target_agents)

        # Parse original TLS programs from network file
        self._original_programs = {}
        if net_file:
            try:
                self._original_programs = parse_original_programs(net_file)
            except Exception as e:
                print(f"  WARNING: Could not parse TLS programs from {net_file}: {e}")

        # Propagate render_mode if present
        if hasattr(env, "render_mode"):
            self.render_mode = env.render_mode
        else:
            self.render_mode = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.env.observation_space(agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.env.action_space(agent)

    @property
    def observation_spaces(self):
        """Dict of observation spaces (SuperSuit 3.9.x may use this)."""
        return {a: self.observation_space(a) for a in self.possible_agents}

    @property
    def action_spaces(self):
        """Dict of action spaces (SuperSuit 3.9.x may use this)."""
        return {a: self.action_space(a) for a in self.possible_agents}

    def reset(self, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        if isinstance(result, tuple):
            observations, infos = result
        else:
            observations = result
            infos = {}

        self.agents = [a for a in self.target_agents
                       if a in self.env.agents]

        # Restore original SUMO programs for non-target TLS
        # This must happen AFTER reset (which starts SUMO and creates
        # TrafficSignal objects that replace original programs)
        if self._original_programs:
            try:
                restore_non_target_programs(
                    self.env, self.target_agents, self._original_programs
                )
            except Exception as e:
                print(f"  WARNING: Could not restore TLS programs: {e}")

        filtered_obs = {a: observations[a] for a in self.agents
                        if a in observations}
        filtered_info = {a: infos.get(a, {}) for a in self.agents}
        return filtered_obs, filtered_info

    def step(self, actions):
        """
        actions: dict of {target_agent: action} from RL policy.
        Non-target agents get default_action (but their TrafficSignal
        objects are patched to ignore it — SUMO runs their programs).
        """
        # Build full action dict: RL actions for targets, default for rest
        # Non-target actions are ignored by the monkey-patched set_next_phase
        full_actions = {}
        target_set = set(self.target_agents)
        for agent in self.env.agents:
            if agent in actions:
                full_actions[agent] = actions[agent]
            else:
                full_actions[agent] = self.default_action

        # Step the underlying env
        result = self.env.step(full_actions)
        if len(result) == 5:
            observations, rewards, terminations, truncations, infos = result
        else:
            observations, rewards, dones, infos = result
            terminations = dones
            truncations = {a: False for a in dones}

        # Filter to only target agents
        self.agents = [a for a in self.target_agents
                       if a in self.env.agents]

        f_obs = {a: observations[a] for a in self.agents if a in observations}
        f_rew = {a: rewards.get(a, 0.0) for a in self.agents}
        f_term = {a: terminations.get(a, False) for a in self.agents}
        f_trunc = {a: truncations.get(a, False) for a in self.agents}
        f_info = {a: infos.get(a, {}) for a in self.agents}

        return f_obs, f_rew, f_term, f_trunc, f_info

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    @property
    def unwrapped(self):
        return self.env.unwrapped
