"""
Zeleni SignaLJ - Original TLS Program Restoration
===================================================
Parses original traffic signal programs from the SUMO .net.xml file
and restores them after sumo-rl replaces them during env creation.

Background:
  sumo-rl's TrafficSignal.build_phases() replaces every TLS's original
  SUMO program with its own phase-based control. This means non-target
  TLS lose their real fixed-time cycling behavior. Without restoration,
  any action sent to non-target TLS (including action=0="keep phase")
  will NOT approximate the original program.

Fix:
  1. Parse original programs from the .net.xml file
  2. After env.reset(), restore them via TraCI for non-target TLS
  3. Monkey-patch non-target TrafficSignal objects so sumo-rl doesn't
     override the restored programs with setRedYellowGreenState calls
"""

import xml.etree.ElementTree as ET


def parse_original_programs(net_file):
    """Parse original TLS programs from a SUMO .net.xml file.

    Returns:
        dict: {tls_id: {"programID": str, "type": str, "offset": int,
                         "phases": [{"duration": int, "state": str}, ...]}}
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    programs = {}
    for tl in root.findall(".//tlLogic"):
        tl_id = tl.get("id")
        phases = []
        for p in tl.findall("phase"):
            phases.append({
                "duration": int(p.get("duration")),
                "state": p.get("state"),
            })
        programs[tl_id] = {
            "programID": tl.get("programID", "0"),
            "type": tl.get("type", "static"),
            "offset": int(tl.get("offset", "0")),
            "phases": phases,
        }

    return programs


def restore_non_target_programs(env, target_agent_ids, original_programs):
    """Restore original SUMO signal programs for non-target TLS.

    After sumo-rl's env.reset(), this function:
    1. Restores the original program for each non-target TLS via TraCI
    2. Monkey-patches TrafficSignal.set_next_phase and .update so that
       sumo-rl doesn't override the restored program

    Args:
        env: The unwrapped SumoEnvironment (or object with .traffic_signals
             and .sumo attributes)
        target_agent_ids: set/list of TLS IDs controlled by RL
        original_programs: dict from parse_original_programs()
    """
    target_set = set(target_agent_ids)

    # Access the underlying SUMO environment
    sumo_env = env
    while hasattr(sumo_env, 'env'):
        sumo_env = sumo_env.env
    if hasattr(sumo_env, 'unwrapped'):
        sumo_env = sumo_env.unwrapped

    # Get the SUMO/libsumo connection
    sumo = sumo_env.sumo if hasattr(sumo_env, 'sumo') else None
    if sumo is None:
        return

    traffic_signals = getattr(sumo_env, 'traffic_signals', {})

    for ts_id, ts_obj in traffic_signals.items():
        if ts_id in target_set:
            continue  # Leave target TLS under RL control

        if ts_id not in original_programs:
            continue  # No original program found (shouldn't happen)

        prog = original_programs[ts_id]

        # 1. Restore the original program via TraCI
        #    Use sumo.trafficlight.Phase() (same approach as sumo-rl itself)
        #    instead of importing traci/libsumo Phase classes directly,
        #    which fails under LIBSUMO_AS_TRACI=1.
        try:
            # Build phase objects using the sumo connection's Phase class
            phases = []
            for p in prog["phases"]:
                phases.append(sumo.trafficlight.Phase(
                    p["duration"], p["state"]
                ))

            # Get the existing Logic object (sumo-rl installed), replace
            # its phases with the original ones (same pattern sumo-rl uses)
            programs = sumo.trafficlight.getAllProgramLogics(ts_id)
            logic = programs[0]
            logic.type = 0  # static
            logic.phases = phases
            sumo.trafficlight.setProgramLogic(ts_id, logic)
            # Re-activate automatic phase cycling: _build_phases() called
            # setRedYellowGreenState() which put TLS in manual mode.
            # setProgramLogic() alone only updates phase definitions.
            # setProgram() is required to resume automatic cycling.
            sumo.trafficlight.setProgram(ts_id, logic.programID)
            # Restore original offset (for signal coordination)
            offset = prog.get("offset", 0)
            if offset != 0:
                try:
                    sumo.trafficlight.setParameter(ts_id, "offset", str(offset))
                except Exception:
                    pass  # offset=0 is common, setParameter may not be available
        except Exception as e:
            print(f"  WARNING: Could not restore program for {ts_id}: {e}")
            continue

        # 2. Monkey-patch the TrafficSignal object so sumo-rl
        #    doesn't override the restored program
        _patch_non_target_ts(ts_obj)


def _patch_non_target_ts(ts_obj):
    """Monkey-patch a TrafficSignal so it doesn't call setRedYellowGreenState.

    After patching:
    - set_next_phase is a no-op (doesn't set state, doesn't update timing)
    - update just increments time counter (no yellow handling)
    - time_to_act will be False after the first step (next_action_time
      stays at initial value, never updated)
    - The SUMO program runs the TLS naturally
    """

    def noop_set_next_phase(new_phase):
        # Complete no-op: don't set state, don't update next_action_time.
        # This ensures time_to_act stays False after the first sim step,
        # so sumo-rl never tries to control this TLS again.
        pass

    def noop_update():
        # Increment time counter (harmless) but skip yellow handling
        # which would call setRedYellowGreenState
        ts_obj.time_since_last_phase_change += 1

    ts_obj.set_next_phase = noop_set_next_phase
    ts_obj.update = noop_update
