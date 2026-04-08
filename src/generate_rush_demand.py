"""
Zeleni SignaLJ - Rush Hour Demand Generator
=============================================
Generates route files for rush-hour training scenarios using the project's
bimodal 24h traffic density curve (demand_math.get_vph).

Three scenarios:
  morning_rush  - 4h window 06:00-10:00, inbound directional bias
  evening_rush  - 4h window 14:00-18:00, outbound directional bias
  offpeak       - 1h window 12:00-13:00, uniform low demand (reference)

Directional asymmetry is approximated by merging two trip batches:
  inbound  (morning): 70% high-fringe-factor + 30% low-fringe-factor
  outbound (evening): 30% high-fringe-factor + 70% low-fringe-factor

High fringe-factor biases origins to network edges (inbound to center).
Low fringe-factor generates local/random trips (approximates outbound).

Usage:
    python src/generate_rush_demand.py --scenario all
    python src/generate_rush_demand.py --scenario morning_rush
    python src/generate_rush_demand.py --scenario evening_rush --output_dir data/routes
"""

import argparse
import os
import sys
import subprocess
import tempfile

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))
from demand_math import get_vph
from generate_demand import write_demand_xml
from config import (
    TOTAL_DAILY_CARS,
    MORNING_RUSH_START, MORNING_RUSH_END, MORNING_RUSH_SECONDS,
    EVENING_RUSH_START, EVENING_RUSH_END, EVENING_RUSH_SECONDS,
    OFFPEAK_SECONDS,
)

# ── Scenario definitions ───────────────────────────────────────────────────
SCENARIOS = {
    "morning_rush": {
        "start_hour": MORNING_RUSH_START,    # 06:00
        "end_hour": MORNING_RUSH_END,        # 10:00
        "duration": MORNING_RUSH_SECONDS,    # 14400s
        "inbound_fraction": 0.70,            # 70% high-fringe (edge -> center)
        "fringe_factor_high": 15,
        "fringe_factor_low": 1,
        "output": "routes_morning_rush.rou.xml",
        "trips": "trips_morning_rush.trips.xml",
    },
    "evening_rush": {
        "start_hour": EVENING_RUSH_START,    # 14:00
        "end_hour": EVENING_RUSH_END,        # 18:00
        "duration": EVENING_RUSH_SECONDS,    # 14400s
        "inbound_fraction": 0.30,            # 30% high-fringe; 70% low-fringe = outbound
        "fringe_factor_high": 15,
        "fringe_factor_low": 1,
        "output": "routes_evening_rush.rou.xml",
        "trips": "trips_evening_rush.trips.xml",
    },
    "offpeak": {
        "start_hour": 12.0,                  # midday
        "end_hour": 13.0,
        "duration": OFFPEAK_SECONDS,         # 3600s
        "inbound_fraction": 0.50,            # No directional bias
        "fringe_factor_high": 5,
        "fringe_factor_low": 5,
        "output": "routes_offpeak.rou.xml",
        "trips": "trips_offpeak.trips.xml",
    },
}


def bimodal_demand_bins(start_hour, end_hour, interval=300):
    """
    Sample the project's bimodal 24h density curve for a time window.

    Maps each simulation time bin to a real-world hour and calls
    get_vph(hour, TOTAL_DAILY_CARS) to get the traffic rate.

    Returns list of (begin_sec, end_sec, rate_vps) tuples.
    """
    duration = int((end_hour - start_hour) * 3600)
    bins = []

    for sim_t in range(0, duration, interval):
        real_hour = start_hour + sim_t / 3600.0
        vph = get_vph(real_hour, TOTAL_DAILY_CARS)
        rate_vps = vph / 3600.0  # vehicles per second
        end_t = min(sim_t + interval, duration)
        bins.append((sim_t, end_t, rate_vps))

    return bins


def _volume_split(bins, fraction):
    """Scale demand bins to a given fraction of total volume."""
    return [(b, e, r * fraction) for b, e, r in bins]


def _generate_trips_only(bins, net_file, output_trips, fringe_factor, seed_offset=0, id_prefix=""):
    """
    Run randomTrips.py for each time bin and merge into output_trips.
    Does NOT run duarouter -- call _route_trips() separately.
    Returns total vehicle count.
    """
    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.exists(random_trips):
        print(f"ERROR: randomTrips.py not found at {random_trips}")
        print("Set SUMO_HOME correctly, e.g.: export SUMO_HOME=/usr/share/sumo")
        sys.exit(1)

    tmp_dir = tempfile.gettempdir()
    all_tmp = []
    total = 0

    for i, (begin, end, rate) in enumerate(bins):
        duration = end - begin
        n_vehicles = int(rate * duration)
        if n_vehicles < 1:
            continue
        period = max(duration / n_vehicles, 0.5)
        total += n_vehicles

        tmp = os.path.join(tmp_dir, f"trips_partial_{seed_offset}_{i}.trips.xml")
        all_tmp.append(tmp)
        cmd = [
            sys.executable, random_trips,
            "-n", net_file,
            "-o", tmp,
            "-b", str(int(begin)),
            "-e", str(int(end)),
            "-p", str(period),
            "--fringe-factor", str(fringe_factor),
            "--seed", str(42 + seed_offset + i),
            "--prefix", f"{id_prefix}{seed_offset}_{i}_",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: randomTrips failed (bin {i}): {result.stderr[:200]}")

    # Merge into output_trips
    with open(output_trips, "w") as out:
        out.write("<trips>\n")
        for tf in all_tmp:
            if not os.path.exists(tf):
                continue
            with open(tf) as f:
                for line in f:
                    line_s = line.strip()
                    if line_s.startswith("<trip "):
                        out.write(f"    {line_s}\n")
            os.remove(tf)
        out.write("</trips>\n")

    return total


def generate_scenario(scenario_name, net_file, output_dir):
    """Generate route file for one rush scenario using the bimodal curve."""
    cfg = SCENARIOS[scenario_name]
    os.makedirs(output_dir, exist_ok=True)

    output_routes = os.path.join(output_dir, cfg["output"])
    output_trips = os.path.join(output_dir, cfg["trips"])

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"  Time window: {cfg['start_hour']:.0f}:00 - {cfg['end_hour']:.0f}:00 "
          f"({cfg['duration']}s = {cfg['duration']/3600:.1f}h)")
    print(f"  Demand source: bimodal 24h curve (demand_math.get_vph)")
    print(f"  Inbound %  : {cfg['inbound_fraction']*100:.0f}%")
    print(f"  Output     : {output_routes}")

    # Sample the bimodal curve for this time window
    full_bins = bimodal_demand_bins(cfg["start_hour"], cfg["end_hour"])

    # Print demand profile summary
    print(f"\n  Demand profile ({len(full_bins)} bins):")
    print(f"    {'Sim time':>10s}  {'Real hour':>10s}  {'Rate (veh/h)':>12s}")
    print(f"    {'-'*38}")
    for i, (b, e, rate) in enumerate(full_bins):
        if i % 4 == 0 or i == len(full_bins) - 1:
            real_h = cfg["start_hour"] + b / 3600.0
            print(f"    {b/60:>7.0f}min  {real_h:>10.1f}h  {rate*3600:>12.0f}")

    inbound_frac = cfg["inbound_fraction"]
    outbound_frac = 1.0 - inbound_frac

    if abs(inbound_frac - 0.5) < 0.01:
        # Symmetric: single batch, no need to merge
        total = write_demand_xml(
            full_bins,
            net_file,
            output_trips,
            output_routes,
            fringe_factor=cfg["fringe_factor_high"],
        )
    else:
        # Asymmetric: merge inbound (high-fringe) + outbound (low-fringe) batches
        inbound_bins = _volume_split(full_bins, inbound_frac)
        outbound_bins = _volume_split(full_bins, outbound_frac)

        tmp_dir = tempfile.gettempdir()
        trips_inbound = os.path.join(tmp_dir, f"trips_{scenario_name}_in.trips.xml")
        trips_outbound = os.path.join(tmp_dir, f"trips_{scenario_name}_out.trips.xml")
        trips_merged = os.path.join(tmp_dir, f"trips_{scenario_name}_merged.trips.xml")

        print(f"\n  [1/3] Generating inbound trips  ({inbound_frac*100:.0f}% volume, "
              f"fringe={cfg['fringe_factor_high']})...")
        _generate_trips_only(
            inbound_bins, net_file, trips_inbound,
            fringe_factor=cfg["fringe_factor_high"],
            seed_offset=0,
            id_prefix="in",
        )

        print(f"\n  [2/3] Generating outbound trips ({outbound_frac*100:.0f}% volume, "
              f"fringe={cfg['fringe_factor_low']})...")
        _generate_trips_only(
            outbound_bins, net_file, trips_outbound,
            fringe_factor=cfg["fringe_factor_low"],
            seed_offset=1000,
            id_prefix="out",
        )

        # Merge trips and sort by departure time
        print("\n  [3/3] Merging and routing combined demand...")
        _merge_trips(trips_inbound, trips_outbound, trips_merged)
        _route_trips(trips_merged, net_file, output_trips, output_routes)

        # Count total vehicles
        total = _count_trips(trips_merged)

        # Clean up temp files
        for f in (trips_inbound, trips_outbound, trips_merged):
            if os.path.exists(f):
                os.remove(f)

    print(f"\n  Done: {total} vehicles -> {output_routes}")
    return output_routes


def _merge_trips(file_a, file_b, output):
    """Merge two trips XML files, sorted by departure time."""
    trips = []

    for fname in (file_a, file_b):
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            for line in f:
                line_s = line.strip()
                if line_s.startswith("<trip "):
                    trips.append(line_s)

    # Sort by depart attribute
    def _depart(trip_line):
        try:
            idx = trip_line.index('depart="') + 8
            end = trip_line.index('"', idx)
            return float(trip_line[idx:end])
        except ValueError:
            return 0.0

    trips.sort(key=_depart)

    with open(output, "w") as f:
        f.write('<trips>\n')
        for t in trips:
            f.write(f"    {t}\n")
        f.write('</trips>\n')


def _route_trips(trips_file, net_file, output_trips, output_routes):
    """Copy merged trips to final location and run DUAROUTER."""
    import shutil
    shutil.copy(trips_file, output_trips)

    cmd = [
        "duarouter",
        "-n", net_file,
        "-t", output_trips,
        "-o", output_routes,
        "--ignore-errors",
        "--no-warnings",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: duarouter issues: {result.stderr[:300]}")
    else:
        print(f"  Routes written to: {output_routes}")


def _count_trips(trips_file):
    """Count <trip ...> lines in a trips file."""
    if not os.path.exists(trips_file):
        return 0
    count = 0
    with open(trips_file) as f:
        for line in f:
            if line.strip().startswith("<trip "):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Generate rush hour traffic demand using the bimodal 24h curve"
    )
    parser.add_argument(
        "--net_file", type=str, default="data/networks/ljubljana.net.xml"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/routes",
        help="Directory for generated route files"
    )
    parser.add_argument(
        "--scenario", type=str, default="all",
        choices=["morning_rush", "evening_rush", "offpeak", "all"],
        help="Which scenario(s) to generate"
    )
    args = parser.parse_args()

    targets = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    for scenario in targets:
        generate_scenario(scenario, args.net_file, args.output_dir)

    print(f"\nAll done. Route files are in {args.output_dir}/")
    print("Next steps:")
    for s in targets:
        print(f"  python src/train.py --scenario {s}")


if __name__ == "__main__":
    main()
