"""
Zeleni SignaLJ - Traffic Demand Generator
==========================================
Generate traffic demand with different temporal profiles.

Modes:
  --profile uniform    Constant rate (for smoke tests / debugging)
  --scenario <name>    Rush-hour scenarios using the bimodal 24h curve

Usage:
    # Quick uniform demand (smoke test)
    python src/generate_demand.py --profile uniform --duration 3600 --peak_vph 800

    # Rush-hour scenarios (bimodal curve, directional asymmetry)
    python src/generate_demand.py --scenario all
    python src/generate_demand.py --scenario morning_rush
    python src/generate_demand.py --scenario evening_rush --output_dir data/routes
"""

import argparse
import subprocess
import os
import sys
import shutil
import numpy as np
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from demand_math import get_vph
from config import (
    TOTAL_DAILY_CARS,
    MORNING_RUSH_START, MORNING_RUSH_END, MORNING_RUSH_SECONDS,
    EVENING_RUSH_START, EVENING_RUSH_END, EVENING_RUSH_SECONDS,
    OFFPEAK_SECONDS, FULL_DAY_SECONDS,
)


def generate_demand_profile(duration, peak_vph, interval=300):
    """
    Generate a list of (begin, end, rate) tuples for uniform demand.
    rate = vehicles per second for that interval.
    interval = seconds per time bin (default 5 min).
    """
    bins = []
    peak_vps = peak_vph / 3600  # vehicles per second at peak

    for t in range(0, duration, interval):
        end = min(t + interval, duration)
        bins.append((t, end, peak_vps))

    return bins


def get_random_trips_path():
    import shutil
    sumo_homes = [
        os.environ.get("SUMO_HOME", ""),
        os.path.expanduser("~/sumo_src"),
        "/usr/share/sumo",
        "/usr/local/share/sumo"
    ]
    for sh in sumo_homes:
        if not sh: continue
        path = os.path.join(sh, "tools", "randomTrips.py")
        if os.path.exists(path): return path
        
    path_in_path = shutil.which("randomTrips.py")
    if path_in_path: return path_in_path
    return None

def write_demand_xml(bins, net_file, output_trips, output_routes, fringe_factor=5, master_seed=42):
    """
    Write trips XML with time-varying demand using randomTrips.py per bin,
    then merge and route.
    """
    random_trips = get_random_trips_path()

    if not random_trips:
        print("ERROR: randomTrips.py not found in SUMO_HOME, ~/sumo_src, or PATH.")
        print("Set SUMO_HOME environment variable correctly.")
        sys.exit(1)

    # Generate trips for each time bin, collect into one file
    all_trips_files = []
    total_vehicles = 0

    print(f"\nDemand profile ({len(bins)} intervals):")
    print(f"  {'Time':>10s}  {'Rate (veh/h)':>12s}  {'Vehicles':>10s}")
    print(f"  {'-'*38}")

    for i, (begin, end, rate) in enumerate(bins):
        interval_duration = end - begin
        n_vehicles = int(rate * interval_duration)
        if n_vehicles < 1:
            continue

        period = interval_duration / n_vehicles if n_vehicles > 0 else 999
        total_vehicles += n_vehicles

        # Show a few representative lines
        if i % 4 == 0 or i == len(bins) - 1:
            t_min = begin / 60
            print(f"  {t_min:>7.0f}min  {rate * 3600:>12.0f}  {n_vehicles:>10d}")

        tmp_dir = tempfile.gettempdir()
        tmp_trips = os.path.join(tmp_dir, f"trips_bin_{i}.trips.xml")
        all_trips_files.append(tmp_trips)

        cmd = [
            sys.executable, "-B", random_trips,
            "-n", net_file,
            "-o", tmp_trips,
            "-b", str(int(begin)),
            "-e", str(int(end)),
            "-p", str(max(period, 0.5)),
            "--fringe-factor", str(fringe_factor),
            "--seed", str(master_seed + i),
            "--prefix", f"b{i}_",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: randomTrips failed for bin {i}: {result.stderr[:2000]}")

    print(f"\n  Total vehicles: {total_vehicles}")

    # Merge all trip files into one
    print("\n  Merging trip files...")
    with open(output_trips, "w") as out:
        out.write('<trips>\n')
        for tf in all_trips_files:
            if not os.path.exists(tf):
                continue
            with open(tf) as f:
                for line in f:
                    line_s = line.strip()
                    if line_s.startswith("<trip "):
                        out.write(f"    {line_s}\n")
            os.remove(tf)
        out.write('</trips>\n')

    # Route with DUAROUTER
    print("  Running DUAROUTER to compute routes...")
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

    return total_vehicles


# ── Rush-hour scenario definitions ────────────────────────────────────────

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
    "full_day": {
        "start_hour": 0.0,
        "end_hour": 24.0,
        "duration": FULL_DAY_SECONDS,        # 86400s
        "output": "routes_full_day.rou.xml",
        "trips": "trips_full_day.trips.xml",
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


def _generate_trips_only(bins, net_file, output_trips, fringe_factor, seed_offset=0, id_prefix="", master_seed=42):
    """
    Run randomTrips.py for each time bin and merge into output_trips.
    Does NOT run duarouter -- call _route_trips() separately.
    Returns total vehicle count.
    """
    random_trips = get_random_trips_path()
    if not random_trips:
        print("ERROR: randomTrips.py not found in SUMO_HOME, ~/sumo_src, or PATH.")
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

        tmp = os.path.join(tmp_dir, f"trips_partial_{master_seed}_{seed_offset}_{i}.trips.xml")
        all_tmp.append(tmp)
        cmd = [
            sys.executable, "-B", random_trips,
            "-n", net_file,
            "-o", tmp,
            "-b", str(int(begin)),
            "-e", str(int(end)),
            "-p", str(period),
            "--fringe-factor", str(fringe_factor),
            "--seed", str(master_seed + seed_offset + i),
            "--prefix", f"{id_prefix}{master_seed}_{seed_offset}_{i}_",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: randomTrips failed (bin {i}): {result.stderr[:2000]}")

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


def _generate_full_day_scenario(net_file, output_dir, master_seed=42, output_suffix=""):
    """Generate a full 24h route file with time-varying directional asymmetry.

    - 06:00-10:00 morning rush: 70% inbound (fringe=15), 30% outbound (fringe=1)
    - 14:00-18:00 evening rush: 30% inbound (fringe=15), 70% outbound (fringe=1)
    - All other hours: 50/50 symmetric (fringe=5)

    Args:
        master_seed: Base seed for randomTrips.py. Different values produce
                     different OD pairs while keeping the same demand curve.
        output_suffix: If set, appended to output filename (e.g. "seed_00"
                       -> routes_full_day_seed_00.rou.xml)
    """
    cfg = SCENARIOS["full_day"]
    os.makedirs(output_dir, exist_ok=True)

    base_routes = cfg["output"]
    base_trips = cfg["trips"]
    if output_suffix:
        base_routes = base_routes.replace(".rou.xml", f"_{output_suffix}.rou.xml")
        base_trips = base_trips.replace(".trips.xml", f"_{output_suffix}.trips.xml")
    output_routes = os.path.join(output_dir, base_routes)
    output_trips = os.path.join(output_dir, base_trips)

    print(f"\n{'='*60}")
    print(f"Scenario: full_day (24h with time-varying directional asymmetry)")
    print(f"  Time window: 00:00 - 24:00 ({cfg['duration']}s = 24h)")
    print(f"  Demand source: bimodal 24h curve (demand_math.get_vph)")
    print(f"  Output: {output_routes}")

    # Generate full 24h demand bins
    full_bins = bimodal_demand_bins(0.0, 24.0)

    # Partition bins into time periods
    morning_bins = []
    evening_bins = []
    offpeak_bins = []

    for b, e, rate in full_bins:
        real_hour = b / 3600.0
        if MORNING_RUSH_START <= real_hour < MORNING_RUSH_END:
            morning_bins.append((b, e, rate))
        elif EVENING_RUSH_START <= real_hour < EVENING_RUSH_END:
            evening_bins.append((b, e, rate))
        else:
            offpeak_bins.append((b, e, rate))

    tmp_dir = tempfile.gettempdir()
    # Use a unique tag per seed so parallel workers don't collide on temp files.
    tag = output_suffix if output_suffix else f"ms{master_seed}"
    all_trip_files = []
    total_vehicles = 0

    # Morning rush: 70% inbound (fringe=15), 30% outbound (fringe=1)
    if morning_bins:
        print(f"\n  [1/5] Morning rush inbound  (70%, fringe=15)...")
        f_in = os.path.join(tmp_dir, f"fd_morning_in_{tag}.trips.xml")
        total_vehicles += _generate_trips_only(
            _volume_split(morning_bins, 0.70), net_file, f_in,
            fringe_factor=15, seed_offset=0, id_prefix="morn_in_",
            master_seed=master_seed)
        all_trip_files.append(f_in)

        print(f"  [2/5] Morning rush outbound (30%, fringe=1)...")
        f_out = os.path.join(tmp_dir, f"fd_morning_out_{tag}.trips.xml")
        total_vehicles += _generate_trips_only(
            _volume_split(morning_bins, 0.30), net_file, f_out,
            fringe_factor=1, seed_offset=1000, id_prefix="morn_out_",
            master_seed=master_seed)
        all_trip_files.append(f_out)

    # Evening rush: 30% inbound (fringe=15), 70% outbound (fringe=1)
    if evening_bins:
        print(f"  [3/5] Evening rush inbound  (30%, fringe=15)...")
        f_in = os.path.join(tmp_dir, f"fd_evening_in_{tag}.trips.xml")
        total_vehicles += _generate_trips_only(
            _volume_split(evening_bins, 0.30), net_file, f_in,
            fringe_factor=15, seed_offset=2000, id_prefix="eve_in_",
            master_seed=master_seed)
        all_trip_files.append(f_in)

        print(f"  [4/5] Evening rush outbound (70%, fringe=1)...")
        f_out = os.path.join(tmp_dir, f"fd_evening_out_{tag}.trips.xml")
        total_vehicles += _generate_trips_only(
            _volume_split(evening_bins, 0.70), net_file, f_out,
            fringe_factor=1, seed_offset=3000, id_prefix="eve_out_",
            master_seed=master_seed)
        all_trip_files.append(f_out)

    # Off-peak: 50/50 symmetric (fringe=5)
    if offpeak_bins:
        print(f"  [5/5] Off-peak symmetric    (50/50, fringe=5)...")
        f_off = os.path.join(tmp_dir, f"fd_offpeak_{tag}.trips.xml")
        total_vehicles += _generate_trips_only(
            offpeak_bins, net_file, f_off,
            fringe_factor=5, seed_offset=4000, id_prefix="off_",
            master_seed=master_seed)
        all_trip_files.append(f_off)

    # Merge all trip files sorted by departure time
    print(f"\n  Merging {len(all_trip_files)} trip files ({total_vehicles} vehicles)...")
    trips = []
    for tf in all_trip_files:
        if not os.path.exists(tf):
            continue
        with open(tf) as f:
            for line in f:
                line_s = line.strip()
                if line_s.startswith("<trip "):
                    trips.append(line_s)
        os.remove(tf)

    def _depart(trip_line):
        try:
            idx = trip_line.index('depart="') + 8
            end_idx = trip_line.index('"', idx)
            return float(trip_line[idx:end_idx])
        except ValueError:
            return 0.0

    trips.sort(key=_depart)
    merged_trips = os.path.join(tmp_dir, f"fd_merged_{tag}.trips.xml")
    with open(merged_trips, "w") as f:
        f.write("<trips>\n")
        for t in trips:
            f.write(f"    {t}\n")
        f.write("</trips>\n")

    _route_trips(merged_trips, net_file, output_trips, output_routes)
    os.remove(merged_trips)

    print(f"\n  Done: {total_vehicles} vehicles -> {output_routes}")
    return output_routes


def generate_scenario(scenario_name, net_file, output_dir, master_seed=42, output_suffix=""):
    """Generate route file for one rush scenario using the bimodal curve.

    Args:
        master_seed: Base seed for randomTrips.py. Different values produce
                     different OD pairs while keeping the same demand curve.
        output_suffix: If set, appended to output filename (e.g. "seed_00"
                       -> routes_morning_rush_seed_00.rou.xml)
    """
    if scenario_name == "full_day":
        return _generate_full_day_scenario(net_file, output_dir,
                                           master_seed=master_seed,
                                           output_suffix=output_suffix)

    cfg = SCENARIOS[scenario_name]
    os.makedirs(output_dir, exist_ok=True)

    base_routes = cfg["output"]
    base_trips = cfg["trips"]
    if output_suffix:
        base_routes = base_routes.replace(".rou.xml", f"_{output_suffix}.rou.xml")
        base_trips = base_trips.replace(".trips.xml", f"_{output_suffix}.trips.xml")
    output_routes = os.path.join(output_dir, base_routes)
    output_trips = os.path.join(output_dir, base_trips)

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

    # Unique tag for temp files (avoids collisions in parallel generation)
    tag = output_suffix if output_suffix else f"ms{master_seed}"

    if abs(inbound_frac - 0.5) < 0.01:
        # Symmetric: single batch, no need to merge
        total = write_demand_xml(
            full_bins,
            net_file,
            output_trips,
            output_routes,
            fringe_factor=cfg["fringe_factor_high"],
            master_seed=master_seed,
        )
    else:
        # Asymmetric: merge inbound (high-fringe) + outbound (low-fringe) batches
        inbound_bins = _volume_split(full_bins, inbound_frac)
        outbound_bins = _volume_split(full_bins, outbound_frac)

        tmp_dir = tempfile.gettempdir()
        trips_inbound = os.path.join(tmp_dir, f"trips_{scenario_name}_in_{tag}.trips.xml")
        trips_outbound = os.path.join(tmp_dir, f"trips_{scenario_name}_out_{tag}.trips.xml")
        trips_merged = os.path.join(tmp_dir, f"trips_{scenario_name}_merged_{tag}.trips.xml")

        print(f"\n  [1/3] Generating inbound trips  ({inbound_frac*100:.0f}% volume, "
              f"fringe={cfg['fringe_factor_high']})...")
        _generate_trips_only(
            inbound_bins, net_file, trips_inbound,
            fringe_factor=cfg["fringe_factor_high"],
            seed_offset=0,
            id_prefix="in",
            master_seed=master_seed,
        )

        print(f"\n  [2/3] Generating outbound trips ({outbound_frac*100:.0f}% volume, "
              f"fringe={cfg['fringe_factor_low']})...")
        _generate_trips_only(
            outbound_bins, net_file, trips_outbound,
            fringe_factor=cfg["fringe_factor_low"],
            seed_offset=1000,
            id_prefix="out",
            master_seed=master_seed,
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


def _generate_one_scenario_variant(args):
    """Worker function for parallel scenario variant generation."""
    scenario_name, net_file, output_dir, seed_idx, master_seed, num_variants = args
    suffix = f"seed_{seed_idx:02d}"
    print(f"\n{'='*60}")
    print(f"Generating {scenario_name} variant {seed_idx+1}/{num_variants} "
          f"(master_seed={master_seed}, worker pid={os.getpid()})")
    path = generate_scenario(
        scenario_name, net_file, output_dir,
        master_seed=master_seed,
        output_suffix=suffix,
    )
    return seed_idx, path


def generate_scenario_variants(scenario_name, net_file, output_dir, num_variants=50,
                               master_seed_stride=10000, num_workers=1):
    """Generate N route files with different random seeds for a single scenario.

    Each file gets master_seed = seed_index * master_seed_stride, producing
    completely different OD pairs while preserving the same demand curve.

    Args:
        scenario_name: One of "morning_rush", "evening_rush", "offpeak", "full_day"
        net_file: Path to SUMO .net.xml
        output_dir: Directory for output route files
        num_variants: Number of route files to generate
        master_seed_stride: Spacing between master seeds (default 10000)
        num_workers: Number of parallel workers (default 1 = sequential)

    Returns:
        List of generated route file paths (ordered by seed index).
    """
    if scenario_name == "full_day":
        return generate_statistical_routes(
            net_file, output_dir,
            num_seeds=num_variants,
            master_seed_stride=master_seed_stride,
            num_workers=num_workers,
        )

    os.makedirs(output_dir, exist_ok=True)
    generated = [None] * num_variants

    work_items = [
        (scenario_name, net_file, output_dir, seed_idx,
         seed_idx * master_seed_stride, num_variants)
        for seed_idx in range(num_variants)
    ]

    effective_workers = min(num_workers, num_variants)
    print(f"Generating {num_variants} {scenario_name} route variants "
          f"with {effective_workers} parallel workers...")

    if effective_workers <= 1:
        for item in work_items:
            seed_idx, path = _generate_one_scenario_variant(item)
            generated[seed_idx] = path
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_generate_one_scenario_variant, item): item[3]
                       for item in work_items}
            completed = 0
            for future in as_completed(futures):
                seed_idx, path = future.result()
                generated[seed_idx] = path
                completed += 1
                print(f"  [{completed}/{num_variants}] seed_{seed_idx:02d} done -> {path}")

    print(f"\n{'='*60}")
    print(f"Generated {len(generated)} {scenario_name} route files in {output_dir}/")
    return generated


def _generate_one_statistical_route(args):
    """Worker function for parallel route generation. Must be module-level for pickling."""
    net_file, output_dir, seed_idx, master_seed, num_seeds = args
    suffix = f"seed_{seed_idx:02d}"
    print(f"\n{'='*60}")
    print(f"Generating route file {seed_idx+1}/{num_seeds} "
          f"(master_seed={master_seed}, worker pid={os.getpid()})")
    path = _generate_full_day_scenario(
        net_file, output_dir,
        master_seed=master_seed,
        output_suffix=suffix,
    )
    return seed_idx, path


def generate_statistical_routes(net_file, output_dir, num_seeds=50,
                                master_seed_stride=10000, num_workers=1):
    """Generate N route files with different random seeds for statistical testing.

    Each file gets master_seed = seed_index * master_seed_stride, producing
    completely different OD pairs while preserving the same bimodal demand curve.

    Args:
        net_file: Path to SUMO .net.xml
        output_dir: Directory for output (e.g. data/routes/statistical-test/)
        num_seeds: Number of route files to generate (default 50)
        master_seed_stride: Spacing between master seeds (default 10000)
        num_workers: Number of parallel workers (default 1 = sequential)

    Returns:
        List of generated route file paths (ordered by seed index).
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = [None] * num_seeds

    work_items = [
        (net_file, output_dir, seed_idx, seed_idx * master_seed_stride, num_seeds)
        for seed_idx in range(num_seeds)
    ]

    effective_workers = min(num_workers, num_seeds)
    print(f"Generating {num_seeds} route files with {effective_workers} parallel workers...")

    if effective_workers <= 1:
        for item in work_items:
            seed_idx, path = _generate_one_statistical_route(item)
            generated[seed_idx] = path
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_generate_one_statistical_route, item): item[2]
                       for item in work_items}
            completed = 0
            for future in as_completed(futures):
                seed_idx, path = future.result()
                generated[seed_idx] = path
                completed += 1
                print(f"  [{completed}/{num_seeds}] seed_{seed_idx:02d} done -> {path}")

    print(f"\n{'='*60}")
    print(f"Generated {len(generated)} route files in {output_dir}/")
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate traffic demand (uniform or rush-hour scenarios)"
    )
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml")

    # Mode selection: --profile OR --scenario OR --statistical_routes (not both)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--profile", type=str, choices=["uniform"],
                      help="Demand temporal profile (uniform for smoke tests)")
    mode.add_argument("--scenario", type=str,
                      choices=["morning_rush", "evening_rush", "offpeak", "full_day", "all"],
                      help="Rush-hour scenario using bimodal 24h curve")
    mode.add_argument("--statistical_routes", type=int, metavar="N",
                      help="Generate N full-day route files with different seeds "
                           "for statistical testing (output to output_dir/)")

    # Variant generation (works with --scenario)
    parser.add_argument("--num_variants", type=int, default=None, metavar="N",
                        help="Generate N route files with different random seeds. "
                             "Requires --scenario. Each variant gets a different "
                             "master_seed producing different OD pairs.")

    # --profile mode arguments
    parser.add_argument("--output_trips", type=str,
                        default="data/routes/trips.trips.xml")
    parser.add_argument("--output_routes", type=str,
                        default="data/routes/routes.rou.xml")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Simulation duration in seconds (profile mode)")
    parser.add_argument("--peak_vph", type=int, default=800,
                        help="Peak vehicles per hour (profile mode)")
    parser.add_argument("--fringe_factor", type=float, default=5.0,
                        help="How much more likely trips start/end at edges")

    # --scenario mode arguments
    parser.add_argument("--output_dir", type=str, default="data/routes",
                        help="Directory for generated route files (scenario mode)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Parallel workers for --statistical_routes / "
                             "--num_variants (default: auto-detect CPU count)")

    args = parser.parse_args()

    # Auto-detect parallel workers from CPU count
    if args.num_workers is None:
        args.num_workers = os.cpu_count() or 4

    if args.statistical_routes:
        generate_statistical_routes(
            args.net_file, args.output_dir,
            num_seeds=args.statistical_routes,
            num_workers=args.num_workers,
        )
        print(f"\nNext steps:")
        print(f"  python src/run_24h.py --route_dir {args.output_dir} --baseline "
              f"--num_runs {args.statistical_routes} ...")
        return

    if args.scenario:
        # Rush-hour scenario mode
        targets = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

        if args.num_variants:
            # Generate N route file variants per scenario
            for scenario in targets:
                generate_scenario_variants(
                    scenario, args.net_file, args.output_dir,
                    num_variants=args.num_variants,
                    num_workers=args.num_workers,
                )
            print(f"\nAll done. Route variants are in {args.output_dir}/")
            print("Next steps:")
            for s in targets:
                print(f"  python src/run_rush_test.py --scenario {s} "
                      f"--route_dir {args.output_dir} --num_runs {args.num_variants}")
        else:
            for scenario in targets:
                generate_scenario(scenario, args.net_file, args.output_dir)

            print(f"\nAll done. Route files are in {args.output_dir}/")
            print("Next steps:")
            for s in targets:
                print(f"  python src/experiment.py --scenario {s}")
    else:
        # Profile mode (default to uniform if neither given)
        profile = args.profile or "uniform"

        print(f"Generating {profile} demand:")
        print(f"  Duration: {args.duration}s ({args.duration/3600:.1f}h)")
        print(f"  Peak rate: {args.peak_vph} veh/h")
        print(f"  Fringe factor: {args.fringe_factor}")

        bins = generate_demand_profile(args.duration, args.peak_vph)
        total = write_demand_xml(
            bins, args.net_file,
            args.output_trips, args.output_routes,
            args.fringe_factor
        )

        # Update sumocfg duration to match
        print(f"\n  NOTE: Update ljubljana.sumocfg <end> to {args.duration}")
        print(f"  Done! {total} vehicles generated.\n")


if __name__ == "__main__":
    main()
