"""
Zeleni SignaLJ - Traffic Demand Generator
==========================================
Generate realistic traffic demand with different temporal profiles.

Profiles:
  uniform   - Constant rate (for smoke tests / debugging)
  rush_hour - Gaussian peak simulating morning or evening rush
  double    - Two peaks (morning + evening, full day simulation)

Usage:
    # Quick uniform demand (smoke test)
    python src/generate_demand.py --profile uniform --duration 3600 --peak_vph 800

    # Morning rush hour (4 hours, peak at 1h mark)
    python src/generate_demand.py --profile rush_hour --duration 14400 --peak_vph 1800

    # Full day with morning + evening peaks
    python src/generate_demand.py --profile double --duration 28800 --peak_vph 1800
"""

import argparse
import subprocess
import os
import sys
import numpy as np


def gaussian_rate(t, peak_time, sigma, peak_rate, base_rate):
    """Gaussian-shaped vehicle spawn rate."""
    return base_rate + (peak_rate - base_rate) * np.exp(
        -0.5 * ((t - peak_time) / sigma) ** 2
    )


def generate_demand_profile(duration, peak_vph, profile, interval=300):
    """
    Generate a list of (begin, end, rate) tuples.
    rate = vehicles per second for that interval.
    interval = seconds per time bin (default 5 min).
    """
    bins = []
    peak_vps = peak_vph / 3600  # vehicles per second at peak

    if profile == "uniform":
        # Constant rate throughout
        for t in range(0, duration, interval):
            end = min(t + interval, duration)
            bins.append((t, end, peak_vps))

    elif profile == "rush_hour":
        # Single Gaussian peak at 1/3 of duration
        # e.g. 4h sim → peak at ~80min, steep ramp up/down
        peak_time = duration * 0.33
        sigma = duration * 0.12  # Steep: ~25% of duration is the "rush"
        base_rate = peak_vps * 0.15  # Off-peak is 15% of peak

        for t in range(0, duration, interval):
            mid = t + interval / 2
            rate = gaussian_rate(mid, peak_time, sigma, peak_vps, base_rate)
            end = min(t + interval, duration)
            bins.append((t, end, rate))

    elif profile == "double":
        # Morning peak at 25%, evening peak at 70% of duration
        peak1_time = duration * 0.25
        peak2_time = duration * 0.70
        sigma = duration * 0.08
        base_rate = peak_vps * 0.12

        for t in range(0, duration, interval):
            mid = t + interval / 2
            r1 = gaussian_rate(mid, peak1_time, sigma, peak_vps, base_rate)
            r2 = gaussian_rate(mid, peak2_time, sigma, peak_vps * 0.85, base_rate)
            rate = max(r1, r2)  # Overlap handled by taking max
            end = min(t + interval, duration)
            bins.append((t, end, rate))

    else:
        raise ValueError(f"Unknown profile: {profile}")

    return bins


def write_demand_xml(bins, net_file, output_trips, output_routes, fringe_factor=5):
    """
    Write trips XML with time-varying demand using randomTrips.py per bin,
    then merge and route.
    """
    sumo_home = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")

    if not os.path.exists(random_trips):
        print(f"ERROR: randomTrips.py not found at {random_trips}")
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

        tmp_trips = f"/tmp/trips_bin_{i}.trips.xml"
        all_trips_files.append(tmp_trips)

        cmd = [
            sys.executable, random_trips,
            "-n", net_file,
            "-o", tmp_trips,
            "-b", str(int(begin)),
            "-e", str(int(end)),
            "-p", str(max(period, 0.5)),
            "--fringe-factor", str(fringe_factor),
            "--seed", str(42 + i),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: randomTrips failed for bin {i}: {result.stderr[:200]}")

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


def main():
    parser = argparse.ArgumentParser(description="Generate traffic demand")
    parser.add_argument("--net_file", type=str,
                        default="data/networks/ljubljana.net.xml")
    parser.add_argument("--output_trips", type=str,
                        default="data/routes/trips.trips.xml")
    parser.add_argument("--output_routes", type=str,
                        default="data/routes/routes.rou.xml")
    parser.add_argument("--profile", type=str, default="uniform",
                        choices=["uniform", "rush_hour", "double"],
                        help="Demand temporal profile")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Simulation duration in seconds")
    parser.add_argument("--peak_vph", type=int, default=800,
                        help="Peak vehicles per hour")
    parser.add_argument("--fringe_factor", type=float, default=5.0,
                        help="How much more likely trips start/end at edges")
    args = parser.parse_args()

    print(f"Generating {args.profile} demand:")
    print(f"  Duration: {args.duration}s ({args.duration/3600:.1f}h)")
    print(f"  Peak rate: {args.peak_vph} veh/h")
    print(f"  Fringe factor: {args.fringe_factor}")

    bins = generate_demand_profile(
        args.duration, args.peak_vph, args.profile
    )
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
