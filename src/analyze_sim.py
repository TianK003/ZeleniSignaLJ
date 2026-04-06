"""
Zeleni SignaLJ - Simulation Analysis
=====================================
Sanity-check a SUMO run: teleport count, edge flows, trip stats.

Usage:
    # Run simulation with stats output first:
    sumo -c data/networks/ljubljana.sumocfg

    # Then analyze:
    python src/analyze_sim.py
    python src/analyze_sim.py --tripinfo results/tripinfo.xml --stats results/sim_stats.xml
"""

import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import os


def parse_stats(stats_file):
    """Parse SUMO statistic-output XML for key KPIs."""
    tree = ET.parse(stats_file)
    root = tree.getroot()

    # Vehicle stats
    vehicles = root.find(".//vehicles")
    teleports = root.find(".//teleports")

    v_loaded = int(vehicles.get("loaded", 0))
    v_inserted = int(vehicles.get("inserted", 0))
    v_running = int(vehicles.get("running", 0))
    v_waiting = int(vehicles.get("waiting", 0))

    tp_total = int(teleports.get("total", 0))
    tp_jam = int(teleports.get("jam", 0))
    tp_yield = int(teleports.get("yield", 0))
    tp_wrong = int(teleports.get("wrongLane", 0))

    teleport_pct = (tp_total / v_inserted * 100) if v_inserted > 0 else 0

    print("=" * 60)
    print("SIMULATION STATISTICS")
    print("=" * 60)
    print(f"  Vehicles loaded:     {v_loaded}")
    print(f"  Vehicles inserted:   {v_inserted}")
    print(f"  Vehicles running:    {v_running} (still in network at end)")
    print(f"  Vehicles waiting:    {v_waiting} (couldn't depart)")
    print()
    print(f"  Teleports total:     {tp_total}  ({teleport_pct:.1f}% of inserted)")
    print(f"    - jam:             {tp_jam}")
    print(f"    - yield:           {tp_yield}")
    print(f"    - wrong lane:      {tp_wrong}")
    print()

    # Sanity check thresholds
    if teleport_pct > 10:
        print("  ⚠ WARNING: >10% teleport rate! Network may have capacity issues.")
        print("    Possible fixes: lower demand (-p value), fix lane connections,")
        print("    increase time-to-teleport, or check for deadlocks in netedit.")
    elif teleport_pct > 5:
        print("  ⚠ CAUTION: 5-10% teleport rate. Acceptable for early testing,")
        print("    but should improve with calibrated demand.")
    else:
        print("  ✓ Teleport rate looks healthy (<5%).")

    return {
        "loaded": v_loaded, "inserted": v_inserted,
        "running": v_running, "waiting": v_waiting,
        "teleports": tp_total, "teleport_pct": teleport_pct,
    }


def parse_tripinfo(tripinfo_file):
    """Parse tripinfo XML into a DataFrame with per-vehicle KPIs."""
    tree = ET.parse(tripinfo_file)
    root = tree.getroot()

    trips = []
    for trip in root.findall("tripinfo"):
        trips.append({
            "id": trip.get("id"),
            "depart": float(trip.get("depart", 0)),
            "arrival": float(trip.get("arrival", -1)),
            "duration": float(trip.get("duration", 0)),
            "routeLength": float(trip.get("routeLength", 0)),
            "waitingTime": float(trip.get("waitingTime", 0)),
            "waitingCount": int(trip.get("waitingCount", 0)),
            "timeLoss": float(trip.get("timeLoss", 0)),
            "departDelay": float(trip.get("departDelay", 0)),
            "vType": trip.get("vType", ""),
        })

    df = pd.DataFrame(trips)

    if len(df) == 0:
        print("  No completed trips found.")
        return df

    print("-" * 60)
    print("TRIP STATISTICS (completed vehicles)")
    print("-" * 60)
    print(f"  Completed trips:       {len(df)}")
    print(f"  Avg duration:          {df['duration'].mean():.1f}s")
    print(f"  Avg waiting time:      {df['waitingTime'].mean():.1f}s")
    print(f"  Avg time loss:         {df['timeLoss'].mean():.1f}s")
    print(f"  Avg depart delay:      {df['departDelay'].mean():.1f}s")
    print(f"  Max waiting time:      {df['waitingTime'].max():.1f}s")
    print(f"  Avg route length:      {df['routeLength'].mean():.0f}m")
    print()

    return df


def parse_edge_data(edge_file):
    """Parse edge-based output to find busiest/quietest roads."""
    tree = ET.parse(edge_file)
    root = tree.getroot()

    edges = []
    for interval in root.findall("interval"):
        for edge in interval.findall("edge"):
            edges.append({
                "begin": float(interval.get("begin", 0)),
                "end": float(interval.get("end", 0)),
                "edge_id": edge.get("id"),
                "entered": int(float(edge.get("entered", 0))),
                "left": int(float(edge.get("left", 0))),
                "density": float(edge.get("density", 0)),
                "occupancy": float(edge.get("occupancy", 0)),
                "waitingTime": float(edge.get("waitingTime", 0)),
                "speed": float(edge.get("speed", 0)),
                "traveltime": float(edge.get("traveltime", 0)),
            })

    df = pd.DataFrame(edges)

    if len(df) == 0:
        print("  No edge data found.")
        return df

    # Aggregate across all time intervals
    agg = df.groupby("edge_id").agg({
        "entered": "sum",
        "waitingTime": "sum",
        "speed": "mean",
        "density": "mean",
    }).sort_values("entered", ascending=False)

    # Filter out internal edges (start with ':')
    agg = agg[~agg.index.str.startswith(":")]

    print("-" * 60)
    print("EDGE FLOW ANALYSIS (non-internal edges)")
    print("-" * 60)
    print(f"  Total edges with traffic: {len(agg[agg['entered'] > 0])}")
    print(f"  Edges with zero flow:     {len(agg[agg['entered'] == 0])}")
    print()

    print("  TOP 15 BUSIEST EDGES:")
    for i, (edge_id, row) in enumerate(agg.head(15).iterrows(), 1):
        print(f"    {i:2d}. {edge_id[:50]:<50s} "
              f"vehicles={row['entered']:>5.0f}  "
              f"avg_speed={row['speed']:.1f}m/s")
    print()

    print("  TOP 10 MOST CONGESTED (by waiting time):")
    congested = agg[agg["entered"] > 10].sort_values("waitingTime", ascending=False)
    for i, (edge_id, row) in enumerate(congested.head(10).iterrows(), 1):
        print(f"    {i:2d}. {edge_id[:50]:<50s} "
              f"wait={row['waitingTime']:>8.0f}s  "
              f"vehicles={row['entered']:>5.0f}")
    print()

    # Save full edge data for later use
    agg.to_csv("results/edge_flows.csv")
    print("  Full edge data saved to results/edge_flows.csv")

    return agg


def main():
    parser = argparse.ArgumentParser(description="Analyze SUMO simulation output")
    parser.add_argument("--stats", type=str, default="results/sim_stats.xml")
    parser.add_argument("--tripinfo", type=str, default="results/tripinfo.xml")
    parser.add_argument("--edges", type=str, default="results/edge_data.xml")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if os.path.exists(args.stats):
        parse_stats(args.stats)
    else:
        print(f"  Stats file not found: {args.stats}")

    if os.path.exists(args.tripinfo):
        df_trips = parse_tripinfo(args.tripinfo)
        if len(df_trips) > 0:
            df_trips.to_csv("results/tripinfo.csv", index=False)
            print("  Trip data saved to results/tripinfo.csv")
    else:
        print(f"  Tripinfo file not found: {args.tripinfo}")

    if os.path.exists(args.edges):
        parse_edge_data(args.edges)
    else:
        print(f"  Edge data file not found: {args.edges}")


if __name__ == "__main__":
    main()
