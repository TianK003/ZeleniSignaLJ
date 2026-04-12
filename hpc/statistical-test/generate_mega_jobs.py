#!/usr/bin/env python3
"""
Generate SLURM scripts for 9 mega-policy + 1 baseline statistical tests.

Two-phase pipeline:
  1. gen_routes.slurm — generate 50 per-seed route files
  2. mega_*.slurm — run 50 replications per condition using per-seed routes

Usage:
    python hpc/statistical-test/generate_mega_jobs.py
    bash hpc/statistical-test/submit_all.sh
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_RUNS = 50
NUM_WORKERS = 50
ROUTE_DIR = "data/routes/statistical-test"

# ── Top 3 models per scenario ─────────────────────────────────────────────

MORNING_MODELS = {
    "M1": {
        "path": "results/experiments/20260408_235440_morningrush_diffwaitingtime_lr1e3_200ep/ppo_shared_policy.zip",
        "desc": "diff-waiting-time, lr=1e-3 (+18.2%)",
    },
    "M2": {
        "path": "results/experiments/20260409_000353_morningrush_pressure_lr1e3_200ep/ppo_shared_policy.zip",
        "desc": "pressure, lr=1e-3 (+17.2%)",
    },
    "M3": {
        "path": "results/experiments/20260408_235311_morningrush_default_lr3e4_200ep/ppo_shared_policy.zip",
        "desc": "queue (default), lr=3e-4 (+17.1%)",
    },
}

EVENING_MODELS = {
    "E1": {
        "path": "results/experiments/20260408_235311_eveningrush_pressure_lr1e3_entanneal_200ep/ppo_shared_policy.zip",
        "desc": "pressure, lr=1e-3, entropy annealing (+15.1%)",
    },
    "E2": {
        "path": "results/experiments/20260408_235311_eveningrush_diffwaitingtime_lr1e3_entanneal_200ep/ppo_shared_policy.zip",
        "desc": "diff-waiting-time, lr=1e-3, entropy annealing (+15.0%)",
    },
    "E3": {
        "path": "results/experiments/20260408_235311_eveningrush_pressure_lr3e4_entanneal_200ep/ppo_shared_policy.zip",
        "desc": "pressure, lr=3e-4, entropy annealing (+12.9%)",
    },
}

# ── SLURM templates ──────────────────────────────────────────────────────

ROUTE_GEN_TEMPLATE = """\
#!/bin/bash
# Zeleni SignaLJ - Generate {num_routes} route files for statistical testing
#SBATCH --job-name=zs_gen_routes
#SBATCH --output=logs/gen_routes_%j.out
#SBATCH --error=logs/gen_routes_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

source hpc/common.sh

echo "Generating {num_routes} route files for statistical testing..."
echo "  Output: {route_dir}/"

# --num_workers auto-detects from cpus-per-task (64 -> all {num_routes} in parallel)
srun python src/generate_demand.py \\
    --statistical_routes {num_routes} \\
    --output_dir {route_dir}

echo "Done. Route files in {route_dir}/"
"""

SLURM_TEMPLATE = """\
#!/bin/bash
# Zeleni SignaLJ - 24h Mega-Policy Statistical Test: {tag}
# {description}
#SBATCH --job-name=zs_mega_{tag}
#SBATCH --output=logs/mega_{tag}_%j.out
#SBATCH --error=logs/mega_{tag}_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting 24h statistical test: {tag}"
echo "  Morning model: {morning_desc}"
echo "  Evening model: {evening_desc}"
echo "  Runs: {num_runs} | Workers: {num_workers} | Per-seed routes: yes"

srun python src/run_24h.py \\
    {model_args}\\
    --route_dir {route_dir} \\
    --num_runs {num_runs} \\
    --num_workers {num_workers} \\
    --output_dir results/statistical-test/{tag} \\
    --tag {tag}
"""


def generate_route_gen_job():
    """Generate a SLURM script for route file generation (phase 1)."""
    content = ROUTE_GEN_TEMPLATE.format(
        num_routes=NUM_RUNS,
        route_dir=ROUTE_DIR,
    )
    path = os.path.join(SCRIPT_DIR, "gen_routes.slurm")
    with open(path, "w") as f:
        f.write(content)
    return path


def generate_mega_job(m_key, e_key):
    """Generate a SLURM script for one mega-policy combination."""
    tag = f"{m_key}{e_key}"
    m = MORNING_MODELS[m_key]
    e = EVENING_MODELS[e_key]

    model_args = (
        f"--model_morning {m['path']} \\\n"
        f"    --model_evening {e['path']} \\\n    "
    )

    content = SLURM_TEMPLATE.format(
        tag=tag,
        description=f"Morning: {m['desc']} | Evening: {e['desc']}",
        morning_desc=m["desc"],
        evening_desc=e["desc"],
        model_args=model_args,
        route_dir=ROUTE_DIR,
        num_runs=NUM_RUNS,
        num_workers=NUM_WORKERS,
    )

    path = os.path.join(SCRIPT_DIR, f"mega_{tag}.slurm")
    with open(path, "w") as f:
        f.write(content)
    return path


def generate_baseline_job():
    """Generate a SLURM script for the baseline (all fixed-time)."""
    tag = "baseline"
    model_args = "--baseline \\\n    "

    content = SLURM_TEMPLATE.format(
        tag=tag,
        description="Baseline: all traffic signals on fixed-time programs (no RL)",
        morning_desc="none (fixed-time)",
        evening_desc="none (fixed-time)",
        model_args=model_args,
        route_dir=ROUTE_DIR,
        num_runs=NUM_RUNS,
        num_workers=NUM_WORKERS,
    )

    path = os.path.join(SCRIPT_DIR, f"mega_{tag}.slurm")
    with open(path, "w") as f:
        f.write(content)
    return path


def main():
    generated = []

    # Phase 1: Route generation job (must run first)
    path = generate_route_gen_job()
    generated.append(path)
    print(f"  Generated: {os.path.basename(path)} (run this FIRST)")

    # Phase 2: 9 mega-policy combinations + 1 baseline
    for m_key in MORNING_MODELS:
        for e_key in EVENING_MODELS:
            path = generate_mega_job(m_key, e_key)
            generated.append(path)
            print(f"  Generated: {os.path.basename(path)}")

    path = generate_baseline_job()
    generated.append(path)
    print(f"  Generated: {os.path.basename(path)}")

    print(f"\nTotal: {len(generated)} SLURM scripts in {SCRIPT_DIR}/")
    print(f"\nPipeline:")
    print(f"  1. sbatch hpc/statistical-test/gen_routes.slurm")
    print(f"     (wait for completion — generates {NUM_RUNS} route files)")
    print(f"  2. bash hpc/statistical-test/submit_all.sh")
    print(f"     (submits all simulation jobs with route dependency)")
    print(f"\nOr submit everything at once:")
    print(f"  bash hpc/statistical-test/submit_all.sh")


if __name__ == "__main__":
    main()
