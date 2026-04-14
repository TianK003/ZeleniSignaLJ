#!/usr/bin/env python3
"""
Generate SLURM scripts for isolated rush-hour statistical tests.

Tests each morning and evening model on 50 different route files to measure
generalization over random traffic patterns (without 24h cascading errors).

Two-phase pipeline per scenario:
  1. gen_routes_<scenario>.slurm — generate 50 per-seed route files
  2. rush_<model_tag>.slurm — run 50 replications per model + baseline

Usage:
    python hpc/statistical-test/generate_rush_jobs.py
    bash hpc/statistical-test/submit_rush.sh
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_RUNS = 50
NUM_WORKERS = 50

# Import model definitions from the mega-policy generator
sys.path.insert(0, SCRIPT_DIR)
from generate_mega_jobs import MORNING_MODELS, EVENING_MODELS

# Output directories
MORNING_ROUTE_DIR = "data/routes/statistical-morning-test"
EVENING_ROUTE_DIR = "data/routes/statistical-evening-test"
RESULTS_BASE = "results/rush-test"

# ── SLURM templates ──────────────────────────────────────────────────────

ROUTE_GEN_TEMPLATE = """\
#!/bin/bash
# Zeleni SignaLJ - Generate {num_routes} {scenario} route variants
#SBATCH --job-name=zs_gen_{scenario}
#SBATCH --output=logs/gen_{scenario}_%j.out
#SBATCH --error=logs/gen_{scenario}_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=64G

source hpc/common.sh

echo "Generating {num_routes} {scenario} route variants..."
echo "  Output: {route_dir}/"

# --num_workers auto-detects from cpus-per-task (64 -> all {num_routes} in parallel)
srun python src/generate_demand.py \\
    --scenario {scenario} \\
    --num_variants {num_routes} \\
    --output_dir {route_dir}

echo "Done. Route files in {route_dir}/"
"""

RUSH_TEST_TEMPLATE = """\
#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: {tag}
# {description}
#SBATCH --job-name=zs_rush_{tag}
#SBATCH --output=logs/rush_{tag}_%j.out
#SBATCH --error=logs/rush_{tag}_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: {tag}"
echo "  Scenario: {scenario}"
echo "  Model: {model_desc}"
echo "  Runs: {num_runs} | Workers: {num_workers}"

srun python src/run_rush_test.py \\
    {model_args}\\
    --scenario {scenario} \\
    --route_dir {route_dir} \\
    --num_runs {num_runs} \\
    --num_workers {num_workers} \\
    --output_dir {results_base}/{tag} \\
    --tag {tag}
"""


def generate_route_gen_jobs():
    """Generate SLURM scripts for route generation (phase 1)."""
    paths = []
    for scenario, route_dir in [
        ("morning_rush", MORNING_ROUTE_DIR),
        ("evening_rush", EVENING_ROUTE_DIR),
    ]:
        content = ROUTE_GEN_TEMPLATE.format(
            num_routes=NUM_RUNS,
            scenario=scenario,
            route_dir=route_dir,
        )
        path = os.path.join(SCRIPT_DIR, f"gen_routes_{scenario}.slurm")
        with open(path, "w") as f:
            f.write(content)
        paths.append(path)
    return paths


def generate_rush_test_jobs():
    """Generate SLURM scripts for rush-hour tests (phase 2)."""
    paths = []

    # Morning models on morning rush
    for m_key, m_info in MORNING_MODELS.items():
        tag = f"{m_key}_morning"
        content = RUSH_TEST_TEMPLATE.format(
            tag=tag,
            description=f"Morning model {m_key}: {m_info['desc']}",
            scenario="morning_rush",
            model_desc=m_info["desc"],
            model_args=f"--model {m_info['path']} \\\n    ",
            route_dir=MORNING_ROUTE_DIR,
            num_runs=NUM_RUNS,
            num_workers=NUM_WORKERS,
            results_base=RESULTS_BASE,
        )
        path = os.path.join(SCRIPT_DIR, f"rush_{tag}.slurm")
        with open(path, "w") as f:
            f.write(content)
        paths.append(path)

    # Evening models on evening rush
    for e_key, e_info in EVENING_MODELS.items():
        tag = f"{e_key}_evening"
        content = RUSH_TEST_TEMPLATE.format(
            tag=tag,
            description=f"Evening model {e_key}: {e_info['desc']}",
            scenario="evening_rush",
            model_desc=e_info["desc"],
            model_args=f"--model {e_info['path']} \\\n    ",
            route_dir=EVENING_ROUTE_DIR,
            num_runs=NUM_RUNS,
            num_workers=NUM_WORKERS,
            results_base=RESULTS_BASE,
        )
        path = os.path.join(SCRIPT_DIR, f"rush_{tag}.slurm")
        with open(path, "w") as f:
            f.write(content)
        paths.append(path)

    # Baselines (one per scenario)
    for scenario, route_dir in [
        ("morning_rush", MORNING_ROUTE_DIR),
        ("evening_rush", EVENING_ROUTE_DIR),
    ]:
        tag = f"baseline_{scenario.split('_')[0]}"
        content = RUSH_TEST_TEMPLATE.format(
            tag=tag,
            description=f"Baseline: fixed-time programs ({scenario})",
            scenario=scenario,
            model_desc="none (fixed-time baseline)",
            model_args="--baseline \\\n    ",
            route_dir=route_dir,
            num_runs=NUM_RUNS,
            num_workers=NUM_WORKERS,
            results_base=RESULTS_BASE,
        )
        path = os.path.join(SCRIPT_DIR, f"rush_{tag}.slurm")
        with open(path, "w") as f:
            f.write(content)
        paths.append(path)

    return paths


def generate_submit_script(route_jobs, test_jobs):
    """Generate submit_rush.sh that handles the dependency chain."""
    content = """\
#!/bin/bash
# Submit all rush-hour generalization tests with dependency chain.
# Phase 1: Generate route files (must complete first)
# Phase 2: Run all model tests + baselines
#
# Usage:
#   bash hpc/statistical-test/submit_rush.sh
#   bash hpc/statistical-test/submit_rush.sh --skip-routes

set -e
cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

SKIP_ROUTES=false
if [ "$1" = "--skip-routes" ]; then
    SKIP_ROUTES=true
    echo "Skipping route generation (assuming routes already exist)"
fi

ROUTE_JIDS=""

if [ "$SKIP_ROUTES" = false ]; then
    echo "Phase 1: Submitting route generation jobs..."
"""
    for rj in route_jobs:
        name = os.path.basename(rj)
        content += f'    JID=$(sbatch --parsable hpc/statistical-test/{name})\n'
        content += f'    echo "  Submitted {name} (job $JID)"\n'
        content += f'    ROUTE_JIDS="$ROUTE_JIDS:$JID"\n'

    content += """
    echo "Waiting for route generation to complete before submitting tests..."
    DEP_FLAG="--dependency=afterok${ROUTE_JIDS}"
else
    DEP_FLAG=""
fi

echo ""
echo "Phase 2: Submitting rush-hour test jobs..."
"""
    for tj in test_jobs:
        name = os.path.basename(tj)
        content += f'JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/{name})\n'
        content += f'echo "  Submitted {name} (job $JID)"\n'

    content += """
echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
echo "Results will be in: results/rush-test/"
"""

    path = os.path.join(SCRIPT_DIR, "submit_rush.sh")
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)
    return path


def main():
    print("Generating rush-hour generalization test SLURM scripts...\n")

    route_jobs = generate_route_gen_jobs()
    for p in route_jobs:
        print(f"  Route gen:  {os.path.basename(p)}")

    test_jobs = generate_rush_test_jobs()
    for p in test_jobs:
        print(f"  Test job:   {os.path.basename(p)}")

    submit_path = generate_submit_script(route_jobs, test_jobs)
    print(f"\n  Submit script: {os.path.basename(submit_path)}")

    print(f"\nTotal: {len(route_jobs)} route gen + {len(test_jobs)} test scripts")
    print(f"\nPipeline:")
    print(f"  bash hpc/statistical-test/submit_rush.sh")
    print(f"  # Or skip route gen if routes exist:")
    print(f"  bash hpc/statistical-test/submit_rush.sh --skip-routes")


if __name__ == "__main__":
    main()
