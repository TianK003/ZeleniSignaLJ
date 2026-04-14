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
    JID=$(sbatch --parsable hpc/statistical-test/gen_routes_morning_rush.slurm)
    echo "  Submitted gen_routes_morning_rush.slurm (job $JID)"
    ROUTE_JIDS="$ROUTE_JIDS:$JID"
    JID=$(sbatch --parsable hpc/statistical-test/gen_routes_evening_rush.slurm)
    echo "  Submitted gen_routes_evening_rush.slurm (job $JID)"
    ROUTE_JIDS="$ROUTE_JIDS:$JID"

    echo "Waiting for route generation to complete before submitting tests..."
    DEP_FLAG="--dependency=afterok${ROUTE_JIDS}"
else
    DEP_FLAG=""
fi

echo ""
echo "Phase 2: Submitting rush-hour test jobs..."
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_M1_morning.slurm)
echo "  Submitted rush_M1_morning.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_M2_morning.slurm)
echo "  Submitted rush_M2_morning.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_M3_morning.slurm)
echo "  Submitted rush_M3_morning.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_M4_morning.slurm)
echo "  Submitted rush_M4_morning.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_M5_morning.slurm)
echo "  Submitted rush_M5_morning.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_E1_evening.slurm)
echo "  Submitted rush_E1_evening.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_E2_evening.slurm)
echo "  Submitted rush_E2_evening.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_E3_evening.slurm)
echo "  Submitted rush_E3_evening.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_E4_evening.slurm)
echo "  Submitted rush_E4_evening.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_E5_evening.slurm)
echo "  Submitted rush_E5_evening.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_baseline_morning.slurm)
echo "  Submitted rush_baseline_morning.slurm (job $JID)"
JID=$(sbatch --parsable $DEP_FLAG hpc/statistical-test/rush_baseline_evening.slurm)
echo "  Submitted rush_baseline_evening.slurm (job $JID)"

echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
echo "Results will be in: results/rush-test/"
