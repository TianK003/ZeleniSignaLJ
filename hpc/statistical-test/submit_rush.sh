#!/bin/bash
# Submit all rush-hour generalization tests with dependency chain.
# Phase 1: Generate route files (morning + evening in parallel)
# Phase 2: Run model tests + baselines (each depends only on its own route gen)
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

MORNING_DEP=""
EVENING_DEP=""

if [ "$SKIP_ROUTES" = false ]; then
    echo "Phase 1: Submitting route generation jobs..."

    MORNING_ROUTE_JID=$(sbatch --parsable hpc/statistical-test/gen_routes_morning_rush.slurm)
    echo "  Submitted gen_routes_morning_rush.slurm (job $MORNING_ROUTE_JID)"

    EVENING_ROUTE_JID=$(sbatch --parsable hpc/statistical-test/gen_routes_evening_rush.slurm)
    echo "  Submitted gen_routes_evening_rush.slurm (job $EVENING_ROUTE_JID)"

    MORNING_DEP="--dependency=afterok:${MORNING_ROUTE_JID}"
    EVENING_DEP="--dependency=afterok:${EVENING_ROUTE_JID}"

    echo ""
    echo "  Morning tests will start after job $MORNING_ROUTE_JID"
    echo "  Evening tests will start after job $EVENING_ROUTE_JID"
fi

echo ""
echo "Phase 2: Submitting rush-hour test jobs..."

# Morning models + baseline (depend on morning route gen only)
for SCRIPT in rush_M1_morning rush_M2_morning rush_M3_morning rush_baseline_morning; do
    JID=$(sbatch --parsable $MORNING_DEP hpc/statistical-test/${SCRIPT}.slurm)
    echo "  Submitted ${SCRIPT}.slurm (job $JID)"
done

# Evening models + baseline (depend on evening route gen only)
for SCRIPT in rush_E1_evening rush_E2_evening rush_E3_evening rush_baseline_evening; do
    JID=$(sbatch --parsable $EVENING_DEP hpc/statistical-test/${SCRIPT}.slurm)
    echo "  Submitted ${SCRIPT}.slurm (job $JID)"
done

echo ""
echo "All 10 jobs submitted. Monitor with: squeue -u $USER"
echo "Results will be in: results/rush-test/"
