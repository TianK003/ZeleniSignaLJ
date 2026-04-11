#!/bin/bash
# Submit all mega-policy statistical test jobs to SLURM
# Two-phase pipeline: generate routes -> run simulations
#
# Usage:
#   bash hpc/statistical-test/submit_all.sh           # full pipeline
#   bash hpc/statistical-test/submit_all.sh --skip-routes  # skip route gen
#   bash hpc/statistical-test/submit_all.sh --skip-routes M1  # filter + skip

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_ROUTES=false

if [ "$1" == "--skip-routes" ]; then
    SKIP_ROUTES=true
    shift
fi

FILTER="${1:-}"

# Phase 1: Generate route files
DEPENDENCY=""
if [ "$SKIP_ROUTES" = false ]; then
    ROUTE_SCRIPT="$SCRIPT_DIR/gen_routes.slurm"
    if [ ! -f "$ROUTE_SCRIPT" ]; then
        echo "ERROR: $ROUTE_SCRIPT not found. Run: python hpc/statistical-test/generate_mega_jobs.py"
        exit 1
    fi
    echo "Phase 1: Submitting route generation job..."
    ROUTE_JOB=$(sbatch --parsable "$ROUTE_SCRIPT")
    echo "  Route generation job ID: $ROUTE_JOB"
    DEPENDENCY="--dependency=afterok:${ROUTE_JOB}"
    echo ""
else
    echo "Skipping route generation (--skip-routes)"
    echo ""
fi

# Phase 2: Submit simulation jobs
echo "Phase 2: Submitting simulation jobs..."
COUNT=0
for f in "$SCRIPT_DIR"/mega_*.slurm; do
    [ -f "$f" ] || continue
    name=$(basename "$f")

    # Apply optional filter
    if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then
        continue
    fi

    echo "  sbatch $DEPENDENCY $f"
    sbatch $DEPENDENCY "$f"
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted $COUNT simulation jobs."
if [ -n "$DEPENDENCY" ]; then
    echo "All simulation jobs depend on route generation job $ROUTE_JOB."
fi
echo "Monitor with: squeue -u \$USER"
