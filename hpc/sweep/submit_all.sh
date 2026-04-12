#!/bin/bash
# Submit route generation + training jobs with dependency chain
set -e
cd "$(git rev-parse --show-toplevel)"
mkdir -p logs

SKIP_ROUTES=false
FILTER="${1:-}"
if [ "$FILTER" = "--skip-routes" ]; then
    SKIP_ROUTES=true; FILTER="${2:-}"; echo "Skipping route generation"
fi

DEP_FLAG=""
if [ "$SKIP_ROUTES" = false ]; then
    echo "Phase 1: Submitting route generation..."
    ROUTE_JID=$(sbatch --parsable hpc/sweep/gen_train_routes.slurm)
    echo "  Submitted gen_train_routes.slurm (job $ROUTE_JID)"
    DEP_FLAG="--dependency=afterok:${ROUTE_JID}"
fi

echo "Phase 2: Submitting training jobs..."
COUNT=0
if [ -z "$FILTER" ] || echo "eveningrush_diffwaitingtime_lr1e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_diffwaitingtime_lr1e3.slurm)
    echo "  Submitted eveningrush_diffwaitingtime_lr1e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_diffwaitingtime_lr1e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_diffwaitingtime_lr1e3_entanneal.slurm)
    echo "  Submitted eveningrush_diffwaitingtime_lr1e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_diffwaitingtime_lr3e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_diffwaitingtime_lr3e3.slurm)
    echo "  Submitted eveningrush_diffwaitingtime_lr3e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_diffwaitingtime_lr3e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_diffwaitingtime_lr3e3_entanneal.slurm)
    echo "  Submitted eveningrush_diffwaitingtime_lr3e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_diffwaitingtime_lr3e4.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_diffwaitingtime_lr3e4.slurm)
    echo "  Submitted eveningrush_diffwaitingtime_lr3e4.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_diffwaitingtime_lr3e4_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_diffwaitingtime_lr3e4_entanneal.slurm)
    echo "  Submitted eveningrush_diffwaitingtime_lr3e4_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_pressure_lr1e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_pressure_lr1e3.slurm)
    echo "  Submitted eveningrush_pressure_lr1e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_pressure_lr1e3_ent002.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_pressure_lr1e3_ent002.slurm)
    echo "  Submitted eveningrush_pressure_lr1e3_ent002.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_pressure_lr1e3_ent01.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_pressure_lr1e3_ent01.slurm)
    echo "  Submitted eveningrush_pressure_lr1e3_ent01.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_pressure_lr1e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_pressure_lr1e3_entanneal.slurm)
    echo "  Submitted eveningrush_pressure_lr1e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_pressure_lr3e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_pressure_lr3e3.slurm)
    echo "  Submitted eveningrush_pressure_lr3e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_pressure_lr3e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_pressure_lr3e3_entanneal.slurm)
    echo "  Submitted eveningrush_pressure_lr3e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_pressure_lr3e4.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_pressure_lr3e4.slurm)
    echo "  Submitted eveningrush_pressure_lr3e4.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_pressure_lr3e4_ent002.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_pressure_lr3e4_ent002.slurm)
    echo "  Submitted eveningrush_pressure_lr3e4_ent002.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_pressure_lr3e4_ent01.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_pressure_lr3e4_ent01.slurm)
    echo "  Submitted eveningrush_pressure_lr3e4_ent01.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_pressure_lr3e4_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_pressure_lr3e4_entanneal.slurm)
    echo "  Submitted eveningrush_pressure_lr3e4_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_queue_lr1e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_queue_lr1e3.slurm)
    echo "  Submitted eveningrush_queue_lr1e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_queue_lr1e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_queue_lr1e3_entanneal.slurm)
    echo "  Submitted eveningrush_queue_lr1e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_queue_lr3e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_queue_lr3e3.slurm)
    echo "  Submitted eveningrush_queue_lr3e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_queue_lr3e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_queue_lr3e3_entanneal.slurm)
    echo "  Submitted eveningrush_queue_lr3e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_queue_lr3e4.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_queue_lr3e4.slurm)
    echo "  Submitted eveningrush_queue_lr3e4.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "eveningrush_queue_lr3e4_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/eveningrush_queue_lr3e4_entanneal.slurm)
    echo "  Submitted eveningrush_queue_lr3e4_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_diffwaitingtime_lr1e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_diffwaitingtime_lr1e3.slurm)
    echo "  Submitted morningrush_diffwaitingtime_lr1e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_diffwaitingtime_lr1e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_diffwaitingtime_lr1e3_entanneal.slurm)
    echo "  Submitted morningrush_diffwaitingtime_lr1e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_diffwaitingtime_lr3e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_diffwaitingtime_lr3e3.slurm)
    echo "  Submitted morningrush_diffwaitingtime_lr3e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_diffwaitingtime_lr3e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_diffwaitingtime_lr3e3_entanneal.slurm)
    echo "  Submitted morningrush_diffwaitingtime_lr3e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_diffwaitingtime_lr3e4.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_diffwaitingtime_lr3e4.slurm)
    echo "  Submitted morningrush_diffwaitingtime_lr3e4.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_diffwaitingtime_lr3e4_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_diffwaitingtime_lr3e4_entanneal.slurm)
    echo "  Submitted morningrush_diffwaitingtime_lr3e4_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_pressure_lr1e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_pressure_lr1e3.slurm)
    echo "  Submitted morningrush_pressure_lr1e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_pressure_lr1e3_ent002.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_pressure_lr1e3_ent002.slurm)
    echo "  Submitted morningrush_pressure_lr1e3_ent002.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_pressure_lr1e3_ent01.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_pressure_lr1e3_ent01.slurm)
    echo "  Submitted morningrush_pressure_lr1e3_ent01.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_pressure_lr1e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_pressure_lr1e3_entanneal.slurm)
    echo "  Submitted morningrush_pressure_lr1e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_pressure_lr3e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_pressure_lr3e3.slurm)
    echo "  Submitted morningrush_pressure_lr3e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_pressure_lr3e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_pressure_lr3e3_entanneal.slurm)
    echo "  Submitted morningrush_pressure_lr3e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_pressure_lr3e4.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_pressure_lr3e4.slurm)
    echo "  Submitted morningrush_pressure_lr3e4.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_pressure_lr3e4_ent002.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_pressure_lr3e4_ent002.slurm)
    echo "  Submitted morningrush_pressure_lr3e4_ent002.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_pressure_lr3e4_ent01.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_pressure_lr3e4_ent01.slurm)
    echo "  Submitted morningrush_pressure_lr3e4_ent01.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_pressure_lr3e4_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_pressure_lr3e4_entanneal.slurm)
    echo "  Submitted morningrush_pressure_lr3e4_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_queue_lr1e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_queue_lr1e3.slurm)
    echo "  Submitted morningrush_queue_lr1e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_queue_lr1e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_queue_lr1e3_entanneal.slurm)
    echo "  Submitted morningrush_queue_lr1e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_queue_lr3e3.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_queue_lr3e3.slurm)
    echo "  Submitted morningrush_queue_lr3e3.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_queue_lr3e3_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_queue_lr3e3_entanneal.slurm)
    echo "  Submitted morningrush_queue_lr3e3_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_queue_lr3e4.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_queue_lr3e4.slurm)
    echo "  Submitted morningrush_queue_lr3e4.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi
if [ -z "$FILTER" ] || echo "morningrush_queue_lr3e4_entanneal.slurm" | grep -q "$FILTER"; then
    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/morningrush_queue_lr3e4_entanneal.slurm)
    echo "  Submitted morningrush_queue_lr3e4_entanneal.slurm (job $JID)"
    COUNT=$((COUNT+1))
fi

echo ""
echo "Submitted $COUNT training jobs. Monitor: squeue -u $USER"
