#!/bin/bash
# Zeleni SignaLJ - Submit all HPC jobs
# Usage: bash hpc/submit_all.sh [filter]
# Examples:
#   bash hpc/submit_all.sh              # submit ALL jobs
#   bash hpc/submit_all.sh pressure     # only pressure reward jobs
#   bash hpc/submit_all.sh 100ep        # only 100-episode jobs
#   bash hpc/submit_all.sh entanneal    # only entropy annealing jobs
#   bash hpc/submit_all.sh curriculum   # only curriculum jobs

FILTER="${1:-}"
COUNT=0

for script in hpc/sweep/*.slurm; do
    if [ -n "$FILTER" ] && [[ ! "$script" == *"$FILTER"* ]]; then
        continue
    fi
    echo "Submitting: $script"
    sbatch "$script"
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted $COUNT jobs. Monitor with: squeue -u \$USER"
