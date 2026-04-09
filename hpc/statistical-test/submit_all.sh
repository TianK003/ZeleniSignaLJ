#!/bin/bash
# Submit all mega-policy statistical test jobs to SLURM
# Usage: bash hpc/statistical-test/submit_all.sh [filter]
# Examples:
#   bash hpc/statistical-test/submit_all.sh           # all 10 jobs
#   bash hpc/statistical-test/submit_all.sh baseline   # only baseline
#   bash hpc/statistical-test/submit_all.sh M1         # only M1* combinations

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILTER="${1:-}"
COUNT=0

echo "Submitting 24h mega-policy statistical test jobs..."
echo ""

for f in "$SCRIPT_DIR"/mega_*.slurm; do
    [ -f "$f" ] || continue
    name=$(basename "$f")

    # Apply optional filter
    if [ -n "$FILTER" ] && [[ "$name" != *"$FILTER"* ]]; then
        continue
    fi

    echo "  sbatch $f"
    sbatch "$f"
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted $COUNT jobs. Monitor with: squeue -u \$USER"
