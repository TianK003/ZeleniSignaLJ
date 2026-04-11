#!/bin/bash
# Zeleni SignaLJ - Generate explanations for all experiments
# Runs collect_states.py + explain.py for every experiment that has a
# trained model but no explanations/ folder yet.
#SBATCH --job-name=zs_explain_all
#SBATCH --output=logs/explain_all_%j.out
#SBATCH --error=logs/explain_all_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

source hpc/common.sh

# Flush output immediately so SLURM logs are readable in real-time
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Generating explanations for all experiments"
echo "  Workers: 16 (each runs its own SUMO)"
echo "  Episodes per experiment: 15"
echo "  Per-experiment logs: logs/explanations/"
echo "============================================"

# Generate route files if missing (needed by collect_states.py)
for scenario in morning_rush evening_rush offpeak; do
    route="data/routes/routes_${scenario}.rou.xml"
    if [ ! -f "$route" ]; then
        echo "Missing $route — generating all scenarios..."
        python src/generate_demand.py --scenario all
        echo "Routes ready."
        break
    fi
done

# Dry run first to show what will be processed
python src/generate_all_explanations.py --dry_run
echo ""

# Run for real
python src/generate_all_explanations.py \
    --num_workers 16 \
    --episodes 15 \
    --log_dir logs/explanations

echo ""
echo "Done. Check logs/explanations/ for per-experiment details."
