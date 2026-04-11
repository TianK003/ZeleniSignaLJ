#!/bin/bash
# Zeleni SignaLJ - Generate explanations for all experiments
# Runs collect_states.py + explain.py for every experiment that has a
# trained model but no explanations/ folder yet.
#SBATCH --job-name=zs_explain_all
#SBATCH --output=logs/explain_all_%j.out
#SBATCH --error=logs/explain_all_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

source hpc/common.sh

# Flush output immediately so logs are readable during execution
export PYTHONUNBUFFERED=1

echo "============================================"
echo "Generating explanations for all experiments"
echo "  Workers: 16 (each runs its own SUMO)"
echo "  Episodes per experiment: 15"
echo "============================================"

# Generate route files if missing (needed by collect_states.py)
if [ ! -f data/routes/routes_morning_rush.rou.xml ] || \
   [ ! -f data/routes/routes_evening_rush.rou.xml ]; then
    echo "Route files missing — generating scenarios first..."
    python src/generate_demand.py --scenario all
    echo "Routes ready."
fi

python src/generate_all_explanations.py \
    --num_workers 16 \
    --episodes 15

echo "Done."
