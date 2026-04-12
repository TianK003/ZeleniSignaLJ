#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: baseline_evening
# Baseline: fixed-time programs (evening_rush)
#SBATCH --job-name=zs_rush_baseline_evening
#SBATCH --output=logs/rush_baseline_evening_%j.out
#SBATCH --error=logs/rush_baseline_evening_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: baseline_evening"
echo "  Scenario: evening_rush"
echo "  Model: none (fixed-time baseline)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --baseline \
    \
    --scenario evening_rush \
    --route_dir data/routes/rush-test-evening \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/baseline_evening \
    --tag baseline_evening
