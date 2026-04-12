#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: E1_evening
# Evening model E1: pressure, lr=1e-3, entropy annealing (+15.1%)
#SBATCH --job-name=zs_rush_E1_evening
#SBATCH --output=logs/rush_E1_evening_%j.out
#SBATCH --error=logs/rush_E1_evening_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: E1_evening"
echo "  Scenario: evening_rush"
echo "  Model: pressure, lr=1e-3, entropy annealing (+15.1%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260408_235311_eveningrush_pressure_lr1e3_entanneal_200ep/ppo_shared_policy.zip \
    \
    --scenario evening_rush \
    --route_dir data/routes/rush-test-evening \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/E1_evening \
    --tag E1_evening
