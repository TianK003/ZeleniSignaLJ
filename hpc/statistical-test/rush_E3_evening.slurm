#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: E3_evening
# Evening model E3: pressure, lr=3e-4, entropy annealing (+12.9%)
#SBATCH --job-name=zs_rush_E3_evening
#SBATCH --output=logs/rush_E3_evening_%j.out
#SBATCH --error=logs/rush_E3_evening_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: E3_evening"
echo "  Scenario: evening_rush"
echo "  Model: pressure, lr=3e-4, entropy annealing (+12.9%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260408_235311_eveningrush_pressure_lr3e4_entanneal_200ep/ppo_shared_policy.zip \
    \
    --scenario evening_rush \
    --route_dir data/routes/rush-test-evening \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/E3_evening \
    --tag E3_evening
