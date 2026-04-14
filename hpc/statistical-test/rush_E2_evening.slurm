#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: E2_evening
# Evening model E2: diff-waiting-time, lr=1e-3, entropy annealing (+7.5%)
#SBATCH --job-name=zs_rush_E2_evening
#SBATCH --output=logs/rush_E2_evening_%j.out
#SBATCH --error=logs/rush_E2_evening_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: E2_evening"
echo "  Scenario: evening_rush"
echo "  Model: diff-waiting-time, lr=1e-3, entropy annealing (+7.5%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260413_005825_66185981_eveningrush_diffwaitingtime_lr1e3_entanneal_3200ep/ppo_shared_policy.zip \
    \
    --scenario evening_rush \
    --route_dir data/routes/statistical-evening-rush \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/E2_evening \
    --tag E2_evening
