#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: E5_evening
# Evening model E5: diff-waiting-time, lr=1e-3 (+2.0%)
#SBATCH --job-name=zs_rush_E5_evening
#SBATCH --output=logs/rush_E5_evening_%j.out
#SBATCH --error=logs/rush_E5_evening_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: E5_evening"
echo "  Scenario: evening_rush"
echo "  Model: diff-waiting-time, lr=1e-3 (+2.0%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260413_005903_66186112_eveningrush_diffwaitingtime_lr1e3_800ep/ppo_shared_policy.zip \
    \
    --scenario evening_rush \
    --route_dir data/routes/statistical-evening-rush \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/E5_evening \
    --tag E5_evening
