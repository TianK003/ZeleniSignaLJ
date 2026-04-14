#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: E4_evening
# Evening model E4: diff-waiting-time, lr=3e-4 (+2.5%)
#SBATCH --job-name=zs_rush_E4_evening
#SBATCH --output=logs/rush_E4_evening_%j.out
#SBATCH --error=logs/rush_E4_evening_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: E4_evening"
echo "  Scenario: evening_rush"
echo "  Model: diff-waiting-time, lr=3e-4 (+2.5%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260413_005816_66185940_eveningrush_diffwaitingtime_lr3e4_4800ep/ppo_shared_policy.zip \
    \
    --scenario evening_rush \
    --route_dir data/routes/statistical-evening-rush \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/E4_evening \
    --tag E4_evening
