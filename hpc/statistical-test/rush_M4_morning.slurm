#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: M4_morning
# Morning model M4: diff-waiting-time, lr=3e-4 (-1.9%)
#SBATCH --job-name=zs_rush_M4_morning
#SBATCH --output=logs/rush_M4_morning_%j.out
#SBATCH --error=logs/rush_M4_morning_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: M4_morning"
echo "  Scenario: morning_rush"
echo "  Model: diff-waiting-time, lr=3e-4 (-1.9%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260413_005844_66186050_morningrush_diffwaitingtime_lr3e4_2400ep/ppo_shared_policy.zip \
    \
    --scenario morning_rush \
    --route_dir data/routes/statistical-morning-test \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/M4_morning \
    --tag M4_morning
