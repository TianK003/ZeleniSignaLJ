#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: M2_morning
# Morning model M2: diff-waiting-time, lr=1e-3 (+1.6%)
#SBATCH --job-name=zs_rush_M2_morning
#SBATCH --output=logs/rush_M2_morning_%j.out
#SBATCH --error=logs/rush_M2_morning_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: M2_morning"
echo "  Scenario: morning_rush"
echo "  Model: diff-waiting-time, lr=1e-3 (+1.6%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260413_005828_66186002_morningrush_diffwaitingtime_lr1e3_3200ep/ppo_shared_policy.zip \
    \
    --scenario morning_rush \
    --route_dir data/routes/statistical-morning-rush \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/M2_morning \
    --tag M2_morning
