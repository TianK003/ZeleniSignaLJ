#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: M1_morning
# Morning model M1: diff-waiting-time, lr=3e-3 (+5.2%)
#SBATCH --job-name=zs_rush_M1_morning
#SBATCH --output=logs/rush_M1_morning_%j.out
#SBATCH --error=logs/rush_M1_morning_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: M1_morning"
echo "  Scenario: morning_rush"
echo "  Model: diff-waiting-time, lr=3e-3 (+5.2%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260413_005856_66186092_morningrush_diffwaitingtime_lr3e3_1600ep/ppo_shared_policy.zip \
    \
    --scenario morning_rush \
    --route_dir data/routes/statistical-morning-test \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/M1_morning \
    --tag M1_morning
