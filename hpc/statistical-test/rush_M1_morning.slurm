#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: M1_morning
# Morning model M1: diff-waiting-time, lr=1e-3 (+18.2%)
#SBATCH --job-name=zs_rush_M1_morning
#SBATCH --output=logs/rush_M1_morning_%j.out
#SBATCH --error=logs/rush_M1_morning_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: M1_morning"
echo "  Scenario: morning_rush"
echo "  Model: diff-waiting-time, lr=1e-3 (+18.2%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260408_235440_morningrush_diffwaitingtime_lr1e3_200ep/ppo_shared_policy.zip \
    \
    --scenario morning_rush \
    --route_dir data/routes/rush-test-morning \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/M1_morning \
    --tag M1_morning
