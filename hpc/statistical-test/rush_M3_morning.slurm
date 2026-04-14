#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: M3_morning
# Morning model M3: queue, lr=3e-3, entropy annealing (-1.4%)
#SBATCH --job-name=zs_rush_M3_morning
#SBATCH --output=logs/rush_M3_morning_%j.out
#SBATCH --error=logs/rush_M3_morning_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: M3_morning"
echo "  Scenario: morning_rush"
echo "  Model: queue, lr=3e-3, entropy annealing (-1.4%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260413_005819_66185977_morningrush_queue_lr3e3_entanneal_4800ep/ppo_shared_policy.zip \
    \
    --scenario morning_rush \
    --route_dir data/routes/statistical-morning-test \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/M3_morning \
    --tag M3_morning
