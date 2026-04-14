#!/bin/bash
# Zeleni SignaLJ - Rush-Hour Generalization Test: M5_morning
# Morning model M5: pressure, lr=3e-4, ent_coef=0.02 (-2.7%)
#SBATCH --job-name=zs_rush_M5_morning
#SBATCH --output=logs/rush_M5_morning_%j.out
#SBATCH --error=logs/rush_M5_morning_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting rush-hour generalization test: M5_morning"
echo "  Scenario: morning_rush"
echo "  Model: pressure, lr=3e-4, ent_coef=0.02 (-2.7%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_rush_test.py \
    --model results/experiments/20260413_003935_66185886_morningrush_pressure_lr3e4_ent002_4800ep/ppo_shared_policy.zip \
    \
    --scenario morning_rush \
    --route_dir data/routes/statistical-morning-rush \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/rush-test/M5_morning \
    --tag M5_morning
