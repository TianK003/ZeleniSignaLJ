#!/bin/bash
# Zeleni SignaLJ - 24h Mega-Policy Statistical Test: M5E2
# Morning: pressure, lr=3e-4, ent_coef=0.02 (-2.7%) | Evening: diff-waiting-time, lr=1e-3, entropy annealing (+7.5%)
#SBATCH --job-name=zs_mega_M5E2
#SBATCH --output=logs/mega_M5E2_%j.out
#SBATCH --error=logs/mega_M5E2_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting 24h statistical test: M5E2"
echo "  Morning model: pressure, lr=3e-4, ent_coef=0.02 (-2.7%)"
echo "  Evening model: diff-waiting-time, lr=1e-3, entropy annealing (+7.5%)"
echo "  Runs: 50 | Workers: 50 | Per-seed routes: yes"

srun python src/run_24h.py \
    --model_morning results/experiments/20260413_003935_66185886_morningrush_pressure_lr3e4_ent002_4800ep/ppo_shared_policy.zip \
    --model_evening results/experiments/20260413_005825_66185981_eveningrush_diffwaitingtime_lr1e3_entanneal_3200ep/ppo_shared_policy.zip \
    \
    --route_dir data/routes/statistical-test \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/statistical-test/M5E2 \
    --tag M5E2
