#!/bin/bash
# Zeleni SignaLJ - 24h Mega-Policy Statistical Test: M4E3
# Morning: diff-waiting-time, lr=3e-4 (-1.9%) | Evening: diff-waiting-time, lr=1e-3 (+4.6%)
#SBATCH --job-name=zs_mega_M4E3
#SBATCH --output=logs/mega_M4E3_%j.out
#SBATCH --error=logs/mega_M4E3_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting 24h statistical test: M4E3"
echo "  Morning model: diff-waiting-time, lr=3e-4 (-1.9%)"
echo "  Evening model: diff-waiting-time, lr=1e-3 (+4.6%)"
echo "  Runs: 50 | Workers: 50 | Per-seed routes: yes"

srun python src/run_24h.py \
    --model_morning results/experiments/20260413_005844_66186050_morningrush_diffwaitingtime_lr3e4_2400ep/ppo_shared_policy.zip \
    --model_evening results/experiments/20260413_003932_66185851_eveningrush_diffwaitingtime_lr1e3_4800ep/ppo_shared_policy.zip \
    \
    --route_dir data/routes/statistical-test \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/statistical-test/M4E3 \
    --tag M4E3
