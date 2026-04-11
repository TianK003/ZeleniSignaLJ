#!/bin/bash
# Zeleni SignaLJ - 24h Mega-Policy Statistical Test: M3E2
# Morning: queue (default), lr=3e-4 (+17.1%) | Evening: diff-waiting-time, lr=1e-3, entropy annealing (+15.0%)
#SBATCH --job-name=zs_mega_M3E2
#SBATCH --output=logs/mega_M3E2_%j.out
#SBATCH --error=logs/mega_M3E2_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting 24h statistical test: M3E2"
echo "  Morning model: queue (default), lr=3e-4 (+17.1%)"
echo "  Evening model: diff-waiting-time, lr=1e-3, entropy annealing (+15.0%)"
echo "  Runs: 50 | Workers: 50 | Per-seed routes: yes"

srun python src/run_24h.py \
    --model_morning results/experiments/20260408_235311_morningrush_default_lr3e4_200ep/ppo_shared_policy.zip \
    --model_evening results/experiments/20260408_235311_eveningrush_diffwaitingtime_lr1e3_entanneal_200ep/ppo_shared_policy.zip \
    \
    --route_dir data/routes/statistical-test \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/statistical-test/M3E2 \
    --tag M3E2
