#!/bin/bash
# Zeleni SignaLJ - 24h Mega-Policy Statistical Test: M1E2
# Morning: diff-waiting-time, lr=1e-3 (+18.2%) | Evening: diff-waiting-time, lr=1e-3, entropy annealing (+15.0%)
#SBATCH --job-name=zs_mega_M1E2
#SBATCH --output=logs/mega_M1E2_%j.out
#SBATCH --error=logs/mega_M1E2_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting 24h statistical test: M1E2"
echo "  Morning model: diff-waiting-time, lr=1e-3 (+18.2%)"
echo "  Evening model: diff-waiting-time, lr=1e-3, entropy annealing (+15.0%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_24h.py \
    --model_morning results/experiments/20260408_235440_morningrush_diffwaitingtime_lr1e3_200ep/ppo_shared_policy.zip \
    --model_evening results/experiments/20260408_235311_eveningrush_diffwaitingtime_lr1e3_entanneal_200ep/ppo_shared_policy.zip \
    \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/statistical-test/M1E2 \
    --tag M1E2
