#!/bin/bash
# Zeleni SignaLJ - 24h Mega-Policy Statistical Test: M2E1
# Morning: pressure, lr=1e-3 (+17.2%) | Evening: pressure, lr=1e-3, entropy annealing (+15.1%)
#SBATCH --job-name=zs_mega_M2E1
#SBATCH --output=logs/mega_M2E1_%j.out
#SBATCH --error=logs/mega_M2E1_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting 24h statistical test: M2E1"
echo "  Morning model: pressure, lr=1e-3 (+17.2%)"
echo "  Evening model: pressure, lr=1e-3, entropy annealing (+15.1%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_24h.py \
    --model_morning results/experiments/20260409_000353_morningrush_pressure_lr1e3_200ep/ppo_shared_policy.zip \
    --model_evening results/experiments/20260408_235311_eveningrush_pressure_lr1e3_entanneal_200ep/ppo_shared_policy.zip \
    \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/statistical-test/M2E1 \
    --tag M2E1
