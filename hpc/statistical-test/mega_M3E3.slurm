#!/bin/bash
# Zeleni SignaLJ - 24h Mega-Policy Statistical Test: M3E3
# Morning: queue (default), lr=3e-4 (+17.1%) | Evening: pressure, lr=3e-4, entropy annealing (+12.9%)
#SBATCH --job-name=zs_mega_M3E3
#SBATCH --output=logs/mega_M3E3_%j.out
#SBATCH --error=logs/mega_M3E3_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting 24h statistical test: M3E3"
echo "  Morning model: queue (default), lr=3e-4 (+17.1%)"
echo "  Evening model: pressure, lr=3e-4, entropy annealing (+12.9%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_24h.py \
    --model_morning results/experiments/20260408_235311_morningrush_default_lr3e4_200ep/ppo_shared_policy.zip \
    --model_evening results/experiments/20260408_235311_eveningrush_pressure_lr3e4_entanneal_200ep/ppo_shared_policy.zip \
    \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/statistical-test/M3E3 \
    --tag M3E3
