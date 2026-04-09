#!/bin/bash
# Zeleni SignaLJ - 24h Mega-Policy Statistical Test: M1E3
# Morning: diff-waiting-time, lr=1e-3 (+18.2%) | Evening: pressure, lr=3e-4, entropy annealing (+12.9%)
#SBATCH --job-name=zs_mega_M1E3
#SBATCH --output=logs/mega_M1E3_%j.out
#SBATCH --error=logs/mega_M1E3_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

source hpc/common.sh

echo "Starting 24h statistical test: M1E3"
echo "  Morning model: diff-waiting-time, lr=1e-3 (+18.2%)"
echo "  Evening model: pressure, lr=3e-4, entropy annealing (+12.9%)"
echo "  Runs: 50 | Workers: 50"

srun python src/run_24h.py \
    --model_morning results/experiments/20260408_235440_morningrush_diffwaitingtime_lr1e3_200ep/ppo_shared_policy.zip \
    --model_evening results/experiments/20260408_235311_eveningrush_pressure_lr3e4_entanneal_200ep/ppo_shared_policy.zip \
    \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/statistical-test/M1E3 \
    --tag M1E3
