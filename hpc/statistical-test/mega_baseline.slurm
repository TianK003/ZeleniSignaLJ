#!/bin/bash
# Zeleni SignaLJ - 24h Mega-Policy Statistical Test: baseline
# Baseline: all traffic signals on fixed-time programs (no RL)
#SBATCH --job-name=zs_mega_baseline
#SBATCH --output=logs/mega_baseline_%j.out
#SBATCH --error=logs/mega_baseline_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting 24h statistical test: baseline"
echo "  Morning model: none (fixed-time)"
echo "  Evening model: none (fixed-time)"
echo "  Runs: 50 | Workers: 50 | Per-seed routes: yes"

srun python src/run_24h.py \
    --baseline \
    \
    --route_dir data/routes/statistical-test \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/statistical-test/baseline \
    --tag baseline
