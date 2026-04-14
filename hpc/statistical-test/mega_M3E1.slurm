#!/bin/bash
# Zeleni SignaLJ - 24h Mega-Policy Statistical Test: M3E1
# Morning: queue, lr=3e-3, entropy annealing (-1.4%) | Evening: diff-waiting-time, lr=1e-3 (+8.2%)
#SBATCH --job-name=zs_mega_M3E1
#SBATCH --output=logs/mega_M3E1_%j.out
#SBATCH --error=logs/mega_M3E1_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G

source hpc/common.sh

echo "Starting 24h statistical test: M3E1"
echo "  Morning model: queue, lr=3e-3, entropy annealing (-1.4%)"
echo "  Evening model: diff-waiting-time, lr=1e-3 (+8.2%)"
echo "  Runs: 50 | Workers: 50 | Per-seed routes: yes"

srun python src/run_24h.py \
    --model_morning results/experiments/20260413_005819_66185977_morningrush_queue_lr3e3_entanneal_4800ep/ppo_shared_policy.zip \
    --model_evening results/experiments/20260413_005825_66185980_eveningrush_diffwaitingtime_lr1e3_3200ep/ppo_shared_policy.zip \
    \
    --route_dir data/routes/statistical-test \
    --num_runs 50 \
    --num_workers 50 \
    --output_dir results/statistical-test/M3E1 \
    --tag M3E1
