#!/bin/bash
# Zeleni SignaLJ - zs_morningrush_diffwaitingtime_lr1e3
#SBATCH --job-name=zs_morningrush_diffwaitingtime_lr1e3
#SBATCH --output=logs/zs_morningrush_diffwaitingtime_lr1e3_%j.out
#SBATCH --error=logs/zs_morningrush_diffwaitingtime_lr1e3_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

source hpc/common.sh

srun python src/experiment.py \
    --num_cpus 16 \
    --episode_count 2400 \
    --scenario morning_rush --reward_fn diff-waiting-time --learning_rate 0.001 --route_dir data/routes/train-morning \
    --tag morningrush_diffwaitingtime_lr1e3_2400ep
