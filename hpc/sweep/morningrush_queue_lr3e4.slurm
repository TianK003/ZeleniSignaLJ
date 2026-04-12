#!/bin/bash
# Zeleni SignaLJ - zs_morningrush_queue_lr3e4
#SBATCH --job-name=zs_morningrush_queue_lr3e4
#SBATCH --output=logs/zs_morningrush_queue_lr3e4_%j.out
#SBATCH --error=logs/zs_morningrush_queue_lr3e4_%j.err
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
    --scenario morning_rush --reward_fn queue --learning_rate 0.0003 --route_dir data/routes/train-morning \
    --tag morningrush_queue_lr3e4_2400ep
