#!/bin/bash
# Zeleni SignaLJ - zs_morningrush_queue_lr3e3
#SBATCH --job-name=zs_morningrush_queue_lr3e3
#SBATCH --output=logs/zs_morningrush_queue_lr3e3_%j.out
#SBATCH --error=logs/zs_morningrush_queue_lr3e3_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G

source hpc/common.sh

srun python src/experiment.py \
    --num_cpus 128 \
    --episode_count 300 \
    --scenario morning_rush --reward_fn queue --learning_rate 0.003 --route_dir data/routes/train-morning \
    --tag morningrush_queue_lr3e3_300ep
