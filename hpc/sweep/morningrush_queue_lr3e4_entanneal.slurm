#!/bin/bash
# Zeleni SignaLJ - zs_morningrush_queue_lr3e4_ea
#SBATCH --job-name=zs_morningrush_queue_lr3e4_ea
#SBATCH --output=logs/zs_morningrush_queue_lr3e4_ea_%j.out
#SBATCH --error=logs/zs_morningrush_queue_lr3e4_ea_%j.err
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
    --scenario morning_rush --reward_fn queue --learning_rate 0.0003 --entropy_annealing --route_dir data/routes/train-morning \
    --tag morningrush_queue_lr3e4_entanneal_2400ep
