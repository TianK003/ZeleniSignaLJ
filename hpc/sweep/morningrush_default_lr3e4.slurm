#!/bin/bash
# Zeleni SignaLJ - zs_morningrush_default_lr3e4
#SBATCH --job-name=zs_morningrush_default_lr3e4
#SBATCH --output=logs/zs_morningrush_default_lr3e4_%j.out
#SBATCH --error=logs/zs_morningrush_default_lr3e4_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G

source hpc/common.sh

# No --reward_fn flag: uses config.py default (REWARD_FN = "queue")
srun python src/experiment.py \
    --num_cpus 128 \
    --episode_count 150 \
    --scenario morning_rush \
    --learning_rate 0.0003 \
    --tag morningrush_default_lr3e4
