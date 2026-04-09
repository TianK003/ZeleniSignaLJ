#!/bin/bash
# Zeleni SignaLJ - zs_eveningrush_default_lr1e3
#SBATCH --job-name=zs_eveningrush_default_lr1e3
#SBATCH --output=logs/zs_eveningrush_default_lr1e3_%j.out
#SBATCH --error=logs/zs_eveningrush_default_lr1e3_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G

source hpc/common.sh

# No --reward_fn flag: uses config.py default (REWARD_FN = "queue")
srun python src/experiment.py \
    --num_cpus 128 \
    --episode_count 200 \
    --scenario evening_rush \
    --learning_rate 0.001 \
    --tag eveningrush_default_lr1e3_200ep
