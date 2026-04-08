#!/bin/bash
# Zeleni SignaLJ - zs_eveningrush_diffwaitingtime_lr1e3
#SBATCH --job-name=zs_eveningrush_diffwaitingtime_lr1e3
#SBATCH --output=logs/zs_eveningrush_diffwaitingtime_lr1e3_%j.out
#SBATCH --error=logs/zs_eveningrush_diffwaitingtime_lr1e3_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G

source hpc/common.sh

srun python src/experiment.py \
    --num_cpus 128 \
    --episode_count 100 \
    --scenario evening_rush --reward_fn diff-waiting-time --learning_rate 0.001 \
    --tag eveningrush_diffwaitingtime_lr1e3_100ep
