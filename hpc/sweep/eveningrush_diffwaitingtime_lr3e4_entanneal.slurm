#!/bin/bash
# Zeleni SignaLJ - zs_eveningrush_diffwaitingtime_lr3e4_ea
#SBATCH --job-name=zs_eveningrush_diffwaitingtime_lr3e4_ea
#SBATCH --output=logs/zs_eveningrush_diffwaitingtime_lr3e4_ea_%j.out
#SBATCH --error=logs/zs_eveningrush_diffwaitingtime_lr3e4_ea_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G

source hpc/common.sh

srun python src/experiment.py \
    --num_cpus 128 \
    --episode_count 150 \
    --scenario evening_rush --reward_fn diff-waiting-time --learning_rate 0.0003 --entropy_annealing \
    --tag eveningrush_diffwaitingtime_lr3e4_entanneal
