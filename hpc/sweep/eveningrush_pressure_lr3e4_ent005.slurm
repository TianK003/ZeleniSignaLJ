#!/bin/bash
# Zeleni SignaLJ - zs_eveningrush_pr_lr3e4_e005
#SBATCH --job-name=zs_eveningrush_pr_lr3e4_e005
#SBATCH --output=logs/zs_eveningrush_pr_lr3e4_e005_%j.out
#SBATCH --error=logs/zs_eveningrush_pr_lr3e4_e005_%j.err
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
    --scenario evening_rush --reward_fn pressure --learning_rate 0.0003 --ent_coef 0.05 --route_dir data/routes/train-evening \
    --tag eveningrush_pressure_lr3e4_ent005_300ep
