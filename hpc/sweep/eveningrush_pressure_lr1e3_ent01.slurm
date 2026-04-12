#!/bin/bash
# Zeleni SignaLJ - zs_eveningrush_pr_lr1e3_e01
#SBATCH --job-name=zs_eveningrush_pr_lr1e3_e01
#SBATCH --output=logs/zs_eveningrush_pr_lr1e3_e01_%j.out
#SBATCH --error=logs/zs_eveningrush_pr_lr1e3_e01_%j.err
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
    --scenario evening_rush --reward_fn pressure --learning_rate 0.001 --ent_coef 0.1 --route_dir data/routes/train-evening \
    --tag eveningrush_pressure_lr1e3_ent01_2400ep
