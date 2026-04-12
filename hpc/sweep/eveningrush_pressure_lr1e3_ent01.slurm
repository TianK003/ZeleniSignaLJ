#!/bin/bash
# Zeleni SignaLJ - zs_eveningrush_pr_lr1e3_e01
#SBATCH --job-name=zs_eveningrush_pr_lr1e3_e01
#SBATCH --output=logs/zs_eveningrush_pr_lr1e3_e01_%j.out
#SBATCH --error=logs/zs_eveningrush_pr_lr1e3_e01_%j.err
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
    --scenario evening_rush --reward_fn pressure --learning_rate 0.001 --ent_coef 0.1 --route_dir data/routes/train-evening \
    --tag eveningrush_pressure_lr1e3_ent01_300ep
