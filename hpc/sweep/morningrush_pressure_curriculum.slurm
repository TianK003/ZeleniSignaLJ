#!/bin/bash
# Zeleni SignaLJ - zs_morningrush_pressure_curr
#SBATCH --job-name=zs_morningrush_pressure_curr
#SBATCH --output=logs/zs_morningrush_pressure_curr_%j.out
#SBATCH --error=logs/zs_morningrush_pressure_curr_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G

source hpc/common.sh

srun python src/experiment.py \
    --num_cpus 128 \
    --episode_count 200 \
    --scenario morning_rush --reward_fn pressure --curriculum --log_curriculum \
    --tag morningrush_pressure_curriculum_200ep
