#!/bin/bash
# Zeleni SignaLJ - Generate route variants for route-randomized training
#SBATCH --job-name=zs_gen_train_routes
#SBATCH --output=logs/gen_train_routes_%j.out
#SBATCH --error=logs/gen_train_routes_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

source hpc/common.sh

echo "Generating 50 morning_rush route variants..."
srun python src/generate_demand.py \
    --scenario morning_rush \
    --num_variants 50 \
    --output_dir data/routes/train-morning

echo "Generating 50 evening_rush route variants..."
srun python src/generate_demand.py \
    --scenario evening_rush \
    --num_variants 50 \
    --output_dir data/routes/train-evening

echo "Done. Route variants generated."
