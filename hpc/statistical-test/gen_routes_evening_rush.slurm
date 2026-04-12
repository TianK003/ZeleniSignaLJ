#!/bin/bash
# Zeleni SignaLJ - Generate 50 evening_rush route variants
#SBATCH --job-name=zs_gen_evening_rush
#SBATCH --output=logs/gen_evening_rush_%j.out
#SBATCH --error=logs/gen_evening_rush_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

source hpc/common.sh

echo "Generating 50 evening_rush route variants..."
echo "  Output: data/routes/rush-test-evening/"

# --num_workers auto-detects from cpus-per-task (64 -> all 50 in parallel)
srun python src/generate_demand.py \
    --scenario evening_rush \
    --num_variants 50 \
    --output_dir data/routes/rush-test-evening

echo "Done. Route files in data/routes/rush-test-evening/"
