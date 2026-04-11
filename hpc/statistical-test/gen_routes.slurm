#!/bin/bash
# Zeleni SignaLJ - Generate 50 route files for statistical testing
#SBATCH --job-name=zs_gen_routes
#SBATCH --output=logs/gen_routes_%j.out
#SBATCH --error=logs/gen_routes_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

source hpc/common.sh

# Flush Python output immediately so SLURM logs are readable in real-time
export PYTHONUNBUFFERED=1

echo "Generating 50 route files for statistical testing..."
echo "  Output: data/routes/statistical-test/"

python src/generate_demand.py \
    --statistical_routes 50 \
    --output_dir data/routes/statistical-test \
    --num_workers 32

echo "Done. Route files in data/routes/statistical-test/"
