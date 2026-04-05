#!/bin/bash
# Zeleni SignaLJ - Vega HPC Training Job
#SBATCH --job-name=zeleni_ppo_train
#SBATCH --output=logs/ppo_train_%j.out
#SBATCH --error=logs/ppo_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Setup
module load Apptainer 2>/dev/null || module load tools/Singularity

export APPTAINER_TMPDIR=/scratch/$USER/apptainer_tmp/${SLURM_JOB_ID}
mkdir -p $APPTAINER_TMPDIR
mkdir -p logs

# Run training in container with GPU passthrough
srun apptainer exec --nv \
    --bind $PWD:/workspace \
    /scratch/$USER/traffic_rl.sif \
    python /workspace/src/train.py \
        --num_envs 32 \
        --total_timesteps 2000000 \
        --checkpoint_dir /workspace/models/

# Cleanup temp directory
rm -rf $APPTAINER_TMPDIR
