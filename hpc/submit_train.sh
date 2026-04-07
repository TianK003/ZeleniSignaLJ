#!/bin/bash
# Zeleni SignaLJ - Vega HPC Training Job
#SBATCH --job-name=zeleni_ppo_train
#SBATCH --output=logs/ppo_train_%j.out
#SBATCH --error=logs/ppo_train_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G

# Make sure logs directory exists
mkdir -p logs

# Activate the local virtual environment (where we pip-installed eclipse-sumo)
# Assuming .venv is in the project root directory
source .venv/bin/activate

# Dynamically map the SUMO environment variables
export SUMO_HOME=$(python -c "import sumo, os; print(os.environ['SUMO_HOME'])")
export LIBSUMO_AS_TRACI="1"

echo "Booting up Zeleni SignaLJ on 32 CPUs..."

# Fire the massive parallel script natively
srun python src/experiment.py \
    --episode_count 100 \
    --curriculum \
    --tag hpc_32cpu_run \
    --num_cpus 64
