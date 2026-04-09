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

# Map SUMO_HOME to a guaranteed Github source checkout because the pip wheel removes tools/
export SUMO_HOME=$HOME/sumo_src
export LIBSUMO_AS_TRACI="1"
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

echo "Booting up Zeleni SignaLJ on 32 CPUs..."

# Fire the massive parallel script natively
srun python src/experiment.py \
    --episode_count 100 \
    --curriculum \
    --tag hpc_32cpu_run \
    --num_cpus 64
