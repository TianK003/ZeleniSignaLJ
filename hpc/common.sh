#!/bin/bash
# Zeleni SignaLJ - Common HPC setup (sourced by all job scripts)

# Activate the local virtual environment
source .venv/bin/activate

# SUMO setup
export SUMO_HOME=$HOME/sumo_src
export LIBSUMO_AS_TRACI="1"
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

# Ensure output directories exist
mkdir -p logs results
