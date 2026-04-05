#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Zeleni SignaLJ - WSL2 Environment Setup
# ══════════════════════════════════════════════════════════════
# Run this script from your project root in WSL2 Ubuntu:
#   chmod +x setup_env.sh && ./setup_env.sh
#
# What this does:
#   1. Installs system prerequisites
#   2. Installs SUMO 1.26.0 from PPA
#   3. Creates Python virtual environment
#   4. Installs all Python dependencies
#   5. Verifies the full stack works
# ══════════════════════════════════════════════════════════════

set -e  # Exit on any error

echo "══════════════════════════════════════════════════"
echo "  Zeleni SignaLJ - Environment Setup"
echo "══════════════════════════════════════════════════"

# ── Step 1: System prerequisites ──
echo ""
echo "[1/5] Installing system prerequisites..."
sudo apt-get update
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    git wget curl \
    build-essential cmake \
    software-properties-common

# ── Step 2: Install SUMO ──
echo ""
echo "[2/5] Installing SUMO..."
sudo add-apt-repository -y ppa:sumo/stable
sudo apt-get update
sudo apt-get install -y sumo sumo-tools sumo-doc

# Set SUMO_HOME (also append to .bashrc for persistence)
export SUMO_HOME="/usr/share/sumo"
export PATH="$PATH:$SUMO_HOME/bin"

if ! grep -q 'SUMO_HOME' ~/.bashrc; then
    echo '' >> ~/.bashrc
    echo '# SUMO environment' >> ~/.bashrc
    echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
    echo 'export PATH="$PATH:$SUMO_HOME/bin"' >> ~/.bashrc
    echo "  Added SUMO_HOME to ~/.bashrc"
fi

# ── Step 3: Create Python virtual environment ──
echo ""
echo "[3/5] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# ── Step 4: Install Python dependencies ──
echo ""
echo "[4/5] Installing Python dependencies..."
pip install -r requirements.txt

# ── Step 5: Verify everything ──
echo ""
echo "[5/5] Verifying installation..."
echo ""

# Check SUMO
SUMO_VERSION=$(sumo --version 2>&1 | head -1)
echo "  SUMO:            $SUMO_VERSION"

# Check Python packages
python3 -c "
import sumo_rl
import stable_baselines3 as sb3
import gymnasium
import torch

print(f'  sumo-rl:         {sumo_rl.__version__}')
print(f'  stable-baselines3: {sb3.__version__}')
print(f'  gymnasium:       {gymnasium.__version__}')
print(f'  PyTorch:         {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:             {torch.cuda.get_device_name(0)}')
"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  To activate the environment in future sessions:"
echo "    source .venv/bin/activate"
echo ""
echo "  Next steps:"
echo "    1. Test SUMO GUI:  sumo-gui"
echo "    2. Test netedit:   netedit"
echo "    3. Download OSM data (see execution plan)"
echo "══════════════════════════════════════════════════"
