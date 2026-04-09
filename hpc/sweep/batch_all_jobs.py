#!/usr/bin/env python3
"""Submit all .slurm jobs in the hpc/ directory via sbatch."""

import subprocess
import glob
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
slurm_files = sorted(glob.glob(os.path.join(script_dir, "*.slurm")))

if not slurm_files:
    print("No .slurm files found.")
    exit(1)

print(f"Submitting {len(slurm_files)} jobs...\n")

for path in slurm_files:
    name = os.path.basename(path)
    result = subprocess.run(["sbatch", path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK]   {name}: {result.stdout.strip()}")
    else:
        print(f"[FAIL] {name}: {result.stderr.strip()}")
