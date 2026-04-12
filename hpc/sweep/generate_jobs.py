"""Generate all HPC SLURM job scripts for the experiment sweep.

Two-phase pipeline:
  1. gen_train_routes.slurm — generate route variants per scenario
  2. Training scripts — use --route_dir for route-randomized training
"""
import os

HPC_DIR = os.path.dirname(os.path.abspath(__file__))

REWARD_FNS = ["queue", "pressure", "diff-waiting-time"]
LEARNING_RATES = [1e-3, 3e-4]
SCENARIOS = ["morning_rush", "evening_rush"]
EPISODES = 300
NUM_ROUTE_VARIANTS = 100

# Route directories for route-randomized training
ROUTE_DIRS = {
    "morning_rush": "data/routes/train-morning",
    "evening_rush": "data/routes/train-evening",
}

def lr_str(lr):
    if lr == 1e-3: return "lr1e3"
    if lr == 3e-4: return "lr3e4"
    return f"lr{lr}"

def reward_str(r):
    return r.replace("-", "")

def scenario_str(s):
    return s.replace("_", "")

def write_script(filename, job_name, extra_args, tag):
    path = os.path.join(HPC_DIR, filename)
    content = f"""#!/bin/bash
# Zeleni SignaLJ - {job_name}
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=96G

source hpc/common.sh

srun python src/experiment.py \\
    --num_cpus 128 \\
    --episode_count {EPISODES} \\
    {extra_args} \\
    --tag {tag}
"""
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)
    return filename


def generate_route_gen_script():
    """Generate a SLURM script that creates route variants for both scenarios."""
    path = os.path.join(HPC_DIR, "gen_train_routes.slurm")
    lines = [
        "#!/bin/bash",
        "# Zeleni SignaLJ - Generate route variants for route-randomized training",
        "#SBATCH --job-name=zs_gen_train_routes",
        "#SBATCH --output=logs/gen_train_routes_%j.out",
        "#SBATCH --error=logs/gen_train_routes_%j.err",
        "#SBATCH --time=04:00:00",
        "#SBATCH --partition=all",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=64",
        "#SBATCH --mem=64G",
        "",
        "source hpc/common.sh",
        "",
    ]
    for scenario, route_dir in ROUTE_DIRS.items():
        lines.append(f'echo "Generating {NUM_ROUTE_VARIANTS} {scenario} route variants..."')
        lines.append(f"srun python src/generate_demand.py \\")
        lines.append(f"    --scenario {scenario} \\")
        lines.append(f"    --num_variants {NUM_ROUTE_VARIANTS} \\")
        lines.append(f"    --output_dir {route_dir}")
        lines.append("")

    lines.append('echo "Done. Route variants generated."')
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return "gen_train_routes.slurm"

scripts = []

# Phase 1: Route generation script
route_script = generate_route_gen_script()
print(f"  Route gen: hpc/sweep/{route_script} (run FIRST)")

# ── Reward x LR x Scenario sweep ──
for scenario in SCENARIOS:
    ss = scenario_str(scenario)
    rd = ROUTE_DIRS[scenario]
    for reward in REWARD_FNS:
        for lr in LEARNING_RATES:
            rs = reward_str(reward)
            ls = lr_str(lr)
            tag = f"{ss}_{rs}_{ls}_{EPISODES}ep"
            job = f"zs_{ss}_{rs}_{ls}"
            fname = f"{ss}_{rs}_{ls}.slurm"
            args = f"--scenario {scenario} --reward_fn {reward} --learning_rate {lr} --route_dir {rd}"
            scripts.append(write_script(fname, job, args, tag))

# ── Entropy annealing variants (pressure + diff-waiting-time, both scenarios) ──
for scenario in SCENARIOS:
    ss = scenario_str(scenario)
    rd = ROUTE_DIRS[scenario]
    for reward in ["pressure", "diff-waiting-time"]:
        for lr in LEARNING_RATES:
            rs = reward_str(reward)
            ls = lr_str(lr)
            tag = f"{ss}_{rs}_{ls}_entanneal_{EPISODES}ep"
            job = f"zs_{ss}_{rs}_{ls}_ea"
            fname = f"{ss}_{rs}_{ls}_entanneal.slurm"
            args = f"--scenario {scenario} --reward_fn {reward} --learning_rate {lr} --entropy_annealing --route_dir {rd}"
            scripts.append(write_script(fname, job, args, tag))

# ── Curriculum learning variants (both scenarios, pressure + queue) ──
for scenario in SCENARIOS:
    ss = scenario_str(scenario)
    rd = ROUTE_DIRS[scenario]
    for reward in ["pressure", "queue"]:
        rs = reward_str(reward)
        tag = f"{ss}_{rs}_curriculum_{EPISODES}ep"
        job = f"zs_{ss}_{rs}_curr"
        fname = f"{ss}_{rs}_curriculum.slurm"
        args = f"--scenario {scenario} --reward_fn {reward} --curriculum --route_dir {rd}"
        scripts.append(write_script(fname, job, args, tag))

# Generate submit script with dependency chain
submit_path = os.path.join(HPC_DIR, "submit_all.sh")
with open(submit_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# Submit route generation + training jobs with dependency chain\n")
    f.write("set -e\n")
    f.write('cd "$(git rev-parse --show-toplevel)"\n')
    f.write("mkdir -p logs\n\n")
    f.write('SKIP_ROUTES=false\n')
    f.write('FILTER="${1:-}"\n')
    f.write('if [ "$FILTER" = "--skip-routes" ]; then\n')
    f.write('    SKIP_ROUTES=true; FILTER="${2:-}"; echo "Skipping route generation"\n')
    f.write('fi\n\n')
    f.write('DEP_FLAG=""\n')
    f.write('if [ "$SKIP_ROUTES" = false ]; then\n')
    f.write('    echo "Phase 1: Submitting route generation..."\n')
    f.write('    ROUTE_JID=$(sbatch --parsable hpc/sweep/gen_train_routes.slurm)\n')
    f.write('    echo "  Submitted gen_train_routes.slurm (job $ROUTE_JID)"\n')
    f.write('    DEP_FLAG="--dependency=afterok:${ROUTE_JID}"\n')
    f.write('fi\n\n')
    f.write('echo "Phase 2: Submitting training jobs..."\n')
    f.write('COUNT=0\n')
    for s in sorted(scripts):
        f.write(f'if [ -z "$FILTER" ] || echo "{s}" | grep -q "$FILTER"; then\n')
        f.write(f'    JID=$(sbatch --parsable $DEP_FLAG hpc/sweep/{s})\n')
        f.write(f'    echo "  Submitted {s} (job $JID)"\n')
        f.write(f'    COUNT=$((COUNT+1))\n')
        f.write(f'fi\n')
    f.write(f'\necho ""\necho "Submitted $COUNT training jobs. Monitor: squeue -u $USER"\n')
os.chmod(submit_path, 0o755)

print(f"\nGenerated {len(scripts)} training scripts + 1 route gen + submit_all.sh")
print(f"All scripts use --route_dir with {NUM_ROUTE_VARIANTS} route variants per scenario.")
for s in sorted(scripts):
    print(f"  hpc/sweep/{s}")
print(f"\nPipeline:")
print(f"  bash hpc/sweep/submit_all.sh          # routes + all training")
print(f"  bash hpc/sweep/submit_all.sh --skip-routes  # skip route gen")
print(f"  bash hpc/sweep/submit_all.sh pressure  # filter by keyword")
print(f"\nResume any experiment with:")
print(f"  python src/experiment.py --resume results/experiments/<run_id>/checkpoints/ppo_model_latest.zip --episode_count 400 ...")
