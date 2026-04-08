"""Generate all HPC SLURM job scripts for the experiment sweep."""
import os

HPC_DIR = os.path.dirname(os.path.abspath(__file__))

REWARD_FNS = ["queue", "pressure", "diff-waiting-time"]
LEARNING_RATES = [1e-3, 3e-4]
SCENARIOS = ["morning_rush", "evening_rush"]
EPISODES = 200

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
#SBATCH --time=16:00:00
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

scripts = []

# ── Reward x LR x Scenario sweep (100ep each) ──
for scenario in SCENARIOS:
    ss = scenario_str(scenario)
    for reward in REWARD_FNS:
        for lr in LEARNING_RATES:
            rs = reward_str(reward)
            ls = lr_str(lr)
            tag = f"{ss}_{rs}_{ls}_{EPISODES}ep"
            job = f"zs_{ss}_{rs}_{ls}"
            fname = f"{ss}_{rs}_{ls}.slurm"
            args = f"--scenario {scenario} --reward_fn {reward} --learning_rate {lr}"
            scripts.append(write_script(fname, job, args, tag))

# ── Entropy annealing variants (pressure + diff-waiting-time, both scenarios) ──
for scenario in SCENARIOS:
    ss = scenario_str(scenario)
    for reward in ["pressure", "diff-waiting-time"]:
        for lr in LEARNING_RATES:
            rs = reward_str(reward)
            ls = lr_str(lr)
            tag = f"{ss}_{rs}_{ls}_entanneal_{EPISODES}ep"
            job = f"zs_{ss}_{rs}_{ls}_ea"
            fname = f"{ss}_{rs}_{ls}_entanneal.slurm"
            args = f"--scenario {scenario} --reward_fn {reward} --learning_rate {lr} --entropy_annealing"
            scripts.append(write_script(fname, job, args, tag))

# ── Curriculum learning variants (both scenarios, pressure + queue) ──
for scenario in SCENARIOS:
    ss = scenario_str(scenario)
    for reward in ["pressure", "queue"]:
        rs = reward_str(reward)
        tag = f"{ss}_{rs}_curriculum_{EPISODES}ep"
        job = f"zs_{ss}_{rs}_curr"
        fname = f"{ss}_{rs}_curriculum.slurm"
        args = f"--scenario {scenario} --reward_fn {reward} --curriculum --log_curriculum"
        scripts.append(write_script(fname, job, args, tag))

print(f"Generated {len(scripts)} SLURM scripts (all {EPISODES} episodes):")
for s in sorted(scripts):
    print(f"  hpc/{s}")
print(f"\nResume any experiment with:")
print(f"  python src/experiment.py --resume results/experiments/<run_id>/checkpoints/ppo_model_XXXXX_steps.zip --episode_count 400 ...")
