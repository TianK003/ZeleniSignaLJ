# Zeleni SignaLJ — Project Context

## What This Project Is
RL-based adaptive traffic signal control for Ljubljana, Slovenia. Hackathon project (Arnes HackathON 2026, 10-day timeline). Team: Ransomware (4 members).

## Target Area
The "Bleiweisova triangle" — three critical intersections in central Ljubljana plus 2 adjacent feeders (5 total):
- Bleiweisova / Tržaška / Aškerčeva (SW corner)
- Bleiweisova / Celovška / Tivolska (NW corner)
- Tivolska / Dunajska / Slovenska (NE corner)
- + 2 adjacent feeder intersections

**Important:** Slovenska cesta through the center is bus-only (since 2013). Do NOT include those traffic lights in RL optimization — they serve a different traffic pattern.

## Controlled Intersections (5 of 37 TLS in network)
Intersection IDs (from netedit):
1. `cluster_12747553827_2030605052_4198388404_4198388407_#10more`
2. `cluster_1727155499_250769435_3145991894_3145991895_#9more`
3. `cluster_10946184173_33632882_4083612498_4898978366_#5more`
4. `cluster_1849709610_6264117642_6264117645_8155872322_#10more`
5. `cluster_1632640893_3884437221_3884437224_4312381314_#8more`

TLS IDs (used in sumo-rl):
1. `joinedS_5154793231_8093399326_8093399327_8093399328_#7more`
2. `joinedS_1951535395_8569909625_8569909627_8569909629_#5more`
3. `joinedS_cluster_10946184173_33632882_4083612498_4898978366_#5more_cluster_4898978371_9307230471_9307230472`
4. `joinedS_16191121_311397806_476283378_6264081028_#12more`
5. `joinedS_8241154017_8241154018_cluster_1632640893_3884437221_3884437224_4312381314_#8more_cluster_8171896855_8171896868_8241143312`

These are defined in `src/config.py`. All other 32 TLS run default fixed-time programs.

## OSM Data Source
Source: Overpass Turbo query (highways + traffic signals + crossings only)
Bounding box (south, west, north, east): `46.04540, 14.49385, 46.05840, 14.50687`

Overpass query:
```
[out:xml][timeout:60];
(
  way["highway"](46.04540,14.49385,46.05840,14.50687);
  node["highway"="traffic_signals"](46.04540,14.49385,46.05840,14.50687);
  node["highway"="crossing"](46.04540,14.49385,46.05840,14.50687);
);
(._;>;);
out body;
```

OSM file: `data/osm/bleiweisova.osm` (gitignored, ~3MB from Overpass, not the 20MB full export)

netconvert command:
```bash
netconvert --osm-files data/osm/bleiweisova.osm \
  --output-file data/networks/ljubljana.net.xml \
  --geometry.remove --ramps.guess \
  --junctions.join --junctions.join-dist 10 \
  --tls.guess-signals --tls.join --tls.ignore-internal-junction-jam \
  --edges.join --osm.turn-lanes true \
  --default.junctions.keep-clear true --default.lanewidth 3.2 \
  --default.speed 13.89 --no-turnarounds true
```

## Tech Stack
- **Simulator:** SUMO 1.26.0 (microscopic traffic sim)
- **RL wrapper:** sumo-rl 1.4.5 (PettingZoo parallel API for multi-agent)
- **Multi-agent vectorization:** SuperSuit 3.9+ (PettingZoo → SB3 VecEnv)
- **RL algorithm:** PPO with parameter sharing via stable-baselines3 2.8.0
- **HPC:** Vega supercomputer (IZUM Slovenia), Apptainer containers, SLURM scheduler
- **Python:** 3.12, PyTorch, pandas, matplotlib, wandb

## Key Architecture Decisions
- **Parameter-sharing PPO** via SuperSuit: the 5 target traffic signals share one policy network. Each signal is treated as a separate sub-environment in a vectorized env. Non-target TLS have their original SUMO programs restored via `src/tls_programs.py` and are monkey-patched so sumo-rl cannot override them. This is critical — without filtering, all 37 TLS share one policy and performance is terrible.
- **Non-target TLS handling** (`src/tls_programs.py`): sumo-rl's `build_phases()` replaces ALL TLS programs on init. To keep non-target TLS running their real signal programs, we: (1) parse original programs from .net.xml before env creation, (2) after each `env.reset()`, restore them via TraCI `setProgramLogic` + `setProgram`, (3) monkey-patch `set_next_phase` to a no-op and `update` to only increment the time counter. Without this, non-target TLS would be stuck on phase 0 (action=0 = "keep current green phase", NOT "run default program").
- **AgentFilterWrapper** (`src/agent_filter.py`): PettingZoo parallel wrapper that exposes only the 5 target agents to the RL algorithm. On `reset()`, it calls `restore_non_target_programs()`. Non-target agents receive `default_action=0` which is ignored by their monkey-patched `set_next_phase`.
- **TimeEncodedObservationFunction** (`src/experiment.py`): Extended observation function that appends `sin(2π·t/86400)` and `cos(2π·t/86400)` to the base observation vector. This gives the agent a smooth, cyclical perception of time-of-day. The global `CURRENT_HOUR` variable in `experiment.py` must be set before each episode (it's set automatically by `experiment.py` and `train.py` based on the scenario). All training, evaluation, and schedule controller use this observation class.
- **`demand_math.py`** (`src/demand_math.py`): Bimodal 24h traffic curve. `get_vph(hour, total_daily_cars)` returns vehicles/hour at a given real-world hour. Two Gaussian peaks at 08:00 (morning rush, σ=1.5h) and 16:00 (evening rush, σ=2.0h) plus a small base noise. Numerically integrates the shape to guarantee exactly `total_daily_cars` vehicles in 24h. Used by curriculum learning and `generate_rush_demand.py`.
- **`generate_rush_demand.py`** (`src/generate_rush_demand.py`): Generates realistic rush-hour route files using the bimodal curve. Three scenarios: `morning_rush` (06:00-10:00, 4h, 70% inbound bias), `evening_rush` (14:00-18:00, 4h, 70% outbound bias), `offpeak` (12:00-13:00, 1h, symmetric). Inbound/outbound directional asymmetry is approximated by merging two trip batches with different `fringe_factor` values. Output files: `routes_morning_rush.rou.xml`, `routes_evening_rush.rou.xml`, `routes_offpeak.rou.xml`.
- **`schedule_controller.py`** (`src/schedule_controller.py`): Production deployment module. `ScheduleController` loads two PPO models (morning + evening) and dispatches to the correct one based on the hour of day. Rush hour windows (06:00-10:00, 14:00-18:00) use RL; all other times use fixed-time programs. Can run full simulation episodes with time-aware control via `run_episode()`.
- **`eval_helper.py`** (`src/eval_helper.py`): Subprocess helper for safe evaluation during curriculum learning. Called by `experiment.py` via `subprocess.run()` to run baseline and RL evaluation in separate SUMO processes (libsumo cannot be re-initialized in the same process).
- **Phase-based control**: agents select from predefined valid phase combinations (not individual lights)
- **Queue-length penalty** as primary reward function: `R(t) = -sum(halted_vehicles)`
- **Baseline**: `fixed_ts=True` in SumoEnvironment — runs real OSM signal programs untouched
- **LIBSUMO_AS_TRACI=1** for 5-8x simulation speedup (in-process vs socket IPC)

## Simulation Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_seconds` | 4200 | Total sim duration = `WARMUP_SECONDS` (600) + `RL_SECONDS` (3600) |
| `WARMUP_SECONDS` | 600 | 10 min of fixed-time SUMO before RL takes over (lets traffic build up) |
| `RL_SECONDS` | 3600 | Duration of RL-controlled phase per episode (1 hour) |
| `delta_time` | 5 | Seconds between agent decisions (action frequency) |
| `yellow_time` | 2 | Duration of yellow phase between green switches |
| `min_green` | 10 | Minimum green phase duration before switching |
| `max_green` | 90 | Maximum green phase duration before forced re-decision |
| `reward_fn` | "queue" | Reward = negative number of halted vehicles per step |
| `TOTAL_DAILY_CARS` | 40000 | Total vehicles in 24h bimodal curve (configurable in `config.py`) |

## PPO Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `learning_rate` | 0.001 | Gradient update step size |
| `n_steps` | 720 | Steps per agent before PPO update (= 1 full episode) |
| `batch_size` | 180 | Mini-batch size for gradient (3600 / 180 = 20 batches) |
| `n_epochs` | 10 | Number of passes over the rollout buffer per update |
| `gamma` | 0.99 | Discount factor (0=myopic, 1=infinite horizon) |
| `gae_lambda` | 0.95 | GAE smoothing between bias and variance |
| `ent_coef` | 0.05 | Entropy bonus — encourages exploration |
| `clip_range` | 0.2 | PPO clipping: limits policy change per update |

## Understanding Timesteps
One SB3 "timestep" = 5 seconds of simulated traffic for 1 traffic signal. Because 5 signals are vectorized via SuperSuit, SB3 counts 5 timesteps per SUMO step.

```
1 episode = num_seconds / delta_time * num_agents
         = 3600 / 5 * 5 = 3600 SB3 timesteps

n_steps = 720 → 720 * 5 agents = 3600 timesteps = exactly 1 episode per PPO update
```

| `--episode_count` | SB3 timesteps | Full episodes | PPO updates |
|-------------------|---------------|---------------|-------------|
| 10 | 36,000 | 10 | 10 |
| 50 | 180,000 | 50 | 50 |
| 100 | 360,000 | 100 | 100 |
| 500 | 1,800,000 | 500 | 500 |

## sumo-rl API Notes (v1.4.5)
- `SumoEnvironment(single_agent=False)` uses parallel dict API:
  - `reset()` → `{ts_id: obs_array, ...}`
  - `step(action_dict)` → `(obs_dict, reward_dict, done_dict, info)` (4 values, NOT 5)
  - `done_dict["__all__"]` signals episode end
  - `env.ts_ids` lists all traffic signal IDs
  - `env.traffic_signals[ts_id]` gives TrafficSignal object with `.observation_space` and `.action_space`
- `sumo_rl.parallel_env()` creates PettingZoo parallel env (for SuperSuit wrapping)
- `fixed_ts=True` runs default signal programs (no RL control) — use for baseline
- There is NO `ts_ids` parameter to select specific intersections. Control is filtered in the action loop.
- `sumo_warnings=False` suppresses TLS compatibility warnings
- **CRITICAL: action=0 gotcha** — In sumo-rl, action=0 means "keep current green phase", NOT "run the default signal program". sumo-rl's `build_phases()` replaces all original SUMO TLS programs on `TrafficSignal.__init__`. Sending action=0 to non-target TLS every 5s would freeze them permanently on phase 0 (one direction always green). This is why we need `tls_programs.py` to restore and protect original programs.

## Code Conventions
- Python code in `src/`, SUMO configs in `data/`, HPC scripts in `hpc/`
- All public-facing documents (README, comments visible to judges) must be in **Slovenian**
- Code comments and internal docs can be in English
- Trained models go to `models/` (gitignored, archived to Zenodo)
- Evaluation CSVs go to `results/` (committed to git)
- Experiment runs go to `results/experiments/<run_id>/` with meta.json, results.csv, training_log.csv, and model checkpoints

## File Paths
- SUMO network: `data/networks/ljubljana.net.xml`
- SUMO config: `data/networks/ljubljana.sumocfg`
- Routes: `data/routes/routes.rou.xml`
- Intersection config: `src/config.py` (TS_IDS, TS_NAMES, all sim/PPO params, rush-hour windows)
- Training: `src/train.py` (standalone, with `--scenario`), `src/experiment.py` (full pipeline)
- Evaluation: `src/evaluate.py` (multi-scenario: morning_rush, evening_rush, offpeak; outputs `results/rush_hour_comparison.csv`)
- Demand generation: `src/generate_demand.py` (uniform, rush_hour, double profiles)
- Rush-hour demand: `src/generate_rush_demand.py` (bimodal curve → morning_rush, evening_rush, offpeak route files)
- Bimodal traffic curve: `src/demand_math.py` (get_vph(hour, total_daily_cars))
- Schedule controller: `src/schedule_controller.py` (time-of-day dispatch: RL in rush hours, fixed-time otherwise)
- Curriculum eval helper: `src/eval_helper.py` (subprocess helper for safe baseline/RL eval during curriculum)
- Simulation analysis: `src/analyze_sim.py` (teleports, edge flows, trip stats)
- Dashboard: `src/dashboard.py` (generates results/dashboard.html)
- Custom rewards: `src/custom_reward.py`
- Agent filter: `src/agent_filter.py` (PettingZoo wrapper, filters to 5 target TLS, restores non-target programs)
- TLS program restoration: `src/tls_programs.py` (parses .net.xml, restores original signal programs via TraCI, monkey-patches non-target TrafficSignal objects)

## Development Environment
- Windows 11 + WSL2 Ubuntu (WSLg for SUMO GUI)
- Python 3.12 venv at `.venv/`
- SUMO_HOME="/usr/share/sumo"
- No NVIDIA GPU (Radeon) — GPU irrelevant, bottleneck is SUMO simulation

## Common Commands
```bash
# Activate environment
source .venv/bin/activate

# Enable fast SUMO (skip TraCI socket overhead)
export LIBSUMO_AS_TRACI=1

# Generate uniform traffic demand (for smoke tests)
python src/generate_demand.py --profile uniform --duration 3600 --peak_vph 800

# Generate rush-hour route files (required for rush-hour training/eval)
python src/generate_rush_demand.py --scenario all
# -> data/routes/routes_morning_rush.rou.xml, routes_evening_rush.rou.xml, routes_offpeak.rou.xml

# Run simulation sanity check
sumo -c data/networks/ljubljana.sumocfg
python src/analyze_sim.py

# Smoke test (quick, ~5min)
python src/experiment.py --episode_count 10 --tag smoke_test

# Train for 50 episodes on uniform demand (~4 min locally)
python src/experiment.py --episode_count 50 --tag local_50ep

# Train on morning rush scenario (requires generate_rush_demand.py first)
python src/experiment.py --scenario morning_rush --episode_count 100 --tag jutro_100ep

# Train with curriculum learning (random hour slices across full day)
python src/experiment.py --episode_count 200 --curriculum --tag curriculum_200ep

# Train with curriculum + per-episode logging (slow but detailed)
python src/experiment.py --episode_count 100 --curriculum --log_curriculum --tag curriculum_log

# Parallel training on N CPUs (for HPC)
python src/experiment.py --episode_count 500 --num_cpus 4 --tag hpc_500ep

# Resume training from checkpoint
python src/experiment.py --episode_count 100 --resume results/experiments/XXXXX/ppo_shared_policy.zip

# 1-hour training run (wall-time limit)
python src/experiment.py --max_hours 1.0 --tag 1h_local

# Raw timesteps (3600 per episode with 5 agents)
python src/experiment.py --total_timesteps 180000 --tag raw_50ep

# Compare all experiments
python src/experiment.py --compare_only

# Generate dashboard
python src/dashboard.py

# Evaluate model across all rush-hour scenarios
python src/evaluate.py --model results/experiments/XXXXX/ppo_shared_policy.zip
# -> results/rush_hour_comparison.csv, results/comparison_summary.csv

# Evaluate only one scenario
python src/evaluate.py --model models/ppo_morning_rush_final.zip --scenario morning_rush

# Standalone train.py (no baseline/eval phases)
python src/train.py --scenario morning_rush --total_timesteps 500000
python src/train.py --scenario morning_rush --resume models/ppo_morning_rush_10000_steps.zip

# Open SUMO GUI
sumo-gui data/networks/ljubljana.sumocfg

# Open network editor
netedit data/networks/ljubljana.net.xml

# Submit to Vega HPC
sbatch hpc/submit_train.sh
```

## References
- Execution plan: `Zeleni_SignaLJ_Execution_Plan.docx` (39 references, 18 sections)
- sumo-rl docs: https://github.com/LucasAlegre/sumo-rl
- SUMO docs: https://sumo.dlr.de/docs/
- stable-baselines3: https://stable-baselines3.readthedocs.io/
- SuperSuit: https://github.com/Farama-Foundation/SuperSuit
