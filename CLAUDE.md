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
- **TimeEncodedObservationFunction** (`src/experiment.py`): Extended observation function that appends `sin(2π·t/86400)` and `cos(2π·t/86400)` to the base observation vector. This gives the agent a smooth, cyclical perception of time-of-day. The global `CURRENT_HOUR` variable in `experiment.py` must be set before each episode (it's set automatically by `experiment.py` based on the scenario). All training, evaluation, and schedule controller use this observation class.
- **`demand_math.py`** (`src/demand_math.py`): Bimodal 24h traffic curve. `get_vph(hour, total_daily_cars)` returns vehicles/hour at a given real-world hour. Two Gaussian peaks at 08:00 (morning rush, σ=1.5h) and 16:00 (evening rush, σ=2.0h) plus a small base noise. Numerically integrates the shape to guarantee exactly `total_daily_cars` vehicles in 24h. Used by curriculum learning and `generate_demand.py`.
- **`generate_demand.py`** (`src/generate_demand.py`): Unified demand generator. Two modes: `--profile uniform` for constant-rate smoke tests, and `--scenario morning_rush|evening_rush|offpeak|all` for realistic rush-hour route files using the bimodal curve. Rush scenarios have directional asymmetry (70% inbound morning / 70% outbound evening) via two trip batches with different `fringe_factor` values. Output files: `routes_morning_rush.rou.xml`, `routes_evening_rush.rou.xml`, `routes_offpeak.rou.xml`. Core function `write_demand_xml()` is imported by `experiment.py`.
- **`schedule_controller.py`** (`src/schedule_controller.py`): Production deployment module. `ScheduleController` loads two PPO models (morning + evening) and dispatches to the correct one based on the hour of day. Rush hour windows (06:00-10:00, 14:00-18:00) use RL; all other times use fixed-time programs. Can run full simulation episodes with time-aware control via `run_episode()`.
- **`eval_helper.py`** (`src/eval_helper.py`): Subprocess helper for safe evaluation during curriculum learning. Called by `experiment.py` via `subprocess.run()` to run baseline and RL evaluation in separate SUMO processes (libsumo cannot be re-initialized in the same process).
- **Interpretability pipeline** (`src/collect_states.py` + `src/explain.py`): Two-step post-hoc analysis of trained models.
  - **`collect_states.py`**: Runs the PPO model (forced to CPU for speed) through N episodes (recommended 10-15, default 12) with randomized hours within the scenario window. Harvests observations, actions, 64-dim latent layer activations, metadata, and **phase definitions** (green phase state strings + controlled links per intersection via TraCI) into `harvested_data.pkl`. Phase info is collected after the first `env.reset()` (TraCI must be live). Supports `--scenario morning_rush|evening_rush|offpeak|uniform` for single-model harvesting and `--megapolicy --model_morning --model_evening` for schedule-controller dispatch. Episode duration capped at 1h regardless of scenario to keep harvest fast.
  - **`explain.py`**: Consumes the pickle and generates three types of visualizations per intersection, all using human-readable names (Kolodvor, Pivovarna, etc.) and Slovenian labels:
    1. **Surrogate Decision Trees** — depth-4 `DecisionTreeClassifier` fit per intersection. Custom renderer (not sklearn's `plot_tree`) with compact subtree-width-based layout. Leaf nodes show which approach directions get green (e.g. "Smer 1 + Smer 3") determined by analyzing phase state strings — links are grouped by source edge, filtered to signalized-only (exclude passthrough), clustered by co-activation pattern. Internal nodes show split conditions with shortened Slovenian feature names. "Pokritost PPO: N%" badge shows tree fidelity.
    2. **SHAP beeswarm plots** — `TreeExplainer` on the surrogate tree, with class names mapped to approach direction labels. Feature names shortened to Slovenian (e.g. "Vrsta pas 2", "Gostota pas 0", "sin(čas)").
    3. **UMAP projections** — 64-dim latent activations projected to 2D, colored by action, time-of-day, and intersection. Axes hidden (dimensions are abstract — no physical meaning). **How to interpret:** closer points = similar internal PPO representations = similar traffic situations. Distinct color clusters = model learned to separate those conditions internally. Mixed colors = model doesn't differentiate along that dimension. Action plot shows if decisions form distinct clusters; time plot shows if model distinguishes rush-hour vs off-peak; intersection plot shows if shared policy differentiates between locations.
  - Observations are zero-padded to max obs size across all 5 intersections; `explain.py` trims each intersection's matrix to its actual feature count before fitting.
  - Output directory structure: `explanations/{category}/{intersection_or_method}.{png,json}`. Categories: `decision-trees/` (5 per-intersection), `shap/` (5 per-intersection), `umap/` (6: 3 colorings x with/without PCA), `t-sne/` (6: same). Each image has a companion JSON with metadata (fidelity, feature importance, embedding coords, etc.).
  - Example: `explanations/decision-trees/kolodvor.png`, `explanations/shap/trzaska.png`, `explanations/umap/umap-actions.png`, `explanations/t-sne/pca-tsne-time.png`.
- **Route randomization** (`--route_dir` in experiment.py): Training on a single route file causes OD-pair overfitting — the model learns policies specific to the training traffic's origin-destination patterns and fails on unseen routes. `--route_dir` points to a directory of route file variants (generated by `generate_demand.py --scenario X --num_variants N`). Each training episode copies a random route file to the training path and forces an env reset, exposing the model to diverse traffic patterns. Compatible with `--curriculum` for combined hour + route randomization. The mechanism piggybacks on the existing curriculum code path (`model._last_obs = None` to force SB3 reset, which makes sumo-rl re-read the route file).
- **Rush-hour generalization test** (`src/run_rush_test.py`): Isolated 4h rush-hour evaluation with 50 different route files. Tests whether a trained model generalizes across random OD pairs WITHOUT the 24h cascading-error problem (where morning rush gridlock contaminates all subsequent windows). Each run is an independent episode with fresh traffic state. Outputs summary.csv for paired statistical comparison (RL vs baseline on the same route file). This is the correct way to measure model quality — the 24h megapolicy test conflates model quality with compounding error sensitivity.
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
- Training: `src/experiment.py` (full pipeline: baseline → train → eval, with `--scenario`, `--curriculum`, `--num_cpus`)
- Evaluation: `src/evaluate.py` (multi-scenario: morning_rush, evening_rush, offpeak; outputs `results/rush_hour_comparison.csv`)
- Demand generation: `src/generate_demand.py` (uniform profile + rush-hour scenarios via bimodal curve)
- Bimodal traffic curve: `src/demand_math.py` (get_vph(hour, total_daily_cars))
- Schedule controller: `src/schedule_controller.py` (time-of-day dispatch: RL in rush hours, fixed-time otherwise)
- Curriculum eval helper: `src/eval_helper.py` (subprocess helper for safe baseline/RL eval during curriculum)
- Rush-hour generalization test: `src/run_rush_test.py` (isolated rush-hour statistical testing — 50 runs with different route files to test model generalization without 24h cascading errors)
- Simulation analysis: `src/analyze_sim.py` (teleports, edge flows, trip stats)
- Dashboard: `src/dashboard.py` (generates results/dashboard.html with 6 tabs including Mega-politike statistical comparison)
- 24h runner: `src/run_24h.py` (24h simulation with dynamic RL/fixed-time switching + multiprocessing)
- Custom rewards: `src/custom_reward.py`
- Agent filter: `src/agent_filter.py` (PettingZoo wrapper, filters to 5 target TLS, restores non-target programs)
- TLS program restoration: `src/tls_programs.py` (parses .net.xml, restores original signal programs via TraCI, monkey-patches non-target TrafficSignal objects)
- State harvester: `src/collect_states.py` (runs trained PPO model on CPU, records obs/actions/latents/phase_info to `harvested_data.pkl`; `--scenario` for single-model, `--megapolicy` for dual-model dispatch)
- Interpretability: `src/explain.py` (from `harvested_data.pkl`: surrogate decision trees with approach-direction labels, SHAP beeswarm plots, UMAP latent projections; all in Slovenian with human-readable intersection names)

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
python src/generate_demand.py --scenario all
# -> data/routes/routes_morning_rush.rou.xml, routes_evening_rush.rou.xml, routes_offpeak.rou.xml

# Run simulation sanity check
sumo -c data/networks/ljubljana.sumocfg
python src/analyze_sim.py

# Smoke test (quick, ~5min)
python src/experiment.py --episode_count 10 --tag smoke_test

# Train for 50 episodes on uniform demand (~4 min locally)
python src/experiment.py --episode_count 50 --tag local_50ep

# Train on morning rush scenario (requires generate_demand.py --scenario first)
python src/experiment.py --scenario morning_rush --episode_count 100 --tag jutro_100ep

# Train with curriculum learning (random hour slices across full day)
python src/experiment.py --episode_count 200 --curriculum --tag curriculum_200ep

# Train with curriculum + per-episode logging (slow but detailed)
python src/experiment.py --episode_count 100 --curriculum --log_curriculum --tag curriculum_log

# Train with route randomization (diverse OD pairs for generalization)
# Step 1: Generate N route variants for the scenario
python src/generate_demand.py --scenario morning_rush --num_variants 20 --output_dir data/routes/morning_rush_variants
# Step 2: Train with route rotation
python src/experiment.py --scenario morning_rush --episode_count 200 --route_dir data/routes/morning_rush_variants --tag morning_routerand_200ep

# Combine route randomization with curriculum
python src/experiment.py --episode_count 200 --curriculum --route_dir data/routes/morning_rush_variants --tag curriculum_routerand

# Parallel training on N CPUs (for HPC)
python src/experiment.py --episode_count 500 --num_cpus 4 --tag hpc_500ep

# Resume training from checkpoint
python src/experiment.py --episode_count 100 --resume results/experiments/XXXXX/ppo_shared_policy.zip

# 1-hour training run (wall-time limit)
python src/experiment.py --max_hours 1.0 --tag 1h_local

# Raw timesteps (3600 per episode with 5 agents)
python src/experiment.py --total_timesteps 180000 --tag raw_50ep

# Interpretability: harvest states from a morning rush model
python src/collect_states.py --model_path results/experiments/XXXXX/ppo_shared_policy.zip --scenario morning_rush --episodes 12

# Interpretability: harvest states from an evening rush model
python src/collect_states.py --model_path results/experiments/XXXXX/ppo_shared_policy.zip --scenario evening_rush --episodes 12

# Interpretability: harvest states from a megapolicy (both models + schedule controller)
python src/collect_states.py --megapolicy --model_morning <morning>.zip --model_evening <evening>.zip --episodes 12 --output_dir results/megapolicy_explain/

# Interpretability: generate SHAP, decision tree, and UMAP visualizations
python src/explain.py --data_path results/experiments/XXXXX/harvested_data.pkl
# -> results/experiments/XXXXX/explanations/{tree_*.png, shap_*.png, umap_*.png}

# Compare all experiments
python src/experiment.py --compare_only

# Generate dashboard
python src/dashboard.py

# Evaluate model across all rush-hour scenarios
python src/evaluate.py --model results/experiments/XXXXX/ppo_shared_policy.zip
# -> results/rush_hour_comparison.csv, results/comparison_summary.csv

# Evaluate only one scenario
python src/evaluate.py --model models/ppo_morning_rush_final.zip --scenario morning_rush

# Rush-hour generalization test (isolated 4h, no 24h cascading errors)
# Step 1: Generate route variants
python src/generate_demand.py --scenario morning_rush --num_variants 50 --output_dir data/routes/rush-test-morning
python src/generate_demand.py --scenario evening_rush --num_variants 50 --output_dir data/routes/rush-test-evening
# Step 2: Run baseline + model tests
python src/run_rush_test.py --baseline --scenario morning_rush --route_dir data/routes/rush-test-morning --num_runs 50 --num_workers 10 --output_dir results/rush-test/baseline_morning
python src/run_rush_test.py --model <model.zip> --scenario morning_rush --route_dir data/routes/rush-test-morning --num_runs 50 --num_workers 10 --output_dir results/rush-test/M1_morning
# HPC: generate all SLURM scripts
python hpc/statistical-test/generate_rush_jobs.py
bash hpc/statistical-test/submit_rush.sh

# Open SUMO GUI
sumo-gui data/networks/ljubljana.sumocfg

# Open network editor
netedit data/networks/ljubljana.net.xml

# Submit to Vega HPC
sbatch hpc/sweep/submit_train.sh
```

## HPC Experiment Sweeps

### Sweep 1 (completed): Single-route training

First sweep: 24 configurations x 200 episodes on 128 CPUs. Trained on a single fixed route file per scenario. Models showed 12-18% improvement on the training route but failed to generalize to unseen traffic patterns (OD-pair overfitting confirmed by rush-hour generalization tests showing -15% to -27% degradation on random routes).

**Matrix:** 3 rewards x 2 LRs x 2 scenarios + entropy annealing + curriculum = 24 jobs.

### Sweep 2 (current): Route-randomized training

44 SLURM scripts in `hpc/sweep/` generated by `hpc/sweep/generate_jobs.py`. All use `--route_dir` with 50 route variants per scenario to prevent OD-pair overfitting. With 300 episodes and random.choice(), each route gets ~6 exposures — enough for the policy to consolidate patterns without memorizing individual routes.

**Matrix:** 3 reward functions (queue, pressure, diff-waiting-time) x 3 learning rates (3e-3, 1e-3, 3e-4) x 2 scenarios = 18 base + 18 entropy annealing + 8 entropy coefficient sweep (pressure only, ent_coef 0.02/0.1 — 0.05 is default, already in base sweep) = **44 jobs**.

**Two-phase pipeline:**
1. `gen_train_routes.slurm` — generates 50 morning + 50 evening route variants
2. All 44 training scripts wait for routes, then run in parallel (128 CPUs, 300 episodes, 24h timeout)

**Checkpoints:** Every 10 episodes, saved as `checkpoints/ppo_policy_10ep.zip`, `ppo_policy_20ep.zip`, etc. with companion `.json` metadata (episode, timesteps, avg reward, baseline comparison). Emergency `ppo_model_latest.zip` always saved via try/finally even on crash/timeout. Final model: `ppo_shared_policy.zip`.

**CLI flags in `experiment.py`:**
- `--reward_fn {queue,pressure,diff-waiting-time,average-speed}` — overrides config.py REWARD_FN
- `--learning_rate FLOAT` — overrides config.py LEARNING_RATE
- `--route_dir DIR` — directory of route file variants for route-randomized training (each episode picks a random route file, training on diverse OD pairs for better generalization)
- `--ent_coef FLOAT` — overrides config.py ENT_COEF
- `--entropy_annealing` — linearly anneal entropy from ent_coef to 0.01 over training
- `--episodes_per_save N` — checkpoint frequency (default: 10)

**Submission:** `bash hpc/sweep/submit_all.sh [--skip-routes] [filter]` — filter by reward name, LR, etc.

All overridden hyperparams are saved in `meta.json` so the dashboard shows them correctly.

## Top Policy Selection & Mega-Policies (Sweep 1 results)

After the first HPC sweep (24 configurations x 200 episodes each on 128 CPUs, single-route training), the top models were selected by overall improvement % over fixed-time baseline (as reported by the dashboard). **Note:** these models overfit to the training route and do not generalize to unseen traffic patterns. Sweep 2 with route randomization aims to fix this.

**Top 3 Morning Rush (06:00-10:00):**
| Rank | Tag | Reward Fn | LR | Improvement |
|------|-----|-----------|-----|------------|
| M1 | `morningrush_diffwaitingtime_lr1e3_200ep` | diff-waiting-time | 1e-3 | +18.2% |
| M2 | `morningrush_pressure_lr1e3_200ep` | pressure | 1e-3 | +17.2% |
| M3 | `morningrush_default_lr3e4_200ep` | queue | 3e-4 | +17.1% |

**Top 3 Evening Rush (14:00-18:00):**
| Rank | Tag | Reward Fn | LR | Extras | Improvement |
|------|-----|-----------|-----|--------|------------|
| E1 | `eveningrush_pressure_lr1e3_entanneal_200ep` | pressure | 1e-3 | entropy annealing | +15.1% |
| E2 | `eveningrush_diffwaitingtime_lr1e3_entanneal_200ep` | diff-waiting-time | 1e-3 | entropy annealing | +15.0% |
| E3 | `eveningrush_pressure_lr3e4_entanneal_200ep` | pressure | 3e-4 | entropy annealing | +12.9% |

**Mega-Policy Concept:** A mega-policy combines one morning model + one evening model with the schedule controller. During rush hours, the corresponding RL model controls the 5 target TLS. Outside rush hours, all TLS run their original SUMO fixed-time programs. This gives 3 x 3 = 9 mega-policy combinations (M1E1 through M3E3).

## Statistical Testing (24h Simulations)

To validate mega-policies with statistical rigor, we run **50 replications** of each mega-policy (+ baseline) on full 24h simulations using a **matched-pairs design** with per-seed random route files.

**Design (matched-pairs):**
- 10 conditions: 9 mega-policies + 1 baseline (all fixed-time)
- 50 runs per condition, each with a unique SUMO seed (1-50)
- **Per-seed route files**: each seed N uses `routes_full_day_seed_{N-1:02d}.rou.xml` — generated with a different `master_seed` in `randomTrips.py`, giving different OD pairs while preserving the same bimodal demand curve
- Same route file for all 10 conditions at each seed → fair paired comparison
- Different route files across seeds → meaningful variance in traffic patterns
- Full 24h bimodal demand with directional asymmetry (70/30 morning inbound, 70/30 evening outbound)
- **Statistical tests**: paired t-test (`ttest_rel`) and Wilcoxon signed-rank on per-seed differences, plus Cohen's d for effect size

**Why matched-pairs?** Previous approach used a single deterministic route file for all runs — SUMO seeds only vary microsimulation noise (gap acceptance, speed jitter), not OD patterns. With `model.predict(deterministic=True)`, this produced near-zero variance. The matched-pairs design with per-seed routes gives real variance while controlling for traffic pattern (same route for baseline and megapolicy at each seed).

**Two-phase HPC pipeline:**
1. `gen_routes.slurm` — generates 50 route files via `generate_demand.py --statistical_routes 50`
2. `mega_*.slurm` — each runs 50 replications using `--route_dir data/routes/statistical-test`

**Baseline uses `fixed_ts=False`**: Both baseline and megapolicy use `fixed_ts=False` with manual program restoration via `_restore_all_programs()`. This ensures identical sumo-rl code paths — the only difference is whether RL activates during rush hours. (Previous baseline used `fixed_ts=True` which takes a different path through sumo-rl's `step()` function.)

**Dynamic RL/Fixed-Time Switching** (`src/run_24h.py`):
The 24h simulation runs a single continuous SUMO environment for 86400 seconds. At each step, the script checks the simulation hour and switches target TLS between RL control and fixed-time control:
- Fixed-time: restores original SUMO programLogic via TraCI, **re-activates automatic cycling via `setProgram()`**, then patches `set_next_phase` to a passthrough (keeps timing alive but doesn't override SUMO's program)
- RL: restores sumo-rl's phase program, un-patches methods, re-syncs TLS state

**CRITICAL: `setProgram()` after `setProgramLogic()`** — sumo-rl's `_build_phases()` calls `setRedYellowGreenState()` which puts TLS into SUMO manual mode. `setProgramLogic()` alone only updates phase definitions but does NOT re-activate automatic phase cycling. Without the `setProgram()` call, TLS get stuck on a single phase permanently. This bug appeared in two places:
1. **Target TLS in `run_24h.py`** (`_switch_to_fixed_time`): Fixed — caused ~5% overall degradation.
2. **Non-target TLS in `tls_programs.py`** (`restore_non_target_programs`): Fixed — caused 32 non-target TLS to be stuck for the entire simulation, resulting in -187% degradation during the night_0006 window (00:00-06:00) where no preceding RL period could compensate. The same fix was also applied to `agent_filter.py`'s warmup loop.

**Time Encoding Compatibility:** With `CURRENT_HOUR = 0.0`, the observation's `time_seconds = sim_step`. At sim_step=21600 (6AM), this matches exactly what the morning model saw during training (CURRENT_HOUR=6.0, sim_step=0). Same for evening model at 14:00.

**Metrics Collected Per Run:**
- Total cumulative reward (all intersections, all timesteps)
- Per-intersection reward, avg queue, avg waiting time
- Per-time-window breakdown (night, morning rush, shoulder day, evening rush, shoulder evening)
- Total teleports, vehicles departed/arrived

**Statistical Analysis:** From 50 paired runs: mean, median, std, 95% CI, paired t-test vs baseline, Wilcoxon signed-rank, Cohen's d (paired).

**Files:**
- `src/run_24h.py` — 24h simulation runner with dynamic switching + multiprocessing
- `src/generate_demand.py --statistical_routes 50` — generates 50 per-seed route files
- `src/generate_demand.py --scenario full_day` — generates single 24h route file
- `hpc/statistical-test/generate_mega_jobs.py` — generates 11 SLURM scripts (1 route gen + 10 simulation jobs, 50 runs each)
- `hpc/statistical-test/gen_routes.slurm` — route generation job (phase 1)
- `hpc/statistical-test/submit_all.sh` — submits all jobs with SLURM dependency chain
- `hpc/statistical-test/mega_*.slurm` — 9 mega + 1 baseline SLURM scripts

**Commands:**
```bash
# Generate 50 per-seed route files for statistical testing
python src/generate_demand.py --statistical_routes 50

# Generate SLURM scripts (route gen + simulation jobs)
python hpc/statistical-test/generate_mega_jobs.py

# Submit all (auto-generates routes first, then simulations)
bash hpc/statistical-test/submit_all.sh

# Or skip route gen if routes already exist
bash hpc/statistical-test/submit_all.sh --skip-routes

# Local smoke test
python src/run_24h.py --baseline --num_runs 2 --num_workers 1 \
    --route_dir data/routes/statistical-test --output_dir /tmp/test_baseline

# Results: results/statistical-test/{M1E1,...,M3E3,baseline}/summary.csv
```

**Estimated HPC Time:** Route generation ~100 min (sequential, 50 files). Simulation ~30-60 min per job (50 runs with 50 parallel workers on 64 CPUs), 8h wall time requested.

## Dashboard

`src/dashboard.py` generates `results/dashboard.html` with 8 tabs. The first 3 tabs provide scenario-specific views.

**Tab structure:**
1. **Jutranja konica (Morning Rush)** — comparison table + charts filtered to morning rush experiments only
2. **Vecerna konica (Evening Rush)** — comparison table + charts filtered to evening rush experiments only
3. **Mega-politike** — 24h statistical test results from `results/statistical-test/`
4. **Krizisca** — per-intersection breakdown across all experiments
5. **Ucenje** — training curves
6. **Hiperparametri** — hyperparameter comparison
7. **Podrobnosti** — experiment detail viewer
8. **Interpretibilnost** — SHAP beeswarm, decision tree, UMAP visualizations

**Mega-politike tab (paired statistics):**
- `load_megapolicy_results()` reads all 10 `summary.csv` files, aligns data by seed using `pd.merge(..., on="seed")`, computes descriptive statistics (mean, median, std, 95% CI via t-distribution) per condition, and runs **paired t-test** (`ttest_rel`), **Wilcoxon signed-rank**, and **paired Cohen's d** for each megapolicy vs baseline
- Tab sections: KPI cards, overall comparison bar chart with CI error bars, 3x3 heatmap with significance stars, per-intersection breakdown, per-time-window breakdown, full statistical significance table, per-megapolicy drill-down

**Chart.js error bars:** Implemented via a custom inline plugin (`errorBarPlugin`) that draws CI whisker lines on bar charts. No external dependency needed.

**Graceful degradation:** If `results/statistical-test/` does not exist, the tab is not rendered. Existing tabs (0-4) are unaffected.

## TODO: Visualization & Demo Plan

### Day 1: HPC Experiments
- [ ] Generate morning rush routes: `python src/generate_demand.py --scenario morning_rush`
- [ ] Submit initial sweep: `bash hpc/sweep/submit_all.sh 100ep` (all 100-episode variants)
- [ ] Check results, identify best reward function
- [ ] Submit longer runs (250/500ep) with the winning reward function
- [ ] Submit entropy annealing variants: `bash hpc/sweep/submit_all.sh entanneal`

### Day 2: Analysis & Refinement
- [ ] Download results from Vega
- [ ] Generate dashboard: `python src/dashboard.py`
- [ ] Analyze per-intersection performance (which improved, which degraded)
- [ ] If results are weak: try lower learning rate (3e-4) or different reward
- [ ] Submit final production runs (500ep, best config, morning + evening rush)
- [ ] Start writing report (methodology, architecture, results)
- [ ] Pre-record SUMO GUI demo video: `python src/evaluate.py --gui --model <best>.zip --scenario morning_rush`

### Day 3: Report & Demo Prep
- [ ] Run evaluate.py on best model across all 3 scenarios (morning, evening, offpeak)
- [ ] Generate final dashboard with all results
- [ ] Finalize report with dashboard screenshots and convergence charts
- [ ] Prepare presentation: map -> architecture -> dashboard -> SUMO demo -> conclusion
- [ ] Test live SUMO GUI demo on presentation hardware
- [ ] Backup: ensure pre-recorded video works if SUMO GUI fails on projector

### Demo Layers (for judges)
1. **Dashboard** (primary, 3-4 min): `results/dashboard.html` — comparison, intersections, training curves, hyperparameters
2. **SUMO GUI** (secondary, 1-2 min): live simulation with best RL model vs. fixed-time baseline
3. **Map context** (30s): `data/media/Observed_intersections.png` — explain the 5 intersections
4. **Summary slide** (1 min): table of scenario x reward function x improvement %

### Key Metrics to Report
- Overall improvement % (RL vs. fixed-time baseline)
- Per-intersection improvement breakdown
- Best reward function and why
- Training convergence (episodes needed, wall time on HPC)
- Queue length reduction during morning/evening rush

## References
- Execution plan: `Zeleni_SignaLJ_Execution_Plan.docx` (39 references, 18 sections)
- sumo-rl docs: https://github.com/LucasAlegre/sumo-rl
- SUMO docs: https://sumo.dlr.de/docs/
- stable-baselines3: https://stable-baselines3.readthedocs.io/
- SuperSuit: https://github.com/Farama-Foundation/SuperSuit
