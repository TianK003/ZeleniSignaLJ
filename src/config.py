"""
Zeleni SignaLJ - Project Configuration
=======================================
All simulation, training, and intersection parameters in one place.
Change values here — they propagate to experiment.py, train.py, evaluate.py.
"""

# ══════════════════════════════════════════
# Target intersections
# ══════════════════════════════════════════
# 1. Tivolska / Slovenska / Dunajska / Trg OF
# 2. Bleiweisova / Tivolska / Celovska / Gosposvetska
# 3. Slovenska / Gosposvetska / Dalmatinova
# 4. Askerceva / Presernova / Groharjeva
# 5. Askerceva / Zoisova / Slovenska / Barjanska

TS_IDS = [
    "joinedS_5154793231_8093399326_8093399327_8093399328_#7more",
    "joinedS_1951535395_8569909625_8569909627_8569909629_#5more",
    "joinedS_cluster_10946184173_33632882_4083612498_4898978366_#5more_cluster_4898978371_9307230471_9307230472",
    "joinedS_16191121_311397806_476283378_6264081028_#12more",
    "joinedS_8241154017_8241154018_cluster_1632640893_3884437221_3884437224_4312381314_#8more_cluster_8171896855_8171896868_8241143312",
]

# Map from human-readable names to TLS IDs (for logging/eval)
TS_NAMES = {
    TS_IDS[0]: "Kolodvor",
    TS_IDS[1]: "Pivovarna",
    TS_IDS[2]: "Slovenska",
    TS_IDS[3]: "Trzaska",
    TS_IDS[4]: "Askerceva",
}

# ══════════════════════════════════════════
# SUMO simulation parameters
# ══════════════════════════════════════════
WARMUP_SECONDS = 600  # 10 minutes of mechanical OSM simulation before RL takes over
RL_SECONDS = 3600     # Episode duration (1 hour of RL-controlled traffic)
NUM_SECONDS = RL_SECONDS + WARMUP_SECONDS    # Total simulation time
DELTA_TIME = 5        # Seconds between agent decisions
YELLOW_TIME = 2       # Yellow phase duration between green switches
MIN_GREEN = 10        # Minimum green phase before switching allowed
MAX_GREEN = 90        # Maximum green phase before forced re-decision
REWARD_FN = "queue"   # Reward = negative halted vehicles per step
TOTAL_DAILY_CARS = 40000 # Configurable number of total cars in a simulated 24h day
CURRICULUM_BASE_NOISE = 0.05
CURRICULUM_SIGMA_1 = 1.5
CURRICULUM_SIGMA_2 = 2.0

# ══════════════════════════════════════════
# PPO hyperparameters
# ══════════════════════════════════════════
LEARNING_RATE = 1e-3  # Gradient update step size
N_STEPS = 720         # Steps per agent before PPO update (= 1 full episode)
BATCH_SIZE = 180      # Mini-batch size (divides evenly into N_STEPS * NUM_AGENTS)
N_EPOCHS = 10         # Passes over rollout buffer per PPO update
GAMMA = 0.99          # Discount factor (0=myopic, 1=infinite horizon)
GAE_LAMBDA = 0.95     # GAE smoothing between bias and variance
ENT_COEF = 0.05       # Entropy bonus — encourages exploration
CLIP_RANGE = 0.2      # PPO clipping: limits policy change per update

# ══════════════════════════════════════════
# Derived constants (do not edit)
# ══════════════════════════════════════════
NUM_AGENTS = len(TS_IDS)
STEPS_PER_EPISODE = (RL_SECONDS // DELTA_TIME) * NUM_AGENTS
