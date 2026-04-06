import sys
import json
import warnings
warnings.filterwarnings('ignore')

from experiment import run_baseline, run_evaluation
from stable_baselines3 import PPO

mode = sys.argv[1]
net_file = sys.argv[2]
route_file = sys.argv[3]
num_seconds = int(sys.argv[4])
out_json = sys.argv[5]

if mode == "baseline":
    rewards, _ = run_baseline(net_file, route_file, num_seconds)
    with open(out_json, "w") as f: 
        json.dump(rewards, f)
        
elif mode == "evaluate":
    model_path = sys.argv[6]
    model = PPO.load(model_path)
    rewards, _ = run_evaluation(net_file, route_file, num_seconds, model)
    with open(out_json, "w") as f: 
        json.dump(rewards, f)
