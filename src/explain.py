"""
Zeleni SignaLJ - Interpretability Module
========================================
Uses the harvested data to generate:
1. SHAP Beeswarm Summary Plots (Feature Attribution)
2. UMAP/t-SNE Scatter Plots (Latent Space Regimes)
3. Decision Tree Flowcharts (Policy Distillation)

Outputs are saved as PNGs in the experiment's explanations/ folder.
"""

import argparse
import os
import sys

# ── SUMO_HOME Detection (HPC Compatibility) ────────────────────────────────
if "SUMO_HOME" not in os.environ:
    for path in [
        os.path.join(os.environ.get("HOME", ""), "sumo_src"),
        "/usr/share/sumo",
        "/usr/local/share/sumo",
        "C:\\Program Files (x86)\\Eclipse\\Sumo" if sys.platform == "win32" else ""
    ]:
        if path and os.path.exists(path):
            os.environ["SUMO_HOME"] = path
            break

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import shap
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import umap

def generate_explanations(data_path):
    run_dir = os.path.dirname(data_path)
    output_dir = os.path.join(run_dir, "explanations")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        
    obs = np.array(data["observations"])
    actions = np.array(data["actions"])
    latents = np.array(data["latents"])
    metadata = pd.DataFrame(data["metadata"])
    feature_maps = data["feature_maps"]
    ts_ids = data["ts_ids"]

    # --- 1. SHAP Feature Attribution ---
    print("  Generating SHAP explanations...")
    # Use a single agent's feature map as a template for global importance
    # (Since the model uses parameter sharing, we can aggregate)
    for ts_id in ts_ids:
        print(f"    Processing {ts_id}...")
        ts_indices = metadata[metadata["ts_id"] == ts_id].index
        if len(ts_indices) == 0: continue
        
        ts_obs = obs[ts_indices]
        f_names = feature_maps[ts_id]
        
        # We'll explain the actions. Discrete actions are tricky, 
        # so we'll treat it as a multi-output regression on logits or 
        # just sample the dominant class.
        # For simplicity and hackathon speed, we use a KernelExplainer on a subset
        def predict_action(x):
            # This is a dummy for now since we have the actions, but SHAP needs a function
            # Real SHAP would use the model, but here we'll use a local surrogate or 
            # similar if we don't want to load the whole model again.
            pass

        # Use SHAP DeepExplainer if possible, but let's stick to simple Tree/Kernel 
        # for harvested data or just summarize the feature correlations.
        # Actually, let's use a simpler approach for the hackathon: 
        # Train a surrogate XGBoost/RandomForest to get SHAP values if needed, 
        # but the Decision Tree pillar covers that rule-based logic.
        
        # Let's do a real SHAP Summary plot on the observations and actions
        # We will use the TreeExplainer on the Decision Tree we build anyway!
        
    # --- 2. Policy Distillation (Decision Trees) ---
    print("  Distilling policy into Decision Trees...")
    for ts_id in ts_ids:
        ts_indices = metadata[metadata["ts_id"] == ts_id].index
        if len(ts_indices) == 0: continue
        
        X = obs[ts_indices]
        y = actions[ts_indices]
        f_names = feature_maps[ts_id]
        
        # Fit tree
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(X, y)
        
        # Plot
        plt.figure(figsize=(20, 10))
        plot_tree(clf, feature_names=f_names, class_names=True, filled=True, rounded=True, fontsize=10)
        plt.title(f"Decision Logic: {ts_id}")
        plt.savefig(os.path.join(output_dir, f"tree_{ts_id}.png"), bbox_inches='tight', dpi=150)
        plt.close()
        
        # Now use SHAP on this tree!
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        
        plt.figure(figsize=(12, 8))
        # Handle multi-class shap_values
        if isinstance(shap_values, list):
            # For each action, show importance
            shap.summary_plot(shap_values, X, feature_names=f_names, show=False)
        else:
            shap.summary_plot(shap_values, X, feature_names=f_names, show=False)
            
        plt.title(f"Feature Importance (SHAP): {ts_id}")
        plt.savefig(os.path.join(output_dir, f"shap_{ts_id}.png"), bbox_inches='tight', dpi=150)
        plt.close()

    # --- 3. Latent Space Visualization (UMAP) ---
    print("  Generating UMAP latent space projections...")
    
    # UMAP on all latents
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(latents)
    
    # Plot 1: Colored by Action
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=actions, palette="viridis", s=10, alpha=0.5)
    plt.title("Latent Space: Colored by Action Choice")
    plt.savefig(os.path.join(output_dir, "umap_actions.png"), dpi=150)
    plt.close()
    
    # Plot 2: Colored by Time of Day (Rush Hour Phases)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=metadata["hour"], palette="coolwarm", s=10, alpha=0.5)
    plt.title("Latent Space: Colored by Time of Day (Rush Hour Regimes)")
    plt.savefig(os.path.join(output_dir, "umap_time.png"), dpi=150)
    plt.close()

    print(f"  All explanations generated in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to harvested_data.pkl")
    args = parser.parse_args()
    
    generate_explanations(args.data_path)
