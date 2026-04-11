"""
Zeleni SignaLJ - Interpretability Module
========================================
Uses the harvested data to generate:
1. SHAP Beeswarm Summary Plots (Feature Attribution via surrogate Decision Tree)
2. UMAP Scatter Plots (Latent Space Regimes)
3. Decision Tree Flowcharts (Policy Distillation)

Outputs are saved as PNGs in the experiment's explanations/ folder.
All intersections are labeled with human-readable names (Kolodvor, Pivovarna, etc.).
Phase actions are described by which approach directions get green.

Requires harvested_data.pkl from collect_states.py as input.
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

import json
import pickle
from collections import deque
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier, export_text
import umap

from config import TS_NAMES


def _save_json(path, data):
    """Save raw data alongside a plot as .json (same name, different extension)."""
    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# Naming helpers
# ══════════════════════════════════════════════════════════════════════════════

def _short_name(ts_id):
    """Map a TLS ID to its human-readable name."""
    return TS_NAMES.get(ts_id, ts_id.split("_")[1] if "_" in ts_id else ts_id)


def _safe_filename(name):
    """Convert a name like 'Kolodvor' to a safe lowercase filename slug."""
    return name.lower().replace(" ", "-").replace("/", "-")


def _shorten_feature(f_name):
    """Make a raw sumo-rl feature name presentable (Slovenian)."""
    if f_name == "MinGreenPassed":
        return "Min. zelena pretečena"
    if f_name == "SinTime":
        return "sin(čas)"
    if f_name == "CosTime":
        return "cos(čas)"
    if f_name.startswith("Phase_"):
        return f"Faza {f_name.split('_', 1)[1]}"
    if f_name.startswith("Density_") or f_name.startswith("Queue_"):
        prefix = "Gostota" if f_name.startswith("Density_") else "Vrsta"
        lane_id = f_name.split("_", 1)[1]
        parts = lane_id.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return f"{prefix} pas {parts[1]}"
        return f"{prefix} {lane_id[-6:]}"
    return f_name


def _shorten_features(f_names):
    """Shorten a list of feature names, disambiguating duplicates."""
    short = [_shorten_feature(f) for f in f_names]
    seen = {}
    result = []
    for s in short:
        if short.count(s) > 1:
            idx = seen.get(s, 0)
            seen[s] = idx + 1
            result.append(f"{s}.{idx}")
        else:
            result.append(s)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Phase analysis — map action indices to real-world signal descriptions
# ══════════════════════════════════════════════════════════════════════════════

def _build_phase_labels(phase_info_for_ts):
    """Analyze green phase state strings and controlled links to create
    human-readable descriptions of what each action does.

    Returns:
        labels: list of str, one per green phase (action index)
        approach_legend: dict mapping approach label -> list of edge IDs
    """
    green_states = phase_info_for_ts["green_states"]
    links = phase_info_for_ts["links"]

    # Group links by source edge (= approach arm of the intersection)
    edge_indices = {}
    for i, link in enumerate(links):
        from_lane = link["from"]
        if not from_lane:
            continue
        edge = from_lane.rsplit("_", 1)[0]
        edge_indices.setdefault(edge, []).append(i)

    # Identify signalized edges: those whose green/red pattern changes across phases.
    # Passthrough links (always G/O) are NOT signalized.
    signalized = {}
    for edge, indices in edge_indices.items():
        patterns = set()
        for gs in green_states:
            pattern = tuple(gs[i] in ('G', 'g') for i in indices if i < len(gs))
            patterns.add(pattern)
        if len(patterns) > 1:
            signalized[edge] = indices

    if not signalized:
        return [f"Akcija {i}" for i in range(len(green_states))], {}

    # Cluster edges that co-activate (identical green/red pattern across all phases).
    # Each cluster = one "approach direction" from the user's perspective.
    edge_activation = {}
    for edge, indices in signalized.items():
        pattern = []
        for gs in green_states:
            any_green = any(gs[i] in ('G', 'g') for i in indices if i < len(gs))
            pattern.append(any_green)
        edge_activation[edge] = tuple(pattern)

    # Group edges with identical activation patterns
    pattern_to_edges = {}
    for edge, pattern in edge_activation.items():
        pattern_to_edges.setdefault(pattern, []).append(edge)

    # Assign numbered labels (Smer 1, Smer 2, ...) sorted by first appearance
    # "First appearance" = the first phase where this group gets green
    def first_green_phase(pattern):
        for i, active in enumerate(pattern):
            if active:
                return i
        return len(pattern)

    sorted_patterns = sorted(pattern_to_edges.keys(), key=first_green_phase)
    group_label = {}  # edge -> "Smer N"
    approach_legend = {}  # "Smer N" -> [edge_ids]
    for idx, pattern in enumerate(sorted_patterns):
        label = f"Smer {idx + 1}"
        approach_legend[label] = pattern_to_edges[pattern]
        for edge in pattern_to_edges[pattern]:
            group_label[edge] = label

    # Build a label per green phase
    labels = []
    for phase_idx, gs in enumerate(green_states):
        active_groups = set()
        for edge, indices in signalized.items():
            any_green = any(gs[i] in ('G', 'g') for i in indices if i < len(gs))
            if any_green and edge in group_label:
                active_groups.add(group_label[edge])

        # Sort by number for consistent ordering
        sorted_groups = sorted(active_groups, key=lambda s: int(s.split()[-1]))
        if sorted_groups:
            labels.append(" + ".join(sorted_groups))
        else:
            labels.append(f"Akcija {phase_idx}")

    return labels, approach_legend


def _fallback_labels(n_actions):
    """Generic labels when phase_info is not available."""
    return [f"Akcija {i}" for i in range(n_actions)]


# ══════════════════════════════════════════════════════════════════════════════
# Custom Decision Tree Renderer
# ══════════════════════════════════════════════════════════════════════════════

def _render_tree(clf, feature_names, class_labels, title, output_path, fidelity):
    """Render a clean, readable decision tree.

    Nodes show:
      - Internal: split condition (e.g. "Vrsta pas 2 ≤ 0.35")
      - Leaf: which approach directions get green (e.g. "Smer 1 + Smer 3")
      - Title: intersection name + PPO coverage %
    """
    tree = clf.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold
    values = tree.value

    # Color palette for actions
    n_classes = len(class_labels)
    cmap = plt.cm.Set3 if n_classes <= 12 else plt.cm.tab20
    colors = [cmap(i / max(n_classes - 1, 1)) for i in range(n_classes)]

    # Compute depths via BFS
    depth_map = {}
    queue = deque([(0, 0)])
    while queue:
        node, d = queue.popleft()
        depth_map[node] = d
        if children_left[node] != children_right[node]:
            queue.append((children_left[node], d + 1))
            queue.append((children_right[node], d + 1))

    max_depth = max(depth_map.values()) if depth_map else 0

    # Compact layout: compute subtree widths bottom-up, then position
    # nodes by splitting allocated horizontal range between children.
    # This packs the tree much tighter than an in-order traversal.
    subtree_w = {}

    def _width(node):
        if children_left[node] == children_right[node]:
            subtree_w[node] = 1.0
        else:
            subtree_w[node] = _width(children_left[node]) + _width(children_right[node])
        return subtree_w[node]

    _width(0)

    x_pos = {}
    y_pos = {}

    def _position(node, x_start, x_end):
        x_pos[node] = (x_start + x_end) / 2
        y_pos[node] = -depth_map[node]
        if children_left[node] != children_right[node]:
            lw = subtree_w[children_left[node]]
            rw = subtree_w[children_right[node]]
            total = lw + rw
            split = x_start + (x_end - x_start) * lw / total
            _position(children_left[node], x_start, split)
            _position(children_right[node], split, x_end)

    _position(0, 0, subtree_w[0])

    y_scale = max_depth or 1
    total_w = subtree_w[0]

    fig_width = max(12, min(20, total_w * 1.5))
    fig_height = max(7, y_scale * 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.2, total_w + 0.2)
    ax.set_ylim(-y_scale - 0.7, 0.8)
    ax.axis('off')

    # Coverage badge at top-left
    ax.text(0.01, 0.99, f"Pokritost PPO: {fidelity:.0%}",
            transform=ax.transAxes, fontsize=13, fontweight='bold',
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f5e9' if fidelity > 0.7 else '#fff3e0',
                      edgecolor='#4caf50' if fidelity > 0.7 else '#ff9800', linewidth=2))

    box_w = 0.7
    box_h = 0.45

    for node_id in range(n_nodes):
        x = x_pos[node_id]
        y = y_pos[node_id]
        is_leaf = children_left[node_id] == children_right[node_id]

        class_counts = values[node_id][0]
        majority = int(np.argmax(class_counts))

        if is_leaf:
            label = class_labels[majority].replace(" + ", " +\n")
            purity = class_counts[majority] / class_counts.sum()
            bg_color = (*colors[majority][:3], 0.3 + 0.6 * purity)
            fontsize = 9
            fontweight = 'bold'
        else:
            feat_idx = features[node_id]
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"x[{feat_idx}]"
            thresh = thresholds[node_id]
            label = f"{feat_name}\n≤ {thresh:.2f}"
            bg_color = (0.95, 0.95, 0.95, 0.9)
            fontsize = 9
            fontweight = 'normal'

        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.08",
            facecolor=bg_color, edgecolor='#555555', linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight)

        if not is_leaf:
            for child, side_label in [(children_left[node_id], "Da"),
                                       (children_right[node_id], "Ne")]:
                cx = x_pos[child]
                cy = y_pos[child]
                ax.annotate(
                    '', xy=(cx, cy + box_h / 2), xytext=(x, y - box_h / 2),
                    arrowprops=dict(arrowstyle='->', color='#777777', lw=1.2)
                )
                mid_x = (x + cx) / 2
                mid_y = (y - box_h / 2 + cy + box_h / 2) / 2
                ax.text(mid_x, mid_y, side_label, fontsize=7, color='#555555',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.8))

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def generate_explanations(data_path):
    run_dir = os.path.dirname(data_path)
    output_dir = os.path.join(run_dir, "explanations")
    tree_dir = os.path.join(output_dir, "decision-trees")
    shap_dir = os.path.join(output_dir, "shap")
    umap_dir = os.path.join(output_dir, "umap")
    tsne_dir = os.path.join(output_dir, "t-sne")
    for d in [output_dir, tree_dir, shap_dir, umap_dir, tsne_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"  Loading data from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    obs = np.array(data["observations"])
    actions = np.array(data["actions"])
    latents = np.array(data["latents"])
    metadata = pd.DataFrame(data["metadata"])
    feature_maps = data["feature_maps"]
    ts_ids = data["ts_ids"]
    phase_info = data.get("phase_info", {})

    metadata["intersection"] = metadata["ts_id"].map(lambda tid: _short_name(tid))

    # --- 1. Policy Distillation (Decision Trees) + SHAP ---
    print("  Distilling policy into Decision Trees + SHAP...")
    for ts_id in ts_ids:
        name = _short_name(ts_id)
        slug = _safe_filename(name)
        print(f"\n    ── {name} ──")
        ts_indices = metadata[metadata["ts_id"] == ts_id].index
        if len(ts_indices) == 0:
            continue

        X = obs[ts_indices]
        y = actions[ts_indices]
        f_names = feature_maps[ts_id]
        short_f_names = _shorten_features(f_names)

        # Trim padded observations to actual feature count
        n_features = len(f_names)
        if X.shape[1] > n_features:
            X = X[:, :n_features]

        # Build phase labels from signal state analysis
        if ts_id in phase_info:
            phase_labels, approach_legend = _build_phase_labels(phase_info[ts_id])
            print(f"    Faze ({len(phase_labels)} zelenih faz):")
            for i, label in enumerate(phase_labels):
                print(f"      Akcija {i} → {label}")
            if approach_legend:
                print(f"    Smeri ({len(approach_legend)} pristopnih skupin):")
                for smer, edges in approach_legend.items():
                    print(f"      {smer}: {len(edges)} priključnih cest")
        else:
            phase_labels = None
            print("    (phase_info ni na voljo — uporabim generične oznake)")

        # Fit surrogate decision tree
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(X, y)
        fidelity = clf.score(X, y)
        print(f"    Pokritost PPO (zvestoba drevesa): {fidelity:.1%}")

        # Map each class (action index) to its phase label
        if phase_labels:
            class_labels = []
            for c in clf.classes_:
                idx = int(c)
                if idx < len(phase_labels):
                    class_labels.append(phase_labels[idx])
                else:
                    class_labels.append(f"Akcija {idx}")
        else:
            class_labels = [f"Akcija {int(c)}" for c in clf.classes_]

        # Render decision tree
        tree_path = os.path.join(tree_dir, f"{slug}.png")
        _render_tree(
            clf, short_f_names, class_labels,
            title=f"Odločitvena logika: {name}",
            output_path=tree_path,
            fidelity=fidelity,
        )
        _save_json(tree_path, {
            "intersection": name,
            "ts_id": ts_id,
            "fidelity": round(fidelity, 4),
            "max_depth": 4,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "classes": [int(c) for c in clf.classes_],
            "class_labels": class_labels,
            "feature_names": short_f_names,
            "tree_text": export_text(clf, feature_names=short_f_names),
            "phase_labels": phase_labels if phase_labels else None,
            "approach_legend": {k: len(v) for k, v in approach_legend.items()} if ts_id in phase_info else None,
        })

        # SHAP on the surrogate tree
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)

        fig = plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values, X, feature_names=short_f_names,
                              class_names=class_labels, show=False)
        else:
            shap.summary_plot(shap_values, X, feature_names=short_f_names, show=False)

        fig.suptitle(f"Pomembnost značilk (SHAP): {name}", fontsize=14, fontweight='bold', y=1.02)
        shap_path = os.path.join(shap_dir, f"{slug}.png")
        plt.savefig(shap_path, bbox_inches='tight', dpi=150)
        plt.close()

        # Compute mean absolute SHAP importance per feature (across all classes)
        if isinstance(shap_values, list):
            mean_abs = np.mean(np.array([np.abs(sv).mean(axis=0) for sv in shap_values]), axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
        mean_abs = np.asarray(mean_abs).flatten()
        feature_importance = sorted(
            zip(short_f_names, [float(v) for v in mean_abs]),
            key=lambda x: x[1], reverse=True
        )
        _save_json(shap_path, {
            "intersection": name,
            "ts_id": ts_id,
            "feature_importance": [{"feature": f, "mean_abs_shap": round(v, 6)} for f, v in feature_importance],
            "class_labels": class_labels,
            "n_samples": int(X.shape[0]),
        })

    # --- 2. Latent Space Visualization ---
    # 4 combinations: UMAP, t-SNE, PCA+UMAP, PCA+t-SNE
    # 3 colorings each: action, hour, intersection
    print("\n  Generating latent space projections (4 methods x 3 colorings)...")

    import warnings

    # PCA pre-processing: keep components explaining 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    latents_pca = pca.fit_transform(latents)
    n_orig = latents.shape[1]
    n_pca = latents_pca.shape[1]
    pca_variance = float(np.sum(pca.explained_variance_ratio_))
    print(f"    PCA: {n_orig}D → {n_pca}D (ohranjeno {pca_variance:.1%} variance)")

    colorings = [
        ("actions",        actions,                    "po izbrani akciji",  "viridis"),
        ("time",           metadata["hour"],           "po uri dneva",       "coolwarm"),
        ("intersections",  metadata["intersection"],   "po križišču",        None),
    ]

    methods = [
        ("umap",      "UMAP",          latents,     False, umap_dir),
        ("tsne",      "t-SNE",         latents,     False, tsne_dir),
        ("pca-umap",  "PCA + UMAP",    latents_pca, True,  umap_dir),
        ("pca-tsne",  "PCA + t-SNE",   latents_pca, True,  tsne_dir),
    ]

    for method_slug, method_name, data_in, used_pca, dest_dir in methods:
        print(f"    {method_name}...")

        if "umap" in method_slug:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="n_jobs value")
                embedding = reducer.fit_transform(data_in)
            method_params = {"n_neighbors": 15, "min_dist": 0.1, "random_state": 42}
        else:
            perplexity = min(30, max(5, data_in.shape[0] // 100))
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                           init="pca", learning_rate="auto")
            embedding = reducer.fit_transform(data_in)
            method_params = {"perplexity": perplexity, "random_state": 42}

        raw_base = {
            "n_points": int(embedding.shape[0]),
            "method": method_name,
            "method_params": method_params,
            "pca_applied": used_pca,
            "pca_dims": n_pca if used_pca else None,
            "pca_variance_retained": round(pca_variance, 4) if used_pca else None,
            "input_dims": int(data_in.shape[1]),
            "embedding": embedding.tolist(),
            "actions": actions.tolist(),
            "hours": metadata["hour"].tolist(),
            "intersections": metadata["intersection"].tolist(),
        }

        for color_slug, color_data, color_label, palette in colorings:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=color_data,
                            palette=palette, s=10, alpha=0.5, ax=ax)
            ax.set_title(f"{method_name}: {color_label}", fontsize=14, fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            # e.g. umap-actions.png, pca-umap-actions.png inside umap/ folder
            fname = f"{method_slug}-{color_slug}.png"
            fpath = os.path.join(dest_dir, fname)
            plt.savefig(fpath, bbox_inches='tight', dpi=150)
            plt.close()
            _save_json(fpath, {**raw_base, "coloring": color_slug})

    print(f"\n  All explanations generated in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to harvested_data.pkl")
    args = parser.parse_args()

    generate_explanations(args.data_path)
