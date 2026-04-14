#!/usr/bin/env python3
"""
Generate presentation-ready plots for rush-hour generalization test results.

Produces:
  1. Box plots: morning models vs baseline (avg_queue, avg_wait, total_reward)
  2. Box plots: evening models vs baseline
  3. 95% CI comparison charts (forest plots) for % improvement
  4. Per-intersection breakdown for top models

Usage:
    python src/rush_test_plots.py
    python src/rush_test_plots.py --results_dir results/rush-test
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "results/rush-test"

MORNING_LABELS = {
    "M1": "M1: diff-wait, lr=3e-3",
    "M2": "M2: diff-wait, lr=1e-3",
    "M3": "M3: queue, lr=3e-3, ent.ann.",
    "M4": "M4: diff-wait, lr=3e-4",
    "M5": "M5: pressure, lr=3e-4",
}
EVENING_LABELS = {
    "E1": "E1: diff-wait, lr=1e-3",
    "E2": "E2: diff-wait, lr=1e-3, ent.ann.",
    "E3": "E3: diff-wait, lr=1e-3",
    "E4": "E4: diff-wait, lr=3e-4",
    "E5": "E5: diff-wait, lr=1e-3",
}

INTERSECTION_NAMES = ["Kolodvor", "Pivovarna", "Slovenska", "Trzaska", "Askerceva"]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_summary(base_dir, tag):
    path = os.path.join(base_dir, tag, "summary.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def paired_stats(rl_series, bl_series):
    """Return dict with improvement %, paired t-test, Wilcoxon, Cohen's d, 95% CI."""
    diff = rl_series.values - bl_series.values
    n = len(diff)
    bl_mean = float(bl_series.mean())
    rl_mean = float(rl_series.mean())
    imp_pct = (rl_mean - bl_mean) / abs(bl_mean) * 100 if bl_mean != 0 else 0.0

    t_stat, t_p = stats.ttest_rel(rl_series, bl_series)
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = 0.0, 1.0

    d_std = float(np.std(diff, ddof=1))
    cohens_d = float(np.mean(diff)) / d_std if d_std > 0 else 0.0

    se = d_std / np.sqrt(n)
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=np.mean(diff), scale=se)

    # CI on improvement %
    ci_low_pct = ci_low / abs(bl_mean) * 100 if bl_mean != 0 else 0.0
    ci_high_pct = ci_high / abs(bl_mean) * 100 if bl_mean != 0 else 0.0

    return {
        "imp_pct": imp_pct,
        "t_stat": float(t_stat), "t_p": float(t_p),
        "w_stat": float(w_stat), "w_p": float(w_p),
        "cohens_d": cohens_d,
        "ci_low_pct": ci_low_pct, "ci_high_pct": ci_high_pct,
        "diff_mean": float(np.mean(diff)),
        "ci_low": float(ci_low), "ci_high": float(ci_high),
    }


# ── plot functions ───────────────────────────────────────────────────────────

def plot_boxplots(baseline_df, model_dfs, model_labels, scenario_title, metric,
                  ylabel, invert=False):
    """Box plot comparing models to baseline for a single metric.

    Args:
        invert: If True, lower is better (queue, wait). Improvement arrows
                point down. For reward, higher (less negative) is better.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    all_data = [baseline_df[metric].values]
    labels = ["Baseline\n(fixed-time)"]
    for key, df in model_dfs.items():
        all_data.append(df[metric].values)
        labels.append(model_labels[key])

    bp = ax.boxplot(all_data, tick_labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(color="#555555"),
                    capprops=dict(color="#555555"),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                    markerfacecolor="#888888", markeredgecolor="none"))

    # Colors: baseline gray, models green/red based on direction
    colors = ["#9ca3af"]  # baseline
    bl_mean = baseline_df[metric].mean()
    for key, df in model_dfs.items():
        m = df[metric].mean()
        if invert:
            better = m < bl_mean
        else:
            better = m > bl_mean
        colors.append("#22c55e" if better else "#ef4444")

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#333333")

    # Mean markers
    means = [d.mean() for d in all_data]
    ax.scatter(range(1, len(means) + 1), means, color="black", marker="D",
               s=30, zorder=5, label="Mean")

    ax.set_title(f"{scenario_title} — {ylabel}", fontsize=14, fontweight="bold",
                 color="#111111", pad=12)
    ax.set_ylabel(ylabel, color="#111111")
    ax.tick_params(colors="#333333")
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    ax.spines["bottom"].set_color("#cccccc")
    ax.spines["left"].set_color("#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, color="#cccccc")
    ax.legend(loc="upper right", fontsize=9, facecolor="white",
              edgecolor="#cccccc", labelcolor="#333333")

    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    return fig


def plot_ci_forest(comparisons, model_labels, scenario_title, metric_key,
                   metric_label):
    """Forest plot: 95% CI of % improvement for each model."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(comparisons) * 0.7 + 1)))

    y_pos = list(range(len(comparisons)))
    names = []
    for i, (key, comp) in enumerate(comparisons.items()):
        s = comp[metric_key]
        mid = s["imp_pct"]
        lo = s["ci_low_pct"]
        hi = s["ci_high_pct"]
        color = "#16a34a" if mid > 0 else "#dc2626"

        ax.errorbar(mid, i, xerr=[[mid - lo], [hi - mid]], fmt="o",
                    color=color, ecolor=color, elinewidth=2, capsize=5,
                    markersize=8, capthick=2)

        sig = ""
        if s["t_p"] < 0.001:
            sig = "***"
        elif s["t_p"] < 0.01:
            sig = "**"
        elif s["t_p"] < 0.05:
            sig = "*"
        label_text = f"{mid:+.1f}% {sig}"
        ax.annotate(label_text, (hi + 0.3, i), fontsize=9, color=color,
                    va="center")

        names.append(model_labels.get(key, key))

    ax.axvline(0, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(f"% change vs baseline ({metric_label})", color="#111111",
                  fontsize=11)
    ax.set_title(f"{scenario_title} — 95% CI of improvement ({metric_label})",
                 fontsize=13, fontweight="bold", color="#111111", pad=12)
    ax.tick_params(colors="#333333")
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    ax.spines["bottom"].set_color("#cccccc")
    ax.spines["left"].set_color("#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3, color="#cccccc")
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


def plot_intersection_breakdown(baseline_df, model_df, model_name, scenario_title):
    """Grouped bar chart: per-intersection avg_queue comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, label in zip(axes, ["queue", "wait"],
                                  ["Avg Queue (vehicles)", "Avg Wait (s)"]):
        bl_means = [baseline_df[f"{metric}_{n}"].mean() for n in INTERSECTION_NAMES]
        rl_means = [model_df[f"{metric}_{n}"].mean() for n in INTERSECTION_NAMES]

        x = np.arange(len(INTERSECTION_NAMES))
        w = 0.35
        bars1 = ax.bar(x - w / 2, bl_means, w, label="Baseline", color="#9ca3af",
                        alpha=0.8, edgecolor="#333333", linewidth=0.5)
        bars2 = ax.bar(x + w / 2, rl_means, w, label=model_name, color="#22c55e",
                        alpha=0.8, edgecolor="#333333", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(INTERSECTION_NAMES, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(label, color="#111111")
        ax.set_title(label, color="#111111", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, facecolor="white", edgecolor="#cccccc",
                  labelcolor="#333333")
        ax.tick_params(colors="#333333")
        ax.set_facecolor("white")
        ax.spines["bottom"].set_color("#cccccc")
        ax.spines["left"].set_color("#cccccc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, color="#cccccc")

    fig.set_facecolor("white")
    fig.suptitle(f"{scenario_title} — {model_name} vs Baseline (per intersection)",
                 fontsize=14, fontweight="bold", color="#111111", y=1.02)
    plt.tight_layout()
    return fig


def print_stats_table(comparisons, model_labels, metric_key, metric_name):
    """Print a formatted statistics table to stdout."""
    print(f"\n{'='*90}")
    print(f"  {metric_name}")
    print(f"{'='*90}")
    print(f"  {'Model':<32} {'Change%':>8} {'p(t)':>9} {'p(W)':>9} "
          f"{'Cohen d':>8} {'95% CI':>18}")
    print(f"  {'-'*32} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*18}")
    for key, comp in comparisons.items():
        s = comp[metric_key]
        sig = ""
        if s["t_p"] < 0.001:
            sig = "***"
        elif s["t_p"] < 0.01:
            sig = " **"
        elif s["t_p"] < 0.05:
            sig = "  *"
        print(f"  {model_labels.get(key, key):<32} {s['imp_pct']:>+7.2f}% "
              f"{s['t_p']:>9.4f} {s['w_p']:>9.4f} {s['cohens_d']:>+8.3f} "
              f"[{s['ci_low_pct']:>+.1f}%, {s['ci_high_pct']:>+.1f}%] {sig}")


# ── main ─────────────────────────────────────────────────────────────────────

def save_fig(fig, out_dir, name):
    """Save figure as PNG and close it."""
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Rush-test visualization")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--output_dir", default="results/rush-test-plots")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    scenarios = {
        "morning_rush": {
            "baseline_tag": "baseline_morning",
            "model_tags": ["M1_morning", "M2_morning", "M3_morning",
                           "M4_morning", "M5_morning"],
            "labels": MORNING_LABELS,
            "title": "Morning Rush (06:00-10:00)",
            "prefix": "morning",
        },
        "evening_rush": {
            "baseline_tag": "baseline_evening",
            "model_tags": ["E1_evening", "E2_evening", "E3_evening",
                           "E4_evening", "E5_evening"],
            "labels": EVENING_LABELS,
            "title": "Evening Rush (14:00-18:00)",
            "prefix": "evening",
        },
    }

    for sc_key, sc in scenarios.items():
        bl_df = load_summary(args.results_dir, sc["baseline_tag"])
        if bl_df is None:
            print(f"WARNING: No baseline for {sc_key}, skipping")
            continue

        model_dfs = {}
        for tag in sc["model_tags"]:
            df = load_summary(args.results_dir, tag)
            if df is not None:
                key = tag.split("_")[0]  # "M1", "E2", etc.
                model_dfs[key] = df

        if not model_dfs:
            print(f"WARNING: No model results for {sc_key}, skipping")
            continue

        prefix = sc["prefix"]

        # ── Compute paired stats ─────────────────────────────────────────
        comparisons = {}
        for key, df in model_dfs.items():
            merged = pd.merge(df, bl_df, on="seed", suffixes=("_rl", "_bl"))
            comparisons[key] = {
                "total_reward": paired_stats(merged["total_reward_rl"],
                                             merged["total_reward_bl"]),
                "avg_queue": paired_stats(merged["avg_queue_rl"],
                                          merged["avg_queue_bl"]),
                "avg_wait": paired_stats(merged["avg_wait_rl"],
                                         merged["avg_wait_bl"]),
            }

        # ── Print tables ─────────────────────────────────────────────────
        print(f"\n{'#'*90}")
        print(f"  {sc['title']}")
        print(f"  Baseline: n={len(bl_df)}, Models: n={len(next(iter(model_dfs.values())))}")
        print(f"{'#'*90}")

        print_stats_table(comparisons, sc["labels"], "total_reward",
                          f"{sc['title']} — Total Reward (higher = better)")
        print_stats_table(comparisons, sc["labels"], "avg_queue",
                          f"{sc['title']} — Avg Queue Length (lower = better)")
        print_stats_table(comparisons, sc["labels"], "avg_wait",
                          f"{sc['title']} — Avg Waiting Time (lower = better)")

        # ── Box plots ────────────────────────────────────────────────────
        fig = plot_boxplots(bl_df, model_dfs, sc["labels"], sc["title"],
                            "avg_queue", "Avg Queue Length (vehicles)", invert=True)
        save_fig(fig, out_dir, f"{prefix}_box_queue")

        fig = plot_boxplots(bl_df, model_dfs, sc["labels"], sc["title"],
                            "avg_wait", "Avg Waiting Time (s)", invert=True)
        save_fig(fig, out_dir, f"{prefix}_box_wait")

        fig = plot_boxplots(bl_df, model_dfs, sc["labels"], sc["title"],
                            "total_reward", "Total Reward (neg. halted vehicles)",
                            invert=False)
        save_fig(fig, out_dir, f"{prefix}_box_reward")

        # ── Forest plots (95% CI) ───────────────────────────────────────
        fig = plot_ci_forest(comparisons, sc["labels"], sc["title"],
                             "total_reward", "Total Reward")
        save_fig(fig, out_dir, f"{prefix}_ci_reward")

        fig = plot_ci_forest(comparisons, sc["labels"], sc["title"],
                             "avg_queue", "Avg Queue")
        save_fig(fig, out_dir, f"{prefix}_ci_queue")

        fig = plot_ci_forest(comparisons, sc["labels"], sc["title"],
                             "avg_wait", "Avg Waiting Time")
        save_fig(fig, out_dir, f"{prefix}_ci_wait")

        # ── Per-intersection breakdown for best model ────────────────────
        best_key = max(comparisons,
                       key=lambda k: comparisons[k]["total_reward"]["imp_pct"])
        fig = plot_intersection_breakdown(bl_df, model_dfs[best_key],
                                          sc["labels"][best_key], sc["title"])
        save_fig(fig, out_dir, f"{prefix}_intersection_{best_key}")

    print(f"\nAll plots saved to {out_dir}/")
    print(f"Open with: xdg-open {out_dir}/")


if __name__ == "__main__":
    main()
