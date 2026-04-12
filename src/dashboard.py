"""
Zeleni SignaLJ - Results Dashboard
====================================
Generates an interactive HTML dashboard comparing experiment runs.
Supports filtering, training curve overlay, per-intersection breakdown,
hyperparameter comparison, and detailed experiment drill-down.

Usage:
    python src/dashboard.py
    python src/dashboard.py --output results/dashboard.html
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

EXPERIMENTS_DIR = "results/experiments"


def load_experiments():
    experiments = []
    if not os.path.exists(EXPERIMENTS_DIR):
        return experiments

    for run_id in sorted(os.listdir(EXPERIMENTS_DIR)):
        run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
        meta_path = os.path.join(run_dir, "meta.json")
        results_path = os.path.join(run_dir, "results.csv")
        log_path = os.path.join(run_dir, "training_log.csv")
        step_log_path = os.path.join(run_dir, "training_steps.csv")

        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Skip experiments that never completed (no results)
        if not os.path.exists(results_path):
            continue
        if meta.get("baseline_total_reward") is None:
            continue

        results = pd.read_csv(results_path).to_dict("records")

        # Episode-level training log
        training_log = None
        if os.path.exists(log_path):
            try:
                df_log = pd.read_csv(log_path)
                if len(df_log) > 0:
                    # Round floats to reduce JSON size
                    for col in df_log.select_dtypes(include='float').columns:
                        df_log[col] = df_log[col].round(2)
                    training_log = df_log.to_dict("records")
            except Exception:
                pass

        # Step-level training log (finer granularity)
        step_log = None
        if os.path.exists(step_log_path):
            try:
                df_step = pd.read_csv(step_log_path)
                if len(df_step) > 0:
                    # Downsample to max 150 points for dashboard performance
                    if len(df_step) > 150:
                        step = max(1, len(df_step) // 150)
                        df_step = df_step.iloc[::step].copy()
                    # Round floats to reduce JSON size
                    for col in df_step.select_dtypes(include='float').columns:
                        df_step[col] = df_step[col].round(3)
                    step_log = df_step.to_dict("records")
            except Exception:
                pass

        # Scan for interpretability explanations (images + JSON)
        # Structure: explanations/{category}/{file}.png|.json
        # Categories: decision-trees, shap, umap, t-sne
        explanations = {}
        expl_dir = os.path.join(run_dir, "explanations")
        if os.path.exists(expl_dir):
            for entry in sorted(os.listdir(expl_dir)):
                sub = os.path.join(expl_dir, entry)
                if os.path.isdir(sub):
                    images = sorted(f for f in os.listdir(sub)
                                    if f.endswith(".png") or f.endswith(".jpg"))
                    jsons = sorted(f for f in os.listdir(sub)
                                   if f.endswith(".json"))
                    if images or jsons:
                        explanations[entry] = {"images": images, "jsons": jsons}
                elif entry.endswith(".png") or entry.endswith(".jpg"):
                    # Legacy: flat files in explanations/
                    explanations.setdefault("_flat", {"images": [], "jsons": []})
                    explanations["_flat"]["images"].append(entry)

        experiments.append({
            "meta": meta,
            "results": results,
            "training_log": training_log,
            "step_log": step_log,
            "explanations": explanations,
        })

    return experiments


def _round_floats(obj, decimals=2):
    """Recursively round floats in nested dicts/lists for smaller JSON."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    return obj


STATISTICAL_TEST_DIR = "results/statistical-test"
RUSH_TEST_DIR = "results/rush-test"

# Model descriptions for megapolicy combinations (from HPC sweep results)
MORNING_MODELS = {
    "M1": "diff-waiting-time, lr=1e-3",
    "M2": "pressure, lr=1e-3",
    "M3": "queue, lr=3e-4",
}
EVENING_MODELS = {
    "E1": "pressure, lr=1e-3, ent. anneal",
    "E2": "diff-waiting-time, lr=1e-3, ent. anneal",
    "E3": "pressure, lr=3e-4, ent. anneal",
}

WINDOW_LABELS = {
    "night_0006": "Noc (00-06)",
    "morning_rush": "Jutranja konica (06-10)",
    "shoulder_day": "Dnevna rama (10-14)",
    "evening_rush": "Vecerna konica (14-18)",
    "shoulder_evening": "Vecerna rama (18-21)",
    "night_2100": "Noc (21-00)",
}
WINDOW_ORDER = ["night_0006", "morning_rush", "shoulder_day",
                "evening_rush", "shoulder_evening", "night_2100"]
INTERSECTION_NAMES = ["Kolodvor", "Pivovarna", "Slovenska", "Trzaska", "Askerceva"]


def _desc_stats(series):
    """Compute mean, median, std, 95% CI for a pandas Series."""
    n = len(series)
    mean = float(series.mean())
    median = float(series.median())
    std = float(series.std(ddof=1))
    se = std / np.sqrt(n) if n > 0 else 0.0
    if se > 0 and n > 1:
        ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    else:
        ci_low, ci_high = mean, mean
    return {
        "mean": mean, "median": median, "std": std,
        "ci_low": float(ci_low), "ci_high": float(ci_high),
    }


def _compare(mega_series, baseline_series):
    """Paired t-test, Wilcoxon signed-rank, Cohen's d for matched pairs."""
    differences = mega_series.values - baseline_series.values
    t_stat, t_p = stats.ttest_rel(mega_series, baseline_series)
    try:
        w_stat, w_p = stats.wilcoxon(differences)
    except ValueError:
        w_stat, w_p = 0.0, 1.0  # all differences identical
    # Cohen's d for paired data
    d_std = float(np.std(differences, ddof=1))
    cohens_d = float(np.mean(differences)) / d_std if d_std > 0 else 0.0
    bl_abs = abs(float(baseline_series.mean()))
    improvement_pct = (
        (float(mega_series.mean()) - float(baseline_series.mean()))
        / bl_abs * 100 if bl_abs > 0 else 0.0)
    return {
        "improvement_pct": float(improvement_pct),
        "paired_t": float(t_stat), "paired_t_p": float(t_p),
        "wilcoxon_w": float(w_stat), "wilcoxon_p": float(w_p),
        "cohens_d": float(cohens_d),
    }


def load_megapolicy_results(base_dir=STATISTICAL_TEST_DIR):
    """Load megapolicy + baseline results and compute statistical comparisons."""
    if not os.path.exists(base_dir):
        return None

    conditions = {}
    for name in sorted(os.listdir(base_dir)):
        d = os.path.join(base_dir, name)
        meta_path = os.path.join(d, "meta.json")
        csv_path = os.path.join(d, "summary.csv")
        if not os.path.isdir(d) or not os.path.exists(csv_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        df = pd.read_csv(csv_path)
        conditions[meta["tag"]] = {"meta": meta, "df": df}

    if "baseline" not in conditions:
        return None

    bl_df = conditions["baseline"]["df"]

    # Build condition stats
    cond_list = []
    for tag, cdata in conditions.items():
        df = cdata["df"]
        is_bl = cdata["meta"].get("baseline", False)
        # Parse morning/evening model from tag
        m_key = tag[:2] if not is_bl else None  # "M1", "M2", "M3"
        e_key = tag[2:] if not is_bl else None  # "E1", "E2", "E3"
        entry = {
            "tag": tag,
            "is_baseline": is_bl,
            "morning_model": MORNING_MODELS.get(m_key, "") if m_key else "",
            "evening_model": EVENING_MODELS.get(e_key, "") if e_key else "",
            "n": len(df),
            "total_reward": _desc_stats(df["total_reward"]),
            "avg_queue": _desc_stats(df["avg_queue"]),
            "avg_wait": _desc_stats(df["avg_wait"]),
            "intersections": {},
            "windows": {},
        }
        for iname in INTERSECTION_NAMES:
            entry["intersections"][iname] = {
                "reward": _desc_stats(df[f"reward_{iname}"]),
                "queue": _desc_stats(df[f"queue_{iname}"]),
                "wait": _desc_stats(df[f"wait_{iname}"]),
            }
        for wname in WINDOW_ORDER:
            entry["windows"][wname] = {
                "reward": _desc_stats(df[f"reward_{wname}"]),
                "queue": _desc_stats(df[f"queue_{wname}"]),
                "wait": _desc_stats(df[f"wait_{wname}"]),
            }
        cond_list.append(entry)

    # Comparisons: each megapolicy vs baseline (paired by seed)
    comparisons = []
    for tag, cdata in conditions.items():
        if cdata["meta"].get("baseline", False):
            continue
        cond_df = cdata["df"]
        # Merge on seed to ensure paired alignment
        merged = pd.merge(
            cond_df[["seed", "total_reward"]],
            bl_df[["seed", "total_reward"]],
            on="seed", suffixes=("_mega", "_bl"))
        comp = {
            "tag": tag,
            **_compare(merged["total_reward_mega"], merged["total_reward_bl"]),
            "intersections": {},
            "windows": {},
        }
        for iname in INTERSECTION_NAMES:
            col = f"reward_{iname}"
            m = pd.merge(cond_df[["seed", col]], bl_df[["seed", col]],
                         on="seed", suffixes=("_mega", "_bl"))
            comp["intersections"][iname] = _compare(
                m[f"{col}_mega"], m[f"{col}_bl"])
        for wname in WINDOW_ORDER:
            col = f"reward_{wname}"
            m = pd.merge(cond_df[["seed", col]], bl_df[["seed", col]],
                         on="seed", suffixes=("_mega", "_bl"))
            comp["windows"][wname] = _compare(
                m[f"{col}_mega"], m[f"{col}_bl"])
        comparisons.append(comp)

    # 3x3 heatmap
    m_keys = ["M1", "M2", "M3"]
    e_keys = ["E1", "E2", "E3"]
    heatmap_values = []
    heatmap_pvalues = []
    heatmap_tags = []
    for mk in m_keys:
        row_v, row_p, row_t = [], [], []
        for ek in e_keys:
            t = mk + ek
            row_t.append(t)
            comp = next((c for c in comparisons if c["tag"] == t), None)
            row_v.append(comp["improvement_pct"] if comp else 0)
            row_p.append(comp["paired_t_p"] if comp else 1)
        heatmap_values.append(row_v)
        heatmap_pvalues.append(row_p)
        heatmap_tags.append(row_t)

    return {
        "conditions": cond_list,
        "comparisons": comparisons,
        "heatmap": {
            "morning_models": m_keys,
            "evening_models": e_keys,
            "morning_descriptions": [MORNING_MODELS[k] for k in m_keys],
            "evening_descriptions": [EVENING_MODELS[k] for k in e_keys],
            "values": heatmap_values,
            "p_values": heatmap_pvalues,
            "tags": heatmap_tags,
        },
        "intersection_names": INTERSECTION_NAMES,
        "window_names": WINDOW_ORDER,
        "window_labels": WINDOW_LABELS,
    }


def load_rush_test_results(base_dir=RUSH_TEST_DIR):
    """Load rush-hour generalization test results and compute statistics."""
    if not os.path.exists(base_dir):
        return None

    # Load all conditions
    conditions = {}
    for name in sorted(os.listdir(base_dir)):
        d = os.path.join(base_dir, name)
        meta_path = os.path.join(d, "meta.json")
        csv_path = os.path.join(d, "summary.csv")
        if not os.path.isdir(d) or not os.path.exists(csv_path):
            continue
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        df = pd.read_csv(csv_path)
        conditions[meta["tag"]] = {"meta": meta, "df": df}

    if not conditions:
        return None

    # Group by scenario
    scenario_groups = {}
    for tag, cdata in conditions.items():
        sc = cdata["meta"].get("scenario", "")
        if sc not in scenario_groups:
            scenario_groups[sc] = {}
        scenario_groups[sc][tag] = cdata

    # Expected policies per scenario
    expected = {
        "morning_rush": {
            "baseline": "baseline_morning",
            "models": ["M1_morning", "M2_morning", "M3_morning"],
            "label": "Jutranja konica (06:00-10:00)",
        },
        "evening_rush": {
            "baseline": "baseline_evening",
            "models": ["E1_evening", "E2_evening", "E3_evening"],
            "label": "Vecerna konica (14:00-18:00)",
        },
    }

    scenarios = {}
    for sc_key, sc_expect in expected.items():
        sc_conds = scenario_groups.get(sc_key, {})
        bl_tag = sc_expect["baseline"]
        if bl_tag not in sc_conds:
            continue  # no baseline = skip scenario

        bl_df = sc_conds[bl_tag]["df"]

        # Warnings for missing policies
        warnings = [t for t in sc_expect["models"] if t not in sc_conds]

        # Build condition stats
        cond_list = []
        for tag, cdata in sc_conds.items():
            df = cdata["df"]
            is_bl = cdata["meta"].get("baseline", False)
            # Parse model key from tag (e.g. "M1" from "M1_morning")
            model_key = tag.split("_")[0] if not is_bl else None
            model_desc = ""
            if model_key:
                model_desc = MORNING_MODELS.get(model_key, EVENING_MODELS.get(model_key, ""))
            entry = {
                "tag": tag,
                "is_baseline": is_bl,
                "model_key": model_key or "",
                "model_desc": model_desc,
                "scenario": sc_key,
                "n": len(df),
                "total_reward": _desc_stats(df["total_reward"]),
                "avg_queue": _desc_stats(df["avg_queue"]),
                "avg_wait": _desc_stats(df["avg_wait"]),
                "intersections": {},
            }
            for iname in INTERSECTION_NAMES:
                entry["intersections"][iname] = {
                    "reward": _desc_stats(df[f"reward_{iname}"]),
                    "queue": _desc_stats(df[f"queue_{iname}"]),
                    "wait": _desc_stats(df[f"wait_{iname}"]),
                }
            cond_list.append(entry)

        # Comparisons: each model vs baseline (paired by seed)
        comparisons = []
        for tag, cdata in sc_conds.items():
            if cdata["meta"].get("baseline", False):
                continue
            cond_df = cdata["df"]
            merged = pd.merge(
                cond_df[["seed", "total_reward", "avg_queue", "avg_wait"]],
                bl_df[["seed", "total_reward", "avg_queue", "avg_wait"]],
                on="seed", suffixes=("_rl", "_bl"))
            comp = {
                "tag": tag,
                **_compare(merged["total_reward_rl"], merged["total_reward_bl"]),
                "kpi_comparisons": {
                    "avg_queue": _compare(merged["avg_queue_rl"], merged["avg_queue_bl"]),
                    "avg_wait": _compare(merged["avg_wait_rl"], merged["avg_wait_bl"]),
                },
                "intersections": {},
            }
            for iname in INTERSECTION_NAMES:
                col = f"reward_{iname}"
                m = pd.merge(cond_df[["seed", col]], bl_df[["seed", col]],
                             on="seed", suffixes=("_rl", "_bl"))
                comp["intersections"][iname] = _compare(
                    m[f"{col}_rl"], m[f"{col}_bl"])
            comparisons.append(comp)

        scenarios[sc_key] = {
            "label": sc_expect["label"],
            "conditions": cond_list,
            "comparisons": comparisons,
            "baseline_tag": bl_tag,
            "warnings": warnings,
        }

    if not scenarios:
        return None

    return {
        "scenarios": scenarios,
        "intersection_names": INTERSECTION_NAMES,
    }


def generate_html(experiments, output_path, megapolicy_data=None,
                  rush_test_data=None):
    # Round all floats to reduce JSON size (82KB → ~30KB)
    rounded = _round_floats(experiments)
    exp_json = json.dumps(rounded, default=str, separators=(',', ':'))
    mega_json = (json.dumps(_round_floats(megapolicy_data, decimals=4),
                            default=str, separators=(',', ':'))
                 if megapolicy_data else 'null')
    rush_json = (json.dumps(_round_floats(rush_test_data, decimals=4),
                            default=str, separators=(',', ':'))
                 if rush_test_data else 'null')

    html = f"""<!DOCTYPE html>
<html lang="sl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zeleni SignaLJ - Nadzorna plosca</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 20px 24px;
    min-height: 100vh;
  }}
  h1 {{ color: #4ade80; margin-bottom: 2px; font-size: 26px; }}
  h2 {{ color: #94a3b8; margin: 16px 0 8px; font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
  .subtitle {{ color: #64748b; margin-bottom: 16px; font-size: 13px; }}
  .header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }}
  .header-right {{ text-align: right; color: #64748b; font-size: 12px; }}

  /* KPI Cards */
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 16px; }}
  .kpi {{
    background: #1e293b; border-radius: 8px; padding: 14px 16px;
    border: 1px solid #334155;
  }}
  .kpi-label {{ color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }}
  .kpi-value {{ font-size: 24px; font-weight: 700; }}
  .kpi-sub {{ color: #64748b; font-size: 11px; margin-top: 2px; }}
  .positive {{ color: #4ade80; }}
  .negative {{ color: #f87171; }}
  .neutral {{ color: #cbd5e1; }}
  .warning {{ color: #fbbf24; }}

  /* Cards */
  .card {{
    background: #1e293b; border-radius: 8px; padding: 14px 16px;
    border: 1px solid #334155; margin-bottom: 10px;
  }}
  .card-title {{ color: #94a3b8; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }}

  /* Tables */
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 7px 10px; text-align: left; border-bottom: 1px solid #1e293b; font-size: 12px; }}
  th {{ color: #94a3b8; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; background: #0f172a; position: sticky; top: 0; cursor: pointer; user-select: none; }}
  th:hover {{ color: #e2e8f0; }}
  tr:hover {{ background: #1e293b; }}
  .table-wrap {{ max-height: 400px; overflow-y: auto; border-radius: 8px; border: 1px solid #334155; }}

  /* Charts */
  .chart-box {{ background: #1e293b; border-radius: 8px; padding: 14px; border: 1px solid #334155; margin-bottom: 10px; }}
  canvas {{ max-height: 300px; }}

  /* Layout */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }}
  @media (max-width: 900px) {{ .grid-2, .grid-3 {{ grid-template-columns: 1fr; }} }}

  /* Badges */
  .badge {{
    display: inline-block; padding: 2px 7px; border-radius: 4px;
    font-size: 11px; font-weight: 600;
  }}
  .badge-green {{ background: #166534; color: #4ade80; }}
  .badge-red {{ background: #7f1d1d; color: #fca5a5; }}
  .badge-gray {{ background: #334155; color: #94a3b8; }}
  .badge-yellow {{ background: #713f12; color: #fbbf24; }}

  /* Controls */
  .controls {{
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
    margin-bottom: 10px;
  }}
  .controls select, .controls input {{
    background: #1e293b; color: #e2e8f0; border: 1px solid #475569;
    padding: 5px 8px; border-radius: 6px; font-size: 12px;
  }}
  .controls label {{ color: #94a3b8; font-size: 11px; }}

  /* Checkbox row */
  .cb-row {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0; }}
  .cb-row label {{
    background: #334155; padding: 3px 8px; border-radius: 5px;
    font-size: 11px; cursor: pointer; user-select: none; transition: all 0.15s;
  }}
  .cb-row label.active {{ background: #166534; color: #4ade80; }}
  .cb-row input {{ display: none; }}

  /* Tabs */
  .tabs {{ display: flex; gap: 2px; margin-bottom: 10px; }}
  .tab {{
    padding: 6px 14px; border-radius: 6px 6px 0 0; cursor: pointer;
    font-size: 12px; font-weight: 500; color: #94a3b8;
    background: #1e293b; border: 1px solid #334155; border-bottom: none;
    transition: all 0.15s;
  }}
  .tab.active {{ background: #334155; color: #e2e8f0; }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  .no-data {{ text-align: center; color: #64748b; padding: 40px; }}
  .info-text {{ color: #64748b; font-size: 12px; font-style: italic; }}

  /* Detail panel */
  .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .detail-item {{ font-size: 12px; }}
  .detail-label {{ color: #64748b; }}
  .detail-value {{ color: #e2e8f0; font-weight: 500; }}

  /* Heatmap-like styling for table cells */
  .heat-good {{ background: rgba(74, 222, 128, 0.15); }}
  .heat-bad {{ background: rgba(248, 113, 113, 0.15); }}

  /* Heatmap for mega-policy matrix */
  .heatmap-grid {{
    display: grid; grid-template-columns: auto repeat(3, 1fr);
    gap: 2px; margin: 8px 0;
  }}
  .heatmap-cell {{
    padding: 14px 8px; text-align: center; border-radius: 4px;
    font-size: 14px; font-weight: 600; cursor: default;
  }}
  .heatmap-header {{
    padding: 8px; text-align: center; color: #94a3b8;
    font-size: 12px; font-weight: 600;
  }}
  .heatmap-sub {{ font-size: 9px; color: #94a3b8; font-weight: 400; display: block; margin-top: 2px; }}
  .sig-marker {{ font-size: 10px; vertical-align: super; color: #fbbf24; }}

  @media print {{
    body {{ background: white; color: #1e293b; padding: 10px; }}
    .card, .chart-box, .kpi {{ border-color: #e2e8f0; background: white; }}
    .kpi-label, .card-title, th {{ color: #475569; }}
    h1 {{ color: #166534; }}
    h2 {{ color: #475569; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Zeleni SignaLJ</h1>
    <p class="subtitle">Nadzorna plosca za primerjavo eksperimentov adaptivnega upravljanja semaforjev</p>
  </div>
  <div class="header-right">
    <div id="genTime"></div>
    <div id="genInfo"></div>
  </div>
</div>

<div id="app"></div>

<script>
const experiments = {exp_json};
const megaData = {mega_json};
const rushData = {rush_json};
const COLORS = ['#4ade80','#60a5fa','#f472b6','#facc15','#a78bfa','#fb923c','#34d399','#f87171','#38bdf8','#c084fc'];
const INT_COLORS = {{'Kolodvor':'#60a5fa','Pivovarna':'#4ade80','Slovenska':'#f472b6','Trzaska':'#facc15','Askerceva':'#a78bfa'}};

// ── Mutable state (must be initialized before render() runs) ──
var intBarInst, intPctInst, intTrendInst;
var trainChartInst, stepChartInst, hpScatter1Inst, hpScatter2Inst;
var megaOverallInst, megaImprovInst;
var megaIntRewardInst, megaIntImprovInst;
var megaWindowRewardInst, megaWindowImprovInst;
var _rushCharts = {{}};
var _tabCharts = {{}};
var _scenarioData = {{}};
var _scenarioSortState = {{}};

const app = document.getElementById('app');
document.getElementById('genTime').textContent = 'Generirano: ' + new Date().toLocaleString('sl-SI');
document.getElementById('genInfo').textContent = experiments.length + ' eksperimentov';

if (experiments.length === 0) {{
  app.innerHTML = '<div class="no-data"><p>Ni podatkov.</p><p style="font-size:13px;margin-top:8px">Zazenite: <code>python src/experiment.py --tag test</code></p></div>';
}} else {{
  render();
}}

function fmt(n, d=0) {{ return n != null ? n.toLocaleString('sl-SI', {{maximumFractionDigits: d}}) : '-'; }}
function fmtPct(n) {{ return n != null ? (n >= 0 ? '+' : '') + n.toFixed(1) + '%' : '-'; }}
function pctClass(n) {{ return n > 0 ? 'positive' : n < 0 ? 'negative' : 'neutral'; }}
function badgeClass(n) {{ return n > 0 ? 'badge-green' : n < -10 ? 'badge-red' : n < 0 ? 'badge-yellow' : 'badge-gray'; }}

function render() {{
  const sorted = [...experiments].sort((a, b) => (a.meta.date || '').localeCompare(b.meta.date || ''));
  const latest = sorted[sorted.length - 1];
  const m = latest.meta;

  // Compute aggregate stats
  const allPcts = experiments.map(e => e.meta.improvement_pct || 0);
  const bestExp = experiments.reduce((b, e) => (e.meta.improvement_pct||0) > (b.meta.improvement_pct||0) ? e : b, experiments[0]);
  const worstExp = experiments.reduce((b, e) => (e.meta.improvement_pct||0) < (b.meta.improvement_pct||0) ? e : b, experiments[0]);
  const avgImprovement = allPcts.reduce((a,b) => a+b, 0) / allPcts.length;
  const totalSteps = experiments.reduce((s, e) => s + (e.meta.actual_timesteps || 0), 0);
  const totalTime = experiments.reduce((s, e) => s + (e.meta.train_time_s || 0), 0);
  const hasPositive = allPcts.some(p => p > 0);

  let h = '';

  // ── KPI CARDS ──
  h += '<div class="kpi-grid">';
  h += `<div class="kpi"><div class="kpi-label">Eksperimenti</div><div class="kpi-value neutral">${{experiments.length}}</div><div class="kpi-sub">skupno stevilo zagonov</div></div>`;
  h += `<div class="kpi"><div class="kpi-label">Najboljse izboljsanje</div><div class="kpi-value ${{pctClass(bestExp.meta.improvement_pct||0)}}">${{fmtPct(bestExp.meta.improvement_pct||0)}}</div><div class="kpi-sub">${{bestExp.meta.tag || bestExp.meta.run_id}}</div></div>`;
  h += `<div class="kpi"><div class="kpi-label">Zadnje izboljsanje</div><div class="kpi-value ${{pctClass(m.improvement_pct||0)}}">${{fmtPct(m.improvement_pct||0)}}</div><div class="kpi-sub">${{m.tag || m.run_id}}</div></div>`;
  h += `<div class="kpi"><div class="kpi-label">Povprecno izboljsanje</div><div class="kpi-value ${{pctClass(avgImprovement)}}">${{fmtPct(avgImprovement)}}</div><div class="kpi-sub">vseh eksperimentov</div></div>`;
  h += `<div class="kpi"><div class="kpi-label">Skupni koraki</div><div class="kpi-value neutral">${{fmt(totalSteps)}}</div><div class="kpi-sub">${{(totalTime/60).toFixed(0)}} min skupni cas</div></div>`;
  h += `<div class="kpi"><div class="kpi-label">Zadnja bazna linija</div><div class="kpi-value neutral">${{fmt(m.baseline_total_reward||0, 0)}}</div><div class="kpi-sub">skupna nagrada (nizja = vec cak.)</div></div>`;
  h += '</div>';

  // ── TABS ──
  h += `<div class="tabs">
    <div class="tab active" onclick="switchTab(0)">Jutranja konica</div>
    <div class="tab" onclick="switchTab(1)">Vecerna konica</div>
    ${{megaData ? '<div class="tab" onclick="switchTab(2)">Mega-politike</div>' : ''}}
    <div class="tab" onclick="switchTab(3)">Krizisca</div>
    <div class="tab" onclick="switchTab(4)">Ucenje</div>
    <div class="tab" onclick="switchTab(5)">Hiperparametri</div>
    <div class="tab" onclick="switchTab(6)">Podrobnosti</div>
    <div class="tab" onclick="switchTab(7)">Interpretibilnost</div>
    ${{rushData ? '<div class="tab" onclick="switchTab(8)">Generalizacija</div>' : ''}}
  </div>`;

  // Filter experiments by scenario
  const morningData = experiments.filter(e => e.meta.scenario === 'morning_rush');
  const eveningData = experiments.filter(e => e.meta.scenario === 'evening_rush');

  // ══════════════════════════════════════════
  // TAB 0: Jutranja konica (morning rush experiments)
  // ══════════════════════════════════════════
  h += '<div class="tab-content active" id="tab0">';
  if (morningData.length === 0) {{
    h += '<div class="no-data"><p>Ni eksperimentov za jutranjo konico.</p><p style="font-size:13px;margin-top:8px">Zazenite: <code>python src/experiment.py --scenario morning_rush --tag ...</code></p></div>';
  }} else {{
    h += renderComparisonTabHTML('morning', morningData);
  }}
  h += '</div>'; // tab0

  // ══════════════════════════════════════════
  // TAB 1: Vecerna konica (evening rush experiments)
  // ══════════════════════════════════════════
  h += '<div class="tab-content" id="tab1">';
  if (eveningData.length === 0) {{
    h += '<div class="no-data"><p>Ni eksperimentov za vecerno konico.</p><p style="font-size:13px;margin-top:8px">Zazenite: <code>python src/experiment.py --scenario evening_rush --tag ...</code></p></div>';
  }} else {{
    h += renderComparisonTabHTML('evening', eveningData);
  }}
  h += '</div>'; // tab1

  // ══════════════════════════════════════════
  // TAB 3: Per-intersection breakdown
  // ══════════════════════════════════════════
  h += '<div class="tab-content" id="tab3">';

  h += `<div class="controls">
    <label>Eksperiment: <select id="intExpSelect" onchange="updateIntersections()">
      ${{experiments.map((e, i) => `<option value="${{i}}" ${{i===experiments.length-1?'selected':''}}>${{e.meta.tag || e.meta.run_id}}</option>`).join('')}}
    </select></label>
  </div>`;

  h += '<div class="grid-2">';
  h += '<div class="chart-box"><div class="card-title">Nagrada po kriziscih (abs. vrednost)</div><canvas id="intBarChart"></canvas></div>';
  h += '<div class="chart-box"><div class="card-title">Izboljsanje po kriziscih (%)</div><canvas id="intPctChart"></canvas></div>';
  h += '</div>';

  // Per-intersection table
  h += '<div class="card"><div class="card-title">Podrobnosti po kriziscih</div><table id="intTable"><thead><tr>';
  h += '<th>Krizisce</th><th>Bazna linija</th><th>RL Agent</th><th>Izboljsanje</th>';
  h += '</tr></thead><tbody></tbody></table></div>';

  // Cross-experiment intersection comparison
  h += '<div class="chart-box"><div class="card-title">Izboljsanje po kriziscih skozi eksperimente</div><canvas id="intTrendChart"></canvas></div>';

  h += '</div>'; // tab3

  // ══════════════════════════════════════════
  // TAB 4: Training curves
  // ══════════════════════════════════════════
  h += '<div class="tab-content" id="tab4">';

  h += '<div class="cb-row" id="curveSelect"></div>';
  h += '<div class="grid-2">';
  h += '<div class="chart-box"><div class="card-title">Nagrada na epizodo (episode-level)</div><canvas id="trainChart"></canvas></div>';
  h += '<div class="chart-box"><div class="card-title">Povprecna nagrada na korak (step-level, glajeno)</div><canvas id="stepChart"></canvas></div>';
  h += '</div>';

  h += '<div class="info-text">Izberite eksperimente zgoraj za prikaz krivulj ucenja. Podatki so na voljo sele po novem zagonu z posodobljenim kodom.</div>';

  h += '</div>'; // tab4

  // ══════════════════════════════════════════
  // TAB 5: Hyperparameter comparison
  // ══════════════════════════════════════════
  h += '<div class="tab-content" id="tab5">';

  h += '<div class="grid-2">';
  h += '<div class="chart-box"><div class="card-title">Koraki vs. izboljsanje</div><canvas id="hpScatter1"></canvas></div>';
  h += '<div class="chart-box"><div class="card-title">Learning rate vs. izboljsanje</div><canvas id="hpScatter2"></canvas></div>';
  h += '</div>';

  h += `<div class="card"><div class="card-title">Hiperparametri po eksperimentih</div>
    <div class="table-wrap">
    <table id="hpTable"><thead><tr>
      <th>Oznaka</th><th>LR</th><th>n_steps</th><th>batch</th><th>gamma</th>
      <th>ent_coef</th><th>clip</th><th>delta_t</th><th>Izboljsanje</th>
    </tr></thead><tbody></tbody></table>
    </div></div>`;

  h += '</div>'; // tab5

  // ══════════════════════════════════════════
  // TAB 6: Detailed experiment info
  // ══════════════════════════════════════════
  h += '<div class="tab-content" id="tab6">';

  h += `<div class="controls">
    <label>Eksperiment: <select id="detailExpSelect" onchange="updateDetails()">
      ${{experiments.map((e, i) => `<option value="${{i}}" ${{i===experiments.length-1?'selected':''}}>${{e.meta.tag || e.meta.run_id}}</option>`).join('')}}
    </select></label>
  </div>`;

  h += '<div id="detailPanel"></div>';

  h += '</div>'; // tab6

  // ══════════════════════════════════════════
  // TAB 2: Mega-politike (statistical test)
  // ══════════════════════════════════════════
  if (megaData) {{
    h += '<div class="tab-content" id="tab2">';

    // ── KPI cards ──
    const mComps = megaData.comparisons;
    const bestMega = mComps.reduce((b, c) => c.improvement_pct > b.improvement_pct ? c : b, mComps[0]);
    const worstMega = mComps.reduce((b, c) => c.improvement_pct < b.improvement_pct ? c : b, mComps[0]);
    const blCond = megaData.conditions.find(c => c.is_baseline);
    const sigCount = mComps.filter(c => c.paired_t_p < 0.05).length;

    h += '<div class="kpi-grid">';
    h += `<div class="kpi"><div class="kpi-label">Stevilo pogojev</div><div class="kpi-value neutral">${{megaData.conditions.length}}</div><div class="kpi-sub">9 mega-politik + bazna linija</div></div>`;
    h += `<div class="kpi"><div class="kpi-label">Najboljsa mega-politika</div><div class="kpi-value ${{pctClass(bestMega.improvement_pct)}}">${{fmtPct(bestMega.improvement_pct)}}</div><div class="kpi-sub">${{bestMega.tag}}</div></div>`;
    h += `<div class="kpi"><div class="kpi-label">Najslabsa mega-politika</div><div class="kpi-value ${{pctClass(worstMega.improvement_pct)}}">${{fmtPct(worstMega.improvement_pct)}}</div><div class="kpi-sub">${{worstMega.tag}}</div></div>`;
    h += `<div class="kpi"><div class="kpi-label">Ponovitve na pogoj</div><div class="kpi-value neutral">${{blCond ? blCond.n : 50}}</div><div class="kpi-sub">razlicni SUMO seedi</div></div>`;
    h += `<div class="kpi"><div class="kpi-label">Bazna linija (povp.)</div><div class="kpi-value neutral">${{blCond ? fmt(blCond.total_reward.mean, 0) : '-'}}</div><div class="kpi-sub">povp. skupna nagrada</div></div>`;
    h += `<div class="kpi"><div class="kpi-label">Stat. znacilne (p&lt;0.05)</div><div class="kpi-value ${{sigCount > 0 ? 'positive' : 'neutral'}}">${{sigCount}} / ${{mComps.length}}</div><div class="kpi-sub">Parni t-test</div></div>`;
    h += '</div>';

    // ── Overall comparison charts ──
    h += '<h2>Primerjava skupne nagrade (24h simulacija)</h2>';
    h += '<div class="grid-2">';
    h += '<div class="chart-box"><div class="card-title">Povprecna skupna nagrada z 95% IZ (abs., nizja = boljsa)</div><canvas id="megaOverallChart"></canvas></div>';
    h += '<div class="chart-box"><div class="card-title">Izboljsanje glede na bazno linijo (%)</div><canvas id="megaImprovChart"></canvas></div>';
    h += '</div>';

    // ── 3x3 Heatmap ──
    h += '<div class="card">';
    h += '<div class="card-title">Matrica mega-politik: jutranji model (vrstica) x vecerni model (stolpec)</div>';
    const hm = megaData.heatmap;
    h += '<div class="heatmap-grid">';
    // Header row
    h += '<div class="heatmap-header"></div>';
    hm.evening_models.forEach((e, j) => {{
      h += `<div class="heatmap-header">${{e}}<span class="heatmap-sub">${{hm.evening_descriptions[j]}}</span></div>`;
    }});
    // Data rows
    hm.morning_models.forEach((m, i) => {{
      h += `<div class="heatmap-header">${{m}}<span class="heatmap-sub">${{hm.morning_descriptions[i]}}</span></div>`;
      hm.evening_models.forEach((e, j) => {{
        const val = hm.values[i][j];
        const pval = hm.p_values[i][j];
        const tag = hm.tags[i][j];
        const sig = pval < 0.001 ? '***' : pval < 0.01 ? '**' : pval < 0.05 ? '*' : '';
        const intensity = Math.min(Math.abs(val) / 15, 1);
        const bg = val >= 0
          ? `rgba(74,222,128,${{(0.15 + intensity * 0.45).toFixed(2)}})`
          : `rgba(248,113,113,${{(0.15 + intensity * 0.45).toFixed(2)}})`;
        h += `<div class="heatmap-cell" style="background:${{bg}}" title="${{tag}}: Parni t-test p=${{pval.toFixed(6)}}">
          ${{fmtPct(val)}}<span class="sig-marker">${{sig}}</span>
        </div>`;
      }});
    }});
    h += '</div>';
    h += '<div class="info-text">* p&lt;0.05 &nbsp; ** p&lt;0.01 &nbsp; *** p&lt;0.001 (Parni t-test)</div>';
    h += '</div>';

    // ── Per-intersection breakdown ──
    h += '<h2>Primerjava po kriziscih</h2>';
    h += '<div class="controls">';
    h += `<label>Mega-politike: <select id="megaIntSelect" onchange="updateMegaIntersections()" multiple size="4" style="min-width:200px">`;
    mComps.forEach((c, i) => {{
      const sel = i < 3 ? 'selected' : '';
      h += `<option value="${{c.tag}}" ${{sel}}>${{c.tag}} (${{fmtPct(c.improvement_pct)}})</option>`;
    }});
    h += '</select></label>';
    h += '<span class="info-text">Drzite Ctrl za izbiro vec politik</span>';
    h += '</div>';
    h += '<div class="grid-2">';
    h += '<div class="chart-box"><div class="card-title">Povprecna nagrada po kriziscih (abs.)</div><canvas id="megaIntRewardChart"></canvas></div>';
    h += '<div class="chart-box"><div class="card-title">Izboljsanje po kriziscih (%)</div><canvas id="megaIntImprovChart"></canvas></div>';
    h += '</div>';

    // ── Per-window breakdown ──
    h += '<h2>Primerjava po casovnih oknih</h2>';
    h += '<div class="controls">';
    h += `<label>Mega-politike: <select id="megaWinSelect" onchange="updateMegaWindows()" multiple size="4" style="min-width:200px">`;
    mComps.forEach((c, i) => {{
      const sel = i < 3 ? 'selected' : '';
      h += `<option value="${{c.tag}}" ${{sel}}>${{c.tag}} (${{fmtPct(c.improvement_pct)}})</option>`;
    }});
    h += '</select></label>';
    h += '<span class="info-text">Drzite Ctrl za izbiro vec politik</span>';
    h += '</div>';
    h += '<div class="grid-2">';
    h += '<div class="chart-box"><div class="card-title">Povprecna nagrada po casovnih oknih (abs.)</div><canvas id="megaWindowRewardChart"></canvas></div>';
    h += '<div class="chart-box"><div class="card-title">Izboljsanje po casovnih oknih (%)</div><canvas id="megaWindowImprovChart"></canvas></div>';
    h += '</div>';

    // ── Statistical significance table ──
    h += '<h2>Statisticna analiza</h2>';
    h += '<div class="card"><div class="card-title">Primerjava z bazno linijo (Parni t-test, Wilcoxon, Cohen d)</div>';
    h += '<div class="table-wrap"><table id="megaStatsTable"><thead><tr>';
    h += '<th>Mega-politika</th><th>Povp. nagrada</th><th>Mediana</th><th>Std</th>';
    h += '<th>95% IZ</th><th>Parni t</th><th>Parni p</th>';
    h += '<th>Wilcoxon W</th><th>Wilcoxon p</th><th>Cohen d</th><th>Izboljsanje</th>';
    h += '</tr></thead><tbody>';
    // Baseline row
    if (blCond) {{
      const s = blCond.total_reward;
      h += `<tr style="background:#1e293b">
        <td><strong>Bazna linija</strong></td>
        <td>${{fmt(s.mean,0)}}</td><td>${{fmt(s.median,0)}}</td><td>${{fmt(s.std,0)}}</td>
        <td>[${{fmt(s.ci_low,0)}}, ${{fmt(s.ci_high,0)}}]</td>
        <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
        <td><span class="badge badge-gray">referenca</span></td>
      </tr>`;
    }}
    // Megapolicy rows
    mComps.forEach(c => {{
      const cond = megaData.conditions.find(x => x.tag === c.tag);
      const s = cond ? cond.total_reward : {{}};
      const pBadge = c.paired_t_p < 0.001 ? 'badge-green' : c.paired_t_p < 0.01 ? 'badge-green' : c.paired_t_p < 0.05 ? 'badge-yellow' : 'badge-gray';
      const pLabel = c.paired_t_p < 0.001 ? '***' : c.paired_t_p < 0.01 ? '**' : c.paired_t_p < 0.05 ? '*' : 'n.s.';
      const wpBadge = c.wilcoxon_p < 0.001 ? 'badge-green' : c.wilcoxon_p < 0.01 ? 'badge-green' : c.wilcoxon_p < 0.05 ? 'badge-yellow' : 'badge-gray';
      const wpLabel = c.wilcoxon_p < 0.001 ? '***' : c.wilcoxon_p < 0.01 ? '**' : c.wilcoxon_p < 0.05 ? '*' : 'n.s.';
      const absD = Math.abs(c.cohens_d);
      const dLabel = absD > 0.8 ? 'Velik' : absD > 0.5 ? 'Srednji' : absD > 0.2 ? 'Majhen' : 'Zanem.';
      const dBadge = absD > 0.8 ? (c.cohens_d > 0 ? 'badge-green' : 'badge-red') : absD > 0.5 ? 'badge-yellow' : 'badge-gray';
      h += `<tr>
        <td><strong>${{c.tag}}</strong></td>
        <td>${{fmt(s.mean||0,0)}}</td><td>${{fmt(s.median||0,0)}}</td><td>${{fmt(s.std||0,0)}}</td>
        <td>[${{fmt(s.ci_low||0,0)}}, ${{fmt(s.ci_high||0,0)}}]</td>
        <td>${{c.paired_t.toFixed(2)}}</td>
        <td><span class="badge ${{pBadge}}">${{c.paired_t_p < 0.0001 ? c.paired_t_p.toExponential(2) : c.paired_t_p.toFixed(4)}} ${{pLabel}}</span></td>
        <td>${{fmt(c.wilcoxon_w,0)}}</td>
        <td><span class="badge ${{wpBadge}}">${{c.wilcoxon_p < 0.0001 ? c.wilcoxon_p.toExponential(2) : c.wilcoxon_p.toFixed(4)}} ${{wpLabel}}</span></td>
        <td><span class="badge ${{dBadge}}">${{c.cohens_d.toFixed(3)}} (${{dLabel}})</span></td>
        <td><span class="badge ${{badgeClass(c.improvement_pct)}}">${{fmtPct(c.improvement_pct)}}</span></td>
      </tr>`;
    }});
    h += '</tbody></table></div></div>';

    // ── Per-megapolicy drill-down ──
    h += '<h2>Podrobnosti mega-politike</h2>';
    h += '<div class="controls">';
    h += `<label>Mega-politika: <select id="megaDetailSelect" onchange="updateMegaDetail()">`;
    mComps.forEach((c, i) => {{
      h += `<option value="${{c.tag}}" ${{i===0?'selected':''}}>${{c.tag}}</option>`;
    }});
    h += '</select></label>';
    h += '</div>';
    h += '<div id="megaDetailPanel"></div>';

    h += '</div>'; // tab2
  }}

  // ══════════════════════════════════════════
  // TAB 7: Interpretibilnost
  // ══════════════════════════════════════════
  h += '<div class="tab-content" id="tab7">';
  const expsWithExpl = experiments.filter(e => e.explanations && Object.keys(e.explanations).length > 0);
  let defaultExplIdx = experiments.findIndex(e => e.explanations && Object.keys(e.explanations).length > 0);
  if (defaultExplIdx < 0) defaultExplIdx = 0;
  h += `<div class="controls">
    <label>Eksperiment: <select id="explExpSelect" onchange="updateExplanations()">
      ${{experiments.map((e, i) => {{
        const has = e.explanations && Object.keys(e.explanations).length > 0;
        const label = (e.meta.tag || e.meta.run_id) + (has ? ' ✓' : '');
        return `<option value="${{i}}" ${{i===defaultExplIdx?'selected':''}}>${{label}}</option>`;
      }}).join('')}}
    </select></label>
    <span class="info-text">${{expsWithExpl.length}} eksperiment(ov) z vizualizacijami</span>
  </div>`;
  h += '<div id="explPanel"></div>';
  h += '</div>'; // tab7

  // ══════════════════════════════════════════
  // TAB 8: Generalizacija (rush-hour generalization tests)
  // ══════════════════════════════════════════
  if (rushData) {{
    h += '<div class="tab-content" id="tab8">';
    h += '<h2 style="color:#4ade80;margin-bottom:4px">Test generalizacije konicnih ur</h2>';
    h += '<p class="info-text" style="margin-bottom:14px">Izolirani 4-urni testi z 50 razlicnimi vzorci prometa na politiko. Meri posplosevalnost modelov brez 24-urnih kaskadnih napak.</p>';

    const rushScenarios = ['morning_rush', 'evening_rush'];
    const RUSH_LABELS = {{'morning_rush': 'Jutranja konica (06:00-10:00)', 'evening_rush': 'Vecerna konica (14:00-18:00)'}};

    rushScenarios.forEach(sk => {{
      const sc = rushData.scenarios[sk];
      if (!sc) return;

      h += `<h2>${{RUSH_LABELS[sk]}}</h2>`;

      // Warnings for missing policies
      if (sc.warnings && sc.warnings.length > 0) {{
        h += `<div class="card" style="border-color:#fbbf24"><div class="card-title" style="color:#fbbf24">Manjkajoce politike</div>`;
        h += `<p style="font-size:12px;color:#fbbf24">${{sc.warnings.join(', ')}} — rezultati niso na voljo</p></div>`;
      }}

      // KPI cards
      const rComps = sc.comparisons;
      const blCond = sc.conditions.find(c => c.is_baseline);
      if (rComps.length > 0) {{
        const bestR = rComps.reduce((b, c) => c.improvement_pct > b.improvement_pct ? c : b, rComps[0]);
        const worstR = rComps.reduce((b, c) => c.improvement_pct < b.improvement_pct ? c : b, rComps[0]);
        const sigCnt = rComps.filter(c => c.paired_t_p < 0.05).length;

        h += '<div class="kpi-grid">';
        h += `<div class="kpi"><div class="kpi-label">Stevilo pogojev</div><div class="kpi-value neutral">${{sc.conditions.length}}</div><div class="kpi-sub">${{rComps.length}} politik + bazna linija</div></div>`;
        h += `<div class="kpi"><div class="kpi-label">Najboljsa politika</div><div class="kpi-value ${{pctClass(bestR.improvement_pct)}}">${{fmtPct(bestR.improvement_pct)}}</div><div class="kpi-sub">${{bestR.tag}}</div></div>`;
        h += `<div class="kpi"><div class="kpi-label">Najslabsa politika</div><div class="kpi-value ${{pctClass(worstR.improvement_pct)}}">${{fmtPct(worstR.improvement_pct)}}</div><div class="kpi-sub">${{worstR.tag}}</div></div>`;
        h += `<div class="kpi"><div class="kpi-label">Ponovitve na pogoj</div><div class="kpi-value neutral">${{blCond ? blCond.n : 50}}</div><div class="kpi-sub">razlicni vzorci prometa</div></div>`;
        h += `<div class="kpi"><div class="kpi-label">Bazna linija (povp.)</div><div class="kpi-value neutral">${{blCond ? fmt(blCond.total_reward.mean, 0) : '-'}}</div><div class="kpi-sub">povp. skupna nagrada</div></div>`;
        h += `<div class="kpi"><div class="kpi-label">Stat. znacilne (p&lt;0.05)</div><div class="kpi-value ${{sigCnt > 0 ? 'positive' : 'neutral'}}">${{sigCnt}} / ${{rComps.length}}</div><div class="kpi-sub">Parni t-test</div></div>`;
        h += '</div>';
      }}

      // Overall comparison charts
      h += '<div class="grid-2">';
      h += `<div class="chart-box"><div class="card-title">Povprecna skupna nagrada z 95% IZ (nizja = boljsa)</div><canvas id="rush_${{sk}}_overall"></canvas></div>`;
      h += `<div class="chart-box"><div class="card-title">Izboljsanje glede na bazno linijo (%)</div><canvas id="rush_${{sk}}_improv"></canvas></div>`;
      h += '</div>';

      // Multi-KPI summary table
      h += '<div class="card"><div class="card-title">Primerjava KPI metrik</div>';
      h += '<div class="table-wrap"><table><thead><tr>';
      h += '<th>Politika</th><th>Nagrada (povp.)</th><th>Nagrada (%)</th>';
      h += '<th>Povp. vrsta</th><th>Vrsta (%)</th>';
      h += '<th>Povp. cakanje (s)</th><th>Cakanje (%)</th>';
      h += '</tr></thead><tbody>';
      if (blCond) {{
        h += `<tr style="background:#1e293b">
          <td><strong>Bazna linija</strong></td>
          <td>${{fmt(blCond.total_reward.mean, 0)}}</td><td><span class="badge badge-gray">referenca</span></td>
          <td>${{fmt(blCond.avg_queue.mean, 1)}}</td><td><span class="badge badge-gray">referenca</span></td>
          <td>${{fmt(blCond.avg_wait.mean, 0)}}</td><td><span class="badge badge-gray">referenca</span></td>
        </tr>`;
      }}
      rComps.forEach(c => {{
        const cond = sc.conditions.find(x => x.tag === c.tag);
        if (!cond) return;
        const qComp = c.kpi_comparisons.avg_queue;
        const wComp = c.kpi_comparisons.avg_wait;
        h += `<tr>
          <td><strong>${{c.tag}}</strong><br><span class="info-text">${{cond.model_desc}}</span></td>
          <td>${{fmt(cond.total_reward.mean, 0)}}</td>
          <td><span class="badge ${{badgeClass(c.improvement_pct)}}">${{fmtPct(c.improvement_pct)}}</span></td>
          <td>${{fmt(cond.avg_queue.mean, 1)}}</td>
          <td><span class="badge ${{badgeClass(-qComp.improvement_pct)}}">${{fmtPct(-qComp.improvement_pct)}}</span></td>
          <td>${{fmt(cond.avg_wait.mean, 0)}}</td>
          <td><span class="badge ${{badgeClass(-wComp.improvement_pct)}}">${{fmtPct(-wComp.improvement_pct)}}</span></td>
        </tr>`;
      }});
      h += '</tbody></table></div></div>';

      // Per-intersection breakdown
      h += '<h2 style="font-size:13px">Primerjava po kriziscih</h2>';
      h += '<div class="controls">';
      h += `<label>Politike: <select id="rush_${{sk}}_intSelect" onchange="updateRushIntersections('${{sk}}')" multiple size="3" style="min-width:200px">`;
      rComps.forEach((c, i) => {{
        const sel = i < 3 ? 'selected' : '';
        h += `<option value="${{c.tag}}" ${{sel}}>${{c.tag}} (${{fmtPct(c.improvement_pct)}})</option>`;
      }});
      h += '</select></label>';
      h += '<span class="info-text">Drzite Ctrl za izbiro vec politik</span>';
      h += '</div>';
      h += '<div class="grid-2">';
      h += `<div class="chart-box"><div class="card-title">Povprecna nagrada po kriziscih (abs.)</div><canvas id="rush_${{sk}}_intReward"></canvas></div>`;
      h += `<div class="chart-box"><div class="card-title">Izboljsanje po kriziscih (%)</div><canvas id="rush_${{sk}}_intImprov"></canvas></div>`;
      h += '</div>';

      // Statistical significance table
      h += '<h2 style="font-size:13px">Statisticna analiza</h2>';
      h += '<div class="card"><div class="card-title">Primerjava z bazno linijo (Parni t-test, Wilcoxon, Cohen d)</div>';
      h += '<div class="table-wrap"><table><thead><tr>';
      h += '<th>Politika</th><th>Povp. nagrada</th><th>Mediana</th><th>Std</th>';
      h += '<th>95% IZ</th><th>Parni t</th><th>Parni p</th>';
      h += '<th>Wilcoxon W</th><th>Wilcoxon p</th><th>Cohen d</th><th>Izboljsanje</th>';
      h += '</tr></thead><tbody>';
      if (blCond) {{
        const s = blCond.total_reward;
        h += `<tr style="background:#1e293b">
          <td><strong>Bazna linija</strong></td>
          <td>${{fmt(s.mean,0)}}</td><td>${{fmt(s.median,0)}}</td><td>${{fmt(s.std,0)}}</td>
          <td>[${{fmt(s.ci_low,0)}}, ${{fmt(s.ci_high,0)}}]</td>
          <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
          <td><span class="badge badge-gray">referenca</span></td>
        </tr>`;
      }}
      rComps.forEach(c => {{
        const cond = sc.conditions.find(x => x.tag === c.tag);
        const s = cond ? cond.total_reward : {{}};
        const pBadge = c.paired_t_p < 0.001 ? 'badge-green' : c.paired_t_p < 0.01 ? 'badge-green' : c.paired_t_p < 0.05 ? 'badge-yellow' : 'badge-gray';
        const pLabel = c.paired_t_p < 0.001 ? '***' : c.paired_t_p < 0.01 ? '**' : c.paired_t_p < 0.05 ? '*' : 'n.s.';
        const wpBadge = c.wilcoxon_p < 0.001 ? 'badge-green' : c.wilcoxon_p < 0.01 ? 'badge-green' : c.wilcoxon_p < 0.05 ? 'badge-yellow' : 'badge-gray';
        const wpLabel = c.wilcoxon_p < 0.001 ? '***' : c.wilcoxon_p < 0.01 ? '**' : c.wilcoxon_p < 0.05 ? '*' : 'n.s.';
        const absD = Math.abs(c.cohens_d);
        const dLabel = absD > 0.8 ? 'Velik' : absD > 0.5 ? 'Srednji' : absD > 0.2 ? 'Majhen' : 'Zanem.';
        const dBadge = absD > 0.8 ? (c.cohens_d > 0 ? 'badge-green' : 'badge-red') : absD > 0.5 ? 'badge-yellow' : 'badge-gray';
        h += `<tr>
          <td><strong>${{c.tag}}</strong></td>
          <td>${{fmt(s.mean||0,0)}}</td><td>${{fmt(s.median||0,0)}}</td><td>${{fmt(s.std||0,0)}}</td>
          <td>[${{fmt(s.ci_low||0,0)}}, ${{fmt(s.ci_high||0,0)}}]</td>
          <td>${{c.paired_t.toFixed(2)}}</td>
          <td><span class="badge ${{pBadge}}">${{c.paired_t_p < 0.0001 ? c.paired_t_p.toExponential(2) : c.paired_t_p.toFixed(4)}} ${{pLabel}}</span></td>
          <td>${{fmt(c.wilcoxon_w,0)}}</td>
          <td><span class="badge ${{wpBadge}}">${{c.wilcoxon_p < 0.0001 ? c.wilcoxon_p.toExponential(2) : c.wilcoxon_p.toFixed(4)}} ${{wpLabel}}</span></td>
          <td><span class="badge ${{dBadge}}">${{c.cohens_d.toFixed(3)}} (${{dLabel}})</span></td>
          <td><span class="badge ${{badgeClass(c.improvement_pct)}}">${{fmtPct(c.improvement_pct)}}</span></td>
        </tr>`;
      }});
      h += '</tbody></table></div></div>';

      // Per-policy drill-down
      h += '<h2 style="font-size:13px">Podrobnosti politike</h2>';
      h += '<div class="controls">';
      h += `<label>Politika: <select id="rush_${{sk}}_detailSelect" onchange="updateRushDetail('${{sk}}')">`;
      rComps.forEach((c, i) => {{
        h += `<option value="${{c.tag}}" ${{i===0?'selected':''}}>${{c.tag}}</option>`;
      }});
      h += '</select></label>';
      h += '</div>';
      h += `<div id="rush_${{sk}}_detailPanel"></div>`;

      h += '<hr style="border-color:#334155;margin:20px 0">';
    }});  // end rushScenarios.forEach

    h += '</div>'; // tab8
  }}

  app.innerHTML = h;

  // Initialize all views
  initComparisonTab('morning', morningData);
  initComparisonTab('evening', eveningData);
  updateIntersections();
  buildCurveSelector();
  updateHyperparams();
  updateDetails();
  updateExplanations();
  if (megaData) {{
    updateMegaOverall();
    updateMegaIntersections();
    updateMegaWindows();
    updateMegaDetail();
  }}
  if (rushData) {{
    ['morning_rush', 'evening_rush'].forEach(sk => {{
      if (rushData.scenarios[sk]) {{
        updateRushOverall(sk);
        updateRushIntersections(sk);
        updateRushDetail(sk);
      }}
    }});
  }}
}}

// ── Tab switching ──
function switchTab(idx) {{
  document.querySelectorAll('.tab').forEach(t => {{
    const onclick = t.getAttribute('onclick') || '';
    const m = onclick.match(/switchTab\\((\\d+)\\)/);
    t.classList.toggle('active', m && parseInt(m[1]) === idx);
  }});
  document.querySelectorAll('.tab-content').forEach(t => {{
    t.classList.toggle('active', t.id === 'tab' + idx);
  }});
}}

// (chart instances and tab state declared above render() call)

// ══════════════════════════════════════════
// TAB 0/1: Scenario-filtered comparison (reusable)
// ══════════════════════════════════════════

function renderComparisonTabHTML(prefix, exps) {{
  let h = '';

  h += `<div class="controls">
    <label>Iskanje: <input type="text" id="${{prefix}}_filterText" placeholder="oznaka ali ID..." oninput="applyScenarioFilters('${{prefix}}')"></label>
    <label>Min korakov: <input type="number" id="${{prefix}}_filterMinSteps" value="0" style="width:80px" oninput="applyScenarioFilters('${{prefix}}')"></label>
    <label>Zadnjih N: <input type="number" id="${{prefix}}_filterLastN" value="" placeholder="vse" style="width:60px" oninput="applyScenarioFilters('${{prefix}}')"></label>
  </div>`;

  // Comparison bar chart
  h += '<div class="grid-2">';
  h += `<div class="chart-box"><div class="card-title">Skupna nagrada: bazna linija vs RL (abs. vrednost, nizja = boljsa)</div><canvas id="${{prefix}}_compChart"></canvas></div>`;
  h += `<div class="chart-box"><div class="card-title">Izboljsanje po eksperimentih (%)</div><canvas id="${{prefix}}_improvChart"></canvas></div>`;
  h += '</div>';

  // Experiment table
  h += `<div class="card" style="overflow-x:auto;">
    <div class="card-title">Eksperimenti (${{exps.length}})</div>
    <div class="table-wrap">
    <table id="${{prefix}}_expTable">
      <thead><tr>
        <th onclick="sortScenarioTable('${{prefix}}', 0)">Oznaka</th>
        <th onclick="sortScenarioTable('${{prefix}}', 1)">Datum</th>
        <th onclick="sortScenarioTable('${{prefix}}', 2)">Koraki</th>
        <th onclick="sortScenarioTable('${{prefix}}', 3)">Bazna linija</th>
        <th onclick="sortScenarioTable('${{prefix}}', 4)">RL Agent</th>
        <th onclick="sortScenarioTable('${{prefix}}', 5)">Izboljsanje</th>
        <th onclick="sortScenarioTable('${{prefix}}', 6)">LR</th>
        <th onclick="sortScenarioTable('${{prefix}}', 7)">Ent. koef.</th>
        <th onclick="sortScenarioTable('${{prefix}}', 8)">Nagradna f.</th>
        <th onclick="sortScenarioTable('${{prefix}}', 9)">Cas</th>
      </tr></thead>
      <tbody></tbody>
    </table>
    </div>
  </div>`;

  return h;
}}

// Store experiment data per prefix for filtering

function initComparisonTab(prefix, exps) {{
  _scenarioData[prefix] = exps;
  applyScenarioFilters(prefix);
}}

function getScenarioFiltered(prefix) {{
  const exps = _scenarioData[prefix] || [];
  const text = (document.getElementById(prefix + '_filterText')?.value || '').toLowerCase();
  const minSteps = parseInt(document.getElementById(prefix + '_filterMinSteps')?.value) || 0;
  const lastN = parseInt(document.getElementById(prefix + '_filterLastN')?.value) || 0;

  let filtered = exps.filter(e => {{
    const tag = (e.meta.tag || e.meta.run_id || '').toLowerCase();
    const steps = e.meta.actual_timesteps || e.meta.total_timesteps || 0;
    return tag.includes(text) && steps >= minSteps;
  }});

  if (lastN > 0) filtered = filtered.slice(-lastN);
  return filtered;
}}

function applyScenarioFilters(prefix) {{
  const filtered = getScenarioFiltered(prefix);
  const labels = filtered.map(e => e.meta.tag || e.meta.run_id);

  // Destroy existing charts for this tab
  if (_tabCharts[prefix + '_comp']) _tabCharts[prefix + '_comp'].destroy();
  if (_tabCharts[prefix + '_improv']) _tabCharts[prefix + '_improv'].destroy();

  // Comparison bar chart
  const ctx1 = document.getElementById(prefix + '_compChart')?.getContext('2d');
  if (ctx1) {{
    _tabCharts[prefix + '_comp'] = new Chart(ctx1, {{
      type: 'bar',
      data: {{
        labels,
        datasets: [
          {{ label: 'Bazna linija', data: filtered.map(e => Math.abs(e.meta.baseline_total_reward||0)), backgroundColor: '#475569', borderRadius: 3 }},
          {{ label: 'RL Agent', data: filtered.map(e => Math.abs(e.meta.rl_total_reward||0)), backgroundColor: '#4ade80', borderRadius: 3 }}
        ]
      }},
      options: chartOpts('Skupna nagrada (abs., nizja = boljsa)')
    }});
  }}

  // Improvement chart
  const ctx2 = document.getElementById(prefix + '_improvChart')?.getContext('2d');
  if (ctx2) {{
    const pcts = filtered.map(e => e.meta.improvement_pct || 0);
    _tabCharts[prefix + '_improv'] = new Chart(ctx2, {{
      type: 'bar',
      data: {{
        labels,
        datasets: [{{
          label: 'Izboljsanje %',
          data: pcts,
          backgroundColor: pcts.map(p => p >= 0 ? '#4ade80' : '#f87171'),
          borderRadius: 3,
        }}]
      }},
      options: {{
        ...chartOpts(''),
        plugins: {{
          ...chartOpts('').plugins,
          annotation: undefined,
        }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45, font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
          y: {{
            ticks: {{ color: '#94a3b8', callback: v => v + '%' }},
            grid: {{ color: '#334155' }},
          }}
        }}
      }}
    }});
  }}

  // Table
  const tbody = document.querySelector('#' + prefix + '_expTable tbody');
  if (tbody) {{
    tbody.innerHTML = filtered.map(e => {{
      const m = e.meta;
      const pct = m.improvement_pct || 0;
      const bc = badgeClass(pct);
      const steps = m.actual_timesteps || m.total_timesteps || 0;
      const hp = m.hyperparams || {{}};
      const date = m.date ? new Date(m.date).toLocaleDateString('sl-SI') : '-';
      return `<tr>
        <td><strong>${{m.tag || m.run_id}}</strong></td>
        <td>${{date}}</td>
        <td>${{fmt(steps)}}</td>
        <td>${{fmt(m.baseline_total_reward||0, 0)}}</td>
        <td>${{fmt(m.rl_total_reward||0, 0)}}</td>
        <td><span class="badge ${{bc}}">${{fmtPct(pct)}}</span></td>
        <td>${{hp.lr || '-'}}</td>
        <td>${{hp.ent_coef || '-'}}</td>
        <td>${{m.reward_fn || hp.reward_fn || '-'}}</td>
        <td>${{fmt(m.train_time_s||0, 0)}}s</td>
      </tr>`;
    }}).join('');
  }}
}}

function sortScenarioTable(prefix, col) {{
  const key = prefix + '_sortCol';
  const keyAsc = prefix + '_sortAsc';
  if (_scenarioSortState[key] === col) _scenarioSortState[keyAsc] = !_scenarioSortState[keyAsc];
  else {{ _scenarioSortState[key] = col; _scenarioSortState[keyAsc] = true; }}
  const asc = _scenarioSortState[keyAsc];

  const tbody = document.querySelector('#' + prefix + '_expTable tbody');
  if (!tbody) return;
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {{
    let va = a.cells[col].textContent.trim().replace(/[^\\d.\\-]/g, '');
    let vb = b.cells[col].textContent.trim().replace(/[^\\d.\\-]/g, '');
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

function chartOpts(title) {{
  return {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
      title: title ? {{ display: true, text: title, color: '#94a3b8', font: {{ size: 12 }} }} : {{ display: false }},
    }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45, font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
    }}
  }};
}}

// ══════════════════════════════════════════
// TAB 3: Intersection breakdown
// ══════════════════════════════════════════
function updateIntersections() {{
  const idx = parseInt(document.getElementById('intExpSelect')?.value || 0);
  const exp = experiments[idx];
  if (!exp || !exp.results) return;

  const results = exp.results;
  const names = results.map(r => r.intersection);
  const blRewards = results.map(r => Math.abs(r.baseline_reward));
  const rlRewards = results.map(r => Math.abs(r.rl_reward));
  const pcts = results.map(r => r.improvement_pct || 0);

  // Bar chart
  if (intBarInst) intBarInst.destroy();
  const ctx1 = document.getElementById('intBarChart')?.getContext('2d');
  if (ctx1) {{
    intBarInst = new Chart(ctx1, {{
      type: 'bar',
      data: {{
        labels: names,
        datasets: [
          {{ label: 'Bazna linija', data: blRewards, backgroundColor: '#475569', borderRadius: 3 }},
          {{ label: 'RL Agent', data: rlRewards, backgroundColor: names.map(n => INT_COLORS[n] || '#4ade80'), borderRadius: 3 }}
        ]
      }},
      options: {{ indexAxis: 'y', responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
          y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 11 }} }}, grid: {{ color: '#1e293b' }} }}
        }}
      }}
    }});
  }}

  // Improvement percentage chart
  if (intPctInst) intPctInst.destroy();
  const ctx2 = document.getElementById('intPctChart')?.getContext('2d');
  if (ctx2) {{
    intPctInst = new Chart(ctx2, {{
      type: 'bar',
      data: {{
        labels: names,
        datasets: [{{
          label: 'Izboljsanje %',
          data: pcts,
          backgroundColor: pcts.map(p => p >= 0 ? '#4ade80' : '#f87171'),
          borderRadius: 3,
        }}]
      }},
      options: {{ indexAxis: 'y', responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', callback: v => v + '%' }}, grid: {{ color: '#334155' }} }},
          y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 11 }} }}, grid: {{ color: '#1e293b' }} }}
        }}
      }}
    }});
  }}

  // Table
  const tbody = document.querySelector('#intTable tbody');
  if (tbody) {{
    tbody.innerHTML = results.map(r => {{
      const pct = r.improvement_pct || 0;
      return `<tr>
        <td><strong>${{r.intersection}}</strong></td>
        <td>${{fmt(r.baseline_reward, 0)}}</td>
        <td>${{fmt(r.rl_reward, 0)}}</td>
        <td><span class="badge ${{badgeClass(pct)}}">${{fmtPct(pct)}}</span></td>
      </tr>`;
    }}).join('');
  }}

  // Cross-experiment intersection trend
  if (intTrendInst) intTrendInst.destroy();
  const ctx3 = document.getElementById('intTrendChart')?.getContext('2d');
  if (ctx3 && experiments.length > 1) {{
    const intNames = experiments[0].results?.map(r => r.intersection) || [];
    const datasets = intNames.map((name, i) => ({{
      label: name,
      data: experiments.map((e, j) => {{
        const r = e.results?.find(r => r.intersection === name);
        return r ? r.improvement_pct || 0 : null;
      }}),
      borderColor: INT_COLORS[name] || COLORS[i],
      backgroundColor: 'transparent',
      tension: 0.3,
      pointRadius: 4,
      borderWidth: 2,
    }}));

    intTrendInst = new Chart(ctx3, {{
      type: 'line',
      data: {{
        labels: experiments.map(e => e.meta.tag || e.meta.run_id),
        datasets,
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
          y: {{ ticks: {{ color: '#94a3b8', callback: v => v + '%' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: 'Izboljsanje %', color: '#94a3b8' }} }}
        }}
      }}
    }});
  }}
}}

// ══════════════════════════════════════════
// TAB 4: Training curves
// ══════════════════════════════════════════
function buildCurveSelector() {{
  const container = document.getElementById('curveSelect');
  if (!container) return;

  const withData = experiments.filter(e =>
    (e.training_log && e.training_log.length > 0) ||
    (e.step_log && e.step_log.length > 0)
  );

  if (withData.length === 0) {{
    container.innerHTML = '<span style="color:#64748b;font-size:12px">Se ni podatkov o ucenju. Zazenite nov eksperiment za prikaz krivulj.</span>';
    return;
  }}

  const shown = withData.slice(-10);
  container.innerHTML = shown.map((e, i) => {{
    const name = e.meta.tag || e.meta.run_id;
    const checked = i >= shown.length - 3 ? 'checked' : '';
    return `<label class="${{checked ? 'active' : ''}}" id="lbl_${{i}}">
      <input type="checkbox" value="${{i}}" ${{checked}} onchange="toggleCurve(this, ${{i}})"> ${{name}}
    </label>`;
  }}).join('');
  window._curveExps = shown;
  updateTrainCharts();
}}

function toggleCurve(cb, idx) {{
  document.getElementById('lbl_' + idx)?.classList.toggle('active', cb.checked);
  updateTrainCharts();
}}

function updateTrainCharts() {{
  const exps = window._curveExps || [];
  const checkboxes = document.querySelectorAll('#curveSelect input:checked');
  const selected = Array.from(checkboxes).map(cb => exps[parseInt(cb.value)]).filter(Boolean);

  // Episode-level chart
  if (trainChartInst) trainChartInst.destroy();
  const ctx1 = document.getElementById('trainChart')?.getContext('2d');
  if (ctx1) {{
    const datasets = selected.filter(e => e.training_log?.length > 0).map((e, i) => ({{
      label: e.meta.tag || e.meta.run_id,
      data: e.training_log.map(l => ({{ x: l.timestep, y: l.reward }})),
      borderColor: COLORS[i % COLORS.length],
      backgroundColor: 'transparent',
      tension: 0.3, pointRadius: 2, borderWidth: 2,
    }}));

    if (datasets.length > 0) {{
      trainChartInst = new Chart(ctx1, {{
        type: 'line', data: {{ datasets }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
          scales: {{
            x: {{ type: 'linear', title: {{ display: true, text: 'Koraki', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
            y: {{ title: {{ display: true, text: 'Nagrada na epizodo', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
          }}
        }}
      }});
    }}
  }}

  // Step-level chart
  if (stepChartInst) stepChartInst.destroy();
  const ctx2 = document.getElementById('stepChart')?.getContext('2d');
  if (ctx2) {{
    const datasets = selected.filter(e => e.step_log?.length > 0).map((e, i) => ({{
      label: e.meta.tag || e.meta.run_id,
      data: e.step_log.map(l => ({{ x: l.timestep, y: l.reward_step_mean }})),
      borderColor: COLORS[i % COLORS.length],
      backgroundColor: 'transparent',
      tension: 0.4, pointRadius: 1, borderWidth: 2,
    }}));

    if (datasets.length > 0) {{
      stepChartInst = new Chart(ctx2, {{
        type: 'line', data: {{ datasets }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
          scales: {{
            x: {{ type: 'linear', title: {{ display: true, text: 'Koraki', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
            y: {{ title: {{ display: true, text: 'Povp. nagrada na korak', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
          }}
        }}
      }});
    }}
  }}
}}

// ══════════════════════════════════════════
// TAB 5: Hyperparameters
// ══════════════════════════════════════════
function updateHyperparams() {{
  // Steps vs improvement scatter
  if (hpScatter1Inst) hpScatter1Inst.destroy();
  const ctx1 = document.getElementById('hpScatter1')?.getContext('2d');
  if (ctx1) {{
    const data = experiments.map((e, i) => ({{
      x: e.meta.actual_timesteps || e.meta.total_timesteps || 0,
      y: e.meta.improvement_pct || 0,
    }}));
    hpScatter1Inst = new Chart(ctx1, {{
      type: 'scatter',
      data: {{
        datasets: [{{
          label: 'Eksperimenti',
          data,
          backgroundColor: data.map(d => d.y >= 0 ? '#4ade80' : '#f87171'),
          pointRadius: 6,
          pointHoverRadius: 9,
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              label: (ctx) => {{
                const e = experiments[ctx.dataIndex];
                return `${{e.meta.tag||e.meta.run_id}}: ${{fmtPct(e.meta.improvement_pct||0)}} (${{fmt(ctx.raw.x)}} korakov)`;
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Koraki', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
          y: {{ title: {{ display: true, text: 'Izboljsanje %', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8', callback: v => v + '%' }}, grid: {{ color: '#334155' }} }}
        }}
      }}
    }});
  }}

  // LR vs improvement scatter
  if (hpScatter2Inst) hpScatter2Inst.destroy();
  const ctx2 = document.getElementById('hpScatter2')?.getContext('2d');
  if (ctx2) {{
    const data = experiments.filter(e => e.meta.hyperparams?.lr).map(e => ({{
      x: e.meta.hyperparams.lr,
      y: e.meta.improvement_pct || 0,
    }}));
    if (data.length > 0) {{
      hpScatter2Inst = new Chart(ctx2, {{
        type: 'scatter',
        data: {{
          datasets: [{{
            label: 'Eksperimenti',
            data,
            backgroundColor: data.map(d => d.y >= 0 ? '#4ade80' : '#f87171'),
            pointRadius: 6,
            pointHoverRadius: 9,
          }}]
        }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ type: 'logarithmic', title: {{ display: true, text: 'Learning Rate', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
            y: {{ title: {{ display: true, text: 'Izboljsanje %', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8', callback: v => v + '%' }}, grid: {{ color: '#334155' }} }}
          }}
        }}
      }});
    }}
  }}

  // HP table
  const tbody = document.querySelector('#hpTable tbody');
  if (tbody) {{
    tbody.innerHTML = experiments.map(e => {{
      const hp = e.meta.hyperparams || {{}};
      const pct = e.meta.improvement_pct || 0;
      return `<tr>
        <td><strong>${{e.meta.tag || e.meta.run_id}}</strong></td>
        <td>${{hp.lr || '-'}}</td>
        <td>${{hp.n_steps || '-'}}</td>
        <td>${{hp.batch_size || '-'}}</td>
        <td>${{hp.gamma || '-'}}</td>
        <td>${{hp.ent_coef || '-'}}</td>
        <td>${{hp.clip_range || '-'}}</td>
        <td>${{hp.delta_time || '-'}}</td>
        <td><span class="badge ${{badgeClass(pct)}}">${{fmtPct(pct)}}</span></td>
      </tr>`;
    }}).join('');
  }}
}}

// ══════════════════════════════════════════
// TAB 6: Experiment details
// ══════════════════════════════════════════
function updateDetails() {{
  const idx = parseInt(document.getElementById('detailExpSelect')?.value || 0);
  const exp = experiments[idx];
  if (!exp) return;
  const m = exp.meta;
  const hp = m.hyperparams || {{}};

  let h = '<div class="grid-2">';

  // Left: meta info
  h += '<div class="card"><div class="card-title">Podatki o eksperimentu</div><div class="detail-grid">';
  const details = [
    ['ID zagona', m.run_id], ['Oznaka', m.tag || '-'],
    ['Datum', m.date ? new Date(m.date).toLocaleString('sl-SI') : '-'],
    ['Koraki (cilj)', fmt(m.total_timesteps)], ['Koraki (dejanski)', fmt(m.actual_timesteps)],
    ['Cas ucenja', (m.train_time_s||0).toFixed(0) + 's'],
    ['Max ur', m.max_hours || 'brez omejitve'],
    ['Trajanje sim.', (m.num_seconds||3600) + 's'],
    ['Pristop', m.approach || '-'],
    ['Filter agentov', m.agent_filter || '-'],
  ];
  details.forEach(([label, value]) => {{
    h += `<div class="detail-item"><span class="detail-label">${{label}}:</span> <span class="detail-value">${{value}}</span></div>`;
  }});
  h += '</div></div>';

  // Right: hyperparameters
  h += '<div class="card"><div class="card-title">Hiperparametri</div><div class="detail-grid">';
  const hpItems = [
    ['Learning rate', hp.lr], ['n_steps', hp.n_steps], ['batch_size', hp.batch_size],
    ['n_epochs', hp.n_epochs], ['gamma', hp.gamma], ['GAE lambda', hp.gae_lambda],
    ['Entropy coef', hp.ent_coef], ['Clip range', hp.clip_range],
    ['delta_time', hp.delta_time], ['yellow_time', hp.yellow_time],
    ['min_green', hp.min_green], ['max_green', hp.max_green],
  ];
  hpItems.forEach(([label, value]) => {{
    h += `<div class="detail-item"><span class="detail-label">${{label}}:</span> <span class="detail-value">${{value != null ? value : '-'}}</span></div>`;
  }});
  h += '</div></div>';

  h += '</div>'; // grid-2

  // Results summary
  h += '<div class="card"><div class="card-title">Rezultati</div>';
  h += `<div class="kpi-grid" style="margin-top:8px">`;
  h += `<div class="kpi"><div class="kpi-label">Bazna linija</div><div class="kpi-value neutral">${{fmt(m.baseline_total_reward||0,0)}}</div></div>`;
  h += `<div class="kpi"><div class="kpi-label">RL Agent</div><div class="kpi-value neutral">${{fmt(m.rl_total_reward||0,0)}}</div></div>`;
  h += `<div class="kpi"><div class="kpi-label">Izboljsanje</div><div class="kpi-value ${{pctClass(m.improvement_pct||0)}}">${{fmtPct(m.improvement_pct||0)}}</div></div>`;
  h += '</div>';
  h += '</div>';

  // TLS IDs
  if (m.ts_names) {{
    h += '<div class="card"><div class="card-title">Kontrolirana krizisca</div>';
    Object.entries(m.ts_names).forEach(([id, name]) => {{
      h += `<div style="font-size:12px;margin:3px 0"><strong>${{name}}</strong>: <span style="color:#64748b;font-size:10px">${{id}}</span></div>`;
    }});
    h += '</div>';
  }}

  document.getElementById('detailPanel').innerHTML = h;
}}

function updateExplanations() {{
  const idx = parseInt(document.getElementById('explExpSelect')?.value || 0);
  const exp = experiments[idx];
  if (!exp) return;

  let h = '';
  const runId = exp.meta.run_id;
  const expl = exp.explanations || {{}};
  const cats = Object.keys(expl);

  if (cats.length === 0) {{
    h = '<div class="no-data"><p>Ni podatkov o interpretibilnosti za ta eksperiment.</p><p style="font-size:13px;margin-top:8px">Zazenite: <code>python src/collect_states.py --model_path results/experiments/' + runId + '/ppo_shared_policy.zip</code><br>Nato: <code>python src/explain.py --data_path results/experiments/' + runId + '/harvested_data.pkl</code></p></div>';
  }} else {{
    // Category display config: title, grid style, image label generator
    const catConfig = {{
      'decision-trees': {{
        title: 'Odlocitvena drevesa (Decision Trees)',
        gridStyle: 'grid-column: span 2;',
        label: f => f.replace('.png','').replace('.jpg',''),
      }},
      'shap': {{
        title: 'Pomembnost znacilk (SHAP)',
        gridStyle: '',
        label: f => f.replace('.png','').replace('.jpg',''),
      }},
      'umap': {{
        title: 'Latentni prostor — UMAP projekcije',
        gridStyle: '',
        label: f => {{
          const n = f.replace('.png','').replace('.jpg','');
          if (n.includes('actions')) return n.includes('pca') ? 'PCA + UMAP: po akcijah' : 'UMAP: po akcijah';
          if (n.includes('intersections')) return n.includes('pca') ? 'PCA + UMAP: po kriziscih' : 'UMAP: po kriziscih';
          if (n.includes('time')) return n.includes('pca') ? 'PCA + UMAP: po casu dneva' : 'UMAP: po casu dneva';
          return n;
        }},
      }},
      't-sne': {{
        title: 'Latentni prostor — t-SNE projekcije',
        gridStyle: '',
        label: f => {{
          const n = f.replace('.png','').replace('.jpg','');
          if (n.includes('actions')) return n.includes('pca') ? 'PCA + t-SNE: po akcijah' : 't-SNE: po akcijah';
          if (n.includes('intersections')) return n.includes('pca') ? 'PCA + t-SNE: po kriziscih' : 't-SNE: po kriziscih';
          if (n.includes('time')) return n.includes('pca') ? 'PCA + t-SNE: po casu dneva' : 't-SNE: po casu dneva';
          return n;
        }},
      }},
    }};

    // Display order: decision-trees first (full-width), then shap, umap, t-sne
    const order = ['decision-trees', 'shap', 'umap', 't-sne'];
    // Add any extra categories not in predefined order
    cats.filter(c => !order.includes(c) && c !== '_flat').forEach(c => order.push(c));

    order.forEach(cat => {{
      if (!expl[cat] || !expl[cat].images || expl[cat].images.length === 0) return;
      const cfg = catConfig[cat] || {{ title: cat, gridStyle: '', label: f => f.replace('.png','') }};
      const imgs = expl[cat].images;

      h += `<div class="card" style="${{cfg.gridStyle}}"><div class="card-title">${{cfg.title}} (${{imgs.length}})</div>`;
      h += '<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap:12px;">';
      imgs.forEach(f => {{
        const label = cfg.label(f);
        h += `<div style="text-align:center;">
                <img src="experiments/${{runId}}/explanations/${{cat}}/${{f}}" style="max-width:100%; border-radius:4px; border:1px solid #334155;" loading="lazy">
                <div class="info-text" style="margin-top:4px">${{label}}</div>
              </div>`;
      }});
      h += '</div></div>';
    }});

    // Legacy flat files (if any)
    if (expl['_flat'] && expl['_flat'].images.length > 0) {{
      h += '<div class="card" style="grid-column: span 2;"><div class="card-title">Vizualizacije</div>';
      expl['_flat'].images.forEach(f => {{
        h += `<div style="margin-bottom:15px; text-align:center;">
                <img src="experiments/${{runId}}/explanations/${{f}}" style="max-width:100%; border-radius:4px; border:1px solid #334155;" loading="lazy">
                <div class="info-text">${{f.replace('.png','')}}</div>
              </div>`;
      }});
      h += '</div>';
    }}
  }}

  document.getElementById('explPanel').innerHTML = h;
}}

// ══════════════════════════════════════════
// TAB 2: Mega-politike
// ══════════════════════════════════════════

// Error bar plugin for Chart.js
const errorBarPlugin = {{
  id: 'errorBars',
  afterDatasetsDraw(chart) {{
    const datasets = chart.data.datasets;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1.5;
    datasets.forEach((ds, dsIdx) => {{
      if (!ds._errorBars) return;
      const meta = chart.getDatasetMeta(dsIdx);
      meta.data.forEach((bar, i) => {{
        if (!ds._errorBars[i]) return;
        const x = bar.x;
        const yLow = chart.scales.y.getPixelForValue(ds._errorBars[i][0]);
        const yHigh = chart.scales.y.getPixelForValue(ds._errorBars[i][1]);
        ctx.beginPath();
        ctx.moveTo(x, yLow); ctx.lineTo(x, yHigh);
        ctx.moveTo(x - 4, yLow); ctx.lineTo(x + 4, yLow);
        ctx.moveTo(x - 4, yHigh); ctx.lineTo(x + 4, yHigh);
        ctx.stroke();
      }});
    }});
    ctx.restore();
  }}
}};

function updateMegaOverall() {{
  if (!megaData) return;
  const conds = megaData.conditions;
  const comps = megaData.comparisons;

  // Sort: baseline first, then megapolicies by tag
  const sorted = [...conds].sort((a, b) => {{
    if (a.is_baseline) return -1;
    if (b.is_baseline) return 1;
    return a.tag.localeCompare(b.tag);
  }});

  const labels = sorted.map(c => c.tag === 'baseline' ? 'Bazna linija' : c.tag);
  const means = sorted.map(c => Math.abs(c.total_reward.mean));
  const errorBars = sorted.map(c => [Math.abs(c.total_reward.ci_high), Math.abs(c.total_reward.ci_low)]);
  const bgColors = sorted.map(c => c.is_baseline ? '#475569' : '#4ade80');

  // Overall bar chart with error bars
  if (megaOverallInst) megaOverallInst.destroy();
  const ctx1 = document.getElementById('megaOverallChart')?.getContext('2d');
  if (ctx1) {{
    const ds = {{
      label: 'Povp. |nagrada|',
      data: means,
      backgroundColor: bgColors,
      borderRadius: 3,
      _errorBars: errorBars,
    }};
    megaOverallInst = new Chart(ctx1, {{
      type: 'bar',
      data: {{ labels, datasets: [ds] }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              afterBody: (items) => {{
                const i = items[0].dataIndex;
                const c = sorted[i];
                return `95% IZ: [${{fmt(c.total_reward.ci_low,0)}}, ${{fmt(c.total_reward.ci_high,0)}}]\\nStd: ${{fmt(c.total_reward.std,0)}}`;
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45, font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: '|Skupna nagrada| (nizja = boljsa)', color: '#94a3b8' }} }}
        }}
      }},
      plugins: [errorBarPlugin]
    }});
  }}

  // Improvement chart (sorted by improvement %)
  if (megaImprovInst) megaImprovInst.destroy();
  const ctx2 = document.getElementById('megaImprovChart')?.getContext('2d');
  if (ctx2) {{
    const sortedComps = [...comps].sort((a, b) => b.improvement_pct - a.improvement_pct);
    megaImprovInst = new Chart(ctx2, {{
      type: 'bar',
      data: {{
        labels: sortedComps.map(c => c.tag),
        datasets: [{{
          label: 'Izboljsanje %',
          data: sortedComps.map(c => c.improvement_pct),
          backgroundColor: sortedComps.map(c => c.improvement_pct >= 0 ? '#4ade80' : '#f87171'),
          borderRadius: 3,
        }}]
      }},
      options: {{
        indexAxis: 'y',
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              afterBody: (items) => {{
                const tag = items[0].label;
                const c = comps.find(x => x.tag === tag);
                if (!c) return '';
                return `Parni t-test p=${{c.paired_t_p < 0.0001 ? c.paired_t_p.toExponential(2) : c.paired_t_p.toFixed(4)}}\\nCohen d=${{c.cohens_d.toFixed(3)}}`;
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', callback: v => v + '%' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: 'Izboljsanje glede na bazno linijo (%)', color: '#94a3b8' }} }},
          y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 11 }} }}, grid: {{ color: '#1e293b' }} }}
        }}
      }}
    }});
  }}
}}

function updateMegaIntersections() {{
  if (!megaData) return;
  const sel = document.getElementById('megaIntSelect');
  if (!sel) return;
  const selected = Array.from(sel.selectedOptions).map(o => o.value);
  if (selected.length === 0) return;

  const intNames = megaData.intersection_names;
  const blCond = megaData.conditions.find(c => c.is_baseline);

  // Reward bar chart
  if (megaIntRewardInst) megaIntRewardInst.destroy();
  const ctx1 = document.getElementById('megaIntRewardChart')?.getContext('2d');
  if (ctx1) {{
    const datasets = [];
    // Baseline dataset
    if (blCond) {{
      datasets.push({{
        label: 'Bazna linija',
        data: intNames.map(n => Math.abs(blCond.intersections[n].reward.mean)),
        backgroundColor: '#475569',
        borderRadius: 3,
      }});
    }}
    // Selected megapolicies
    selected.forEach((tag, idx) => {{
      const cond = megaData.conditions.find(c => c.tag === tag);
      if (!cond) return;
      datasets.push({{
        label: tag,
        data: intNames.map(n => Math.abs(cond.intersections[n].reward.mean)),
        backgroundColor: COLORS[idx % COLORS.length],
        borderRadius: 3,
      }});
    }});
    megaIntRewardInst = new Chart(ctx1, {{
      type: 'bar',
      data: {{ labels: intNames, datasets }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: '|Nagrada| (nizja = boljsa)', color: '#94a3b8' }} }}
        }}
      }}
    }});
  }}

  // Improvement % chart
  if (megaIntImprovInst) megaIntImprovInst.destroy();
  const ctx2 = document.getElementById('megaIntImprovChart')?.getContext('2d');
  if (ctx2) {{
    const datasets = selected.map((tag, idx) => {{
      const comp = megaData.comparisons.find(c => c.tag === tag);
      if (!comp) return null;
      return {{
        label: tag,
        data: intNames.map(n => comp.intersections[n].improvement_pct),
        backgroundColor: COLORS[idx % COLORS.length],
        borderRadius: 3,
      }};
    }}).filter(Boolean);
    megaIntImprovInst = new Chart(ctx2, {{
      type: 'bar',
      data: {{ labels: intNames, datasets }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
          y: {{ ticks: {{ color: '#94a3b8', callback: v => v + '%' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: 'Izboljsanje (%)', color: '#94a3b8' }} }}
        }}
      }}
    }});
  }}
}}

function updateMegaWindows() {{
  if (!megaData) return;
  const sel = document.getElementById('megaWinSelect');
  if (!sel) return;
  const selected = Array.from(sel.selectedOptions).map(o => o.value);
  if (selected.length === 0) return;

  const winNames = megaData.window_names;
  const winLabels = megaData.window_labels;
  const labels = winNames.map(w => winLabels[w] || w);
  const blCond = megaData.conditions.find(c => c.is_baseline);

  // Reward bar chart
  if (megaWindowRewardInst) megaWindowRewardInst.destroy();
  const ctx1 = document.getElementById('megaWindowRewardChart')?.getContext('2d');
  if (ctx1) {{
    const datasets = [];
    if (blCond) {{
      datasets.push({{
        label: 'Bazna linija',
        data: winNames.map(w => Math.abs(blCond.windows[w].reward.mean)),
        backgroundColor: '#475569',
        borderRadius: 3,
      }});
    }}
    selected.forEach((tag, idx) => {{
      const cond = megaData.conditions.find(c => c.tag === tag);
      if (!cond) return;
      datasets.push({{
        label: tag,
        data: winNames.map(w => Math.abs(cond.windows[w].reward.mean)),
        backgroundColor: COLORS[idx % COLORS.length],
        borderRadius: 3,
      }});
    }});
    megaWindowRewardInst = new Chart(ctx1, {{
      type: 'bar',
      data: {{ labels, datasets }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45, font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: '|Nagrada| (nizja = boljsa)', color: '#94a3b8' }} }}
        }}
      }}
    }});
  }}

  // Improvement chart
  if (megaWindowImprovInst) megaWindowImprovInst.destroy();
  const ctx2 = document.getElementById('megaWindowImprovChart')?.getContext('2d');
  if (ctx2) {{
    const datasets = selected.map((tag, idx) => {{
      const comp = megaData.comparisons.find(c => c.tag === tag);
      if (!comp) return null;
      return {{
        label: tag,
        data: winNames.map(w => comp.windows[w].improvement_pct),
        backgroundColor: COLORS[idx % COLORS.length],
        borderRadius: 3,
      }};
    }}).filter(Boolean);
    megaWindowImprovInst = new Chart(ctx2, {{
      type: 'bar',
      data: {{ labels, datasets }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45, font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
          y: {{ ticks: {{ color: '#94a3b8', callback: v => v + '%' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: 'Izboljsanje (%)', color: '#94a3b8' }} }}
        }}
      }}
    }});
  }}
}}

function updateMegaDetail() {{
  if (!megaData) return;
  const sel = document.getElementById('megaDetailSelect');
  if (!sel) return;
  const tag = sel.value;
  const comp = megaData.comparisons.find(c => c.tag === tag);
  const cond = megaData.conditions.find(c => c.tag === tag);
  const panel = document.getElementById('megaDetailPanel');
  if (!comp || !cond || !panel) return;

  let h = '';

  // Model info
  h += '<div class="card"><div class="card-title">Modeli</div>';
  h += `<div class="detail-grid">`;
  h += `<div class="detail-item"><span class="detail-label">Jutranji model:</span> <span class="detail-value">${{tag.substring(0,2)}} — ${{cond.morning_model}}</span></div>`;
  h += `<div class="detail-item"><span class="detail-label">Vecerni model:</span> <span class="detail-value">${{tag.substring(2)}} — ${{cond.evening_model}}</span></div>`;
  h += `<div class="detail-item"><span class="detail-label">Ponovitve:</span> <span class="detail-value">${{cond.n}}</span></div>`;
  h += `<div class="detail-item"><span class="detail-label">Skupno izboljsanje:</span> <span class="detail-value ${{pctClass(comp.improvement_pct)}}">${{fmtPct(comp.improvement_pct)}}</span></div>`;
  h += '</div></div>';

  // Per-intersection table
  h += '<div class="card"><div class="card-title">Primerjava po kriziscih</div>';
  h += '<div class="table-wrap"><table><thead><tr>';
  h += '<th>Krizisce</th><th>Bazna (povp.)</th><th>Mega (povp.)</th><th>Izboljsanje</th>';
  h += '<th>Parni p</th><th>Wilcoxon p</th><th>Cohen d</th>';
  h += '</tr></thead><tbody>';
  const blCond = megaData.conditions.find(c => c.is_baseline);
  megaData.intersection_names.forEach(iname => {{
    const ic = comp.intersections[iname];
    const blR = blCond ? blCond.intersections[iname].reward.mean : 0;
    const mR = cond.intersections[iname].reward.mean;
    const pB = ic.paired_t_p < 0.001 ? 'badge-green' : ic.paired_t_p < 0.01 ? 'badge-green' : ic.paired_t_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const pL = ic.paired_t_p < 0.001 ? '***' : ic.paired_t_p < 0.01 ? '**' : ic.paired_t_p < 0.05 ? '*' : 'n.s.';
    const wB = ic.wilcoxon_p < 0.001 ? 'badge-green' : ic.wilcoxon_p < 0.01 ? 'badge-green' : ic.wilcoxon_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const wL = ic.wilcoxon_p < 0.001 ? '***' : ic.wilcoxon_p < 0.01 ? '**' : ic.wilcoxon_p < 0.05 ? '*' : 'n.s.';
    const absD = Math.abs(ic.cohens_d);
    const dL = absD > 0.8 ? 'Velik' : absD > 0.5 ? 'Srednji' : absD > 0.2 ? 'Majhen' : 'Zanem.';
    const dB = absD > 0.8 ? (ic.cohens_d > 0 ? 'badge-green' : 'badge-red') : absD > 0.5 ? 'badge-yellow' : 'badge-gray';
    h += `<tr>
      <td><strong style="color:${{INT_COLORS[iname]||'#e2e8f0'}}">${{iname}}</strong></td>
      <td>${{fmt(blR, 0)}}</td>
      <td>${{fmt(mR, 0)}}</td>
      <td><span class="badge ${{badgeClass(ic.improvement_pct)}}">${{fmtPct(ic.improvement_pct)}}</span></td>
      <td><span class="badge ${{pB}}">${{ic.paired_t_p < 0.0001 ? ic.paired_t_p.toExponential(2) : ic.paired_t_p.toFixed(4)}} ${{pL}}</span></td>
      <td><span class="badge ${{wB}}">${{ic.wilcoxon_p < 0.0001 ? ic.wilcoxon_p.toExponential(2) : ic.wilcoxon_p.toFixed(4)}} ${{wL}}</span></td>
      <td><span class="badge ${{dB}}">${{ic.cohens_d.toFixed(3)}} (${{dL}})</span></td>
    </tr>`;
  }});
  h += '</tbody></table></div></div>';

  // Per-window table
  h += '<div class="card"><div class="card-title">Primerjava po casovnih oknih</div>';
  h += '<div class="table-wrap"><table><thead><tr>';
  h += '<th>Casovno okno</th><th>Bazna (povp.)</th><th>Mega (povp.)</th><th>Izboljsanje</th>';
  h += '<th>Parni p</th><th>Wilcoxon p</th><th>Cohen d</th>';
  h += '</tr></thead><tbody>';
  megaData.window_names.forEach(wname => {{
    const wc = comp.windows[wname];
    const blR = blCond ? blCond.windows[wname].reward.mean : 0;
    const mR = cond.windows[wname].reward.mean;
    const wLabel = megaData.window_labels[wname] || wname;
    const pB = wc.paired_t_p < 0.001 ? 'badge-green' : wc.paired_t_p < 0.01 ? 'badge-green' : wc.paired_t_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const pL = wc.paired_t_p < 0.001 ? '***' : wc.paired_t_p < 0.01 ? '**' : wc.paired_t_p < 0.05 ? '*' : 'n.s.';
    const wB = wc.wilcoxon_p < 0.001 ? 'badge-green' : wc.wilcoxon_p < 0.01 ? 'badge-green' : wc.wilcoxon_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const wL = wc.wilcoxon_p < 0.001 ? '***' : wc.wilcoxon_p < 0.01 ? '**' : wc.wilcoxon_p < 0.05 ? '*' : 'n.s.';
    const absD = Math.abs(wc.cohens_d);
    const dL = absD > 0.8 ? 'Velik' : absD > 0.5 ? 'Srednji' : absD > 0.2 ? 'Majhen' : 'Zanem.';
    const dB = absD > 0.8 ? (wc.cohens_d > 0 ? 'badge-green' : 'badge-red') : absD > 0.5 ? 'badge-yellow' : 'badge-gray';
    // Highlight rush hour rows
    const rowStyle = (wname === 'morning_rush' || wname === 'evening_rush') ? 'background:#1e293b' : '';
    h += `<tr style="${{rowStyle}}">
      <td><strong>${{wLabel}}</strong></td>
      <td>${{fmt(blR, 0)}}</td>
      <td>${{fmt(mR, 0)}}</td>
      <td><span class="badge ${{badgeClass(wc.improvement_pct)}}">${{fmtPct(wc.improvement_pct)}}</span></td>
      <td><span class="badge ${{pB}}">${{wc.paired_t_p < 0.0001 ? wc.paired_t_p.toExponential(2) : wc.paired_t_p.toFixed(4)}} ${{pL}}</span></td>
      <td><span class="badge ${{wB}}">${{wc.wilcoxon_p < 0.0001 ? wc.wilcoxon_p.toExponential(2) : wc.wilcoxon_p.toFixed(4)}} ${{wL}}</span></td>
      <td><span class="badge ${{dB}}">${{wc.cohens_d.toFixed(3)}} (${{dL}})</span></td>
    </tr>`;
  }});
  h += '</tbody></table></div></div>';

  panel.innerHTML = h;
}}

// ══════════════════════════════════════════
// TAB 8: Generalizacija (rush-hour tests)
// ══════════════════════════════════════════

function updateRushOverall(sk) {{
  if (!rushData || !rushData.scenarios[sk]) return;
  const sc = rushData.scenarios[sk];
  const conds = sc.conditions;
  const comps = sc.comparisons;

  // Sort: baseline first, then by tag
  const sorted = [...conds].sort((a, b) => {{
    if (a.is_baseline) return -1;
    if (b.is_baseline) return 1;
    return a.tag.localeCompare(b.tag);
  }});

  const labels = sorted.map(c => c.is_baseline ? 'Bazna linija' : c.tag);
  const means = sorted.map(c => Math.abs(c.total_reward.mean));
  const errorBars = sorted.map(c => [Math.abs(c.total_reward.ci_high), Math.abs(c.total_reward.ci_low)]);
  const bgColors = sorted.map(c => c.is_baseline ? '#475569' : '#4ade80');

  // Overall bar chart with error bars
  const k1 = sk + '_overall';
  if (_rushCharts[k1]) _rushCharts[k1].destroy();
  const ctx1 = document.getElementById('rush_' + sk + '_overall')?.getContext('2d');
  if (ctx1) {{
    _rushCharts[k1] = new Chart(ctx1, {{
      type: 'bar',
      data: {{ labels, datasets: [{{
        label: 'Povp. |nagrada|',
        data: means,
        backgroundColor: bgColors,
        borderRadius: 3,
        _errorBars: errorBars,
      }}] }},
      options: {{
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              afterBody: (items) => {{
                const i = items[0].dataIndex;
                const c = sorted[i];
                return `95% IZ: [${{fmt(c.total_reward.ci_low,0)}}, ${{fmt(c.total_reward.ci_high,0)}}]\\nStd: ${{fmt(c.total_reward.std,0)}}`;
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45, font: {{ size: 10 }} }}, grid: {{ color: '#1e293b' }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: '|Skupna nagrada| (nizja = boljsa)', color: '#94a3b8' }} }}
        }}
      }},
      plugins: [errorBarPlugin]
    }});
  }}

  // Improvement chart
  const k2 = sk + '_improv';
  if (_rushCharts[k2]) _rushCharts[k2].destroy();
  const ctx2 = document.getElementById('rush_' + sk + '_improv')?.getContext('2d');
  if (ctx2) {{
    const sortedComps = [...comps].sort((a, b) => b.improvement_pct - a.improvement_pct);
    _rushCharts[k2] = new Chart(ctx2, {{
      type: 'bar',
      data: {{
        labels: sortedComps.map(c => c.tag),
        datasets: [{{
          label: 'Izboljsanje %',
          data: sortedComps.map(c => c.improvement_pct),
          backgroundColor: sortedComps.map(c => c.improvement_pct >= 0 ? '#4ade80' : '#f87171'),
          borderRadius: 3,
        }}]
      }},
      options: {{
        indexAxis: 'y',
        responsive: true,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              afterBody: (items) => {{
                const tag = items[0].label;
                const c = comps.find(x => x.tag === tag);
                if (!c) return '';
                return `Parni t-test p=${{c.paired_t_p < 0.0001 ? c.paired_t_p.toExponential(2) : c.paired_t_p.toFixed(4)}}\\nCohen d=${{c.cohens_d.toFixed(3)}}`;
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', callback: v => v + '%' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: 'Izboljsanje glede na bazno linijo (%)', color: '#94a3b8' }} }},
          y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 11 }} }}, grid: {{ color: '#1e293b' }} }}
        }}
      }}
    }});
  }}
}}

function updateRushIntersections(sk) {{
  if (!rushData || !rushData.scenarios[sk]) return;
  const sc = rushData.scenarios[sk];
  const sel = document.getElementById('rush_' + sk + '_intSelect');
  if (!sel) return;
  const selected = Array.from(sel.selectedOptions).map(o => o.value);
  if (selected.length === 0) return;

  const intNames = rushData.intersection_names;
  const blCond = sc.conditions.find(c => c.is_baseline);

  // Reward bar chart
  const k1 = sk + '_intReward';
  if (_rushCharts[k1]) _rushCharts[k1].destroy();
  const ctx1 = document.getElementById('rush_' + sk + '_intReward')?.getContext('2d');
  if (ctx1) {{
    const datasets = [];
    if (blCond) {{
      datasets.push({{
        label: 'Bazna linija',
        data: intNames.map(n => Math.abs(blCond.intersections[n].reward.mean)),
        backgroundColor: '#475569',
        borderRadius: 3,
      }});
    }}
    selected.forEach((tag, idx) => {{
      const cond = sc.conditions.find(c => c.tag === tag);
      if (!cond) return;
      datasets.push({{
        label: tag,
        data: intNames.map(n => Math.abs(cond.intersections[n].reward.mean)),
        backgroundColor: COLORS[idx % COLORS.length],
        borderRadius: 3,
      }});
    }});
    _rushCharts[k1] = new Chart(ctx1, {{
      type: 'bar',
      data: {{ labels: intNames, datasets }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: '|Nagrada| (nizja = boljsa)', color: '#94a3b8' }} }}
        }}
      }}
    }});
  }}

  // Improvement % chart
  const k2 = sk + '_intImprov';
  if (_rushCharts[k2]) _rushCharts[k2].destroy();
  const ctx2 = document.getElementById('rush_' + sk + '_intImprov')?.getContext('2d');
  if (ctx2) {{
    const datasets = selected.map((tag, idx) => {{
      const comp = sc.comparisons.find(c => c.tag === tag);
      if (!comp) return null;
      return {{
        label: tag,
        data: intNames.map(n => comp.intersections[n].improvement_pct),
        backgroundColor: COLORS[idx % COLORS.length],
        borderRadius: 3,
      }};
    }}).filter(Boolean);
    _rushCharts[k2] = new Chart(ctx2, {{
      type: 'bar',
      data: {{ labels: intNames, datasets }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
          y: {{ ticks: {{ color: '#94a3b8', callback: v => v + '%' }}, grid: {{ color: '#334155' }},
               title: {{ display: true, text: 'Izboljsanje (%)', color: '#94a3b8' }} }}
        }}
      }}
    }});
  }}
}}

function updateRushDetail(sk) {{
  if (!rushData || !rushData.scenarios[sk]) return;
  const sc = rushData.scenarios[sk];
  const sel = document.getElementById('rush_' + sk + '_detailSelect');
  if (!sel) return;
  const tag = sel.value;
  const comp = sc.comparisons.find(c => c.tag === tag);
  const cond = sc.conditions.find(c => c.tag === tag);
  const panel = document.getElementById('rush_' + sk + '_detailPanel');
  if (!comp || !cond || !panel) return;

  let h = '';

  // Model info
  h += '<div class="card"><div class="card-title">Model</div>';
  h += `<div class="detail-grid">`;
  h += `<div class="detail-item"><span class="detail-label">Oznaka:</span> <span class="detail-value">${{cond.model_key}}</span></div>`;
  h += `<div class="detail-item"><span class="detail-label">Opis:</span> <span class="detail-value">${{cond.model_desc}}</span></div>`;
  h += `<div class="detail-item"><span class="detail-label">Ponovitve:</span> <span class="detail-value">${{cond.n}}</span></div>`;
  h += `<div class="detail-item"><span class="detail-label">Skupno izboljsanje:</span> <span class="detail-value ${{pctClass(comp.improvement_pct)}}">${{fmtPct(comp.improvement_pct)}}</span></div>`;
  h += '</div></div>';

  // Per-intersection table
  const blCond = sc.conditions.find(c => c.is_baseline);
  h += '<div class="card"><div class="card-title">Primerjava po kriziscih (nagrada)</div>';
  h += '<div class="table-wrap"><table><thead><tr>';
  h += '<th>Krizisce</th><th>Bazna (povp.)</th><th>Politika (povp.)</th><th>Izboljsanje</th>';
  h += '<th>Parni p</th><th>Wilcoxon p</th><th>Cohen d</th>';
  h += '</tr></thead><tbody>';
  rushData.intersection_names.forEach(iname => {{
    const ic = comp.intersections[iname];
    const blR = blCond ? blCond.intersections[iname].reward.mean : 0;
    const mR = cond.intersections[iname].reward.mean;
    const pB = ic.paired_t_p < 0.001 ? 'badge-green' : ic.paired_t_p < 0.01 ? 'badge-green' : ic.paired_t_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const pL = ic.paired_t_p < 0.001 ? '***' : ic.paired_t_p < 0.01 ? '**' : ic.paired_t_p < 0.05 ? '*' : 'n.s.';
    const wB = ic.wilcoxon_p < 0.001 ? 'badge-green' : ic.wilcoxon_p < 0.01 ? 'badge-green' : ic.wilcoxon_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const wL = ic.wilcoxon_p < 0.001 ? '***' : ic.wilcoxon_p < 0.01 ? '**' : ic.wilcoxon_p < 0.05 ? '*' : 'n.s.';
    const absD = Math.abs(ic.cohens_d);
    const dL = absD > 0.8 ? 'Velik' : absD > 0.5 ? 'Srednji' : absD > 0.2 ? 'Majhen' : 'Zanem.';
    const dB = absD > 0.8 ? (ic.cohens_d > 0 ? 'badge-green' : 'badge-red') : absD > 0.5 ? 'badge-yellow' : 'badge-gray';
    h += `<tr>
      <td><strong style="color:${{INT_COLORS[iname]||'#e2e8f0'}}">${{iname}}</strong></td>
      <td>${{fmt(blR, 0)}}</td>
      <td>${{fmt(mR, 0)}}</td>
      <td><span class="badge ${{badgeClass(ic.improvement_pct)}}">${{fmtPct(ic.improvement_pct)}}</span></td>
      <td><span class="badge ${{pB}}">${{ic.paired_t_p < 0.0001 ? ic.paired_t_p.toExponential(2) : ic.paired_t_p.toFixed(4)}} ${{pL}}</span></td>
      <td><span class="badge ${{wB}}">${{ic.wilcoxon_p < 0.0001 ? ic.wilcoxon_p.toExponential(2) : ic.wilcoxon_p.toFixed(4)}} ${{wL}}</span></td>
      <td><span class="badge ${{dB}}">${{ic.cohens_d.toFixed(3)}} (${{dL}})</span></td>
    </tr>`;
  }});
  h += '</tbody></table></div></div>';

  // Additional KPI comparison (queue and wait)
  h += '<div class="card"><div class="card-title">Dodatne metrike</div>';
  h += '<div class="table-wrap"><table><thead><tr>';
  h += '<th>Metrika</th><th>Bazna (povp.)</th><th>Politika (povp.)</th><th>Izboljsanje</th>';
  h += '<th>Parni p</th><th>Cohen d</th>';
  h += '</tr></thead><tbody>';
  const kpis = [
    ['Povp. vrsta', 'avg_queue', blCond ? blCond.avg_queue.mean : 0, cond.avg_queue.mean, comp.kpi_comparisons.avg_queue, 1],
    ['Povp. cakanje (s)', 'avg_wait', blCond ? blCond.avg_wait.mean : 0, cond.avg_wait.mean, comp.kpi_comparisons.avg_wait, 0],
  ];
  kpis.forEach(([label, key, blVal, mVal, kc, dec]) => {{
    const pB = kc.paired_t_p < 0.001 ? 'badge-green' : kc.paired_t_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const pL = kc.paired_t_p < 0.001 ? '***' : kc.paired_t_p < 0.01 ? '**' : kc.paired_t_p < 0.05 ? '*' : 'n.s.';
    // Negate for queue/wait (lower is better)
    const dispPct = -kc.improvement_pct;
    const dispD = -kc.cohens_d;
    const absD = Math.abs(dispD);
    const dL = absD > 0.8 ? 'Velik' : absD > 0.5 ? 'Srednji' : absD > 0.2 ? 'Majhen' : 'Zanem.';
    const dB = absD > 0.8 ? (dispD > 0 ? 'badge-green' : 'badge-red') : absD > 0.5 ? 'badge-yellow' : 'badge-gray';
    h += `<tr>
      <td><strong>${{label}}</strong></td>
      <td>${{fmt(blVal, dec)}}</td>
      <td>${{fmt(mVal, dec)}}</td>
      <td><span class="badge ${{badgeClass(dispPct)}}">${{fmtPct(dispPct)}}</span></td>
      <td><span class="badge ${{pB}}">${{kc.paired_t_p < 0.0001 ? kc.paired_t_p.toExponential(2) : kc.paired_t_p.toFixed(4)}} ${{pL}}</span></td>
      <td><span class="badge ${{dB}}">${{dispD.toFixed(3)}} (${{dL}})</span></td>
    </tr>`;
  }});
  h += '</tbody></table></div></div>';

  panel.innerHTML = h;
}}

</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved to: {output_path}")


def find_incomplete_experiments():
    """Find experiments that have a trained model but no evaluation results."""
    incomplete = []
    if not os.path.exists(EXPERIMENTS_DIR):
        return incomplete

    for run_id in sorted(os.listdir(EXPERIMENTS_DIR)):
        run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
        meta_path = os.path.join(run_dir, "meta.json")
        model_path = os.path.join(run_dir, "ppo_shared_policy.zip")
        results_path = os.path.join(run_dir, "results.csv")

        if not os.path.exists(meta_path) or not os.path.exists(model_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        if os.path.exists(results_path) and meta.get("baseline_total_reward") is not None:
            continue

        incomplete.append(run_id)

    return incomplete


def prompt_supplement(incomplete):
    """Ask user whether to run supplement_missing_results.py first.

    Returns True if supplement was run (dashboard should reload experiments).
    """
    print(f"\nFound {len(incomplete)} experiment(s) with trained models but no evaluation results:")
    for rid in incomplete[:10]:
        print(f"  - {rid}")
    if len(incomplete) > 10:
        print(f"  ... and {len(incomplete) - 10} more")

    print(f"\nThese won't appear in the dashboard without evaluation.")
    print(f"Options:")
    print(f"  [y] Run evaluation now (python src/supplement_missing_results.py --num-workers 4)")
    print(f"  [n] Skip — generate dashboard with available results only")

    try:
        choice = input("\nEvaluate missing experiments? [y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if choice not in ("y", "yes"):
        return False

    import subprocess
    import sys

    # Ask for worker count
    try:
        workers_input = input("Number of parallel workers (default 4, max ~your CPU cores): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        workers_input = ""

    num_workers = int(workers_input) if workers_input.isdigit() and int(workers_input) > 0 else 4

    print(f"\nRunning supplement_missing_results.py with {num_workers} workers...\n")

    result = subprocess.run(
        [sys.executable, "src/supplement_missing_results.py", "--num-workers", str(num_workers)],
        env={**os.environ, "LIBSUMO_AS_TRACI": os.environ.get("LIBSUMO_AS_TRACI", "1")},
    )

    if result.returncode != 0:
        print(f"\nSupplement script exited with code {result.returncode}.")
        print("Generating dashboard with whatever results are available.\n")

    return True


def find_missing_explanations():
    """Find experiments that have a trained model but no explanations/ folder."""
    missing = []
    if not os.path.exists(EXPERIMENTS_DIR):
        return missing

    for run_id in sorted(os.listdir(EXPERIMENTS_DIR)):
        run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
        model_path = os.path.join(run_dir, "ppo_shared_policy.zip")
        expl_dir = os.path.join(run_dir, "explanations")

        if os.path.exists(model_path) and not os.path.isdir(expl_dir):
            missing.append(run_id)

    return missing


def prompt_generate_explanations(missing):
    """Ask user whether to generate explanations for experiments missing them."""
    print(f"\nFound {len(missing)} experiment(s) with trained models but no explanations:")
    for rid in missing[:10]:
        print(f"  - {rid}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")

    print(f"\nThe Interpretibilnost tab will be empty for these experiments.")
    print(f"Options:")
    print(f"  [y] Generate explanations now (collect_states + explain.py)")
    print(f"  [n] Skip — generate dashboard without explanations")

    try:
        choice = input("\nGenerate missing explanations? [y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if choice not in ("y", "yes"):
        return False

    import subprocess as _sp

    try:
        workers_input = input("Number of parallel workers (default 4, max ~your CPU cores): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        workers_input = ""

    num_workers = int(workers_input) if workers_input.isdigit() and int(workers_input) > 0 else 4

    print(f"\nRunning generate_all_explanations.py with {num_workers} workers...\n")

    result = _sp.run(
        [sys.executable, "src/generate_all_explanations.py",
         "--num_workers", str(num_workers), "--episodes", "12"],
        env={**os.environ, "LIBSUMO_AS_TRACI": os.environ.get("LIBSUMO_AS_TRACI", "1")},
    )

    if result.returncode != 0:
        print(f"\nExplanation generation exited with code {result.returncode}.")
        print("Generating dashboard with whatever explanations are available.\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate results dashboard")
    parser.add_argument("--output", type=str, default="results/dashboard.html")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Skip the incomplete-experiment check and generate "
                             "dashboard with available results only.")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if not args.no_prompt:
        incomplete = find_incomplete_experiments()
        if incomplete:
            prompt_supplement(incomplete)

        missing_expl = find_missing_explanations()
        if missing_expl:
            prompt_generate_explanations(missing_expl)

    experiments = load_experiments()
    megapolicy_data = load_megapolicy_results()
    rush_test_data = load_rush_test_results()
    generate_html(experiments, args.output, megapolicy_data=megapolicy_data,
                  rush_test_data=rush_test_data)


if __name__ == "__main__":
    main()
