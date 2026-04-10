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

        experiments.append({
            "meta": meta,
            "results": results,
            "training_log": training_log,
            "step_log": step_log,
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
    """Welch's t-test, Mann-Whitney U, Cohen's d between two Series."""
    t_stat, t_p = stats.ttest_ind(mega_series, baseline_series, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(
        mega_series, baseline_series, alternative='two-sided')
    s1 = float(mega_series.std(ddof=1))
    s2 = float(baseline_series.std(ddof=1))
    pooled_std = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    cohens_d = ((mega_series.mean() - baseline_series.mean()) / pooled_std
                if pooled_std > 0 else 0.0)
    bl_abs = abs(float(baseline_series.mean()))
    improvement_pct = (
        (float(mega_series.mean()) - float(baseline_series.mean()))
        / bl_abs * 100 if bl_abs > 0 else 0.0)
    return {
        "improvement_pct": float(improvement_pct),
        "welch_t": float(t_stat), "welch_p": float(t_p),
        "mann_whitney_u": float(u_stat), "mann_whitney_p": float(u_p),
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

    # Comparisons: each megapolicy vs baseline
    comparisons = []
    for tag, cdata in conditions.items():
        if cdata["meta"].get("baseline", False):
            continue
        df = cdata["df"]
        comp = {
            "tag": tag,
            **_compare(df["total_reward"], bl_df["total_reward"]),
            "intersections": {},
            "windows": {},
        }
        for iname in INTERSECTION_NAMES:
            comp["intersections"][iname] = _compare(
                df[f"reward_{iname}"], bl_df[f"reward_{iname}"])
        for wname in WINDOW_ORDER:
            comp["windows"][wname] = _compare(
                df[f"reward_{wname}"], bl_df[f"reward_{wname}"])
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
            row_p.append(comp["welch_p"] if comp else 1)
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


def generate_html(experiments, output_path, megapolicy_data=None):
    # Round all floats to reduce JSON size (82KB → ~30KB)
    rounded = _round_floats(experiments)
    exp_json = json.dumps(rounded, default=str, separators=(',', ':'))
    mega_json = (json.dumps(_round_floats(megapolicy_data, decimals=4),
                            default=str, separators=(',', ':'))
                 if megapolicy_data else 'null')

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
const COLORS = ['#4ade80','#60a5fa','#f472b6','#facc15','#a78bfa','#fb923c','#34d399','#f87171','#38bdf8','#c084fc'];
const INT_COLORS = {{'Kolodvor':'#60a5fa','Pivovarna':'#4ade80','Slovenska':'#f472b6','Trzaska':'#facc15','Askerceva':'#a78bfa'}};

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
    <div class="tab active" onclick="switchTab(0)">Primerjava</div>
    <div class="tab" onclick="switchTab(1)">Krizisca</div>
    <div class="tab" onclick="switchTab(2)">Ucenje</div>
    <div class="tab" onclick="switchTab(3)">Hiperparametri</div>
    <div class="tab" onclick="switchTab(4)">Podrobnosti</div>
    ${{megaData ? '<div class="tab" onclick="switchTab(5)">Mega-politike</div>' : ''}}
  </div>`;

  // ══════════════════════════════════════════
  // TAB 0: Experiment Comparison
  // ══════════════════════════════════════════
  h += '<div class="tab-content active" id="tab0">';

  h += `<div class="controls">
    <label>Iskanje: <input type="text" id="filterText" placeholder="oznaka ali ID..." oninput="applyFilters()"></label>
    <label>Min korakov: <input type="number" id="filterMinSteps" value="0" style="width:80px" oninput="applyFilters()"></label>
    <label>Zadnjih N: <input type="number" id="filterLastN" value="" placeholder="vse" style="width:60px" oninput="applyFilters()"></label>
  </div>`;

  // Comparison bar chart
  h += '<div class="grid-2">';
  h += '<div class="chart-box"><div class="card-title">Skupna nagrada: bazna linija vs RL (abs. vrednost, nizja = boljsa)</div><canvas id="compChart"></canvas></div>';
  h += '<div class="chart-box"><div class="card-title">Izboljsanje po eksperimentih (%)</div><canvas id="improvChart"></canvas></div>';
  h += '</div>';

  // Experiment table
  h += `<div class="card" style="overflow-x:auto;">
    <div class="card-title">Vsi eksperimenti</div>
    <div class="table-wrap">
    <table id="expTable">
      <thead><tr>
        <th onclick="sortTable(0)">Oznaka</th>
        <th onclick="sortTable(1)">Datum</th>
        <th onclick="sortTable(2)">Koraki</th>
        <th onclick="sortTable(3)">Bazna linija</th>
        <th onclick="sortTable(4)">RL Agent</th>
        <th onclick="sortTable(5)">Izboljsanje</th>
        <th onclick="sortTable(6)">LR</th>
        <th onclick="sortTable(7)">Ent. koef.</th>
        <th onclick="sortTable(8)">Cas</th>
      </tr></thead>
      <tbody></tbody>
    </table>
    </div>
  </div>`;

  h += '</div>'; // tab0

  // ══════════════════════════════════════════
  // TAB 1: Per-intersection breakdown
  // ══════════════════════════════════════════
  h += '<div class="tab-content" id="tab1">';

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

  h += '</div>'; // tab1

  // ══════════════════════════════════════════
  // TAB 2: Training curves
  // ══════════════════════════════════════════
  h += '<div class="tab-content" id="tab2">';

  h += '<div class="cb-row" id="curveSelect"></div>';
  h += '<div class="grid-2">';
  h += '<div class="chart-box"><div class="card-title">Nagrada na epizodo (episode-level)</div><canvas id="trainChart"></canvas></div>';
  h += '<div class="chart-box"><div class="card-title">Povprecna nagrada na korak (step-level, glajeno)</div><canvas id="stepChart"></canvas></div>';
  h += '</div>';

  h += '<div class="info-text">Izberite eksperimente zgoraj za prikaz krivulj ucenja. Podatki so na voljo sele po novem zagonu z posodobljenim kodom.</div>';

  h += '</div>'; // tab2

  // ══════════════════════════════════════════
  // TAB 3: Hyperparameter comparison
  // ══════════════════════════════════════════
  h += '<div class="tab-content" id="tab3">';

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

  h += '</div>'; // tab3

  // ══════════════════════════════════════════
  // TAB 4: Detailed experiment info
  // ══════════════════════════════════════════
  h += '<div class="tab-content" id="tab4">';

  h += `<div class="controls">
    <label>Eksperiment: <select id="detailExpSelect" onchange="updateDetails()">
      ${{experiments.map((e, i) => `<option value="${{i}}" ${{i===experiments.length-1?'selected':''}}>${{e.meta.tag || e.meta.run_id}}</option>`).join('')}}
    </select></label>
  </div>`;

  h += '<div id="detailPanel"></div>';

  h += '</div>'; // tab4

  // ══════════════════════════════════════════
  // TAB 5: Mega-politike (statistical test)
  // ══════════════════════════════════════════
  if (megaData) {{
    h += '<div class="tab-content" id="tab5">';

    // ── KPI cards ──
    const mComps = megaData.comparisons;
    const bestMega = mComps.reduce((b, c) => c.improvement_pct > b.improvement_pct ? c : b, mComps[0]);
    const worstMega = mComps.reduce((b, c) => c.improvement_pct < b.improvement_pct ? c : b, mComps[0]);
    const blCond = megaData.conditions.find(c => c.is_baseline);
    const sigCount = mComps.filter(c => c.welch_p < 0.05).length;

    h += '<div class="kpi-grid">';
    h += `<div class="kpi"><div class="kpi-label">Stevilo pogojev</div><div class="kpi-value neutral">${{megaData.conditions.length}}</div><div class="kpi-sub">9 mega-politik + bazna linija</div></div>`;
    h += `<div class="kpi"><div class="kpi-label">Najboljsa mega-politika</div><div class="kpi-value ${{pctClass(bestMega.improvement_pct)}}">${{fmtPct(bestMega.improvement_pct)}}</div><div class="kpi-sub">${{bestMega.tag}}</div></div>`;
    h += `<div class="kpi"><div class="kpi-label">Najslabsa mega-politika</div><div class="kpi-value ${{pctClass(worstMega.improvement_pct)}}">${{fmtPct(worstMega.improvement_pct)}}</div><div class="kpi-sub">${{worstMega.tag}}</div></div>`;
    h += `<div class="kpi"><div class="kpi-label">Ponovitve na pogoj</div><div class="kpi-value neutral">${{blCond ? blCond.n : 50}}</div><div class="kpi-sub">razlicni SUMO seedi</div></div>`;
    h += `<div class="kpi"><div class="kpi-label">Bazna linija (povp.)</div><div class="kpi-value neutral">${{blCond ? fmt(blCond.total_reward.mean, 0) : '-'}}</div><div class="kpi-sub">povp. skupna nagrada</div></div>`;
    h += `<div class="kpi"><div class="kpi-label">Stat. znacilne (p&lt;0.05)</div><div class="kpi-value ${{sigCount > 0 ? 'positive' : 'neutral'}}">${{sigCount}} / ${{mComps.length}}</div><div class="kpi-sub">Welch t-test</div></div>`;
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
        h += `<div class="heatmap-cell" style="background:${{bg}}" title="${{tag}}: Welch p=${{pval.toFixed(6)}}">
          ${{fmtPct(val)}}<span class="sig-marker">${{sig}}</span>
        </div>`;
      }});
    }});
    h += '</div>';
    h += '<div class="info-text">* p&lt;0.05 &nbsp; ** p&lt;0.01 &nbsp; *** p&lt;0.001 (Welch t-test)</div>';
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
    h += '<div class="card"><div class="card-title">Primerjava z bazno linijo (Welch t-test, Mann-Whitney U, Cohen d)</div>';
    h += '<div class="table-wrap"><table id="megaStatsTable"><thead><tr>';
    h += '<th>Mega-politika</th><th>Povp. nagrada</th><th>Mediana</th><th>Std</th>';
    h += '<th>95% IZ</th><th>Welch t</th><th>Welch p</th>';
    h += '<th>M-W U</th><th>M-W p</th><th>Cohen d</th><th>Izboljsanje</th>';
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
      const pBadge = c.welch_p < 0.001 ? 'badge-green' : c.welch_p < 0.01 ? 'badge-green' : c.welch_p < 0.05 ? 'badge-yellow' : 'badge-gray';
      const pLabel = c.welch_p < 0.001 ? '***' : c.welch_p < 0.01 ? '**' : c.welch_p < 0.05 ? '*' : 'n.s.';
      const mwpBadge = c.mann_whitney_p < 0.001 ? 'badge-green' : c.mann_whitney_p < 0.01 ? 'badge-green' : c.mann_whitney_p < 0.05 ? 'badge-yellow' : 'badge-gray';
      const mwpLabel = c.mann_whitney_p < 0.001 ? '***' : c.mann_whitney_p < 0.01 ? '**' : c.mann_whitney_p < 0.05 ? '*' : 'n.s.';
      const absD = Math.abs(c.cohens_d);
      const dLabel = absD > 0.8 ? 'Velik' : absD > 0.5 ? 'Srednji' : absD > 0.2 ? 'Majhen' : 'Zanem.';
      const dBadge = absD > 0.8 ? (c.cohens_d > 0 ? 'badge-green' : 'badge-red') : absD > 0.5 ? 'badge-yellow' : 'badge-gray';
      h += `<tr>
        <td><strong>${{c.tag}}</strong></td>
        <td>${{fmt(s.mean||0,0)}}</td><td>${{fmt(s.median||0,0)}}</td><td>${{fmt(s.std||0,0)}}</td>
        <td>[${{fmt(s.ci_low||0,0)}}, ${{fmt(s.ci_high||0,0)}}]</td>
        <td>${{c.welch_t.toFixed(2)}}</td>
        <td><span class="badge ${{pBadge}}">${{c.welch_p < 0.0001 ? c.welch_p.toExponential(2) : c.welch_p.toFixed(4)}} ${{pLabel}}</span></td>
        <td>${{fmt(c.mann_whitney_u,0)}}</td>
        <td><span class="badge ${{mwpBadge}}">${{c.mann_whitney_p < 0.0001 ? c.mann_whitney_p.toExponential(2) : c.mann_whitney_p.toFixed(4)}} ${{mwpLabel}}</span></td>
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

    h += '</div>'; // tab5
  }}

  app.innerHTML = h;

  // Initialize all views
  applyFilters();
  updateIntersections();
  buildCurveSelector();
  updateHyperparams();
  updateDetails();
  if (megaData) {{
    updateMegaOverall();
    updateMegaIntersections();
    updateMegaWindows();
    updateMegaDetail();
  }}
}}

// ── Tab switching ──
function switchTab(idx) {{
  document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', i === idx));
  document.querySelectorAll('.tab-content').forEach((t, i) => t.classList.toggle('active', i === idx));
}}

// ── Chart instances (var, not let — avoids temporal dead zone when
//    render() is called before this line during initial script execution) ──
var compChartInst, improvChartInst, intBarInst, intPctInst, intTrendInst;
var trainChartInst, stepChartInst, hpScatter1Inst, hpScatter2Inst;
var megaOverallInst, megaImprovInst;
var megaIntRewardInst, megaIntImprovInst;
var megaWindowRewardInst, megaWindowImprovInst;

// ══════════════════════════════════════════
// TAB 0: Filtering & comparison
// ══════════════════════════════════════════
function getFiltered() {{
  const text = (document.getElementById('filterText')?.value || '').toLowerCase();
  const minSteps = parseInt(document.getElementById('filterMinSteps')?.value) || 0;
  const lastN = parseInt(document.getElementById('filterLastN')?.value) || 0;

  let filtered = experiments.filter(e => {{
    const tag = (e.meta.tag || e.meta.run_id || '').toLowerCase();
    const steps = e.meta.actual_timesteps || e.meta.total_timesteps || 0;
    return tag.includes(text) && steps >= minSteps;
  }});

  if (lastN > 0) filtered = filtered.slice(-lastN);
  return filtered;
}}

function applyFilters() {{
  const filtered = getFiltered();
  const labels = filtered.map(e => e.meta.tag || e.meta.run_id);

  // Comparison bar chart
  if (compChartInst) compChartInst.destroy();
  const ctx1 = document.getElementById('compChart')?.getContext('2d');
  if (ctx1) {{
    compChartInst = new Chart(ctx1, {{
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
  if (improvChartInst) improvChartInst.destroy();
  const ctx2 = document.getElementById('improvChart')?.getContext('2d');
  if (ctx2) {{
    const pcts = filtered.map(e => e.meta.improvement_pct || 0);
    improvChartInst = new Chart(ctx2, {{
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
  const tbody = document.querySelector('#expTable tbody');
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
        <td>${{fmt(m.train_time_s||0, 0)}}s</td>
      </tr>`;
    }}).join('');
  }}
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
// TAB 1: Intersection breakdown
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
// TAB 2: Training curves
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
// TAB 3: Hyperparameters
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
// TAB 4: Experiment details
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

// ── Table sorting ──
var sortCol = -1, sortAsc = true;
function sortTable(col) {{
  if (sortCol === col) sortAsc = !sortAsc;
  else {{ sortCol = col; sortAsc = true; }}

  const tbody = document.querySelector('#expTable tbody');
  if (!tbody) return;
  const rows = Array.from(tbody.rows);
  rows.sort((a, b) => {{
    let va = a.cells[col].textContent.trim().replace(/[^\\d.\\-]/g, '');
    let vb = b.cells[col].textContent.trim().replace(/[^\\d.\\-]/g, '');
    const na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return sortAsc ? na - nb : nb - na;
    return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

// ══════════════════════════════════════════
// TAB 5: Mega-politike
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
                return `Welch p=${{c.welch_p < 0.0001 ? c.welch_p.toExponential(2) : c.welch_p.toFixed(4)}}\\nCohen d=${{c.cohens_d.toFixed(3)}}`;
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
  h += '<th>Welch p</th><th>M-W p</th><th>Cohen d</th>';
  h += '</tr></thead><tbody>';
  const blCond = megaData.conditions.find(c => c.is_baseline);
  megaData.intersection_names.forEach(iname => {{
    const ic = comp.intersections[iname];
    const blR = blCond ? blCond.intersections[iname].reward.mean : 0;
    const mR = cond.intersections[iname].reward.mean;
    const pB = ic.welch_p < 0.001 ? 'badge-green' : ic.welch_p < 0.01 ? 'badge-green' : ic.welch_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const pL = ic.welch_p < 0.001 ? '***' : ic.welch_p < 0.01 ? '**' : ic.welch_p < 0.05 ? '*' : 'n.s.';
    const mwB = ic.mann_whitney_p < 0.001 ? 'badge-green' : ic.mann_whitney_p < 0.01 ? 'badge-green' : ic.mann_whitney_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const mwL = ic.mann_whitney_p < 0.001 ? '***' : ic.mann_whitney_p < 0.01 ? '**' : ic.mann_whitney_p < 0.05 ? '*' : 'n.s.';
    const absD = Math.abs(ic.cohens_d);
    const dL = absD > 0.8 ? 'Velik' : absD > 0.5 ? 'Srednji' : absD > 0.2 ? 'Majhen' : 'Zanem.';
    const dB = absD > 0.8 ? (ic.cohens_d > 0 ? 'badge-green' : 'badge-red') : absD > 0.5 ? 'badge-yellow' : 'badge-gray';
    h += `<tr>
      <td><strong style="color:${{INT_COLORS[iname]||'#e2e8f0'}}">${{iname}}</strong></td>
      <td>${{fmt(blR, 0)}}</td>
      <td>${{fmt(mR, 0)}}</td>
      <td><span class="badge ${{badgeClass(ic.improvement_pct)}}">${{fmtPct(ic.improvement_pct)}}</span></td>
      <td><span class="badge ${{pB}}">${{ic.welch_p < 0.0001 ? ic.welch_p.toExponential(2) : ic.welch_p.toFixed(4)}} ${{pL}}</span></td>
      <td><span class="badge ${{mwB}}">${{ic.mann_whitney_p < 0.0001 ? ic.mann_whitney_p.toExponential(2) : ic.mann_whitney_p.toFixed(4)}} ${{mwL}}</span></td>
      <td><span class="badge ${{dB}}">${{ic.cohens_d.toFixed(3)}} (${{dL}})</span></td>
    </tr>`;
  }});
  h += '</tbody></table></div></div>';

  // Per-window table
  h += '<div class="card"><div class="card-title">Primerjava po casovnih oknih</div>';
  h += '<div class="table-wrap"><table><thead><tr>';
  h += '<th>Casovno okno</th><th>Bazna (povp.)</th><th>Mega (povp.)</th><th>Izboljsanje</th>';
  h += '<th>Welch p</th><th>M-W p</th><th>Cohen d</th>';
  h += '</tr></thead><tbody>';
  megaData.window_names.forEach(wname => {{
    const wc = comp.windows[wname];
    const blR = blCond ? blCond.windows[wname].reward.mean : 0;
    const mR = cond.windows[wname].reward.mean;
    const wLabel = megaData.window_labels[wname] || wname;
    const pB = wc.welch_p < 0.001 ? 'badge-green' : wc.welch_p < 0.01 ? 'badge-green' : wc.welch_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const pL = wc.welch_p < 0.001 ? '***' : wc.welch_p < 0.01 ? '**' : wc.welch_p < 0.05 ? '*' : 'n.s.';
    const mwB = wc.mann_whitney_p < 0.001 ? 'badge-green' : wc.mann_whitney_p < 0.01 ? 'badge-green' : wc.mann_whitney_p < 0.05 ? 'badge-yellow' : 'badge-gray';
    const mwL = wc.mann_whitney_p < 0.001 ? '***' : wc.mann_whitney_p < 0.01 ? '**' : wc.mann_whitney_p < 0.05 ? '*' : 'n.s.';
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
      <td><span class="badge ${{pB}}">${{wc.welch_p < 0.0001 ? wc.welch_p.toExponential(2) : wc.welch_p.toFixed(4)}} ${{pL}}</span></td>
      <td><span class="badge ${{mwB}}">${{wc.mann_whitney_p < 0.0001 ? wc.mann_whitney_p.toExponential(2) : wc.mann_whitney_p.toFixed(4)}} ${{mwL}}</span></td>
      <td><span class="badge ${{dB}}">${{wc.cohens_d.toFixed(3)}} (${{dL}})</span></td>
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

    experiments = load_experiments()
    megapolicy_data = load_megapolicy_results()
    generate_html(experiments, args.output, megapolicy_data=megapolicy_data)


if __name__ == "__main__":
    main()
