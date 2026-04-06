"""
Zeleni SignaLJ - Results Dashboard
====================================
Generates an interactive HTML dashboard comparing experiment runs.

Usage:
    python src/dashboard.py
    python src/dashboard.py --output results/dashboard.html
"""

import argparse
import json
import os

import pandas as pd

EXPERIMENTS_DIR = "results/experiments"


def load_experiments():
    """Load all experiment metadata and results."""
    experiments = []

    if not os.path.exists(EXPERIMENTS_DIR):
        return experiments

    for run_id in sorted(os.listdir(EXPERIMENTS_DIR)):
        meta_path = os.path.join(EXPERIMENTS_DIR, run_id, "meta.json")
        results_path = os.path.join(EXPERIMENTS_DIR, run_id, "results.csv")
        log_path = os.path.join(EXPERIMENTS_DIR, run_id, "training_log.csv")

        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        results = None
        if os.path.exists(results_path):
            results = pd.read_csv(results_path).to_dict("records")

        training_log = None
        if os.path.exists(log_path):
            training_log = pd.read_csv(log_path).to_dict("records")

        experiments.append({
            "meta": meta,
            "results": results,
            "training_log": training_log,
        })

    return experiments


def generate_html(experiments, output_path):
    """Generate a self-contained HTML dashboard."""

    # Prepare data for JS
    exp_json = json.dumps(experiments, default=str)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zeleni SignaLJ - Results Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 24px;
  }}
  h1 {{ color: #4ade80; margin-bottom: 8px; font-size: 28px; }}
  h2 {{ color: #94a3b8; margin: 24px 0 12px; font-size: 18px; }}
  .subtitle {{ color: #64748b; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{
    background: #1e293b; border-radius: 12px; padding: 20px;
    border: 1px solid #334155;
  }}
  .card-label {{ color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card-value {{ font-size: 32px; font-weight: 700; margin-top: 4px; }}
  .positive {{ color: #4ade80; }}
  .negative {{ color: #f87171; }}
  .neutral {{ color: #94a3b8; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ font-size: 14px; }}
  .chart-container {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; margin-bottom: 16px; }}
  canvas {{ max-height: 400px; }}
  .run-selector {{ margin-bottom: 16px; }}
  .run-selector select {{
    background: #1e293b; color: #e2e8f0; border: 1px solid #475569;
    padding: 8px 12px; border-radius: 8px; font-size: 14px; cursor: pointer;
  }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 12px; font-weight: 600;
  }}
  .badge-green {{ background: #166534; color: #4ade80; }}
  .badge-red {{ background: #7f1d1d; color: #fca5a5; }}
  .badge-gray {{ background: #334155; color: #94a3b8; }}
  .no-data {{ text-align: center; color: #64748b; padding: 60px 20px; }}
  .no-data p {{ font-size: 18px; margin-bottom: 8px; }}
</style>
</head>
<body>

<h1>Zeleni SignaLJ</h1>
<p class="subtitle">Primerjava eksperimentov adaptivnega upravljanja semaforjev</p>

<div id="app"></div>

<script>
const experiments = {exp_json};

const app = document.getElementById('app');

if (experiments.length === 0) {{
  app.innerHTML = `
    <div class="no-data">
      <p>Ni podatkov o eksperimentih.</p>
      <p style="font-size:14px">Zaženite: <code>python src/experiment.py --tag test</code></p>
    </div>`;
}} else {{
  renderDashboard();
}}

function renderDashboard() {{
  // Summary KPIs from latest run
  const latest = experiments[experiments.length - 1];
  const meta = latest.meta;
  const bestImprovement = experiments.reduce((best, e) =>
    (e.meta.improvement_pct || 0) > (best.meta.improvement_pct || 0) ? e : best
  , experiments[0]);

  let html = `
    <div class="grid">
      <div class="card">
        <div class="card-label">Skupaj eksperimentov</div>
        <div class="card-value neutral">${{experiments.length}}</div>
      </div>
      <div class="card">
        <div class="card-label">Zadnji eksperiment</div>
        <div class="card-value neutral" style="font-size:18px">${{meta.tag || meta.run_id}}</div>
      </div>
      <div class="card">
        <div class="card-label">Najboljše izboljšanje</div>
        <div class="card-value ${{(bestImprovement.meta.improvement_pct || 0) >= 0 ? 'positive' : 'negative'}}">
          ${{(bestImprovement.meta.improvement_pct || 0).toFixed(1)}}%
        </div>
      </div>
      <div class="card">
        <div class="card-label">Zadnje izboljšanje</div>
        <div class="card-value ${{(meta.improvement_pct || 0) >= 0 ? 'positive' : 'negative'}}">
          ${{(meta.improvement_pct || 0).toFixed(1)}}%
        </div>
      </div>
    </div>

    <h2>Primerjava vseh eksperimentov</h2>
    <div class="chart-container">
      <canvas id="comparisonChart"></canvas>
    </div>

    <h2>Tabela eksperimentov</h2>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Oznaka</th>
            <th>Koraki</th>
            <th>Bazna linija</th>
            <th>RL Agent</th>
            <th>Izboljšanje</th>
            <th>Čas učenja</th>
          </tr>
        </thead>
        <tbody>
          ${{experiments.map(e => {{
            const m = e.meta;
            const pct = (m.improvement_pct || 0);
            const badge = pct > 0 ? 'badge-green' : pct < 0 ? 'badge-red' : 'badge-gray';
            return `<tr>
              <td>${{m.tag || m.run_id}}</td>
              <td>${{(m.total_timesteps || 0).toLocaleString()}}</td>
              <td>${{(m.baseline_total_reward || 0).toFixed(1)}}</td>
              <td>${{(m.rl_total_reward || 0).toFixed(1)}}</td>
              <td><span class="badge ${{badge}}">${{pct >= 0 ? '+' : ''}}${{pct.toFixed(1)}}%</span></td>
              <td>${{(m.train_time_s || 0).toFixed(0)}}s</td>
            </tr>`;
          }}).join('')}}
        </tbody>
      </table>
    </div>

    <h2>Rezultati po križiščih (zadnji eksperiment)</h2>
    <div class="chart-container">
      <canvas id="intersectionChart"></canvas>
    </div>
  `;

  // Training curve selector (if any experiment has training logs)
  const expsWithLogs = experiments.filter(e => e.training_log && e.training_log.length > 0);
  if (expsWithLogs.length > 0) {{
    html += `
      <h2>Krivulja učenja</h2>
      <div class="run-selector">
        <select id="trainingRunSelect" onchange="updateTrainingChart()">
          ${{expsWithLogs.map((e, i) =>
            `<option value="${{i}}">${{e.meta.tag || e.meta.run_id}}</option>`
          ).join('')}}
        </select>
      </div>
      <div class="chart-container">
        <canvas id="trainingChart"></canvas>
      </div>
    `;
  }}

  app.innerHTML = html;

  // ── Comparison bar chart ──
  const compCtx = document.getElementById('comparisonChart').getContext('2d');
  new Chart(compCtx, {{
    type: 'bar',
    data: {{
      labels: experiments.map(e => e.meta.tag || e.meta.run_id),
      datasets: [
        {{
          label: 'Bazna linija (fiksni čas)',
          data: experiments.map(e => Math.abs(e.meta.baseline_total_reward || 0)),
          backgroundColor: '#475569',
          borderRadius: 4,
        }},
        {{
          label: 'RL Agent',
          data: experiments.map(e => Math.abs(e.meta.rl_total_reward || 0)),
          backgroundColor: '#4ade80',
          borderRadius: 4,
        }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: '#94a3b8' }} }},
        title: {{ display: true, text: 'Skupna nagrada (absolutna vrednost, nižja = boljša)', color: '#94a3b8' }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
        y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
      }}
    }}
  }});

  // ── Per-intersection chart (latest run) ──
  if (latest.results) {{
    const intCtx = document.getElementById('intersectionChart').getContext('2d');
    new Chart(intCtx, {{
      type: 'bar',
      data: {{
        labels: latest.results.map(r => r.intersection),
        datasets: [
          {{
            label: 'Bazna linija',
            data: latest.results.map(r => Math.abs(r.baseline_reward)),
            backgroundColor: '#475569',
            borderRadius: 4,
          }},
          {{
            label: 'RL Agent',
            data: latest.results.map(r => Math.abs(r.rl_reward)),
            backgroundColor: '#4ade80',
            borderRadius: 4,
          }}
        ]
      }},
      options: {{
        responsive: true,
        indexAxis: 'y',
        plugins: {{
          legend: {{ labels: {{ color: '#94a3b8' }} }},
          title: {{ display: true, text: 'Nagrada po križišču (absolutna vrednost, nižja = boljša)', color: '#94a3b8' }}
        }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }}
        }}
      }}
    }});
  }}

  // ── Training curve ──
  if (expsWithLogs.length > 0) {{
    window._expsWithLogs = expsWithLogs;
    updateTrainingChart();
  }}
}}

function updateTrainingChart() {{
  const select = document.getElementById('trainingRunSelect');
  if (!select) return;
  const exp = window._expsWithLogs[parseInt(select.value)];
  const log = exp.training_log;

  const canvas = document.getElementById('trainingChart');
  if (window._trainingChartInstance) {{
    window._trainingChartInstance.destroy();
  }}

  const ctx = canvas.getContext('2d');
  window._trainingChartInstance = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: log.map(l => l.episode),
      datasets: [{{
        label: 'Povprečna nagrada na epizodo',
        data: log.map(l => l.avg_reward),
        borderColor: '#4ade80',
        backgroundColor: 'rgba(74, 222, 128, 0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 1,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: '#94a3b8' }} }},
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'Epizoda', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#1e293b' }} }},
        y: {{ title: {{ display: true, text: 'Nagrada', color: '#94a3b8' }}, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
      }}
    }}
  }});
}}
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate results dashboard")
    parser.add_argument("--output", type=str, default="results/dashboard.html")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    experiments = load_experiments()
    generate_html(experiments, args.output)


if __name__ == "__main__":
    main()
