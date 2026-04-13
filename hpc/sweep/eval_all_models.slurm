#!/bin/bash
# Zeleni SignaLJ - Evaluate all trained models with episode count > 500
# - Generates ONE fresh route file per scenario (only if missing)
# - Runs baseline ONCE per scenario (cached)
# - Runs RL-only for each model (no redundant baselines)
#SBATCH --job-name=zs_eval_all
#SBATCH --output=logs/zs_eval_all_%j.out
#SBATCH --error=logs/zs_eval_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source hpc/common.sh

MORNING_ROUTE="data/routes/routes_morning_rush.rou.xml"
EVENING_ROUTE="data/routes/routes_evening_rush.rou.xml"
BASELINE_MORNING="results/baseline_morning_rush.csv"
BASELINE_EVENING="results/baseline_evening_rush.csv"

# ── Phase 1: Generate route files ONLY if they don't already exist ──
echo "=== Phase 1: Ensuring evaluation route files exist ==="
if [ -f "$MORNING_ROUTE" ]; then
    echo "  [SKIP] $MORNING_ROUTE already exists (re-using)"
else
    echo "  [GEN]  Generating $MORNING_ROUTE..."
    python src/generate_demand.py --scenario morning_rush
fi

if [ -f "$EVENING_ROUTE" ]; then
    echo "  [SKIP] $EVENING_ROUTE already exists (re-using)"
else
    echo "  [GEN]  Generating $EVENING_ROUTE..."
    python src/generate_demand.py --scenario evening_rush
fi

# Hard-lock route files read-only — no eval job can ever overwrite them
chmod 444 "$MORNING_ROUTE" "$EVENING_ROUTE"
echo "  Routes locked read-only (chmod 444)"
echo ""

# ── Phase 2: Run baseline ONCE per scenario (skipped if already cached) ──
echo "=== Phase 2: Running fixed-time baseline (once per scenario) ==="
if [ -f "$BASELINE_MORNING" ]; then
    echo "  [SKIP] Morning baseline already cached: $BASELINE_MORNING"
else
    echo "  [RUN]  Morning baseline..."
    python src/evaluate.py --scenario morning_rush --output "$BASELINE_MORNING"
fi

if [ -f "$BASELINE_EVENING" ]; then
    echo "  [SKIP] Evening baseline already cached: $BASELINE_EVENING"
else
    echo "  [RUN]  Evening baseline..."
    python src/evaluate.py --scenario evening_rush --output "$BASELINE_EVENING"
fi
echo ""

# ── Phase 3: Evaluate RL only for each model ──
echo "=== Phase 3: Evaluating RL models (no redundant baselines) ==="
EVAL_LOG="results/eval_all_summary.csv"
echo "run_id,scenario,episodes,status,output_csv" > "$EVAL_LOG"

count=0
skipped=0
failed=0
succeeded=0

for run_dir in results/experiments/*/; do
    run_id=$(basename "$run_dir")

    ep_part="${run_id##*_}"          # e.g. "4800ep"
    ep_count="${ep_part%ep}"

    if ! [[ "$ep_count" =~ ^[0-9]+$ ]] || [ "$ep_count" -le 500 ]; then
        skipped=$((skipped+1))
        continue
    fi

    if [[ "$run_id" == *morningrush* ]]; then
        scenario="morning_rush"
    elif [[ "$run_id" == *eveningrush* ]]; then
        scenario="evening_rush"
    else
        echo "[SKIP] $run_id: unknown scenario"
        skipped=$((skipped+1))
        continue
    fi

    model_path="${run_dir}ppo_shared_policy.zip"
    if [ ! -f "$model_path" ]; then
        model_path="${run_dir}checkpoints/ppo_model_latest.zip"
        if [ ! -f "$model_path" ]; then
            echo "[SKIP] $run_id: no model file"
            skipped=$((skipped+1))
            continue
        fi
    fi

    count=$((count+1))
    output_csv="${run_dir}eval_rl.csv"

    echo ""
    echo "=== [$count] Evaluating $run_id ($scenario, ${ep_count}ep) ==="

    if python src/evaluate.py \
        --model "$model_path" \
        --scenario "$scenario" \
        --skip_baseline \
        --output "$output_csv"; then
        echo "$run_id,$scenario,$ep_count,OK,$output_csv" >> "$EVAL_LOG"
        succeeded=$((succeeded+1))
    else
        echo "[FAIL] $run_id: evaluate.py exited non-zero"
        echo "$run_id,$scenario,$ep_count,FAIL," >> "$EVAL_LOG"
        failed=$((failed+1))
    fi
done

echo ""
echo "==================================================="
echo "Evaluation complete: $succeeded succeeded, $failed failed, $skipped skipped"
echo "Baseline (cached, run once):  $BASELINE_MORNING, $BASELINE_EVENING"
echo "Per-model RL results:         results/experiments/<run_id>/eval_rl.csv"
echo "Summary index:                $EVAL_LOG"
echo "==================================================="
