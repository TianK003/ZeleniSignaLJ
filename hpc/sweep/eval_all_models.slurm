#!/bin/bash
# Zeleni SignaLJ - Evaluate all trained models with episode count > 500
# - Phase 1: Generate route files ONLY if missing (chmod 444 to lock)
# - Phase 2: Run baseline ONCE per scenario (cached)
# - Phase 3: Discover all eligible experiments, then evaluate 32 in parallel
#SBATCH --job-name=zs_eval_all
#SBATCH --output=logs/zs_eval_all_%j.out
#SBATCH --error=logs/zs_eval_all_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G

source hpc/common.sh

PARALLEL=32
MORNING_ROUTE="data/routes/routes_morning_rush.rou.xml"
EVENING_ROUTE="data/routes/routes_evening_rush.rou.xml"
BASELINE_MORNING="results/baseline_morning_rush.csv"
BASELINE_EVENING="results/baseline_evening_rush.csv"
EVAL_LOG="results/eval_all_summary.csv"

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

# ── Phase 3a: Discover all eligible experiments ──
echo "=== Phase 3a: Discovering eligible experiments ==="
JOBS_FILE=$(mktemp)
trap "rm -f $JOBS_FILE" EXIT

for run_dir in results/experiments/*/; do
    run_id=$(basename "$run_dir")

    ep_part="${run_id##*_}"
    ep_count="${ep_part%ep}"

    if ! [[ "$ep_count" =~ ^[0-9]+$ ]] || [ "$ep_count" -le 500 ]; then
        continue
    fi

    if [[ "$run_id" == *morningrush* ]]; then
        scenario="morning_rush"
    elif [[ "$run_id" == *eveningrush* ]]; then
        scenario="evening_rush"
    else
        continue
    fi

    model_path="${run_dir}ppo_shared_policy.zip"
    if [ ! -f "$model_path" ]; then
        model_path="${run_dir}checkpoints/ppo_model_latest.zip"
        [ ! -f "$model_path" ] && continue
    fi

    output_csv="${run_dir}eval_rl.csv"
    # Pipe-separated record: run_id|scenario|ep_count|model_path|output_csv
    echo "${run_id}|${scenario}|${ep_count}|${model_path}|${output_csv}" >> "$JOBS_FILE"
done

JOB_COUNT=$(wc -l < "$JOBS_FILE")
echo "  Discovered $JOB_COUNT eligible experiments:"
echo ""
awk -F'|' '{printf "    %-70s  %-13s  %sep\n", $1, $2, $3}' "$JOBS_FILE"
echo ""

if [ "$JOB_COUNT" -eq 0 ]; then
    echo "No eligible experiments to evaluate. Exiting."
    exit 0
fi

# ── Phase 3b: Evaluate in parallel (32 simultaneous SUMO instances) ──
echo "=== Phase 3b: Evaluating $JOB_COUNT models, $PARALLEL in parallel ==="
echo "run_id,scenario,episodes,status,output_csv" > "$EVAL_LOG"

START=$(date +%s)

# Each line of $JOBS_FILE is processed by an independent Python subprocess.
# `xargs -P` limits concurrency. Single-line appends to $EVAL_LOG are atomic
# under POSIX (line < PIPE_BUF = 4096 bytes), so no flock needed.
# After RL eval succeeds, finalize_for_dashboard.py merges baseline+RL into
# results.csv + meta.json so dashboard.py can ingest the experiment.
export BASELINE_MORNING BASELINE_EVENING EVAL_LOG

cat "$JOBS_FILE" | xargs -P "$PARALLEL" -I {} bash -c '
    IFS="|" read -r run_id scenario ep_count model_path output_csv <<< "{}"
    log_file="${output_csv%.csv}.log"
    exp_dir=$(dirname "$output_csv")

    # Pick the matching cached baseline CSV
    if [ "$scenario" = "morning_rush" ]; then
        baseline_csv="$BASELINE_MORNING"
    else
        baseline_csv="$BASELINE_EVENING"
    fi

    # Phase A: RL evaluation (writes per-intersection JSON in output_csv)
    if ! python src/evaluate.py \
            --model "$model_path" \
            --scenario "$scenario" \
            --skip_baseline \
            --output "$output_csv" > "$log_file" 2>&1; then
        echo "[FAIL] $run_id (eval, see $log_file)"
        echo "${run_id},${scenario},${ep_count},FAIL_EVAL," >> "$EVAL_LOG"
        exit 0  # do not fail the whole xargs job
    fi

    # Phase B: Finalize for dashboard (writes results.csv + updates meta.json)
    if python hpc/sweep/finalize_for_dashboard.py \
            --baseline_csv "$baseline_csv" \
            --rl_csv "$output_csv" \
            --experiment_dir "$exp_dir" >> "$log_file" 2>&1; then
        echo "[OK]   $run_id"
        echo "${run_id},${scenario},${ep_count},OK,${output_csv}" >> "$EVAL_LOG"
    else
        echo "[FAIL] $run_id (finalize, see $log_file)"
        echo "${run_id},${scenario},${ep_count},FAIL_FINALIZE,${output_csv}" >> "$EVAL_LOG"
    fi
'

ELAPSED=$(( $(date +%s) - START ))
SUCCEEDED=$(grep -c ",OK," "$EVAL_LOG" || echo 0)
FAILED=$(grep -c ",FAIL," "$EVAL_LOG" || echo 0)

echo ""
echo "==================================================="
echo "Evaluation complete in $((ELAPSED/60))min ($((ELAPSED%60))s)"
echo "  $SUCCEEDED succeeded, $FAILED failed (out of $JOB_COUNT discovered)"
echo "  Baseline (cached, run once): $BASELINE_MORNING, $BASELINE_EVENING"
echo "  Per-model RL results:        results/experiments/<run_id>/eval_rl.csv"
echo "  Per-model logs:              results/experiments/<run_id>/eval_rl.log"
echo "  Summary index:               $EVAL_LOG"
echo "==================================================="
