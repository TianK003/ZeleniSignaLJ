#!/bin/bash
# Zeleni SignaLJ - Evaluate all trained models with episode count > 500
# Generates ONE fresh route file per scenario, then evaluates every model on it.
# All morning models compared on the same route; all evening models compared on the same route.
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

# ── Phase 1: Generate route files ONLY if they don't already exist ──
# After this, all eval jobs only READ these files (evaluate.py never writes routes).
MORNING_ROUTE="data/routes/routes_morning_rush.rou.xml"
EVENING_ROUTE="data/routes/routes_evening_rush.rou.xml"

echo "=== Phase 1: Ensuring evaluation route files exist ==="
if [ -f "$MORNING_ROUTE" ]; then
    echo "  [SKIP] $MORNING_ROUTE already exists (will be re-used as-is)"
else
    echo "  [GEN]  Generating $MORNING_ROUTE..."
    python src/generate_demand.py --scenario morning_rush
fi

if [ -f "$EVENING_ROUTE" ]; then
    echo "  [SKIP] $EVENING_ROUTE already exists (will be re-used as-is)"
else
    echo "  [GEN]  Generating $EVENING_ROUTE..."
    python src/generate_demand.py --scenario evening_rush
fi

# Make routes read-only as a hard guarantee against accidental writes
chmod 444 "$MORNING_ROUTE" "$EVENING_ROUTE"
echo "  Routes locked read-only (chmod 444)"
echo ""

# ── Phase 2: Evaluate all eligible models ──
EVAL_LOG="results/eval_all_summary.csv"
echo "run_id,scenario,episodes,status,output_csv" > "$EVAL_LOG"

count=0
skipped=0
failed=0
succeeded=0

for run_dir in results/experiments/*/; do
    run_id=$(basename "$run_dir")

    # Extract episode count from "_NNNNep" suffix at end of directory name
    ep_part="${run_id##*_}"          # e.g. "4800ep" or "200ep"
    ep_count="${ep_part%ep}"         # strip "ep" suffix

    # Skip if not numeric or <= 500
    if ! [[ "$ep_count" =~ ^[0-9]+$ ]] || [ "$ep_count" -le 500 ]; then
        skipped=$((skipped+1))
        continue
    fi

    # Determine scenario from directory name
    if [[ "$run_id" == *morningrush* ]]; then
        scenario="morning_rush"
    elif [[ "$run_id" == *eveningrush* ]]; then
        scenario="evening_rush"
    else
        echo "[SKIP] $run_id: unknown scenario"
        skipped=$((skipped+1))
        continue
    fi

    # Skip if no final model exists
    model_path="${run_dir}ppo_shared_policy.zip"
    if [ ! -f "$model_path" ]; then
        # Fall back to latest checkpoint
        model_path="${run_dir}checkpoints/ppo_model_latest.zip"
        if [ ! -f "$model_path" ]; then
            echo "[SKIP] $run_id: no model file"
            skipped=$((skipped+1))
            continue
        fi
    fi

    count=$((count+1))
    output_csv="${run_dir}eval_results.csv"

    echo ""
    echo "=== [$count] Evaluating $run_id ($scenario, ${ep_count}ep) ==="

    if python src/evaluate.py \
        --model "$model_path" \
        --scenario "$scenario" \
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
echo "Evaluation complete: $succeeded succeeded, $failed failed, $skipped skipped (<=500ep or no model)"
echo "Summary: $EVAL_LOG"
echo "==================================================="
