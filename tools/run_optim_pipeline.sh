#!/usr/bin/env bash
# tools/run_optim_pipeline.sh
# Unified pipeline to run all parameter optimization and rule coverage tools (Issue #5).

set -e

# --- Configuration ---
INPUT_DIR="${1:-data_samples/DOC_LINE_CATEG}"
CONFIG="${2:-setup/config_langID.txt}"
OUT_BASE="${3:-sweep_output_full}"
TRIALS="${4:-400}"         # For RF and Optuna
SOBOL_N="${5:-256}"        # N parameter for Sobol (Note: total evals = N * (D+2))
MORRIS_R="${6:-10}"        # Trajectories for Morris

echo "============================================================"
echo " ATRIUM ALTO Post-Process : Unified Optimization Pipeline"
echo "============================================================"
echo " Input Data   : $INPUT_DIR"
echo " Config File  : $CONFIG"
echo " Output Base  : $OUT_BASE"
echo " ML Trials    : $TRIALS (RF/Optuna)"
echo " Sobol N      : $SOBOL_N"
echo "------------------------------------------------------------"

# Ensure we have the required dependencies
echo ">> Checking/Installing dependencies..."
pip install -r setup/requirements-sweep.txt -q

mkdir -p "$OUT_BASE"

# ---------------------------------------------------------
# 1. Rule Coverage
# ---------------------------------------------------------
echo ""
echo "[1/6] Running Rule Coverage Report..."
python tools/rule_coverage_report.py \
    --input-dir "$INPUT_DIR" \
    --config "$CONFIG" \
    --output "$OUT_BASE/rule_coverage.json"

# ---------------------------------------------------------
# 2. Sklearn (Random Forest MDI & Permutation)
# ---------------------------------------------------------
echo ""
echo "[2/6] Running Sklearn (Random Forest) Sweep..."
python tools/const_importance_sweep.py \
    --input-dir "$INPUT_DIR" \
    --config "$CONFIG" \
    --output-dir "$OUT_BASE/rf_sweep" \
    --backend sklearn \
    --metric macro_f1 \
    --n-trials "$TRIALS"

# ---------------------------------------------------------
# 3. Optuna (fANOVA)
# ---------------------------------------------------------
#echo ""
#echo "[3/6] Running Optuna (fANOVA) Sweep..."
#python tools/const_importance_sweep.py \
#    --input-dir "$INPUT_DIR" \
#    --config "$CONFIG" \
#    --output-dir "$OUT_BASE/optuna_sweep" \
#    --backend optuna \
#    --sampler random \
#    --metric macro_f1 \
#    --n-trials "$TRIALS"

# ---------------------------------------------------------
# 4. SALib Morris (Screening & Interaction)
# ---------------------------------------------------------
echo ""
echo "[4/6] Running SALib Morris Screening Sweep..."
python tools/const_importance_sweep.py \
    --input-dir "$INPUT_DIR" \
    --config "$CONFIG" \
    --output-dir "$OUT_BASE/morris_sweep" \
    --backend morris \
    --metric macro_f1 \
    --morris-r "$MORRIS_R"

# ---------------------------------------------------------
# 5. SALib Sobol (Global Sensitivity)
# ---------------------------------------------------------
echo ""
echo "[5/6] Running SALib Sobol Sweep (Computationally Heavy)..."
python tools/const_importance_sweep.py \
    --input-dir "$INPUT_DIR" \
    --config "$CONFIG" \
    --output-dir "$OUT_BASE/sobol_sweep" \
    --backend sobol \
    --metric macro_f1 \
    --sobol-n "$SOBOL_N"

# ---------------------------------------------------------
# 6. Cross-Backend Consensus
# ---------------------------------------------------------
echo ""
if [ -f "tools/importance_consensus.py" ]; then
    echo "[6/6] Generating Cross-Backend Parameter Consensus..."
    python tools/importance_consensus.py \
        "$OUT_BASE/rf_sweep" \
        "$OUT_BASE/optuna_sweep" \
        "$OUT_BASE/morris_sweep" \
        "$OUT_BASE/sobol_sweep" \
        --out "$OUT_BASE/importance_consensus.json" \
        --top-k 10
else
    echo "[6/6] tools/importance_consensus.py not found! Skipping consensus aggregation."
fi

echo ""
echo "============================================================"
echo " ✅ Pipeline Complete!"
echo " Results successfully saved to: $OUT_BASE"
echo " Open $OUT_BASE/importance_consensus.json to view the robust parameters."
echo "============================================================"
