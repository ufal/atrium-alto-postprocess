#!/usr/bin/env bash
# tools/run_optim_pipeline.sh
# Unified pipeline to run all parameter optimization and rule coverage tools (Issue #5).

# -u: an unset variable is a bug, not an empty string. -o pipefail: a failing
# stage inside a pipe must not be masked by a succeeding `tee`/`head`.
set -euo pipefail

# --- Configuration ---
INPUT_DIR="${1:-data_samples/DOC_LINE_CATEG}"
CONFIG="${2:-setup/config.txt}"
OUT_BASE="${3:-sweep_output_full}"
TRIALS="${4:-400}"         # For RF and Optuna
SOBOL_N="${5:-256}"        # N parameter for Sobol (Note: total evals = N * (D+2))
MORRIS_R="${6:-10}"        # Trajectories for Morris

# --- Gold objective (issue #30) ---
# Without these every stage scores the re-categorisation against the pipeline's
# OWN stored labels. tools/gold/GOLD.md states the consequence plainly: the
# shipped configuration is optimal by construction, and a genuine accuracy
# improvement scores as pure damage. Set both, or read the banner below.
#
#   GOLD_SIDECAR=tools/gold/sidecars/issue30_gold_2067.csv \
#   GOLD_COLUMN=gold_categ ./tools/run_optim_pipeline.sh <input-dir>
GOLD_SIDECAR="${GOLD_SIDECAR:-}"
GOLD_COLUMN="${GOLD_COLUMN:-}"

# Expanded below as ${GOLD_ARGS[@]+"${GOLD_ARGS[@]}"} rather than "${GOLD_ARGS[@]}":
# bash before 4.4 treats the latter as an unset variable under `set -u` and aborts
# on an empty array, which is the default path.
GOLD_ARGS=()
if [ -n "$GOLD_SIDECAR" ] || [ -n "$GOLD_COLUMN" ]; then
    if [ -z "$GOLD_SIDECAR" ] || [ -z "$GOLD_COLUMN" ]; then
        echo "error: set BOTH GOLD_SIDECAR and GOLD_COLUMN, or neither." >&2
        echo "       A sidecar without a column is scored self-referentially." >&2
        exit 2
    fi
    if [ ! -f "$GOLD_SIDECAR" ]; then
        echo "error: GOLD_SIDECAR does not exist: $GOLD_SIDECAR" >&2
        exit 2
    fi
    GOLD_ARGS=(--gold-sidecar "$GOLD_SIDECAR" --gold-column "$GOLD_COLUMN")
fi

echo "============================================================"
echo " ATRIUM ALTO Post-Process : Unified Optimization Pipeline"
echo "============================================================"
echo " Input Data   : $INPUT_DIR"
echo " Config File  : $CONFIG"
echo " Output Base  : $OUT_BASE"
if [ ${#GOLD_ARGS[@]} -gt 0 ]; then
    echo " Objective    : agreement with gold ($GOLD_SIDECAR :: $GOLD_COLUMN)"
else
    echo " Objective    : SELF-REFERENTIAL (no gold)"
fi
echo " ML Trials    : $TRIALS (RF/Optuna)"
echo " Sobol N      : $SOBOL_N"
echo "------------------------------------------------------------"

# --- Interpreter guard ------------------------------------------------------
# setup/requirements-sweep.txt floors scikit-learn at 1.9 and matplotlib at
# 3.11.1, and BOTH are published for Python >= 3.11 only. On Python 3.10 pip
# cannot resolve them and aborts this whole script (`set -e`) with a wall of
# candidate versions that never mentions the interpreter -- which is the actual
# cause. Check it here so the message names the problem.
#
# This is not a sweep-specific requirement: CI runs 3.11, the Dockerfile is
# python:3.11-slim, and setup/requirements-test.txt needs pandas>=3.0.5, which
# is also 3.11+. A 3.10 venv cannot install this repo's test dependencies either.
PY_BIN="${PYTHON:-python}"
REQUIRED_MINOR=11
read -r PY_MAJOR PY_MINOR PY_FULL <<< "$("$PY_BIN" -c 'import sys; print(sys.version_info[0], sys.version_info[1], sys.version.split()[0])')"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt "$REQUIRED_MINOR" ]; }; then
    {
        echo ""
        echo "ERROR: the sweep needs Python >= 3.${REQUIRED_MINOR}; this interpreter is ${PY_FULL}."
        echo "       ($PY_BIN -> $("$PY_BIN" -c 'import sys; print(sys.executable)'))"
        echo ""
        echo "  Cause: scikit-learn >= 1.8 and matplotlib >= 3.11 ship no Python 3.10"
        echo "         wheels, so 'scikit-learn>=1.9.0,<1.10' resolves to nothing."
        echo ""
        echo "  Fix:   recreate the venv on 3.11, which is what CI and the Docker"
        echo "         image already use:"
        echo ""
        echo "           python3.11 -m venv venv-alto && . venv-alto/bin/activate"
        echo "           pip install -r setup/requirements.txt -r setup/requirements-sweep.txt"
        echo ""
        echo "  If you must stay on 3.10, the sweep CODE is compatible with older"
        echo "  releases (it uses only RandomForestRegressor, permutation_importance"
        echo "  and pyplot). Install them yourself and re-run with SKIP_DEP_INSTALL=1:"
        echo ""
        echo "           pip install 'scikit-learn>=1.4,<1.8' 'matplotlib>=3.8,<3.11' optuna SALib"
        echo "           SKIP_DEP_INSTALL=1 $0 $*"
        echo ""
        echo "  Note that this tunes production constants against a different library"
        echo "  set than production runs on. See setup/requirements-sweep.txt."
        echo ""
    } >&2
    exit 1
fi

# Ensure we have the required dependencies
if [ -n "${SKIP_DEP_INSTALL:-}" ]; then
    echo ">> Skipping dependency install (SKIP_DEP_INSTALL set)."
else
    echo ">> Checking/Installing dependencies..."
    if ! pip install -r setup/requirements-sweep.txt -q; then
        {
            echo ""
            echo "ERROR: could not install setup/requirements-sweep.txt on Python ${PY_FULL}."
            echo "       Re-run with SKIP_DEP_INSTALL=1 once the four packages are present,"
            echo "       or see the compatibility note in setup/requirements-sweep.txt."
            echo ""
        } >&2
        exit 1
    fi
fi

if [ ${#GOLD_ARGS[@]} -eq 0 ]; then
    echo ""
    echo "############################################################"
    echo "#  WARNING: running WITHOUT a gold set.                     #"
    echo "############################################################"
    echo "#  Every stage below scores the re-score against the        #"
    echo "#  pipeline's OWN stored categories. That objective is      #"
    echo "#  circular: the shipped config is optimal by construction, #"
    echo "#  and a genuine accuracy improvement scores as damage.     #"
    echo "#                                                           #"
    echo "#  Importance rankings remain meaningful; any 'best_config' #"
    echo "#  and every PRUNE/KEEP verdict do NOT.                     #"
    echo "#                                                           #"
    echo "#  Fix: GOLD_SIDECAR=tools/gold/sidecars/issue30_gold_2067.csv"
    echo "#       GOLD_COLUMN=gold_categ                              #"
    echo "#  See tools/gold/GOLD.md.                                  #"
    echo "############################################################"
    echo ""
fi

mkdir -p "$OUT_BASE"

# ---------------------------------------------------------
# 1. Rule Coverage
# ---------------------------------------------------------
echo ""
echo "[1/6] Running Rule Coverage Report..."
"$PY_BIN" tools/rule_coverage_report.py \
    --input-dir "$INPUT_DIR" \
    --config "$CONFIG" \
    ${GOLD_ARGS[@]+"${GOLD_ARGS[@]}"} \
    --output "$OUT_BASE/rule_coverage.json"

# ---------------------------------------------------------
# 2. Sklearn (Random Forest MDI & Permutation)
# ---------------------------------------------------------
echo ""
echo "[2/6] Running Sklearn (Random Forest) Sweep..."
"$PY_BIN" tools/const_importance_sweep.py \
    --input-dir "$INPUT_DIR" \
    ${GOLD_ARGS[@]+"${GOLD_ARGS[@]}"} \
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
"$PY_BIN" tools/const_importance_sweep.py \
    --input-dir "$INPUT_DIR" \
    ${GOLD_ARGS[@]+"${GOLD_ARGS[@]}"} \
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
"$PY_BIN" tools/const_importance_sweep.py \
    --input-dir "$INPUT_DIR" \
    ${GOLD_ARGS[@]+"${GOLD_ARGS[@]}"} \
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
    "$PY_BIN" tools/importance_consensus.py \
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
