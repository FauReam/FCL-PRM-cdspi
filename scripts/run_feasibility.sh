#!/usr/bin/env bash
# 2-day feasibility analysis: run head-only (H1 control) then full-FT (H2 test)
# Each: 500 samples/client, 2 rounds.  Total wall-clock < 1h.
set -euo pipefail

cd /home/jiayu/FCL-PRM
PYTHON=venv/bin/python3
LOGDIR=experiments/feasibility_analysis
mkdir -p "$LOGDIR"

echo "============================================================"
echo "FCL-PRM Feasibility Analysis"
echo "Started: $(date)"
echo "============================================================"

# ---- Stage 1: H1 control (head-only, should show CD-SPI ≈ 0) ----
echo ""
echo "[Stage 1/3] H1 control: head-only 2-round"
echo "Expected: CD-SPI sym ≈ 0, CKA ≈ 1.0, EVR ≈ 0"
echo "----------------------------------------"
$PYTHON scripts/run_federated.py \
    --config configs/feasibility_head_1.4b.yaml \
    --rounds 2 \
    2>&1 | tee "$LOGDIR/head_only.log"
echo "[Stage 1] DONE at $(date)"

# ---- Stage 2: H2 test (full-FT, MAKE-OR-BREAK) ----
echo ""
echo "[Stage 2/3] H2 test: full-FT 2-round"
echo "Expected: CD-SPI sym > 0.02 for feasibility"
echo "----------------------------------------"
$PYTHON scripts/run_federated.py \
    --config configs/feasibility_full_1.4b.yaml \
    --rounds 2 \
    2>&1 | tee "$LOGDIR/full_ft.log"
echo "[Stage 2] DONE at $(date)"

# ---- Stage 3: extract key metrics ----
echo ""
echo "[Stage 3/3] Extracting CD-SPI comparison"
echo "============================================================"
echo "FEASIBILITY REPORT"
echo "Generated: $(date)"
echo "============================================================"
echo ""
echo "--- H1 (head-only) ---"
grep -E "\[CD-SPI sym\]|\[PCA EVR\]|\[CKA\]|Summary" "$LOGDIR/head_only.log" | tail -10
echo ""
echo "--- H2 (full-FT) ---"
grep -E "\[CD-SPI sym\]|\[PCA EVR\]|\[CKA\]|Summary" "$LOGDIR/full_ft.log" | tail -10
echo ""
echo "--- VERDICT ---"
H1_CDSPI=$(grep "\[CD-SPI sym\]" "$LOGDIR/head_only.log" | tail -1 | grep -oP 'mean=\K[0-9.]+' || echo "N/A")
H2_CDSPI=$(grep "\[CD-SPI sym\]" "$LOGDIR/full_ft.log" | tail -1 | grep -oP 'mean=\K[0-9.]+' || echo "N/A")
echo "H1 CD-SPI sym: $H1_CDSPI  (expect ≈ 0)"
echo "H2 CD-SPI sym: $H2_CDSPI  (expect > 0.02 for feasibility)"
if [ "$H2_CDSPI" != "N/A" ] && [ "$H1_CDSPI" != "N/A" ]; then
    if (( $(echo "$H2_CDSPI > 0.02" | bc -l) )); then
        echo ""
        echo "✅ FEASIBLE: Full-FT CD-SPI ($H2_CDSPI) exceeds noise floor."
        echo "   Proceed to full 10-round experiments."
    else
        echo ""
        echo "❌ NOT FEASIBLE: Full-FT CD-SPI ($H2_CDSPI) at noise level."
        echo "   Direction may be falsified. Consider KILL."
    fi
fi
echo "============================================================"
