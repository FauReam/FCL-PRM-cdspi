#!/usr/bin/env bash
# FCL-PRM 实验批量启动器 — 2026-06-18 修复版
# Fix: simulator.py OptimizedModule unwrap + centralized --rounds
# 用法: nohup bash scripts/run_batch_20260618.sh &> experiments/batch_$(date +%Y%m%d_%H%M%S).log &
set -euo pipefail

cd /home/jiayu/FCL-PRM
PYTHON=venv/bin/python3
BATCH_ID=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="experiments/batch_${BATCH_ID}.log"

log()   { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }
log_sep(){ log "============================================================"; }

log_sep
log "FCL-PRM batch — 2026-06-18 (fixed OptimizedModule + --rounds)"
log "Batch ID: ${BATCH_ID}"
log "Hardware: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo GPU)"
log "Start: $(date)"
log_sep

# ============================================================
# Stage 0: H2 feasibility quick test (~20min)
# ============================================================
log ""
log "▶ [Stage 0] H2 feasibility: full-FT 500samp 2r"
log "  Config: configs/feasibility_full_1.4b.yaml"
log "  Expected: CD-SPI sym > 0.02 → thesis viable"
log "  Log: experiments/feasibility_analysis/full_ft.log"

mkdir -p experiments/feasibility_analysis

set +e
$PYTHON scripts/run_federated.py \
    --config configs/feasibility_full_1.4b.yaml \
    >> experiments/feasibility_analysis/full_ft.log 2>&1
EXIT=$?
set -e

if [ $EXIT -eq 0 ]; then
    CDSPI=$(grep "\[CD-SPI sym\]" experiments/feasibility_analysis/full_ft.log | \
        tail -1 | grep -oP 'mean=\K[0-9.]+' || echo "N/A")
    log "✅ Stage 0 完成 — CD-SPI sym = ${CDSPI}"
    if [ "$(echo "$CDSPI > 0.02" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
        log "🟢 H2 PASSED: CD-SPI sym > 0.02 — thesis viable, continuing"
    else
        log "🔴 H2 LOW: CD-SPI sym <= 0.02 — review needed but continuing"
    fi
else
    log "❌ Stage 0 失败 (exit=$EXIT) — check log, continuing anyway"
fi

# GPU cleanup
$PYTHON -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# ============================================================
# Stage 1: M2 centralized baseline (~3h)
# ============================================================
log ""
log "▶ [Stage 1] M2 centralized baseline"
log "  Config: configs/m2_centralized_full_1.4b.yaml"
log "  Log: experiments/m2_centralized_full_1.4b_5r/run_${BATCH_ID}.log"

mkdir -p experiments/m2_centralized_full_1.4b_5r

STAGE_START=$(date +%s)
set +e
$PYTHON scripts/train_centralized_prm.py \
    --config configs/m2_centralized_full_1.4b.yaml \
    >> "experiments/m2_centralized_full_1.4b_5r/run_${BATCH_ID}.log" 2>&1
EXIT=$?
set -e
STAGE_ELAPSED=$((($(date +%s) - STAGE_START) / 60))

if [ $EXIT -eq 0 ]; then
    log "✅ Stage 1 完成 (${STAGE_ELAPSED}m)"
else
    log "❌ Stage 1 失败 (exit=$EXIT, ${STAGE_ELAPSED}m)"
fi

$PYTHON -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# ============================================================
# Stage 2-5: M3 federated capacity continuum
# ============================================================
declare -A STAGE_INFO
STAGE_INFO=(
    ["m3_fedavg_full_1.4b"]="H2 MAKE-OR-BREAK | 22h"
    ["m3_fedavg_lora_r8_1.4b"]="H1 replication | 17h"
    ["m3_fedavg_lora_r64_1.4b"]="H2 termination-anchor | 17h"
    ["m3_fedavg_lora_r256_1.4b"]="H2 boundary | 17h"
)

STAGE_NUM=1
for config in \
    m3_fedavg_full_1.4b \
    m3_fedavg_lora_r8_1.4b \
    m3_fedavg_lora_r64_1.4b \
    m3_fedavg_lora_r256_1.4b; \
do
    STAGE_NUM=$((STAGE_NUM + 1))
    IFS='|' read -r PHASE EST <<< "${STAGE_INFO[$config]}"
    PHASE=$(echo "$PHASE" | xargs)
    EST=$(echo "$EST" | xargs)

    CFG_FILE="configs/${config}.yaml"
    CFG_LOG="experiments/${config}_5r/run_${BATCH_ID}.log"
    mkdir -p "experiments/${config}_5r"

    log ""
    log "▶ [Stage ${STAGE_NUM}] ${config} (${PHASE}, ~${EST})"
    log "  Config: ${CFG_FILE}"
    log "  Log: ${CFG_LOG}"

    STAGE_START=$(date +%s)
    set +e
    $PYTHON scripts/run_federated.py \
        --config "$CFG_FILE" \
        --rounds 2 \
        >> "$CFG_LOG" 2>&1
    EXIT=$?
    set -e
    STAGE_ELAPSED=$((($(date +%s) - STAGE_START) / 3600))
    STAGE_REM=$(((($(date +%s) - STAGE_START) % 3600) / 60))

    if [ $EXIT -eq 0 ]; then
        log "✅ Stage ${STAGE_NUM} 完成 (${STAGE_ELAPSED}h${STAGE_REM}m)"
        # Extract CD-SPI summary
        CDSPI=$(grep "\[CD-SPI sym\]" "$CFG_LOG" | tail -1 | \
            grep -oP 'mean=\K[0-9.]+' || echo "N/A")
        log "   CD-SPI sym = ${CDSPI}"
    else
        log "❌ Stage ${STAGE_NUM} 失败 (exit=$EXIT, ${STAGE_ELAPSED}h${STAGE_REM}m)"
    fi

    $PYTHON -c "import torch; torch.cuda.empty_cache(); \
        print(f'GPU cleanup: {torch.cuda.memory_allocated()/1e9:.1f}GB')" \
        2>/dev/null || true
done

# ============================================================
# Summary
# ============================================================
log_sep
log "Batch complete"
log "End: $(date)"
log "Logs: experiments/batch_${BATCH_ID}.log"
log_sep

echo ""
echo ">>> CD-SPI Summary <<<"
for logfile in \
    experiments/m3_fedavg_full_1.4b_5r/run_${BATCH_ID}.log \
    experiments/m3_fedavg_lora_r8_1.4b_5r/run_${BATCH_ID}.log \
    experiments/m3_fedavg_lora_r64_1.4b_5r/run_${BATCH_ID}.log \
    experiments/m3_fedavg_lora_r256_1.4b_5r/run_${BATCH_ID}.log; \
do
    cfg=$(basename $(dirname "$logfile") | sed 's/_5r//')
    if [ -f "$logfile" ]; then
        CDSPI=$(grep "\[CD-SPI sym\]" "$logfile" | tail -1 | \
            grep -oP 'mean=\K[0-9.]+' || echo "N/A")
        EVR=$(grep "\[PCA EVR\]" "$logfile" | tail -1 | \
            grep -oP 'first=\K[0-9.]+' || echo "N/A")
        echo "  ${cfg}: CD-SPI=${CDSPI}  EVR=${EVR}"
    fi
done
echo "  (head-only baseline: CD-SPI=0.0011 EVR=0.0000 — from 06-16 batch)"
