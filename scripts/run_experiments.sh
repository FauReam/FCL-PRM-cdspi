#!/usr/bin/env bash
# FCL-PRM 实验批量启动器 — 5 configs × 5 rounds，后台安全运行
# 用法: nohup bash scripts/run_experiments.sh &> experiments/batch_$(date +%Y%m%d_%H%M%S).log &
# 关闭终端后继续运行，用 tail -f experiments/batch_*.log 查看进度
set -euo pipefail

cd /home/jiayu/FCL-PRM
PYTHON=venv/bin/python3
BATCH_ID=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="experiments/batch_${BATCH_ID}.log"

log()   { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }
log_sep(){ log "============================================================"; }

log_sep
log "FCL-PRM 5-config × 2-round 实验批次"
log "批次ID: ${BATCH_ID}"
log "硬件: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo GPU)"
log "开始时间: $(date)"
log "预计总耗时: ~8 天"
log_sep

# 实验队列: config name | rounds | 预计耗时
declare -A QUEUE
QUEUE=(
    ["m3_fedavg_head_1.4b"]="2     16h   H1 baseline"
    ["m3_fedavg_full_1.4b"]="2     22h   H2 MAKE-OR-BREAK"
    ["m3_fedavg_lora_r8_1.4b"]="2     17h   H1 replication"
    ["m3_fedavg_lora_r256_1.4b"]="2     17h   H2 boundary"
    ["m2_centralized_full_1.4b"]="2      3h   sanity check"
)

TOTAL=0
PASSED=0
FAILED=0
START_TIME=$(date +%s)

for config in \
    m3_fedavg_head_1.4b \
    m3_fedavg_full_1.4b \
    m3_fedavg_lora_r8_1.4b \
    m3_fedavg_lora_r256_1.4b \
    m2_centralized_full_1.4b; \
do
    TOTAL=$((TOTAL + 1))
    IFS=' ' read -r ROUNDS EST PHASE <<< "${QUEUE[$config]}"

    CFG_FILE="configs/${config}.yaml"
    CFG_LOG="experiments/${config}_5r/run_${BATCH_ID}.log"
    mkdir -p "experiments/${config}_5r"

    log ""
    log "▶ [${TOTAL}/5] 启动: ${config}  (${ROUNDS}r, 预计${EST})"
    log "  阶段: ${PHASE}"
    log "  日志: ${CFG_LOG}"

    CFG_START=$(date +%s)

    # 判断入口点
    if echo "$config" | grep -q "centralized"; then
        SCRIPT="scripts/train_centralized_prm.py"
    else
        SCRIPT="scripts/run_federated.py"
    fi

    # 运行 — 错误时继续下一个，不中断整批
    set +e
    "$PYTHON" "$SCRIPT" --config "$CFG_FILE" --rounds "$ROUNDS" \
        >> "$CFG_LOG" 2>&1
    EXIT_CODE=$?
    set -e

    CFG_ELAPSED=$(($(date +%s) - CFG_START))
    CFG_H=$((CFG_ELAPSED / 3600))
    CFG_M=$(((CFG_ELAPSED % 3600) / 60))

    if [ $EXIT_CODE -eq 0 ]; then
        PASSED=$((PASSED + 1))
        log "✅ [${TOTAL}/5] 完成: ${config}  (耗时 ${CFG_H}h${CFG_M}m)"
    else
        FAILED=$((FAILED + 1))
        log "❌ [${TOTAL}/5] 失败: ${config}  (exit=${EXIT_CODE}, 耗时 ${CFG_H}h${CFG_M}m)"
        log "  检查日志: tail -100 ${CFG_LOG}"
        # 如果是 full-FT 失败 → 致命
        if [ "$config" = "m3_fedavg_full_1.4b" ]; then
            log "🔴 H2 生死实验失败！查看 CD-SPI 数据后决定是否 KILL。"
        fi
    fi

    # GPU cleanup between configs — prevent OOM cascade
    "$PYTHON" -c "import torch; torch.cuda.empty_cache(); \
        print(f'GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated, \
        {torch.cuda.memory_reserved()/1e9:.1f}GB reserved')" \
        >> "$CFG_LOG" 2>&1 || true
done

TOTAL_ELAPSED=$(($(date +%s) - START_TIME))
TOTAL_D=$((TOTAL_ELAPSED / 86400))
TOTAL_H=$(((TOTAL_ELAPSED % 86400) / 3600))

log_sep
log "批次完成"
log "通过: ${PASSED}/${TOTAL}  失败: ${FAILED}/${TOTAL}"
log "总耗时: ${TOTAL_D}d${TOTAL_H}h"
log "结束时间: $(date)"
log_sep

# 快速摘要
echo ""
echo ">>> CD-SPI 快速摘要 <<<"
for config in \
    m3_fedavg_head_1.4b \
    m3_fedavg_full_1.4b \
    m3_fedavg_lora_r8_1.4b \
    m3_fedavg_lora_r256_1.4b; \
do
    CFG_LOG="experiments/${config}_5r/run_${BATCH_ID}.log"
    if [ -f "$CFG_LOG" ]; then
        CDSPI=$(grep "\[CD-SPI sym\]" "$CFG_LOG" | tail -1 | grep -oP 'mean=\K[0-9.]+' || echo "N/A")
        EVR=$(grep "\[PCA EVR\]" "$CFG_LOG" | tail -1 | grep -oP 'first=\K[0-9.]+' || echo "N/A")
        CKA=$(grep "\[CKA\]" "$CFG_LOG" | tail -1 | grep -oP 'mean=\K[0-9.]+' || echo "N/A")
        echo "  ${config}: CD-SPI=${CDSPI}  EVR=${EVR}  CKA=${CKA}"
    fi
done
