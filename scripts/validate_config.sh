#!/usr/bin/env bash
# Run any FCL-PRM config for 2-round validation checkpoint.
# Usage: bash scripts/validate_config.sh configs/m3_fedavg_head_1.4b.yaml
#        bash scripts/validate_config.sh --all   # run all 17 configs sequentially
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/venv/bin/python3"
RUND_SCRIPT="${PROJECT_DIR}/scripts/run_federated.py"
CENT_SCRIPT="${PROJECT_DIR}/scripts/train_centralized_prm.py"
ROUNDS=2

run_one() {
    local config="$1"
    local name
    name="$(basename "$config" .yaml)"
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] Validating: ${name} (${ROUNDS} rounds)"
    echo "============================================================"

    if echo "$config" | grep -q "centralized"; then
        "$PYTHON" "$CENT_SCRIPT" --config "$config" --rounds "$ROUNDS"
    else
        "$PYTHON" "$RUND_SCRIPT" --config "$config" --rounds "$ROUNDS"
    fi

    echo "[$(date '+%H:%M:%S')] ${name} DONE"
    echo ""
}

if [ "${1:-}" = "--all" ]; then
    echo "Running ALL 17 configs for ${ROUNDS} rounds each..."
    echo "Estimated total: ~205 hours. Make sure nohup/screen is used!"
    echo ""
    for cfg in configs/m2_*.yaml configs/m3_fedavg_head_1.4b.yaml \
               configs/m3_fedavg_lora_r8_1.4b.yaml \
               configs/m3_fedavg_lora_r64_1.4b.yaml \
               configs/m3_fedavg_lora_r128_1.4b.yaml \
               configs/m3_fedavg_lora_r256_1.4b.yaml \
               configs/m3_fedavg_partialft_last2_1.4b.yaml \
               configs/m3_fedavg_partialft_last4_1.4b.yaml \
               configs/m3_fedavg_partialft_last8_1.4b.yaml \
               configs/m3_fedavg_partialft_mlp_1.4b.yaml \
               configs/m3_fedavg_partialft_attn_1.4b.yaml \
               configs/m3_fedavg_full_1.4b.yaml \
               configs/m3_fedavg_head_1.4b_identity.yaml \
               configs/m3_hard_fedavg_1.4b.yaml \
               configs/m3_fedavg_partialft_1.4b.yaml \
               configs/smoke_versaprm.yaml; do
        run_one "$cfg"
    done
    echo "ALL DONE at $(date)"
elif [ $# -ge 1 ]; then
    run_one "$1"
else
    echo "Usage: bash scripts/validate_config.sh <config.yaml>"
    echo "       bash scripts/validate_config.sh --all"
    exit 1
fi
