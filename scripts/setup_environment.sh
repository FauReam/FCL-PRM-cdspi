#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# FCL-PRM 环境准备脚本
# ──────────────────────────────────────────────────────────────────────────────
# 用法:
#   bash scripts/setup_environment.sh                    # 默认（HuggingFace 官方源）
#   bash scripts/setup_environment.sh --mirror hf        # 使用 HuggingFace 国内镜像
#   bash scripts/setup_environment.sh --mirror modelscope # 使用 ModelScope 镜像
#
# 功能:
#   1. 安装 Python 依赖
#   2. 下载 Pythia-1.4B 模型（预训练 backbone）
#   3. 下载 VersaPRM 数据集
#   4. 下载 ProcessBench 数据集（如有）
#   5. 验证完整性
#
# 国内用户建议: bash scripts/setup_environment.sh --mirror hf
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=============================================="
echo "  FCL-PRM 环境准备脚本"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# ── 解析参数 ────────────────────────────────────────────────────────────────
MIRROR="${1:-}"
if [[ "$MIRROR" == "--mirror" && "${2:-}" == "hf" ]]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "[INFO] 使用 HuggingFace 国内镜像: $HF_ENDPOINT"
elif [[ "$MIRROR" == "--mirror" && "${2:-}" == "modelscope" ]]; then
    echo "[INFO] 使用 ModelScope（当前仅支持 HF 镜像）"
    export HF_ENDPOINT="https://hf-mirror.com"
else
    echo "[INFO] 使用 HuggingFace 官方源"
fi

# ── 1. 安装 Python 依赖 ────────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────"
echo " [1/5] 安装 Python 依赖"
echo "──────────────────────────────────────────────"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "[INFO] 创建虚拟环境: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q
pip install -r "$PROJECT_DIR/requirements.txt" -q
echo "[OK] Python 依赖安装完成"

# ── 2. 下载 Pythia-1.4B 模型 ────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────"
echo " [2/5] 下载 Pythia-1.4B 模型"
echo "──────────────────────────────────────────────"
HF_MODEL="${HF_MODEL:-EleutherAI/pythia-1.4b}"
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface/hub}"
python3 -c "
import sys, os
os.environ['HF_ENDPOINT'] = '${HF_ENDPOINT:-https://huggingface.co}'
from transformers import AutoModel, AutoTokenizer
model_name = '$HF_MODEL'
print(f'  下载模型: {model_name}')
print(f'  缓存目录: {os.environ.get(\"HF_HOME\", os.path.expanduser(\"~/.cache/huggingface/hub\"))}')
print()
try:
    print('  [1/2] 下载 tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, resume_download=True)
    print(f'  [OK] Tokenizer 下载完成: {type(tokenizer).__name__}')
    print()
    print('  [2/2] 下载 model (BF16)...')
    import torch
    model = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16)
    print(f'  [OK] 模型下载完成: {type(model).__name__}, {sum(p.numel() for p in model.parameters())/1e9:.2f}B 参数')
    print()
    print('  [验证] 运行模型 forward 测试...')
    encoded = tokenizer('Hello world', return_tensors='pt')
    with torch.no_grad():
        out = model(**encoded)
    print(f'  [OK] Forward 测试通过: last_hidden shape={out.last_hidden_state.shape}')
except Exception as e:
    print(f'[ERROR] 模型下载失败: {e}')
    sys.exit(1)
"
echo "[OK] Pythia-1.4B 模型就绪"

# ── 3. 下载 VersaPRM 数据集 ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────"
echo " [3/5] 下载 VersaPRM 数据集"
echo "──────────────────────────────────────────────"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/versaprm}"
mkdir -p "$DATA_DIR"

python3 -c "
import sys, os, json
from pathlib import Path
from urllib.request import urlopen

data_dir = Path('$DATA_DIR')
target_file = data_dir / 'versa_prm.jsonl'

if target_file.exists() and target_file.stat().st_size > 1e7:
    print(f'  [SKIP] 数据集已存在: {target_file} ({target_file.stat().st_size / 1e6:.1f} MB)')
    sys.exit(0)

# Try mirror first, fallback to official
urls = [
    'https://hf-mirror.com/datasets/fau-ream/versaprm/resolve/main/versa_prm_steps.jsonl',
    'https://huggingface.co/datasets/fau-ream/versaprm/resolve/main/versa_prm_steps.jsonl',
]

downloaded = False
for url in urls:
    try:
        print(f'  正在下载: {url}')
        response = urlopen(url, timeout=300)
        total_size = int(response.headers.get('Content-Length', 0))
        chunk_size = 1024 * 1024  # 1MB
        downloaded_size = 0
        with open(target_file, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    pct = downloaded_size * 100 / total_size
                else:
                    pct = 0
                print(f'    下载进度: {downloaded_size / 1e6:.1f} MB / {total_size / 1e6:.1f} MB ({pct:.0f}%)', end='\r')
        print()
        print(f'  [OK] 下载完成: {target_file} ({downloaded_size / 1e6:.1f} MB)')
        downloaded = True
        break
    except Exception as e:
        print(f'  [WARN] 下载失败 ({url}): {e}')
        continue

if not downloaded:
    print(f'[ERROR] 所有下载源均失败')
    sys.exit(1)
"
echo "[OK] VersaPRM 数据集就绪"

# ── 4. 下载 ProcessBench 数据集 ────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────"
echo " [4/5] 下载 ProcessBench 数据集"
echo "──────────────────────────────────────────────"
PB_DIR="${PB_DIR:-$PROJECT_DIR/data/processbench}"
mkdir -p "$PB_DIR"

python3 -c "
import sys, os
from pathlib import Path
from urllib.request import urlopen

pb_dir = Path('$PB_DIR')
files = {
    'processbench.jsonl': 'https://huggingface.co/datasets/fau-ream/ProcessBench/resolve/main/processbench.jsonl',
    'processbench_steps.jsonl': 'https://huggingface.co/datasets/fau-ream/ProcessBench/resolve/main/processbench_steps.jsonl',
}

for name, url in files.items():
    target = pb_dir / name
    if target.exists() and target.stat().st_size > 1e5:
        print(f'  [SKIP] {name} 已存在')
        continue
    try:
        print(f'  正在下载: {name}...')
        response = urlopen(url, timeout=120)
        total = int(response.headers.get('Content-Length', 0))
        chunk = 1024 * 1024
        downloaded = 0
        with open(target, 'wb') as f:
            while True:
                data = response.read(chunk)
                if not data:
                    break
                f.write(data)
                downloaded += len(data)
        print(f'  [OK] {name} ({downloaded / 1e6:.1f} MB)')
    except Exception as e:
        print(f'  [WARN] {name} 下载失败: {e}')
        continue
"
echo "[OK] ProcessBench 数据集处理完成"

# ── 5. 验证完整性 ───────────────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────"
echo " [5/5] 验证完整性"
echo "──────────────────────────────────────────────"
python3 -c "
import sys
sys.path.insert(0, '.')
import torch
from pathlib import Path

errors = []

# Check model cache
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained('$HF_MODEL', local_files_only=True)
    print(f'  [OK] 模型配置加载成功: {config.model_type}')
except Exception as e:
    errors.append(f'模型未缓存: {e}')

# Check VersaPRM
vp = Path('$DATA_DIR/versa_prm.jsonl')
if vp.exists() and vp.stat().st_size > 1e7:
    line_count = sum(1 for _ in open(vp))
    print(f'  [OK] VersaPRM: {vp} ({line_count} 条样本)')
else:
    errors.append(f'VersaPRM 不完整: {vp}')

# Check ProcessBench
pb_dir = Path('$PB_DIR')
for fname in ['processbench.jsonl', 'processbench_steps.jsonl']:
    f = pb_dir / fname
    if f.exists():
        lc = sum(1 for _ in open(f))
        print(f'  [OK] ProcessBench: {f.name} ({lc} 条)')
    else:
        errors.append(f'ProcessBench 缺少: {fname}')

if errors:
    print()
    for e in errors:
        print(f'  [WARN] {e}')
    print('  [WARN] 部分组件未就绪，请检查网络后重试')
else:
    print()
    print('  [OK] 所有组件验证通过')
" 2>&1

echo ""
echo "=============================================="
echo "  FCL-PRM 环境准备完成"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""
echo "下一步："
echo "  集中式基线实验："
echo "    python scripts/train_centralized_prm.py --config configs/m2_centralized_full_1.4b.yaml"
echo "  联邦容量连续谱："
echo "    python scripts/run_federated.py --config configs/m3_fedavg_head_1.4b.yaml"
echo "  烟雾测试（快速验证）："
echo "    python scripts/run_federated.py --config configs/smoke_versaprm.yaml --rounds 3"
