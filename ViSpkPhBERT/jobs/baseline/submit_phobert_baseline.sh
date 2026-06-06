#!/usr/bin/env bash
#SBATCH --job-name=PHOBERT_BL
#SBATCH --chdir=/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit
#SBATCH --output=/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit/logs/slurm-%x-%j.out
#SBATCH --error=/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit/logs/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:4
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -Eeuo pipefail

REQUIRED_VRAM=28000

PROJECT_ROOT="/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit"
DATASTORE_ROOT="/datastore/cndt_phungdtm/KLTN_GRIT/Nhan"
VENV_PATH="$DATASTORE_ROOT/venv_kltn/bin/activate"

ENTRYPOINT="phobert.py"
SCRIPT_PATH="$PROJECT_ROOT/uit-train/$ENTRYPOINT"
RUN_NAME="new-baseline"

TRAIN_DIR="$PROJECT_ROOT/uit-vsfc/train"
DEV_DIR="$PROJECT_ROOT/uit-vsfc/dev"
TEST_DIR="$PROJECT_ROOT/uit-vsfc/test"

OUTPUT_DIR="$PROJECT_ROOT/uit-models/$RUN_NAME"
LOG_DIR="$PROJECT_ROOT/logs"
CACHE_ROOT="$DATASTORE_ROOT/hf_cache/$RUN_NAME/${SLURM_JOB_ID:-manual}"

module clear -f || true
module load shared python312 || true

if [[ ! -f "$VENV_PATH" ]]; then
    echo "[ERROR] VENV_PATH not found: $VENV_PATH" >&2
    exit 1
fi

source "$VENV_PATH"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CACHE_ROOT"

for path in "$SCRIPT_PATH" "$TRAIN_DIR" "$DEV_DIR" "$TEST_DIR"; do
    if [[ ! -e "$path" ]]; then
        echo "[ERROR] Required path not found: $path" >&2
        exit 1
    fi
done

unset CUDA_VISIBLE_DEVICES

JOB_ID="${SLURM_JOB_ID:-manual}"
CHECK_OUT=$(/usr/local/bin/gpu_check.sh "$REQUIRED_VRAM" "$JOB_ID")
EXIT_CODE=$?

if [[ "$EXIT_CODE" -eq 10 ]]; then
    echo "$CHECK_OUT"
    exit 0
elif [[ "$EXIT_CODE" -eq 11 ]]; then
    echo "$CHECK_OUT"
    exit 1
fi

BEST_GPU="$CHECK_OUT"
echo "✅ Job $JOB_ID bắt đầu trên GPU: $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job$JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job$JOB_ID"

rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

export CUDA_VISIBLE_DEVICES="$BEST_GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=warning
export HF_LOCAL_ONLY="${LOCAL_FILES_ONLY:-0}"
export PHOBERT_OUTPUT_DIR="$OUTPUT_DIR"

echo "[RUNTIME] date=$(date) host=$(hostname) job_id=$JOB_ID"
echo "[RUNTIME] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[RUNTIME] REQUIRED_VRAM=$REQUIRED_VRAM"
echo "[RUNTIME] CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "[RUNTIME] CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"
echo "[PATH] PROJECT_ROOT=$PROJECT_ROOT"
echo "[PATH] VENV_PATH=$VENV_PATH"
echo "[PATH] SCRIPT_PATH=$SCRIPT_PATH"
echo "[PATH] OUTPUT_DIR=$OUTPUT_DIR"
echo "[PATH] TRAIN_DIR=$TRAIN_DIR"
echo "[PATH] DEV_DIR=$DEV_DIR"
echo "[PATH] TEST_DIR=$TEST_DIR"
echo "[CONFIG] HF_LOCAL_ONLY=$HF_LOCAL_ONLY"

python - <<'PYCHECK'
import importlib.util
import sys

print(f"[PYTHON] executable={sys.executable}", flush=True)

for pkg in ["torch", "transformers", "pandas", "sklearn", "numpy", "underthesea"]:
    if importlib.util.find_spec(pkg) is None:
        raise SystemExit(f"[ERROR] {pkg} is not importable")

import torch
print(f"[TORCH] version={torch.__version__} cuda_build={torch.version.cuda}", flush=True)
print(f"[CUDA] available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}", flush=True)
PYCHECK

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true

python -u "$SCRIPT_PATH"
