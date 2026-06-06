#!/usr/bin/env bash
#SBATCH --job-name=ablate-wiki
#SBATCH --chdir=/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit
#SBATCH --output=/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit/logs/slurm-%x-%j.out
#SBATCH --error=/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit/logs/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:l40:4
#SBATCH --mem=38G
#SBATCH --time=72:00:00

set -Eeuo pipefail

# =========================================================
# SLURM / GPU
# =========================================================
# L40 usable vRAM limit is below 44GB, so keep REQUIRED_VRAM conservative.
# If you increase batch_size/max_length a lot, switch to A100:
#   #SBATCH --gres=mps:a100:4
#   REQUIRED_VRAM=48000
REQUIRED_VRAM="${REQUIRED_VRAM:-32000}"

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT="${PROJECT_ROOT:-/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit}"
DATASTORE_ROOT="${DATASTORE_ROOT:-/datastore/cndt_phungdtm/KLTN_GRIT/Nhan}"
VENV_PATH="${VENV_PATH:-$DATASTORE_ROOT/venv_kltn/bin/activate}"

ENTRYPOINT="ablation_stage1_wiki.py"

# Support both locations:
#   uit-train/ablation_stage1_wiki.py
#   uit-train/entrypoints/ablation_stage1_wiki.py
if [[ -f "$PROJECT_ROOT/uit-train/$ENTRYPOINT" ]]; then
    SCRIPT_PATH="$PROJECT_ROOT/uit-train/$ENTRYPOINT"
elif [[ -f "$PROJECT_ROOT/uit-train/entrypoints/$ENTRYPOINT" ]]; then
    SCRIPT_PATH="$PROJECT_ROOT/uit-train/entrypoints/$ENTRYPOINT"
else
    echo "[ERROR] Cannot find $ENTRYPOINT in:" >&2
    echo "        $PROJECT_ROOT/uit-train/$ENTRYPOINT" >&2
    echo "        $PROJECT_ROOT/uit-train/entrypoints/$ENTRYPOINT" >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-ablations/ablation_stage1_wiki}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/uit-models/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# Your wiki folder may be either:
#   /datastore/.../Nhan/wiki_vi_20231101
# or:
#   /datastore/.../Nhan/uit/wiki_vi_20231101
if [[ -d "${WIKI_LOCAL_DIR:-}" ]]; then
    DATA_DIR="$WIKI_LOCAL_DIR"
elif [[ -d "$DATASTORE_ROOT/wiki_vi_20231101" ]]; then
    DATA_DIR="$DATASTORE_ROOT/wiki_vi_20231101"
elif [[ -d "$PROJECT_ROOT/wiki_vi_20231101" ]]; then
    DATA_DIR="$PROJECT_ROOT/wiki_vi_20231101"
else
    echo "[ERROR] Cannot find wiki_vi_20231101 dataset folder." >&2
    echo "Checked:" >&2
    echo "  ${WIKI_LOCAL_DIR:-<WIKI_LOCAL_DIR not set>}" >&2
    echo "  $DATASTORE_ROOT/wiki_vi_20231101" >&2
    echo "  $PROJECT_ROOT/wiki_vi_20231101" >&2
    exit 1
fi

CACHE_ROOT="${CACHE_ROOT:-$DATASTORE_ROOT/hf_cache/ablation_stage1_wiki/${SLURM_JOB_ID:-manual}}"

# =========================================================
# TRAINING CONFIG
# =========================================================
SEED="${SEED:-42}"

# For Wiki stage-1, start conservative.
# Full wiki + implicit SNN can be slow.
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-3000}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

SPIKING_LAYERS="${SPIKING_LAYERS:-4}"
ANDERSON_MAX_ITER="${ANDERSON_MAX_ITER:-30}"
ANDERSON_TOL="${ANDERSON_TOL:-1e-3}"
ANDERSON_M="${ANDERSON_M:-5}"
NEUMANN_K="${NEUMANN_K:-5}"
THRESHOLD="${THRESHOLD:-0.5}"
SURROGATE_ALPHA="${SURROGATE_ALPHA:-2.0}"

LAMBDA_EMBEDDING="${LAMBDA_EMBEDDING:-0.2}"
LAMBDA_HIDDEN="${LAMBDA_HIDDEN:-1.0}"
LAMBDA_LAST="${LAMBDA_LAST:-1.0}"

SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"

RESUME="${RESUME:-}"

# Usually set LOCAL_FILES_ONLY=1 if PhoBERT is already cached.
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
NO_AMP="${NO_AMP:-0}"

# =========================================================
# MODULE / VENV
# =========================================================
module clear -f || true
module load shared python312 || true

if [[ ! -f "$VENV_PATH" ]]; then
    echo "[ERROR] VENV_PATH not found: $VENV_PATH" >&2
    exit 1
fi

source "$VENV_PATH"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CACHE_ROOT"

# =========================================================
# BASIC CHECKS
# =========================================================
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "[ERROR] SCRIPT_PATH not found: $SCRIPT_PATH" >&2
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "[ERROR] DATA_DIR not found: $DATA_DIR" >&2
    exit 1
fi

if [[ ! -f "$DATA_DIR/state.json" || ! -f "$DATA_DIR/dataset_info.json" ]]; then
    echo "[ERROR] DATA_DIR does not look like a HuggingFace save_to_disk dataset: $DATA_DIR" >&2
    echo "Expected state.json and dataset_info.json" >&2
    exit 1
fi

# =========================================================
# GPU CHECK
# =========================================================
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

# This follows the private MPS directory pattern in the server guide.
export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-job$JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log-job$JOB_ID"

cleanup_mps_dirs() {
    echo "[CLEANUP] Removing MPS directories for job ${JOB_ID}"
    rm -rf "${CUDA_MPS_PIPE_DIRECTORY:-}" "${CUDA_MPS_LOG_DIRECTORY:-}" || true
}
trap cleanup_mps_dirs EXIT

rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

export CUDA_VISIBLE_DEVICES="$BEST_GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# =========================================================
# ENV
# =========================================================
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=warning

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

LOCAL_FLAG=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    LOCAL_FLAG+=(--local_files_only)
fi

AMP_FLAG=()
if [[ "$NO_AMP" == "1" ]]; then
    AMP_FLAG+=(--no_amp)
fi

RESUME_FLAG=()
if [[ -n "$RESUME" ]]; then
    RESUME_FLAG+=(--resume "$RESUME")
fi

# =========================================================
# RUNTIME INFO
# =========================================================
echo "[RUNTIME] date=$(date)"
echo "[RUNTIME] host=$(hostname)"
echo "[RUNTIME] job_id=$JOB_ID"
echo "[RUNTIME] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[RUNTIME] REQUIRED_VRAM=$REQUIRED_VRAM"
echo "[RUNTIME] CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:-UNSET}"
echo "[RUNTIME] CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "[RUNTIME] CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"

echo "[PATH] PROJECT_ROOT=$PROJECT_ROOT"
echo "[PATH] DATASTORE_ROOT=$DATASTORE_ROOT"
echo "[PATH] VENV_PATH=$VENV_PATH"
echo "[PATH] SCRIPT_PATH=$SCRIPT_PATH"
echo "[PATH] DATA_DIR=$DATA_DIR"
echo "[PATH] OUTPUT_DIR=$OUTPUT_DIR"
echo "[PATH] CACHE_ROOT=$CACHE_ROOT"

echo "[CONFIG] SEED=$SEED"
echo "[CONFIG] EPOCHS=$EPOCHS"
echo "[CONFIG] MAX_STEPS=$MAX_STEPS"
echo "[CONFIG] MAX_SAMPLES=$MAX_SAMPLES"
echo "[CONFIG] BATCH_SIZE=$BATCH_SIZE"
echo "[CONFIG] MAX_LENGTH=$MAX_LENGTH"
echo "[CONFIG] SPIKING_LAYERS=$SPIKING_LAYERS"
echo "[CONFIG] ANDERSON_MAX_ITER=$ANDERSON_MAX_ITER"
echo "[CONFIG] ANDERSON_TOL=$ANDERSON_TOL"
echo "[CONFIG] NEUMANN_K=$NEUMANN_K"
echo "[CONFIG] THRESHOLD=$THRESHOLD"

# =========================================================
# PYTHON CHECK
# =========================================================
python - <<'PYCHECK'
import importlib.util
import sys

print(f"[PYTHON] executable={sys.executable}", flush=True)

required = ["torch", "transformers", "numpy", "datasets"]
for pkg in required:
    if importlib.util.find_spec(pkg) is None:
        raise SystemExit(f"[ERROR] {pkg} is not importable")

import torch
print(f"[TORCH] version={torch.__version__} cuda_build={torch.version.cuda}", flush=True)
print(f"[CUDA] available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    print(f"[CUDA] current_device={torch.cuda.current_device()} name={torch.cuda.get_device_name(0)}", flush=True)
PYCHECK

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true

# =========================================================
# RUN
# =========================================================
python -u "$SCRIPT_PATH" \
    --seed "$SEED" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "vinai/phobert-base-v2" \
    --epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --max_samples "$MAX_SAMPLES" \
    --max_length "$MAX_LENGTH" \
    --num_workers "$NUM_WORKERS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --spiking_layers "$SPIKING_LAYERS" \
    --anderson_max_iter "$ANDERSON_MAX_ITER" \
    --anderson_tol "$ANDERSON_TOL" \
    --anderson_m "$ANDERSON_M" \
    --neumann_k "$NEUMANN_K" \
    --threshold "$THRESHOLD" \
    --surrogate_alpha "$SURROGATE_ALPHA" \
    --lambda_embedding "$LAMBDA_EMBEDDING" \
    --lambda_hidden "$LAMBDA_HIDDEN" \
    --lambda_last "$LAMBDA_LAST" \
    --save_every_steps "$SAVE_EVERY_STEPS" \
    --log_every_steps "$LOG_EVERY_STEPS" \
    "${LOCAL_FLAG[@]}" \
    "${AMP_FLAG[@]}" \
    "${RESUME_FLAG[@]}"
