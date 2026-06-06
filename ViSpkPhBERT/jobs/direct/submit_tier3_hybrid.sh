#!/usr/bin/env bash
#SBATCH --job-name=T3_HD8665
#SBATCH --chdir=/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit
#SBATCH --output=/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit/logs/slurm-%x-%j.out
#SBATCH --error=/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit/logs/slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:4
#SBATCH --mem=44G
#SBATCH --time=72:00:00

set -Eeuo pipefail

REQUIRED_VRAM=40000

PROJECT_ROOT="/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit"
DATASTORE_ROOT="/datastore/cndt_phungdtm/KLTN_GRIT/Nhan"
VENV_PATH="$DATASTORE_ROOT/venv_kltn/bin/activate"

ENTRYPOINT="tier3_hybrid.py"
SCRIPT_PATH="$PROJECT_ROOT/uit-train/entrypoints/$ENTRYPOINT"
RUN_NAME="tier3_hybrid"
SEED="${SEED:-42}"

TRAIN_PATH="$PROJECT_ROOT/cache/train_segmented.parquet"
DEV_PATH="$PROJECT_ROOT/cache/dev_segmented.parquet"
TEST_PATH="$PROJECT_ROOT/cache/test_segmented.parquet"
TEACHER_CKPT="$PROJECT_ROOT/uit-models/phobert_vsfc/best_model.pth"

OUTPUT_DIR="$PROJECT_ROOT/uit-models/$RUN_NAME/seed_$SEED"
LOG_DIR="$PROJECT_ROOT/logs"
CACHE_ROOT="$DATASTORE_ROOT/hf_cache/$RUN_NAME/${SLURM_JOB_ID:-manual}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
SPIKING_LAYERS="${SPIKING_LAYERS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_LENGTH="${MAX_LENGTH:-256}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ALPHA="${ALPHA:-0.3}"
TEMPERATURE_KD="${TEMPERATURE_KD:-4.0}"
LAMBDA_F="${LAMBDA_F:-0.1}"
LAMBDA_E="${LAMBDA_E:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
WARMUP_CURRICULUM_RATIO="${WARMUP_CURRICULUM_RATIO:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
ANDERSON_MAX_ITER="${ANDERSON_MAX_ITER:-50}"
ANDERSON_TOL="${ANDERSON_TOL:-1e-4}"
ANDERSON_M="${ANDERSON_M:-5}"
NEUMANN_K="${NEUMANN_K:-10}"
THRESHOLD="${THRESHOLD:-1.0}"
SURROGATE_ALPHA="${SURROGATE_ALPHA:-2.0}"

module clear -f || true
module load shared python312 || true

if [[ ! -f "$VENV_PATH" ]]; then
    echo "[ERROR] VENV_PATH not found: $VENV_PATH" >&2
    exit 1
fi

source "$VENV_PATH"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CACHE_ROOT"

for path in "$SCRIPT_PATH" "$TRAIN_PATH" "$DEV_PATH" "$TEST_PATH" "$TEACHER_CKPT"; do
    if [[ ! -f "$path" ]]; then
        echo "[ERROR] Required file not found: $path" >&2
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

LOCAL_FLAG=()
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
    LOCAL_FLAG+=(--local_files_only)
fi

AMP_FLAG=()
if [[ "${NO_AMP:-0}" == "1" ]]; then
    AMP_FLAG+=(--no_amp)
fi

echo "[RUNTIME] date=$(date) host=$(hostname) job_id=$JOB_ID"
echo "[RUNTIME] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[RUNTIME] REQUIRED_VRAM=$REQUIRED_VRAM"
echo "[RUNTIME] CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "[RUNTIME] CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"
echo "[PATH] PROJECT_ROOT=$PROJECT_ROOT"
echo "[PATH] VENV_PATH=$VENV_PATH"
echo "[PATH] SCRIPT_PATH=$SCRIPT_PATH"
echo "[PATH] OUTPUT_DIR=$OUTPUT_DIR"
echo "[PATH] TEACHER_CKPT=$TEACHER_CKPT"

python - <<'PYCHECK'
import importlib.util
import sys

print(f"[PYTHON] executable={sys.executable}", flush=True)

for pkg in ["torch", "transformers", "pandas", "sklearn", "numpy"]:
    if importlib.util.find_spec(pkg) is None:
        raise SystemExit(f"[ERROR] {pkg} is not importable")

import torch
print(f"[TORCH] version={torch.__version__} cuda_build={torch.version.cuda}", flush=True)
print(f"[CUDA] available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}", flush=True)
PYCHECK

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true

python -u "$SCRIPT_PATH" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --spiking_layers "$SPIKING_LAYERS" \
    --output_dir "$OUTPUT_DIR" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --train_path "$TRAIN_PATH" \
    --dev_path "$DEV_PATH" \
    --test_path "$TEST_PATH" \
    --max_length "$MAX_LENGTH" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --alpha "$ALPHA" \
    --temperature_kd "$TEMPERATURE_KD" \
    --lambda_f "$LAMBDA_F" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --num_workers "$NUM_WORKERS" \
    --anderson_max_iter "$ANDERSON_MAX_ITER" \
    --anderson_tol "$ANDERSON_TOL" \
    --anderson_m "$ANDERSON_M" \
    --neumann_k "$NEUMANN_K" \
    --threshold "$THRESHOLD" \
    --surrogate_alpha "$SURROGATE_ALPHA" \
    --warmup_curriculum_ratio "$WARMUP_CURRICULUM_RATIO" \
    --lambda_e "$LAMBDA_E" \
    "${LOCAL_FLAG[@]}" \
    "${AMP_FLAG[@]}"
