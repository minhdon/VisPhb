#!/usr/bin/env bash
#SBATCH --job-name=T2_2S
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

REQUIRED_VRAM="${REQUIRED_VRAM:-28000}"

PROJECT_ROOT="/datastore/cndt_phungdtm/KLTN_GRIT/Nhan/uit"
DATASTORE_ROOT="/datastore/cndt_phungdtm/KLTN_GRIT/Nhan"
VENV_PATH="$DATASTORE_ROOT/venv_kltn/bin/activate"

SEED="${SEED:-42}"
RUN_NAME="tier2_implicit_2stage"
ENTRYPOINT="tier2_implicit_2stage.py"

SCRIPT_PATH="$PROJECT_ROOT/uit-train/entrypoints/$ENTRYPOINT"

SPIKING_LAYERS="${SPIKING_LAYERS:-4}"

STAGE1_CKPT="${STAGE1_CKPT:-$PROJECT_ROOT/uit-models/stage1_wiki_implicit/spk${SPIKING_LAYERS}/seed_${SEED}/best_stage1_wiki_implicit_seed${SEED}.pt}"
TEACHER_CKPT="${TEACHER_CKPT:-$PROJECT_ROOT/uit-models/phobert_vsfc/best_model.pth}"

if [[ ! -f "$TEACHER_CKPT" && -f "$PROJECT_ROOT/uit-models/models/phobert_vsfc/best_model.pth" ]]; then
    TEACHER_CKPT="$PROJECT_ROOT/uit-models/models/phobert_vsfc/best_model.pth"
fi

TRAIN_PATH="${TRAIN_PATH:-$PROJECT_ROOT/cache/train_segmented.parquet}"
DEV_PATH="${DEV_PATH:-$PROJECT_ROOT/cache/dev_segmented.parquet}"
TEST_PATH="${TEST_PATH:-$PROJECT_ROOT/cache/test_segmented.parquet}"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/uit-models/$RUN_NAME/spk${SPIKING_LAYERS}/seed_${SEED}}"
LOG_DIR="$PROJECT_ROOT/logs"
CACHE_ROOT="$DATASTORE_ROOT/hf_cache/$RUN_NAME/spk${SPIKING_LAYERS}/seed_${SEED}/${SLURM_JOB_ID:-manual}"

TASK_KD_EPOCHS="${TASK_KD_EPOCHS:-15}"
PREDICTION_KD_EPOCHS="${PREDICTION_KD_EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-0}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

TEMPERATURE_KD="${TEMPERATURE_KD:-4.0}"
ALPHA_CE="${ALPHA_CE:-0.5}"
LAMBDA_F="${LAMBDA_F:-0.1}"

ALPHA_CE_FINAL="${ALPHA_CE_FINAL:-0.5}"
LAMBDA_F_FINAL="${LAMBDA_F_FINAL:-0.0}"

ANDERSON_MAX_ITER="${ANDERSON_MAX_ITER:-30}"
ANDERSON_TOL="${ANDERSON_TOL:-1e-3}"
ANDERSON_M="${ANDERSON_M:-5}"
NEUMANN_K="${NEUMANN_K:-5}"
THRESHOLD="${THRESHOLD:-0.5}"
SURROGATE_ALPHA="${SURROGATE_ALPHA:-2.0}"
EQUILIBRIUM_METRIC="${EQUILIBRIUM_METRIC:-both}"
FIRING_RATE_TOL="${FIRING_RATE_TOL:-1e-4}"
MIN_EQUILIBRIUM_ITER="${MIN_EQUILIBRIUM_ITER:-3}"

SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-0}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-1000}"

LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
NO_AMP="${NO_AMP:-0}"
RESUME="${RESUME:-}"
LOAD_STAGE1_STRICT="${LOAD_STAGE1_STRICT:-1}"

MEASURE_GPU_ENERGY="${MEASURE_GPU_ENERGY:-0}"
MAX_ENERGY_BATCHES="${MAX_ENERGY_BATCHES:-0}"

module clear -f || true
module load shared python312 || true

if [[ ! -f "$VENV_PATH" ]]; then
    echo "[ERROR] VENV_PATH not found: $VENV_PATH" >&2
    exit 1
fi

source "$VENV_PATH"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CACHE_ROOT"

for path in "$SCRIPT_PATH" "$TRAIN_PATH" "$DEV_PATH" "$TEST_PATH" "$TEACHER_CKPT" "$STAGE1_CKPT"; do
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

STRICT_FLAG=()
if [[ "$LOAD_STAGE1_STRICT" == "1" ]]; then
    STRICT_FLAG+=(--load_stage1_strict)
fi

ENERGY_FLAG=()
if [[ "$MEASURE_GPU_ENERGY" == "1" ]]; then
    ENERGY_FLAG+=(--measure_gpu_energy --max_energy_batches "$MAX_ENERGY_BATCHES")
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
echo "[PATH] STAGE1_CKPT=$STAGE1_CKPT"
echo "[PATH] TEACHER_CKPT=$TEACHER_CKPT"
echo "[CONFIG] SEED=$SEED SPIKING_LAYERS=$SPIKING_LAYERS TASK_KD_EPOCHS=$TASK_KD_EPOCHS PREDICTION_KD_EPOCHS=$PREDICTION_KD_EPOCHS"

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
    --model_name "vinai/phobert-base-v2" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --stage1_ckpt "$STAGE1_CKPT" \
    --train_path "$TRAIN_PATH" \
    --dev_path "$DEV_PATH" \
    --test_path "$TEST_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --text_column text \
    --label_column label \
    --task_kd_epochs "$TASK_KD_EPOCHS" \
    --prediction_kd_epochs "$PREDICTION_KD_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --num_workers "$NUM_WORKERS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --temperature_kd "$TEMPERATURE_KD" \
    --alpha_ce "$ALPHA_CE" \
    --lambda_f "$LAMBDA_F" \
    --alpha_ce_final "$ALPHA_CE_FINAL" \
    --lambda_f_final "$LAMBDA_F_FINAL" \
    --spiking_layers "$SPIKING_LAYERS" \
    --anderson_max_iter "$ANDERSON_MAX_ITER" \
    --anderson_tol "$ANDERSON_TOL" \
    --anderson_m "$ANDERSON_M" \
    --neumann_k "$NEUMANN_K" \
    --threshold "$THRESHOLD" \
    --surrogate_alpha "$SURROGATE_ALPHA" \
    --equilibrium_metric "$EQUILIBRIUM_METRIC" \
    --firing_rate_tol "$FIRING_RATE_TOL" \
    --min_equilibrium_iter "$MIN_EQUILIBRIUM_ITER" \
    --save_every_steps "$SAVE_EVERY_STEPS" \
    --log_every_steps "$LOG_EVERY_STEPS" \
    "${LOCAL_FLAG[@]}" \
    "${AMP_FLAG[@]}" \
    "${RESUME_FLAG[@]}" \
    "${STRICT_FLAG[@]}" \
    "${ENERGY_FLAG[@]}"
