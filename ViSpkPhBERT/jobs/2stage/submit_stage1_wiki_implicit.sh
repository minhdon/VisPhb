#!/usr/bin/env bash
#SBATCH --job-name=S1_IMP
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
RUN_NAME="stage1_wiki_implicit"
ENTRYPOINT="stage1_wiki_implicit.py"

SCRIPT_PATH="$PROJECT_ROOT/uit-train/entrypoints/$ENTRYPOINT"
DATA_DIR="${DATA_DIR:-$DATASTORE_ROOT/wiki_vi_20231101_segmented}"

SPIKING_LAYERS="${SPIKING_LAYERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/uit-models/$RUN_NAME/spk${SPIKING_LAYERS}/seed_${SEED}}"
LOG_DIR="$PROJECT_ROOT/logs"
CACHE_ROOT="$DATASTORE_ROOT/hf_cache/$RUN_NAME/spk${SPIKING_LAYERS}/seed_${SEED}/${SLURM_JOB_ID:-manual}"

EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

ANDERSON_MAX_ITER="${ANDERSON_MAX_ITER:-30}"
ANDERSON_TOL="${ANDERSON_TOL:-1e-3}"
ANDERSON_M="${ANDERSON_M:-5}"
NEUMANN_K="${NEUMANN_K:-5}"
THRESHOLD="${THRESHOLD:-0.5}"
SURROGATE_ALPHA="${SURROGATE_ALPHA:-2.0}"
EQUILIBRIUM_METRIC="${EQUILIBRIUM_METRIC:-both}"
FIRING_RATE_TOL="${FIRING_RATE_TOL:-1e-4}"
MIN_EQUILIBRIUM_ITER="${MIN_EQUILIBRIUM_ITER:-3}"

LAMBDA_EMBEDDING="${LAMBDA_EMBEDDING:-0.2}"
LAMBDA_HIDDEN="${LAMBDA_HIDDEN:-1.0}"
LAMBDA_LAST="${LAMBDA_LAST:-1.0}"

SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-1000}"

LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
NO_AMP="${NO_AMP:-0}"
RESUME="${RESUME:-}"

module clear -f || true
module load shared python312 || true

if [[ ! -f "$VENV_PATH" ]]; then
    echo "[ERROR] VENV_PATH not found: $VENV_PATH" >&2
    exit 1
fi

source "$VENV_PATH"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$CACHE_ROOT"

for path in "$SCRIPT_PATH" "$DATA_DIR/state.json" "$DATA_DIR/dataset_info.json"; do
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

echo "[RUNTIME] date=$(date) host=$(hostname) job_id=$JOB_ID"
echo "[RUNTIME] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[RUNTIME] REQUIRED_VRAM=$REQUIRED_VRAM"
echo "[RUNTIME] CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "[RUNTIME] CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"
echo "[PATH] PROJECT_ROOT=$PROJECT_ROOT"
echo "[PATH] SCRIPT_PATH=$SCRIPT_PATH"
echo "[PATH] DATA_DIR=$DATA_DIR"
echo "[PATH] OUTPUT_DIR=$OUTPUT_DIR"
echo "[CONFIG] SEED=$SEED SPIKING_LAYERS=$SPIKING_LAYERS MAX_STEPS=$MAX_STEPS"

python - <<'PYCHECK'
import importlib.util
import sys
print(f"[PYTHON] executable={sys.executable}", flush=True)
for pkg in ["torch", "transformers", "datasets", "pandas", "sklearn", "numpy"]:
    if importlib.util.find_spec(pkg) is None:
        raise SystemExit(f"[ERROR] {pkg} is not importable")
import torch
print(f"[TORCH] version={torch.__version__} cuda_build={torch.version.cuda}", flush=True)
print(f"[CUDA] available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}", flush=True)
PYCHECK

nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true

python -u "$SCRIPT_PATH" \
    --seed "$SEED" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "vinai/phobert-base-v2" \
    --epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
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
    --equilibrium_metric "$EQUILIBRIUM_METRIC" \
    --firing_rate_tol "$FIRING_RATE_TOL" \
    --min_equilibrium_iter "$MIN_EQUILIBRIUM_ITER" \
    --lambda_embedding "$LAMBDA_EMBEDDING" \
    --lambda_hidden "$LAMBDA_HIDDEN" \
    --lambda_last "$LAMBDA_LAST" \
    --save_every_steps "$SAVE_EVERY_STEPS" \
    --log_every_steps "$LOG_EVERY_STEPS" \
    "${LOCAL_FLAG[@]}" \
    "${AMP_FLAG[@]}" \
    "${RESUME_FLAG[@]}"
