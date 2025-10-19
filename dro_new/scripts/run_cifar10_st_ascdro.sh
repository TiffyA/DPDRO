#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts}"
DATA_ROOT_DEFAULT="$(dirname "${PROJECT_ROOT}")/imagenet"

EPSILON="${1:-1.0}"
shift || true
DELTA="${DELTA:-0.1}"
ETA="${ETA:-0.05}"
BETA="${BETA:-0.5}"
RHO="${RHO:-0.5}"
LAMBDA0="${LAMBDA0:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-192}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-2048}"
MIN_STEPS="${MIN_STEPS:-512}"
C_CONST="${C_CONST:-1.0}"
LOSS_SCALE="${LOSS_SCALE:-${LAMBDA0}}"
MODEL_DEPTH="${MODEL_DEPTH:-50}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EXP_CLIP="${EXP_CLIP:-8.0}"
CE_WARMUP_STEPS="${CE_WARMUP_STEPS:-1024}"
CE_WARMUP_LR="${CE_WARMUP_LR:-}" 
LOG_FILE="${LOG_FILE:-}" 

DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_DEFAULT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/imagenet_ascdro_eps_${EPSILON}}"

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=(
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --epsilon "${EPSILON}"
  --delta "${DELTA}"
  --eta "${ETA}"
  --beta "${BETA}"
  --rho "${RHO}"
  --lambda0 "${LAMBDA0}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --max-steps "${MAX_STEPS}"
  --min-steps "${MIN_STEPS}"
  --loss-scale "${LOSS_SCALE}"
  --model-depth "${MODEL_DEPTH}"
  --image-size "${IMAGE_SIZE}"
  --c "${C_CONST}"
  --exp-clip "${EXP_CLIP}"
  --grad-clip "${GRAD_CLIP}"
  --ce-warmup-steps "${CE_WARMUP_STEPS}"
)

if [[ -n "${CE_WARMUP_LR}" ]]; then
  EXTRA_ARGS+=("--ce-warmup-lr" "${CE_WARMUP_LR}")
fi

if [[ -n "${LOG_FILE}" ]]; then
  EXTRA_ARGS+=("--log-file" "${LOG_FILE}")
fi

python "${PROJECT_ROOT}/training/train_cifar10_st_ascdro.py" \
  "${EXTRA_ARGS[@]}" \
  "$@"
