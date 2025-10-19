#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts}"
DATA_ROOT_DEFAULT="$(dirname "${PROJECT_ROOT}")/imagenet"

EPSILON="${1:-8}"
shift || true

DELTA="${DELTA:-0.1}"
LR="${LR:-0.05}"
MOMENTUM="${MOMENTUM:-0.9}"
NESTEROV="${NESTEROV:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-2048}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-128}"
LOG_INTERVAL="${LOG_INTERVAL:-64}"
MODEL_DEPTH="${MODEL_DEPTH:-50}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
SEED="${SEED:-42}"
LOG_FILE="${LOG_FILE:-}"

DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_DEFAULT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/imagenet_dpsgd_eps_${EPSILON}}"

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=(
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --epsilon "${EPSILON}"
  --delta "${DELTA}"
  --lr "${LR}"
  --momentum "${MOMENTUM}"
  --weight-decay "${WEIGHT_DECAY}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --max-steps "${MAX_STEPS}"
  --eval-interval "${EVAL_INTERVAL}"
  --log-interval "${LOG_INTERVAL}"
  --model-depth "${MODEL_DEPTH}"
  --image-size "${IMAGE_SIZE}"
  --seed "${SEED}"
)

if [[ "${NESTEROV}" == "1" ]]; then
  EXTRA_ARGS+=("--nesterov")
fi

if [[ -n "${LOG_FILE}" ]]; then
  EXTRA_ARGS+=("--log-file" "${LOG_FILE}")
fi

if [[ "${VERBOSE:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--verbose")
fi

python "${PROJECT_ROOT}/training/train_cifar10_st_dpsgd.py" "${EXTRA_ARGS[@]}" "$@"
