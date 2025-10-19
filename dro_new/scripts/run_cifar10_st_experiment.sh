#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts}"
RUN_SCRIPT="${SCRIPT_DIR}/run_cifar10_st_ascdro.sh"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_cifar10_st_ascdro.py"

: "${DATA_ROOT:=$(dirname "${PROJECT_ROOT}")/imagenet}"
: "${EPS_LIST:=0.5 1 2 4 8}"
: "${ETA_LIST:=0.05}"
: "${DELTA:=0.1}"
: "${BATCH_SIZE:=192}"
: "${NUM_WORKERS:=4}"
: "${MAX_STEPS:=2048}"
: "${MIN_STEPS:=512}"
: "${LOSS_SCALE:=${LAMBDA0}}"
: "${C_CONST:=1.0}"
: "${RHO:=0.5}"
: "${LAMBDA0:=0.1}"
: "${BETA:=0.5}"
: "${MODEL_DEPTH:=50}"
: "${IMAGE_SIZE:=224}"
: "${GRAD_CLIP:=1.0}"
: "${EXP_CLIP:=8.0}"
: "${CE_WARMUP_STEPS:=1024}"
: "${CE_WARMUP_LR:=}"
: "${LOG_FILE:=}"
: "${BASE_OUTPUT:=${PROJECT_ROOT}/runs/imagenet_sweeps}"
: "${PLOT_OUTPUT:=${PROJECT_ROOT}/plots/imagenet_accuracy.png}"
: "${GPU_IDS:=0 1}"
: "${PARALLEL_JOBS:=}"
: "${EPS_GPU_MAP:=}"

mkdir -p "${BASE_OUTPUT}"

if [[ -n "${EPS_GPU_MAP}" ]]; then
  IFS=';' read -ra MAP_ENTRIES <<< "${EPS_GPU_MAP}"
  eta_value=${ETA_LIST%% *}
  for entry in "${MAP_ENTRIES[@]}"; do
    [[ -z "${entry}" ]] && continue
    gpu="${entry%%:*}"
    eps_chunk="${entry#*:}"
    IFS=',' read -ra EPS_ARRAY <<< "${eps_chunk}"
    for eps in "${EPS_ARRAY[@]}"; do
      RUN_DIR="${BASE_OUTPUT}/eta_${eta_value}/eps_${eps}"
      echo "[RUN] epsilon=${eps} gpu=${gpu} -> ${RUN_DIR}"
      OUTPUT_DIR="${RUN_DIR}" \
      DATA_ROOT="${DATA_ROOT}" \
      ETA="${eta_value}" \
      DELTA="${DELTA}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      NUM_WORKERS="${NUM_WORKERS}" \
      MAX_STEPS="${MAX_STEPS}" \
      MIN_STEPS="${MIN_STEPS}" \
      LOSS_SCALE="${LOSS_SCALE}" \
      C_CONST="${C_CONST}" \
      RHO="${RHO}" \
      LAMBDA0="${LAMBDA0}" \
      BETA="${BETA}" \
      GRAD_CLIP="${GRAD_CLIP}" \
      EXP_CLIP="${EXP_CLIP}" \
      CE_WARMUP_STEPS="${CE_WARMUP_STEPS}" \
      CE_WARMUP_LR="${CE_WARMUP_LR}" \
      LOG_FILE="${LOG_FILE}" \
      MODEL_DEPTH="${MODEL_DEPTH}" \
      IMAGE_SIZE="${IMAGE_SIZE}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
        "${RUN_SCRIPT}" "${eps}" --verbose &
    done
  done

  if [[ -n "$(jobs -pr)" ]]; then
    wait
  fi

  python "${PLOT_SCRIPT}" \
    --base-dir "${BASE_OUTPUT}" \
    --group-key eta \
    --output "${PLOT_OUTPUT}" \
    --title "ImageNet ASCDRO Accuracy"

  echo "Experiment complete. Plot saved to ${PLOT_OUTPUT}."
  exit 0
fi

read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS}"
if [[ ${#GPU_IDS_ARRAY[@]} -eq 0 ]]; then
  GPU_IDS_ARRAY=(0)
fi
MAX_PARALLEL=${PARALLEL_JOBS:-${#GPU_IDS_ARRAY[@]}}
if [[ ${MAX_PARALLEL} -lt 1 ]]; then
  MAX_PARALLEL=1
fi

declare -a ACTIVE_PIDS=()
JOB_INDEX=0

for eta in ${ETA_LIST}; do
  for eps in ${EPS_LIST}; do
    gpu_index=$(( JOB_INDEX % ${#GPU_IDS_ARRAY[@]} ))
    gpu_id="${GPU_IDS_ARRAY[$gpu_index]}"
    RUN_DIR="${BASE_OUTPUT}/eta_${eta}/eps_${eps}"
    echo "[RUN] epsilon=${eps}, eta=${eta}, gpu=${gpu_id} -> ${RUN_DIR}"
    OUTPUT_DIR="${RUN_DIR}" \
    DATA_ROOT="${DATA_ROOT}" \
    ETA="${eta}" \
    DELTA="${DELTA}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    MAX_STEPS="${MAX_STEPS}" \
    MIN_STEPS="${MIN_STEPS}" \
    LOSS_SCALE="${LOSS_SCALE}" \
    C_CONST="${C_CONST}" \
    RHO="${RHO}" \
    LAMBDA0="${LAMBDA0}" \
    BETA="${BETA}" \
    GRAD_CLIP="${GRAD_CLIP}" \
    EXP_CLIP="${EXP_CLIP}" \
    CE_WARMUP_STEPS="${CE_WARMUP_STEPS}" \
    CE_WARMUP_LR="${CE_WARMUP_LR}" \
    LOG_FILE="${LOG_FILE}" \
    MODEL_DEPTH="${MODEL_DEPTH}" \
    IMAGE_SIZE="${IMAGE_SIZE}" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      "${RUN_SCRIPT}" "${eps}" --verbose &

    ACTIVE_PIDS+=($!)
    ((JOB_INDEX++))

    if [[ ${#ACTIVE_PIDS[@]} -ge ${MAX_PARALLEL} ]]; then
      wait -n || true
      TMP_PIDS=()
      for pid in "${ACTIVE_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          TMP_PIDS+=("${pid}")
        fi
      done
      ACTIVE_PIDS=("${TMP_PIDS[@]}")
    fi
  done
done

if [[ ${#ACTIVE_PIDS[@]} -gt 0 ]]; then
  wait
fi

python "${PLOT_SCRIPT}" \
  --base-dir "${BASE_OUTPUT}" \
  --group-key eta \
  --output "${PLOT_OUTPUT}" \
  --title "ImageNet ASCDRO Accuracy"

echo "Experiment complete. Plot saved to ${PLOT_OUTPUT}."
