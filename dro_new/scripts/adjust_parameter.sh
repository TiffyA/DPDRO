#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts}"
DATA_ROOT_DEFAULT="$(dirname "${PROJECT_ROOT}")/imagenet"
DATA_ROOT="${DATA_ROOT:-${DATA_ROOT_DEFAULT}}"
OUT_BASE="${PROJECT_ROOT}/runs/parameter_sweep"
RESULTS_FILE="${OUT_BASE}/results_resnet20.csv"
mkdir -p "${OUT_BASE}"

ACC_LOG="${OUT_BASE}/para_acc.csv"

HEADER="eta,rho,depth,image_size,batch_size,loss_scale,warmup_steps,step,val_acc,train_acc,val_loss,train_loss,output_dir,log_file"
ROW_FORMAT="%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"

if [[ ! -s "${RESULTS_FILE}" ]]; then
  printf "%s\n" "$HEADER" > "${RESULTS_FILE}"
fi

if [[ ! -s "${ACC_LOG}" ]]; then
  printf "%s\n" "$HEADER" > "${ACC_LOG}"
fi

format_token() {
  echo "$1" | tr '.' 'p'
}

declare -a ETA_LIST=(0.02 0.05 0.1 0.2 0.4)
declare -a RHO_LIST=(0.05 0.1 0.5 1.0)
declare -a DEPTH_LIST=(50)
declare -a IMAGE_SIZE_LIST=(224)
declare -a BATCH_LIST=(192 224 256)
declare -a LOSS_LIST=(0.00025 0.0005 0.001)
declare -a WARMUP_LIST=(1024 1536 2048)

desired_acc=${DESIRED_ACC:-60}
best_acc=-1
best_cfg=""

GPU_IDS_OVERRIDE=${GPU_IDS_OVERRIDE:-"0 1"}
read -r -a GPU_IDS <<< "${GPU_IDS_OVERRIDE}"
GPU_COUNT=${#GPU_IDS[@]}
PER_GPU_JOBS=${PER_GPU_JOBS:-1}

# Track running jobs per GPU
declare -A GPU_OCCUPIED
for gpu in "${GPU_IDS[@]}"; do
  GPU_OCCUPIED["${gpu}"]=0
done

# Map pid to metadata
declare -A PID_INFO

check_completed_jobs() {
  local finished=()
  for pid in "${!PID_INFO[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    wait "$pid" || true
    finished+=("$pid")
  done
  for pid in "${finished[@]}"; do
    handle_completion "$pid"
  done
}

wait_for_slot() {
  while true; do
    check_completed_jobs
    for gpu in "${GPU_IDS[@]}"; do
      if (( GPU_OCCUPIED[$gpu] < PER_GPU_JOBS )); then
        SELECTED_GPU=$gpu
        return
      fi
    done
    sleep 5
  done
}

handle_completion() {
  local pid=$1
  local meta=${PID_INFO[$pid]-}
  if [[ -z "${meta}" ]]; then
    return
  fi
  unset PID_INFO[$pid]
  local gpu run_id eta rho depth image_size batch loss_scale warmup output_dir log_file metrics_path
  IFS=';' read -r gpu run_id eta rho depth image_size batch loss_scale warmup output_dir log_file metrics_path <<< "$meta"
  GPU_OCCUPIED[$gpu]=$((GPU_OCCUPIED[$gpu]-1))

  if [[ ! -f "$metrics_path" ]]; then
    echo "[WARN] metrics missing for $run_id"
    return
  fi

  read step val_acc train_acc val_loss train_loss <<<"$(python - <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
history = data.get("history") or []
if history:
    final = history[-1]
    step = final.get("step", float('nan'))
    val_acc = final.get("val_acc", float('nan'))
    train_acc = final.get("train_acc", float('nan'))
    val_loss = final.get("val_loss", float('nan'))
    train_loss = final.get("train_loss", float('nan'))
else:
    step = val_acc = train_acc = val_loss = train_loss = float('nan')
print(step, val_acc, train_acc, val_loss, train_loss)
PY
"$metrics_path" 2>/dev/null)"

  local row
  row=$(printf "$ROW_FORMAT" \
    "$eta" "$rho" "$depth" "$image_size" "$batch" "$loss_scale" "$warmup" "$step" \
    "$val_acc" "$train_acc" "$val_loss" "$train_loss" "$output_dir" "$log_file")

  printf "%s\n" "$row" >> "$RESULTS_FILE"
  printf "%s\n" "$row" >> "$ACC_LOG"

  finite=$(python - "$val_acc" <<'PY'
import sys, math
try:
    val = float(sys.argv[1])
except ValueError:
    print('no')
else:
    print('yes' if math.isfinite(val) else 'no')
PY
)

  if [[ "$finite" == "yes" ]]; then
    better=$(python - "$val_acc" "$best_acc" <<'PY'
import sys
val = float(sys.argv[1])
best = float(sys.argv[2])
print('yes' if val > best else 'no')
PY
)
    if [[ "$better" == "yes" ]]; then
      best_acc="$val_acc"
      best_cfg="eta=$eta rho=$rho depth=$depth image_size=$image_size batch=$batch loss_scale=$loss_scale warmup=$warmup output=$output_dir"
    fi

    reached=$(python - "$val_acc" "$desired_acc" <<'PY'
import sys
val = float(sys.argv[1])
threshold = float(sys.argv[2])
print('yes' if val >= threshold else 'no')
PY
)
    if [[ "$reached" == "yes" ]]; then
      echo "[SUCCESS] Achieved target accuracy $val_acc% with configuration: $best_cfg"
      echo "Best configuration: $best_cfg" > "$OUT_BASE/best_config.txt"
      exit 0
    fi
  fi
}

launch_training() {
  local eta=$1 rho=$2 depth=$3 image_size=$4 batch=$5 loss_scale=$6 warmup=$7
  wait_for_slot
  local gpu=$SELECTED_GPU
  local eta_token=$(format_token "$eta")
  local rho_token=$(format_token "$rho")
  local loss_token=$(format_token "$loss_scale")
  local run_id="eta_${eta_token}_rho_${rho_token}_depth_${depth}_ims_${image_size}_bs_${batch}_ls_${loss_token}_warm_${warmup}"
  local output_dir="${OUT_BASE}/${run_id}"
  local log_file="${output_dir}/training.log"
  local metrics_path="${output_dir}/metrics_ascdro.json"

  if [[ -f "$metrics_path" ]]; then
    echo "[INFO] $run_id already exists, skipping"
    return
  fi

  echo "[LAUNCH] eta=$eta rho=$rho depth=$depth image_size=$image_size batch=$batch loss_scale=$loss_scale warmup=$warmup gpu=$gpu"

  CUDA_VISIBLE_DEVICES="$gpu" \
  RHO="$rho" \
  BATCH_SIZE="$batch" \
  MODEL_DEPTH="$depth" \
  IMAGE_SIZE="$image_size" \
  ETA="$eta" \
  LOSS_SCALE="$loss_scale" \
  CE_WARMUP_STEPS="$warmup" \
  CE_WARMUP_LR="$eta" \
  MIN_STEPS=1024 \
  MAX_STEPS=4096 \
  GRAD_CLIP=1.0 \
  EXP_CLIP=8.0 \
  EPSILON=8 \
  DATA_ROOT="$DATA_ROOT" \
  OUTPUT_DIR="$output_dir" \
  LOG_FILE="$log_file" \
    "$PROJECT_ROOT/scripts/run_cifar10_st_ascdro.sh" 8 --verbose &
  local pid=$!
  GPU_OCCUPIED[$gpu]=$((GPU_OCCUPIED[$gpu]+1))
  PID_INFO[$pid]="$gpu;${run_id};${eta};${rho};${depth};${image_size};${batch};${loss_scale};${warmup};${output_dir};${log_file};${metrics_path}"
}

for eta in "${ETA_LIST[@]}"; do
  for rho in "${RHO_LIST[@]}"; do
    for depth in "${DEPTH_LIST[@]}"; do
      for image_size in "${IMAGE_SIZE_LIST[@]}"; do
        for batch in "${BATCH_LIST[@]}"; do
          for loss_scale in "${LOSS_LIST[@]}"; do
            for warmup in "${WARMUP_LIST[@]}"; do
              launch_training "$eta" "$rho" "$depth" "$image_size" "$batch" "$loss_scale" "$warmup"
            done
          done
        done
      done
    done
  done
done

while [[ ${#PID_INFO[@]} -gt 0 ]]; do
  check_completed_jobs
  sleep 5
done

if [[ -n "$best_cfg" ]]; then
  echo "Best configuration: $best_cfg (val_acc=${best_acc}%)" > "$OUT_BASE/best_config.txt"
  echo "[INFO] Best configuration so far: $best_cfg (val_acc=${best_acc}%)"
else
  echo "[INFO] No successful runs recorded" > "$OUT_BASE/best_config.txt"
fi
