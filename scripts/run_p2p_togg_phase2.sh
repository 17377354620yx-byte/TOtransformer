#!/usr/bin/env bash
set -euo pipefail

P2P_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$P2P_PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

P2P_CONDA_ENV="${P2P_CONDA_ENV:-geo_v3}"
P2P_MAX_EPOCH="${P2P_MAX_EPOCH:-150}"
P2P_LEARNING_RATE="${P2P_LEARNING_RATE:-1e-4}"
P2P_CHECKPOINT_NAME="${P2P_CHECKPOINT_NAME:-epoch-${P2P_MAX_EPOCH}.pth.tar}"
PROFILE="togg_phase2"
RUN_NAME="${P2P_RUN_NAME:-rtor_a3_${PROFILE}_seed7351}"
RUN_DIR="output/geotransformer.p2p_liver.${RUN_NAME}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_p2p_togg_phase2.sh train|test|all [TRAIN_ARGS...]

Examples:
  bash scripts/run_p2p_togg_phase2.sh all
  bash scripts/run_p2p_togg_phase2.sh train
  bash scripts/run_p2p_togg_phase2.sh train --resume
  bash scripts/run_p2p_togg_phase2.sh test

Training protocol:
  TOGGT runs start from random initialization. --warm_start and training-time
  --snapshot are rejected. --resume is allowed only for the same run.

Environment overrides:
  CUDA_VISIBLE_DEVICES=1 P2P_CONDA_ENV=geo_v3 \
    bash scripts/run_p2p_togg_phase2.sh all

  P2P_CHECKPOINT_NAME=best.pth.tar \
    bash scripts/run_p2p_togg_phase2.sh test
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

action="$1"
shift
train_args=("$@")
if [[ "$action" != "train" && "$action" != "test" && "$action" != "all" ]]; then
  usage >&2
  exit 2
fi

for argument in "${train_args[@]}"; do
  if [[ "$argument" == "--warm_start" || "$argument" == "--snapshot" || "$argument" == --snapshot=* ]]; then
    echo "TOGGT protocol requires training from scratch; do not use --warm_start or --snapshot." >&2
    exit 2
  fi
done

train_model() {
  echo "[train] profile=${PROFILE} run=${RUN_NAME} gpu=${CUDA_VISIBLE_DEVICES} env=${P2P_CONDA_ENV}"
  P2P_RUN_NAME="$RUN_NAME" conda run --no-capture-output -n "$P2P_CONDA_ENV" \
    python experiments/geotransformer.p2p_liver/trainval.py \
    --architecture rtor_a3 \
    --interaction_profile "$PROFILE" \
    --max_epoch "$P2P_MAX_EPOCH" \
    --lr "$P2P_LEARNING_RATE" \
    --log_steps 10 \
    "${train_args[@]}"
}

test_model() {
  local checkpoint="${P2P_SNAPSHOT:-${RUN_DIR}/snapshots/${P2P_CHECKPOINT_NAME}}"
  local checkpoint_tag
  checkpoint_tag="$(basename -- "$checkpoint" .pth.tar)"
  local result_dir="${RUN_DIR}/evaluation_${checkpoint_tag}"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing trained checkpoint: $checkpoint" >&2
    echo "Override it with P2P_SNAPSHOT=/path/to/checkpoint.pth.tar" >&2
    exit 1
  fi
  mkdir -p "$result_dir"
  echo "[test] profile=${PROFILE} checkpoint=${checkpoint} gpu=${CUDA_VISIBLE_DEVICES} env=${P2P_CONDA_ENV}"
  for noise in none 2 4; do
    P2P_RUN_NAME="$RUN_NAME" conda run --no-capture-output -n "$P2P_CONDA_ENV" \
      python experiments/geotransformer.p2p_liver/test.py \
      --architecture rtor_a3 \
      --interaction_profile "$PROFILE" \
      --snapshot "$checkpoint" \
      --dataset in_silico \
      --noise "$noise" \
      --output "${result_dir}/in_silico_noise_${noise}.json"
  done
  P2P_RUN_NAME="$RUN_NAME" conda run --no-capture-output -n "$P2P_CONDA_ENV" \
    python experiments/geotransformer.p2p_liver/test.py \
    --architecture rtor_a3 \
    --interaction_profile "$PROFILE" \
    --snapshot "$checkpoint" \
    --dataset in_vitro \
    --noise none \
    --output "${result_dir}/in_vitro_noise_none.json"
}

case "$action" in
  train) train_model ;;
  test) test_model ;;
  all)
    train_model
    test_model
    ;;
esac
