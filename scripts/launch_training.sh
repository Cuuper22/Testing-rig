#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${CONFIG_PATH:-configs/distributed.yaml}
TRAINING_SCRIPT=${TRAINING_SCRIPT:-src/training/train.py}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

print_env() {
  cat <<EOM
Launching distributed training with the following parameters:
  CONFIG_PATH=${CONFIG_PATH}
  TRAINING_SCRIPT=${TRAINING_SCRIPT}
  MASTER_ADDR=${MASTER_ADDR}
  MASTER_PORT=${MASTER_PORT}
  NODE_RANK=${NODE_RANK}
  WORLD_SIZE=${WORLD_SIZE}
  NPROC_PER_NODE=${NPROC_PER_NODE}
EOM
}

print_env

if [[ ${WORLD_SIZE} -gt 1 || ${NPROC_PER_NODE} -gt 1 ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    torchrun \
      --nnodes="${WORLD_SIZE}" \
      --node_rank="${NODE_RANK}" \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      "${TRAINING_SCRIPT}" --config "${CONFIG_PATH}" \
      --node-rank "${NODE_RANK}" --world-size "${WORLD_SIZE}" \
      --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT}"
  else
    echo "[WARN] torchrun not found; falling back to single-process execution." >&2
    python "${TRAINING_SCRIPT}" --config "${CONFIG_PATH}" \
      --node-rank "${NODE_RANK}" --world-size "${WORLD_SIZE}" \
      --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT}"
  fi
else
  python "${TRAINING_SCRIPT}" --config "${CONFIG_PATH}" \
    --node-rank "${NODE_RANK}" --world-size "${WORLD_SIZE}" \
    --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT}"
fi
