#!/usr/bin/env bash
# 项目 B - Leap Hand 掌内重定向 baseline
# 用法: nohup bash /root/autodl-tmp/b1_train_baseline.sh > /root/autodl-tmp/mjx_train_baseline.log 2>&1 &
set -euo pipefail
source /root/autodl-tmp/mjx_activate.sh

LOG=/root/autodl-tmp/mjx_train_baseline.log
exec > >(tee -a "$LOG") 2>&1

echo "=== B1 baseline start $(date) ==="

# 训练前检查：不抢占他人 GPU 任务
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [[ "${USED}" -gt 1000 ]]; then
  echo "WARN: GPU memory already ${USED} MiB in use — abort to protect other jobs."
  echo "Re-run when GPU is free, or lower NUM_ENVS."
  exit 2
fi

ENV_NAME="${ENV_NAME:-LeapCubeReorient}"
NUM_ENVS="${NUM_ENVS:-4096}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-200000000}"
SEED="${SEED:-42}"
DR="${DR:-true}"
LOGDIR="${LOGDIR:-/root/autodl-tmp/logs}"

echo "env=$ENV_NAME num_envs=$NUM_ENVS num_timesteps=$NUM_TIMESTEPS seed=$SEED DR=$DR"
nvidia-smi | head -15

DR_FLAG=""
if [[ "$DR" == "true" ]]; then
  DR_FLAG="--domain_randomization"
fi

train-jax-ppo \
  --env_name "$ENV_NAME" \
  --num_envs "$NUM_ENVS" \
  --num_timesteps "$NUM_TIMESTEPS" \
  --seed "$SEED" \
  --impl jax \
  --logdir "$LOGDIR" \
  --num_videos 3 \
  $DR_FLAG \
  2>&1

echo "=== B1 baseline done $(date) ==="
