#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="${MINIGPT_ENV:-minigpt-crop}"
GPU_ID="${MINIGPT_GPU_ID:-0}"
PORT="${MINIGPT_PORT:-7861}"
SERVER_NAME="${MINIGPT_SERVER_NAME:-127.0.0.1}"

required=(
  "models/vicuna-7b/config.json"
  "models/checkpoint_0.pth"
  "models/eva_vit_g.pth"
  "models/opus-mt-en-zh/config.json"
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少模型文件：$path" >&2
    echo "请按照 models/README.md 放置权重后重试。" >&2
    exit 1
  fi
done

echo "正在启动智农卫士：http://${SERVER_NAME}:${PORT}/"
conda run -n "$ENV_NAME" --no-capture-output python -u demo_v2.py \
  --cfg-path eval_configs/minigptv2_checkpoint_0_eval.yaml \
  --gpu-id "$GPU_ID" \
  --server-name "$SERVER_NAME" \
  --server-port "$PORT" \
  --inbrowser
