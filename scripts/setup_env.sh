#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-minigpt-crop}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "未找到 conda，请先安装 Miniconda 或 Anaconda。" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env update --name "$ENV_NAME" --file "$PROJECT_ROOT/environment.yml" --prune
else
  conda env create --name "$ENV_NAME" --file "$PROJECT_ROOT/environment.yml"
fi

conda run -n "$ENV_NAME" python -c "import torch, torchvision, transformers, peft, bitsandbytes, gradio, cv2, timm, sentencepiece; print('环境验证通过'); print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"
