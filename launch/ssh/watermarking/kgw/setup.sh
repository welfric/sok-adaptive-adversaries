#!/usr/bin/env bash
# Python environment for launch/ssh/watermarking/kgw/run.py (Watermark Stealing vs KGW).
#
# Versions follow upstream env.yaml (python 3.10, pydantic 2.5.3, accelerate 0.23.0,
# datasets 2.16.1, nltk 3.8.1, openai 1.6.1, protobuf 4.24.4, sentencepiece 0.1.99,
# sacremoses 0.0.53, pyyaml 6.0.1). Where env.yaml leaves a package unpinned or the pin no
# longer installs cleanly, the choice is ours:
#   torch 2.1.2 (env.yaml: 2.1.0; 2.1.1+ is what transformers needs for SDPA attention)
#   transformers 4.38.2 (unpinned upstream; Feb 2024, covers Llama-2, Mistral, Gemma, Qwen2)
#   huggingface_hub 0.23.4, pyarrow 15.0.2, httpx 0.27.2, numpy<2 (compatibility with the above)
# FlashAttention, gradio and neptune are not installed (see DEVIATIONS in run.py).
#
# Usage, from the repository root:
#   bash launch/ssh/watermarking/kgw/setup.sh            # creates .venv-kgw/
#   SOK_VENV=/path/to/venv bash launch/ssh/watermarking/kgw/setup.sh
#   TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash ...   # NVIDIA driver older than 530
set -euo pipefail

VENV="${SOK_VENV:-$(pwd)/.venv-kgw}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.10 "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python")
elif command -v python3.10 >/dev/null 2>&1; then
  python3.10 -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
else
  echo "Needs uv (https://docs.astral.sh/uv/) or python3.10 on PATH." >&2
  exit 1
fi

"${PIP[@]}" --index-url "$TORCH_INDEX" torch==2.1.2
"${PIP[@]}" \
  "numpy<2" \
  transformers==4.38.2 \
  accelerate==0.23.0 \
  datasets==2.16.1 \
  pyarrow==15.0.2 \
  huggingface_hub==0.23.4 \
  pydantic==2.5.3 \
  pyyaml==6.0.1 \
  nltk==3.8.1 \
  openai==1.6.1 \
  httpx==0.27.2 \
  protobuf==4.24.4 \
  sentencepiece==0.1.99 \
  sacremoses==0.0.53 \
  json5 matplotlib scikit-learn scipy tqdm

"$VENV/bin/python" - <<'EOF'
import torch, transformers, datasets
print("torch", torch.__version__, "CUDA build", torch.version.cuda, "GPU visible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "datasets", datasets.__version__)
EOF
echo "Environment ready: $VENV"
