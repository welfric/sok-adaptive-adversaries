#!/usr/bin/env bash
# Python environment for launch/ssh/privacy/memguard/run.py (LiRA vs MemGuard).
#
# Versions follow the HAMP repository's README (Python 3.8, tensorflow 2.12.0, scikit-learn 1.0.2,
# scipy 1.7, numpy 1.22.4, pandas 1.5.0). MemGuard's optimisation is TensorFlow/Keras code
# and runs on the CPU here (a small MLP, one sample at a time), so the CPU build of TensorFlow is
# installed. PyTorch is not needed: MemGuard is applied to the saved logits of the HAMP LiRA run.
# On Apple silicon (local smoke tests only) tensorflow-macos 2.12.0, Python 3.10 and
# numpy 1.23.5 / scipy 1.9.3 are used instead.
#
# Usage, from the repository root:
#   bash launch/ssh/privacy/memguard/setup.sh            # creates .venv-memguard/
#   SOK_VENV=/path/to/venv bash launch/ssh/privacy/memguard/setup.sh
set -euo pipefail

VENV="${SOK_VENV:-$(pwd)/.venv-memguard}"
PYVER=3.8
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  PYVER=3.10  # local smoke tests only: Python 3.8 dependencies do not build on recent macOS
fi

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYVER" "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python")
elif command -v python3.8 >/dev/null 2>&1; then
  python3.8 -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  PIP=("$VENV/bin/python" -m pip install)
else
  echo "Needs uv (https://docs.astral.sh/uv/) or python3.8 on PATH." >&2
  exit 1
fi

if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  TF=(tensorflow-macos==2.12.0)
  NUMPY=numpy==1.23.5  # tensorflow-macos 2.12 fails to import with numpy 1.22.4
  SCIPY=scipy==1.9.3   # scipy 1.7.3 requires numpy < 1.23
else
  TF=(tensorflow-cpu==2.12.0)
  NUMPY=numpy==1.22.4
  SCIPY=scipy==1.7.3
fi

"${PIP[@]}" "${TF[@]}" \
  "$NUMPY" \
  "$SCIPY" \
  scikit-learn==1.0.2 \
  pandas==1.5.0 \
  matplotlib==3.7.5 \
  h5py==3.8.0 \
  gdown==5.2.0

"$VENV/bin/python" - <<'EOF'
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy, scipy, sklearn, tensorflow as tf, keras
print("tensorflow", tf.__version__, "keras", keras.__version__, "numpy", numpy.__version__,
      "scipy", scipy.__version__, "scikit-learn", sklearn.__version__)
print("CPU cores:", os.cpu_count())
EOF
echo "Environment ready: $VENV"
