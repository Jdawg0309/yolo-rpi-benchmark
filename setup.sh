#!/usr/bin/env bash
set -e

echo "🔧 Setting up YOLO Raspberry Pi Benchmark"

# ---- System dependencies ----
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# ---- Python virtual environment ----
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

# ---- Python tooling ----
pip install --upgrade pip setuptools wheel

# ---- PyTorch (ARM64, CPU-only) ----
echo "📦 Installing PyTorch (ARM64 CPU wheels)"
pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cpu

# ---- Remaining dependencies ----
pip install -r requirements.txt

# ---- Model weights ----
if [ ! -f "yolo11n.pt" ]; then
  echo "📥 Downloading YOLOv11n weights"
  python - <<EOF
from ultralytics import YOLO
YOLO("yolo11n.pt")
EOF
fi

# ---- Environment summary ----
echo ""
echo "✅ Setup complete"
echo "Python: $(python --version)"
echo "Torch: $(python - <<EOF
import torch; print(torch.__version__)
EOF
)"

echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python benchmark.py --cpu --data <dataset.yaml>"
