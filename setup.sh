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

# ---- Python packages ----
pip install --upgrade pip
pip install -r requirements.txt

# ---- Download YOLOv11 model ----
yolo download model=yolo11n.pt

echo ""
echo "✅ Setup complete"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python benchmark.py --cpu --data <dataset.yaml>"
