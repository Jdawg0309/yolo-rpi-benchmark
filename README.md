# YOLO Raspberry Pi Benchmark

Reproducible benchmarking harness for evaluating YOLOv11 inference accuracy,
latency, and system behavior on Raspberry Pi.

This repository provides a **CPU baseline** using Ultralytics validation.
Support for accelerator backends (e.g. Hailo) is planned.

---

## What this measures

For a fixed model, dataset, and configuration, the benchmark reports:

- **Accuracy**
  - mAP@0.5
  - mAP@0.5:0.95 (COCO-style)
- **Performance**
  - End-to-end latency (preprocess + inference + NMS)
  - Estimated FPS (batch = 1)
- **System behavior**
  - Mean and peak CPU utilization
  - CPU temperature (when available)

Results are written to CSV and log files for reproducibility and comparison.

---

## Requirements

- Raspberry Pi (tested on Raspberry Pi 5)
- Python 3
- Linux (Raspberry Pi OS or Ubuntu)

No prior PyTorch or YOLO installation is required.

---

## Setup (one-shot)

```bash
./setup.sh
