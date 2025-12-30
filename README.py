# YOLOv11n Embedded Inference Benchmark

## Overview

This repository provides an **academic benchmarking harness** for evaluating **YOLOv11n** object detection performance on embedded Linux platforms.

The benchmark is designed to produce **reproducible, terminology-precise, and system-aware measurements** for:

- **CPU-only inference** using PyTorch on ARM CPUs (software baseline)
- **NPU-accelerated inference** using the **Hailo-8 or Hailo-8L** neural processing unit

The reference platform is the **Raspberry Pi 5**, but the methodology applies to **similar Linux-based ARM systems** that expose standard userspace telemetry (e.g., CPU utilization and temperature).

This tool is intended for **controlled experimentation and comparative analysis**, not for certification benchmarking (e.g., MLPerf).

---

## Motivation

Object detection benchmarks on embedded platforms often conflate:

- algorithmic cost vs. system overhead,
- framework-level preprocessing/postprocessing vs. inference execution,
- and accelerator speedups vs. end-to-end latency.

This benchmark explicitly separates these components and enforces **clear terminology and explicit assumptions**, enabling results to be:

- compared fairly,
- reproduced by others,
- and interpreted correctly in an academic context.

---

## Benchmark Scope

### What is measured

For each run, the benchmark records:

#### Accuracy
- **mAP@0.5**
- **mAP@0.5:0.95**

#### Performance
- Preprocessing latency (ms)
- Inference latency (ms)
- **Postprocessing latency (ms, includes NMS execution)**
- Total end-to-end latency per image (ms)
- Estimated throughput (FPS, derived from per-image latency)

#### Model characteristics
- Parameter count (millions)
- GFLOPs (theoretical, computed via graph tracing)

#### System behavior (CPU-only runs)
- Mean and peak CPU utilization (%)
- CPU core count
- CPU package temperature (start / max / end)

> **Note:** GPU and PMIC thermal zones are not logged. CPU package temperature is used as a proxy for thermal throttling behavior during CPU-only inference.

#### Bookkeeping metadata
- Dataset name and image count
- Run type and experiment tag
- Code version identifier
- Timestamp

All metrics are written to **CSV** and **plain-text logs** for downstream analysis.

---

## Terminology

To avoid ambiguity, the following terms are used consistently:

- **ARM CPU-only inference**  
  PyTorch-based execution on ARM CPU cores with no hardware accelerator involved.

- **NPU inference**  
  Execution of the YOLOv11n model offloaded to a Hailo-8 or Hailo-8L device using a compiled HEF file.

- **End-to-end latency**  
  Preprocessing + inference + postprocessing (including NMS).

- **Inference-only latency**  
  Model forward-pass execution time only.

- **Baseline**  
  The ARM CPU-only inference configuration used as a reference point for comparisons.

---

## Hardware Configuration (Reference Platform)

| Component        | Specification                                  |
|------------------|-----------------------------------------------|
| Board            | Raspberry Pi 5                                |
| CPU              | ARM Cortex-A76 (4 cores, ARM64)               |
| Memory           | 8 GB LPDDR4X                                  |
| Cooling          | Active (fan-assisted)                         |
| NPU (optional)   | Hailo-8 or Hailo-8L                           |
| OS               | Linux (ARM64), kernel ≥ 6.1                   |
| Power Supply     | 5V / 5A recommended                           |

Performance and thermal behavior may vary with kernel version, cooling, and power delivery.

---

## Dataset Requirements

The benchmark expects a **Ultralytics-compatible dataset YAML** file.

### Canonical dataset
- **COCO 2017 validation set (`val2017`)**
- 5,000 validation images with ground-truth annotations

### Annotation assumption (important)

This benchmark **assumes a 1:1 correspondence between validation images and ground-truth annotations**, as required by Ultralytics evaluation.

- For datasets other than COCO `val2017`, users are responsible for ensuring annotation completeness and correctness.
- This tool **does not validate annotation integrity**.
- Reported mAP values are only meaningful when annotations are complete.

### Image counting

The benchmark reports the **number of images evaluated**, not the number of labeled instances.

---

## Installation

Installation steps are intentionally explicit to support reproducibility.

### System requirements
- Architecture: ARM64
- OS: Linux
- Python: 3.9+

### Virtual environment (recommended)

```bash
python3 -m venv yolo_bench
source yolo_bench/bin/activate
pip install --upgrade pip
```

### Python dependencies

Install pinned dependencies:

```bash
pip install -r requirements.txt
```

### PyTorch on ARM64

On Raspberry Pi and other ARM64 systems:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Model weights

Place the YOLOv11n weights in the repository root:

```text
yolo11n.pt
```

---

## Methodology Notes

- **Warm-up:** For reportable results, perform warm-up evaluations before timing measurements.
- **Runs:** Use `--runs ≥ 3` for publication-quality results.
- **Statistics:** Mean and standard deviation are reported. Confidence intervals are not computed.
- **NMS configuration:** Uses Ultralytics defaults (IoU threshold = 0.6, confidence threshold = 0.001, max detections = 300).
- **GFLOPs:** Computed via static graph tracing (THOP); reflects theoretical compute cost, not runtime performance.
- **CPU sampling:** CPU utilization and temperature are sampled asynchronously during evaluation.

---

## Usage

All behavior is controlled via an explicit CLI.

### View help
```bash
python benchmark.py --help
```

### CPU-only smoke test

```bash
python benchmark.py   --cpu   --data path/to/coco_val50.yaml   --runs 1   --run-type smoke   --tag smoke_test   --code-version v1.0
```

### CPU baseline (reportable)

```bash
python benchmark.py   --cpu   --data path/to/coco_val2017.yaml   --runs 3   --run-type canonical   --tag cpu_baseline   --code-version v1.0
```

### NPU-accelerated inference

```bash
python benchmark.py   --npu   --hef yolo11n.hef   --data path/to/coco_val2017.yaml   --runs 3   --run-type experiment   --tag hailo_npu   --code-version v1.0
```

---

## Output Artifacts

```text
outputs/
├── csv/
│   └── yolo_benchmarks.csv
└── log/
    └── yolo11n_academic_results.txt
```

### CSV schema (summary)

Each row corresponds to one benchmark run and includes:

- Run metadata (run_type, tag, code_version, timestamp)
- System configuration (backend, device, architecture)
- Model metrics (params_M, gflops)
- Accuracy metrics (map50, map50_95)
- Timing metrics (preprocess_ms, inference_ms, postprocess_ms, total_latency_ms)
- System behavior (cpu_util_mean_percent, cpu_util_peak_percent, temperatures)

The CSV format may evolve; code version identifiers should be used to track schema changes.

---

## Reproducibility Guidelines

For reproducible results:

1. Fix dataset, image size, batch size, and model weights
2. Use `--runs ≥ 3` for reported measurements
3. Allow the system to idle before benchmarking to stabilize thermals
4. Record kernel version, cooling configuration, and ambient conditions
5. Do not mix CPU-only and NPU results without explicit labeling

---

## Intended Use

This benchmark is intended for:

- academic coursework and research projects
- embedded AI performance analysis
- hardware/software co-design studies

It is **not** intended as a replacement for standardized benchmarks such as MLPerf.

---

## License

This project is provided for academic and research use.
Model weights and runtimes are subject to Ultralytics and Hailo licensing terms.
