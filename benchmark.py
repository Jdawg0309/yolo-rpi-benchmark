#!/usr/bin/env python3
"""
YOLOv11n Academic Benchmark
CPU (Ultralytics) and NPU (Hailo HEF)

Hardware:
    CPU: Raspberry Pi 5 (ARM64)
    NPU: Hailo-8 / Hailo-8L

Dataset:
    COCO val2017 or subsets thereof

Notes:
- Uses Ultralytics `model.val()` for CPU benchmarking
- Dataset YAML MUST explicitly point to val images/labels
- Parameters are computed dynamically from PyTorch tensors
- GFLOPs are computed dynamically by tracing the model graph (thop)
- CPU thermals and utilization are logged passively
"""

import argparse
import time
import platform
import statistics
import threading
from pathlib import Path

import torch
import psutil
import pandas as pd
import yaml
from ultralytics import YOLO
from thop import profile

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

OUTPUT_DIR = Path("outputs")
CSV_DIR = OUTPUT_DIR / "csv"
LOG_DIR = OUTPUT_DIR / "log"

for d in (OUTPUT_DIR, CSV_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser()

parser.add_argument("--cpu", action="store_true")
parser.add_argument("--npu", action="store_true")
parser.add_argument("--hef", type=str)

parser.add_argument("--data", type=str, required=True)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--runs", type=int, default=1)

# 🔹 NEW: run metadata
parser.add_argument(
    "--run-type",
    type=str,
    choices=["canonical", "experiment", "smoke"],
    default="experiment",
    help="Classification of this run for analysis discipline",
)

parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="Optional experiment label (e.g. batch4, nms_off, baseline)",
)

parser.add_argument(
    "--code-version",
    type=str,
    default="v1.0",
    help="Manual code version or git hash",
)

args = parser.parse_args()

if args.cpu == args.npu:
    raise ValueError("Choose exactly one backend: --cpu or --npu")

if args.npu and not args.hef:
    raise ValueError("--npu requires --hef")

DATA_YAML = Path(args.data).expanduser().resolve()
if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_NAME = "yolo11n.pt"
DEVICE = "cpu"
CONF_THRESHOLD = 0.001
IOU_THRESHOLD = 0.6

CSV_PATH = CSV_DIR / "yolo_benchmarks.csv"
TXT_PATH = LOG_DIR / "yolo11n_academic_results.txt"
THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")

# =============================================================================
# UTILITIES
# =============================================================================

def read_cpu_temp_c():
    if not THERMAL_PATH.exists():
        return None
    try:
        return int(THERMAL_PATH.read_text().strip()) / 1000.0
    except Exception:
        return None

def count_parameters(torch_model):
    return sum(p.numel() for p in torch_model.parameters())

def compute_gflops(torch_model, imgsz):
    torch_model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    flops, _ = profile(torch_model, inputs=(dummy,), verbose=False)
    return flops / 1e9

def count_dataset_images(yaml_path: Path) -> int:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    root = yaml_path.parent
    base = Path(cfg.get("path", root))

    val = cfg.get("val")
    if val is None:
        return 0

    val_path = (base / val).resolve()

    if val_path.is_dir():
        return len(list(val_path.rglob("*.jpg")))
    if val_path.is_file():
        return sum(1 for _ in open(val_path))

    return 0

# =============================================================================
# CPU UTILIZATION SAMPLER
# =============================================================================

def sample_cpu_util(stop_event, samples, interval=0.2):
    while not stop_event.is_set():
        samples.append(psutil.cpu_percent(interval=None))
        time.sleep(interval)

# =============================================================================
# CPU BENCHMARK
# =============================================================================

def run_cpu():
    model = YOLO(MODEL_NAME)
    torch_model = model.model

    params_m = count_parameters(torch_model) / 1e6
    gflops = compute_gflops(torch_model, args.imgsz)

    latencies = []
    cpu_samples = []

    stop_event = threading.Event()
    sampler = threading.Thread(
        target=sample_cpu_util,
        args=(stop_event, cpu_samples),
        daemon=True,
    )

    temp_start = read_cpu_temp_c()
    temp_max = temp_start or 0.0

    sampler.start()

    for run in range(args.runs):
        print(f"[INFO] Evaluating {DATA_YAML.name} (run {run+1}/{args.runs})")

        t0 = time.time()
        metrics = model.val(
            data=str(DATA_YAML),
            imgsz=args.imgsz,
            batch=args.batch,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
            plots=False,
        )
        elapsed_min = (time.time() - t0) / 60.0

        speed = metrics.speed
        total_ms = speed["preprocess"] + speed["inference"] + speed["postprocess"]
        latencies.append(total_ms)

        temp_now = read_cpu_temp_c()
        if temp_now:
            temp_max = max(temp_max, temp_now)

        if run == args.runs - 1:
            final_metrics = metrics
            final_speed = speed
            final_total_ms = total_ms
            final_elapsed_min = elapsed_min

    stop_event.set()
    sampler.join()

    temp_end = read_cpu_temp_c()

    mean_ms = statistics.mean(latencies) if args.runs > 1 else None
    std_ms = statistics.stdev(latencies) if args.runs > 1 else None

    num_images = count_dataset_images(DATA_YAML)

    result = {
        # Run metadata
        "run_type": args.run_type,
        "experiment_tag": args.tag,
        "code_version": args.code_version,

        # System configuration
        "backend": "cpu",
        "device": "rpi5_cpu",
        "architecture": platform.machine(),
        "model": "yolo11n",
        "imgsz": args.imgsz,
        "batch": args.batch,

        # Model characteristics
        "params_M": round(params_m, 2),
        "gflops": round(gflops, 2),

        # Accuracy
        "map50": round(final_metrics.box.map50, 4),
        "map50_95": round(final_metrics.box.map, 4),

        # Timing
        "preprocess_ms": round(final_speed["preprocess"], 2),
        "inference_ms": round(final_speed["inference"], 2),
        "postprocess_ms": round(final_speed["postprocess"], 2),
        "total_latency_ms": round(final_total_ms, 2),
        "total_latency_ms_mean": round(mean_ms, 2) if mean_ms else None,
        "total_latency_ms_std": round(std_ms, 2) if std_ms else None,
        "fps_estimated": round(1000.0 / final_total_ms, 2),

        # System behavior
        "cpu_util_mean_percent": round(statistics.mean(cpu_samples), 1),
        "cpu_util_peak_percent": round(max(cpu_samples), 1),
        "cpu_cores": psutil.cpu_count(logical=True),
        "temp_start_c": round(temp_start, 1) if temp_start else None,
        "temp_max_c": round(temp_max, 1),
        "temp_end_c": round(temp_end, 1) if temp_end else None,

        # Dataset + bookkeeping
        "dataset": DATA_YAML.name,
        "num_images": num_images,
        "eval_minutes": round(final_elapsed_min, 2),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),

        "note": (
            "Accuracy on subsets is indicative only. "
            "FPS derived from per-image latency. "
            "CPU timing includes preprocessing, inference, and NMS."
        ),
    }

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Run type: {args.run_type.upper()} ({args.tag})")
    print(f"Device: Raspberry Pi 5 (CPU)")
    print(f"Dataset: {DATA_YAML.name} ({num_images} images)")
    print(f"Batch size: {args.batch}")
    print(f"Latency: {final_total_ms:.1f} ms/image")
    print(f"Throughput: ~{result['fps_estimated']} FPS")
    print(f"CPU usage: {result['cpu_util_mean_percent']}% avg")
    print(f"Temperature: {temp_start} → {temp_max} °C")
    print("========================\n")

    return result

# =============================================================================
# RUN + SAVE
# =============================================================================

result = run_cpu()
df_new = pd.DataFrame([result])

if CSV_PATH.exists():
    df = pd.concat([pd.read_csv(CSV_PATH), df_new], ignore_index=True)
else:
    df = df_new

df.to_csv(CSV_PATH, index=False)

with open(TXT_PATH, "a") as f:
    f.write("=" * 80 + "\n")
    for k, v in result.items():
        f.write(f"{k}: {v}\n")

print("✓ Benchmark complete")
print(f"✓ CSV → {CSV_PATH}")
print(f"✓ TXT → {TXT_PATH}")
print(df_new)
