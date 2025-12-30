#!/usr/bin/env python3
"""
YOLOv11n Academic Benchmark for Embedded Systems

Benchmarks YOLOv11n on embedded Linux platforms (primarily Raspberry Pi 5, ARM64 CPU).
Measures full evaluation pipeline performance (preprocess → inference → postprocess/NMS),
accuracy (mAP), and basic system telemetry (CPU utilization, CPU package temperature).

Key features:
- CPU-only PyTorch inference benchmarking via Ultralytics
- Warm-up runs to reduce one-time initialization effects
- Multi-run statistics (mean/std for latency)
- CSV + TXT logging for traceability

Notes / assumptions:
- Postprocess time reported by Ultralytics includes NMS.
- Validation annotation completeness is assumed (Ultralytics requirement).
- Temperature reading uses /sys/class/thermal/thermal_zone0/temp (Linux-specific).
"""

import argparse
import platform
import statistics
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import psutil
import torch
import yaml
from thop import profile
from ultralytics import YOLO, __version__ as ultralytics_version


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

parser = argparse.ArgumentParser(
    description=(
        "YOLOv11n Embedded Benchmark (CPU)\n\n"
        "Runs Ultralytics validation on the specified dataset and records:\n"
        "- Accuracy: mAP@0.5, mAP@0.5:0.95\n"
        "- Timing: preprocess/inference/postprocess/total latency\n"
        "- Throughput: FPS derived from per-image latency\n"
        "- System: CPU utilization samples + CPU package temperature (if available)\n"
    ),
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        "Examples:\n"
        "  Canonical baseline (1 warm-up, 3 measured):\n"
        "    python benchmark.py --cpu --data coco.yaml --warmup 1 --runs 3 --run-type canonical --tag cpu_baseline\n\n"
        "  Smoke test:\n"
        "    python benchmark.py --cpu --data coco_small.yaml --warmup 0 --runs 1 --run-type smoke --tag smoke\n"
    ),
)

parser.add_argument("--cpu", action="store_true", help="Run CPU-only inference (baseline).")

# Placeholder flags for future NPU mode (not implemented in this script)
parser.add_argument("--npu", action="store_true", help="[PLACEHOLDER] NPU mode (not implemented).")
parser.add_argument("--hef", type=str, help="[PLACEHOLDER] Path to HEF file (NPU mode).")

parser.add_argument("--data", type=str, required=True, help="Path to Ultralytics dataset YAML.")
parser.add_argument("--imgsz", type=int, default=640, help="Input image size (square). Default: 640")
parser.add_argument("--batch", type=int, default=1, help="Batch size. Default: 1")
parser.add_argument("--runs", type=int, default=3, help="Number of measured runs. Default: 3")
parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up runs. Default: 1")

parser.add_argument(
    "--run-type",
    type=str,
    choices=["canonical", "experiment", "smoke"],
    default="experiment",
    help="Run classification (recorded in outputs).",
)
parser.add_argument("--tag", type=str, default="", help="Optional experiment label.")
parser.add_argument("--code-version", type=str, default="v1.0", help="Code version / git hash string.")

args = parser.parse_args()

if args.cpu == args.npu:
    raise ValueError("Exactly one backend must be specified: --cpu or --npu")
if args.npu:
    raise ValueError("NPU mode is a placeholder in this script (not implemented).")

DATA_YAML = Path(args.data).expanduser().resolve()
if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")


# =============================================================================
# CONSTANTS (explicit for reproducibility)
# =============================================================================

MODEL_NAME = "yolo11n.pt"
DEVICE = "cpu"

# Detection/eval params (explicitly recorded in outputs)
CONF_THRESHOLD = 0.001
IOU_THRESHOLD = 0.6
MAX_DETECTIONS = 300

CSV_PATH = CSV_DIR / "yolo_benchmarks.csv"
TXT_PATH = LOG_DIR / "yolo11n_academic_results.txt"
THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")

SCHEMA_VERSION = "1.1"  # bumped: renamed num_images -> num_eval_images and added env fields


# =============================================================================
# UTILITIES
# =============================================================================

def read_cpu_temp_c() -> Optional[float]:
    """Read CPU package temperature in Celsius, if available."""
    if not THERMAL_PATH.exists():
        return None
    try:
        return int(THERMAL_PATH.read_text(encoding="utf-8").strip()) / 1000.0
    except Exception:
        return None


def count_parameters(torch_model: torch.nn.Module) -> int:
    """Total parameter count (includes non-trainable too; matches .parameters() traversal)."""
    return sum(p.numel() for p in torch_model.parameters())


def compute_gflops(torch_model: torch.nn.Module, imgsz: int) -> float:
    """Theoretical GFLOPs for one forward pass via THOP graph tracing."""
    torch_model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    flops, _ = profile(torch_model, inputs=(dummy,), verbose=False)
    return flops / 1e9


def count_dataset_images(yaml_path: Path) -> int:
    """
    Count validation images referenced by the dataset YAML.

    IMPORTANT:
    - This is an image-count heuristic (directory jpg count OR filelist line count).
    - It does NOT validate label/annotation completeness.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    root = yaml_path.parent
    base = Path(cfg.get("path", root))
    val = cfg.get("val")

    if not val:
        return 0

    val_path = (base / val).expanduser().resolve()

    if val_path.is_dir():
        # Common for Ultralytics datasets; COCO is often like images/val2017
        return len(list(val_path.rglob("*.jpg"))) + len(list(val_path.rglob("*.jpeg"))) + len(list(val_path.rglob("*.png")))
    if val_path.is_file():
        with open(val_path, "r", encoding="utf-8") as fp:
            return sum(1 for line in fp if line.strip())

    return 0


# =============================================================================
# CPU UTILIZATION SAMPLER
# =============================================================================

def sample_cpu_util(stop_event: threading.Event, samples: List[float], interval: float = 0.25) -> None:
    """
    Sample CPU utilization periodically.

    Uses psutil.cpu_percent(interval=interval) which blocks for `interval` seconds
    and reports utilization over that window (less 'first-sample 0%' weirdness).
    """
    # Prime once (psutil's first call can be 0.0 if interval=None)
    _ = psutil.cpu_percent(interval=interval)

    while not stop_event.is_set():
        samples.append(psutil.cpu_percent(interval=interval))


# =============================================================================
# CPU BENCHMARK
# =============================================================================

def run_cpu() -> Dict[str, Any]:
    """Run CPU-only benchmark and return a single result dict (one CSV row)."""
    # Load model
    if not Path(MODEL_NAME).exists():
        raise FileNotFoundError(f"Model weights not found in repo root: {MODEL_NAME}")

    model = YOLO(MODEL_NAME)
    torch_model = model.model
    torch_model.eval()

    # Model characteristics
    params_m = count_parameters(torch_model) / 1e6
    gflops = compute_gflops(torch_model, args.imgsz)

    # Start system sampling
    cpu_samples: List[float] = []
    stop_event = threading.Event()
    sampler = threading.Thread(target=sample_cpu_util, args=(stop_event, cpu_samples), daemon=True)
    sampler.start()

    temp_start = read_cpu_temp_c()
    temp_max = temp_start or 0.0

    # Warm-up (not measured)
    if args.warmup > 0:
        print(f"[INFO] Warm-up: {args.warmup} run(s) (not timed)")
    for _ in range(args.warmup):
        _ = model.val(
            data=str(DATA_YAML),
            imgsz=args.imgsz,
            batch=args.batch,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            device=DEVICE,
            verbose=False,
            plots=False,
        )
        t = read_cpu_temp_c()
        if t is not None:
            temp_max = max(temp_max, t)

    # Measured runs
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    latencies_ms: List[float] = []
    final_metrics = None
    final_speed = None
    final_total_ms = None
    final_elapsed_min = None

    print(f"[INFO] Measured runs: {args.runs}")
    for i in range(args.runs):
        print(f"  [INFO] Run {i+1}/{args.runs} ...")

        t0 = time.time()
        metrics = model.val(
            data=str(DATA_YAML),
            imgsz=args.imgsz,
            batch=args.batch,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DETECTIONS,
            device=DEVICE,
            verbose=False,
            plots=False,
        )
        elapsed_min = (time.time() - t0) / 60.0

        speed = metrics.speed
        total_ms = float(speed["preprocess"] + speed["inference"] + speed["postprocess"])
        latencies_ms.append(total_ms)

        t = read_cpu_temp_c()
        if t is not None:
            temp_max = max(temp_max, t)

        final_metrics = metrics
        final_speed = speed
        final_total_ms = total_ms
        final_elapsed_min = elapsed_min

    # Stop sampling
    stop_event.set()
    sampler.join(timeout=2.0)
    temp_end = read_cpu_temp_c()

    # Dataset stats (heuristic count)
    num_eval_images = count_dataset_images(DATA_YAML)

    # Latency stats (per-image end-to-end)
    mean_ms = statistics.mean(latencies_ms) if len(latencies_ms) >= 2 else None
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) >= 3 else None  # intentional: discourage tiny-n std

    # Defensive: cpu_samples could be empty if something weird happens
    cpu_mean = statistics.mean(cpu_samples) if cpu_samples else None
    cpu_peak = max(cpu_samples) if cpu_samples else None

    # Environment / versions
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,

        # Run metadata
        "run_type": args.run_type,
        "experiment_tag": args.tag,
        "code_version": args.code_version,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),

        # Software environment (useful for reproducibility)
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics_version,

        # System configuration
        "backend": "cpu",
        "device": "rpi5_cpu",
        "architecture": platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=True),

        # Model + eval configuration
        "model": "yolo11n",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "max_detections": MAX_DETECTIONS,

        # Model characteristics
        "params_M": round(params_m, 2),
        "gflops": round(gflops, 2),

        # Accuracy
        "map50": round(float(final_metrics.box.map50), 4),
        "map50_95": round(float(final_metrics.box.map), 4),

        # Timing (ms per image)
        "preprocess_ms": round(float(final_speed["preprocess"]), 2),
        "inference_ms": round(float(final_speed["inference"]), 2),
        "postprocess_ms": round(float(final_speed["postprocess"]), 2),  # includes NMS
        "total_latency_ms": round(float(final_total_ms), 2),

        # Multi-run timing
        "total_latency_ms_mean": round(mean_ms, 2) if mean_ms is not None else None,
        "total_latency_ms_std": round(std_ms, 2) if std_ms is not None else None,

        # Throughput (derived)
        "fps_estimated": round(1000.0 / float(final_total_ms), 2),

        # System behavior
        "cpu_util_mean_percent": round(cpu_mean, 1) if cpu_mean is not None else None,
        "cpu_util_peak_percent": round(cpu_peak, 1) if cpu_peak is not None else None,
        "temp_start_c": round(temp_start, 1) if temp_start is not None else None,
        "temp_max_c": round(temp_max, 1) if temp_max is not None else None,
        "temp_end_c": round(temp_end, 1) if temp_end is not None else None,

        # Dataset bookkeeping
        "dataset": DATA_YAML.name,
        "num_eval_images": num_eval_images,
        "eval_minutes_last_run": round(float(final_elapsed_min), 2),
    }

    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    result = run_cpu()
    df_new = pd.DataFrame([result])

    if CSV_PATH.exists():
        df = pd.concat([pd.read_csv(CSV_PATH), df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(CSV_PATH, index=False)

    with open(TXT_PATH, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        for k, v in result.items():
            f.write(f"{k}: {v}\n")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Run Type: {result['run_type'].upper()} ({result['experiment_tag']})")
    print(f"Model: YOLOv11n @ {result['imgsz']}x{result['imgsz']}")
    print(f"Dataset: {result['dataset']} ({result['num_eval_images']} eval images)")
    print(f"Latency: {result['total_latency_ms']:.1f} ms/image (end-to-end)")
    print(f"Throughput: ~{result['fps_estimated']:.1f} FPS")
    print(f"Accuracy: mAP50={result['map50']:.3f}, mAP50-95={result['map50_95']:.3f}")
    if result["cpu_util_mean_percent"] is not None:
        print(f"CPU Utilization: {result['cpu_util_mean_percent']}% avg")
    if result["temp_start_c"] is not None and result["temp_end_c"] is not None:
        print(f"Temperature: {result['temp_start_c']}°C → {result['temp_end_c']}°C (max {result['temp_max_c']}°C)")
    print("=" * 60)
    print(f"CSV saved to: {CSV_PATH}")
    print(f"Log saved to: {TXT_PATH}")
    print("=" * 60 + "\n")
