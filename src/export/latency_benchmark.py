"""
src/export/latency_benchmark.py
───────────────────────────────
ONNX latency + file-size benchmark for the Track-2 hardware-viability gate
(Generic_Traning_Plan §4.4 / §5.3 Milestone 3).

Gate (defaults match the plan):
  latency_p50_ms <= 100  (≈10 FPS minimum)
  weights_mb     <= 30

Reports p50/p95/p99 over `--iters` measured iterations after `--warmup`,
on CPU by default (the conservative proxy for embedded ARM / RPi). Use
--device cuda for an NVIDIA Jetson-style estimate.

Outputs alongside the .onnx file:
  <ckpt-dir>/latency.json

Usage:
  python src/export/latency_benchmark.py --target student
  python src/export/latency_benchmark.py --target specialists --device cuda
  python src/export/latency_benchmark.py --onnx runs/student/bina_v1/weights/best.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]

LATENCY_GATE_MS = 100.0   # plan §4.4 / §5.3
SIZE_GATE_MB = 30.0


def _import_ort():
    try:
        import onnxruntime as ort  # noqa: F401
        return __import__("onnxruntime")
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "onnxruntime is required for benchmarking. "
            "Install with `pip install onnxruntime` or "
            "`pip install onnxruntime-gpu`."
        ) from e


def benchmark(
    onnx_path: Path,
    *,
    device: str = "cpu",
    iters: int = 500,
    warmup: int = 100,
    imgsz: int = 640,
) -> dict:
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    ort = _import_ort()
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    chosen_provider = sess.get_providers()[0]

    inp = sess.get_inputs()[0]
    in_name = inp.name
    in_shape = list(inp.shape)
    # Replace dynamic dims with concrete values
    in_shape = [imgsz if isinstance(d, str) or d is None or d <= 0 else d
                for d in in_shape]
    # Common YOLO ONNX input: (1, 3, H, W)
    if len(in_shape) == 4 and (in_shape[2] in (-1, 0) or in_shape[3] in (-1, 0)):
        in_shape[2] = imgsz
        in_shape[3] = imgsz
    if len(in_shape) == 4 and in_shape[0] in (-1, 0):
        in_shape[0] = 1

    x = np.random.rand(*in_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        sess.run(None, {in_name: x})

    timings_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    t = np.asarray(timings_ms)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    p50 = float(np.percentile(t, 50))
    p95 = float(np.percentile(t, 95))
    p99 = float(np.percentile(t, 99))

    latency_ok = p50 <= LATENCY_GATE_MS
    size_ok = size_mb <= SIZE_GATE_MB

    result = {
        "onnx_path": str(onnx_path),
        "provider": chosen_provider,
        "device": device,
        "input_shape": in_shape,
        "iters": iters,
        "warmup": warmup,
        "latency_ms": {
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "mean": float(t.mean()),
            "std": float(t.std()),
        },
        "weights_mb": size_mb,
        "gates": {
            f"latency_p50_<={LATENCY_GATE_MS}ms": latency_ok,
            f"weights_<={SIZE_GATE_MB}MB": size_ok,
        },
        "passed": latency_ok and size_ok,
    }
    out = onnx_path.parent / "latency.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  → {out}")
    return result


def _summarize(name: str, r: dict) -> str:
    g = r["gates"]
    verdict = "PASS" if r["passed"] else "FAIL"
    lat_ok = list(g.values())[0]
    size_ok = list(g.values())[1]
    return (f"  [{name}] {verdict}  provider={r['provider']}  "
            f"p50={r['latency_ms']['p50']:.1f}ms{'✓' if lat_ok else '✗'}  "
            f"p95={r['latency_ms']['p95']:.1f}ms  "
            f"size={r['weights_mb']:.2f}MB{'✓' if size_ok else '✗'}")


def benchmark_specialists(device: str, iters: int, warmup: int,
                          imgsz: int) -> list[dict]:
    out = []
    for cond in CONDITIONS:
        onnx = (ROOT / "runs" / "specialists" / f"specialist_{cond}"
                / "weights" / "best.onnx")
        if not onnx.exists():
            print(f"  skip {cond}: no ONNX at {onnx}")
            continue
        print(f"\n[{cond}]")
        r = benchmark(onnx, device=device, iters=iters, warmup=warmup,
                      imgsz=imgsz)
        print(_summarize(cond, r))
        out.append(r)
    return out


def benchmark_student(device: str, iters: int, warmup: int,
                      imgsz: int) -> dict | None:
    onnx = ROOT / "runs" / "student" / "bina_v1" / "weights" / "best.onnx"
    if not onnx.exists():
        print(f"  no student ONNX at {onnx}")
        return None
    print("\n[student]")
    r = benchmark(onnx, device=device, iters=iters, warmup=warmup, imgsz=imgsz)
    print(_summarize("student", r))
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["specialists", "student", "all"],
                        default="all")
    parser.add_argument("--onnx", default=None,
                        help="explicit .onnx path to benchmark")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if args.onnx:
        r = benchmark(Path(args.onnx), device=args.device,
                      iters=args.iters, warmup=args.warmup, imgsz=args.imgsz)
        print(_summarize(Path(args.onnx).name, r))
    else:
        if args.target in ("specialists", "all"):
            benchmark_specialists(args.device, args.iters, args.warmup, args.imgsz)
        if args.target in ("student", "all"):
            benchmark_student(args.device, args.iters, args.warmup, args.imgsz)
