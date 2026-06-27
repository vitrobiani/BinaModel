"""
src/train/sweep.py
──────────────────
Phase 1 multi-architecture specialist sweep (Generic_Traning_Plan Part 2).

For each (architecture, condition) pair:
  1. HPO mini-run         — Optuna RandomSampler over lr0/weight_decay/batch
                            on a 10% subset, 5–10 epochs/trial (plan §5.1).
                            Writes runs/hpo/<arch>/<cond>_best.json.
  2. Full train           — picks up HPO-best automatically via train_specialist
                            → runs/sweep/<arch>/specialist_<cond>/weights/best.pt
  3. Threshold calibration — PR-curve → smallest conf with P>=0.95 & R>=0.60
                            on val → runs/sweep/<arch>/specialist_<cond>/threshold.json
  4. KPI gate              — mAP@0.5>=0.85 AND test P>=0.95 AND test R>=0.60
                            → runs/sweep/<arch>/specialist_<cond>/kpi_gate.json

Per-pair verdicts are streamed into runs/sweep/results.json after each pair
completes, so the file is always current even if the run is interrupted.

The promotion step (src/train/promote.py) picks the per-condition winner and
copies it into runs/specialists/specialist_<cond>/ for downstream phases.

Usage:
  # Default: sweep yolo26s × yolo26x × rtdetr-l over all 6 conditions
  python src/train/sweep.py

  # Restrict scope
  python src/train/sweep.py --archs yolo26s rtdetr-l --conditions recession caries

  # Skip HPO (use pipeline defaults or pre-existing HPO outputs)
  python src/train/sweep.py --skip-hpo

  # Resume interrupted training within each pair
  python src/train/sweep.py --resume
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from train.train_specialist import train_specialist                  # noqa: E402
from train.hpo import run_hpo                                        # noqa: E402
from validation.threshold_finder import find_specialist_threshold    # noqa: E402
from validation.kpi_gate import evaluate_specialist, write_manifest  # noqa: E402

PIPELINE_CFG = ROOT / "configs" / "pipeline.yaml"
RESULTS_JSON = ROOT / "runs" / "sweep" / "results.json"
# Each subprocess-spawned pair writes its result here so the parent driver
# can read it without parsing stdout. Lets us survive CUDA OOMs / crashes
# without losing whole-sweep state.
PAIR_RESULTS_DIR = ROOT / "runs" / "sweep" / "_pairs"

# Phase 1A — Ultralytics-API compatible backbones (plan §2.1).
# yolo26x dropped: at imgsz=640 it OOMs on a 4070 12GB even at small batch
# once VRAM fragments. The marginal accuracy gain vs yolo26s rarely
# justifies the 4-6h/condition training cost.
# rtdetr-l dropped: Ultralytics' RTDETR class internally upscales to 1280px
# (multi-scale=0 doesn't override), needs ~21GB VRAM on 4070 12GB so CUDA
# pages to system RAM (iter times 10-80s vs yolo26s's 0.1s). Also DETR-family
# convergence needs 50-300 epochs, making 5-epoch HPO meaningless. Net cost
# was ~6 days for this arch alone.
ULTRALYTICS_ARCHS = ["yolo26s"]
# Phase 1B — torchvision + HuggingFace adapters (Faster R-CNN, DETR).
EXTENDED_ARCHS = ["frcnn-r50", "detr-r50"]
ALL_ARCHS = ULTRALYTICS_ARCHS + EXTENDED_ARCHS

CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]


def _is_skip_architecture(exc: BaseException) -> bool:
    """Detect SkipArchitecture without importing it eagerly (it lives in the
    HF DETR adapter, which we don't want to import unless DETR is used)."""
    return type(exc).__name__ == "SkipArchitecture"


def _release_cuda() -> None:
    """Force GC + CUDA cache release between sub-steps of a pair.

    Each sub-step (HPO, train, threshold, KPI) loads its own Ultralytics
    model. Without explicit cleanup, the prior model's tensors stay alive
    on host RAM and VRAM until natural GC, which on plaque-sized val sets
    has produced MemoryError in threshold_finder. Cheap to call (~1ms)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        pass


# ── Per-pair execution ──────────────────────────────────────────────────────


def sweep_one(
    arch: str,
    condition: str,
    *,
    skip_hpo: bool = False,
    hpo_trials: int = 20,
    hpo_epochs: int = 8,
    hpo_fraction: float = 0.10,
    hpo_seed: int = 42,
    resume: bool = False,
    train_overrides: dict | None = None,
) -> dict:
    """Run HPO → train → threshold → KPI gate for one (arch, condition).
    Each sub-step is wrapped in try/except so a single failing pair doesn't
    abort the whole sweep."""
    result: dict = {"arch": arch, "condition": condition}
    t0 = time.time()

    # 1. HPO ─────────────────────────────────────────────────────────────────
    if skip_hpo:
        result["hpo"] = "skipped"
    else:
        try:
            run_hpo(
                condition,
                arch=arch,
                n_trials=hpo_trials,
                epochs=hpo_epochs,
                fraction=hpo_fraction,
                seed=hpo_seed,
            )
            result["hpo"] = "ok"
        except Exception as e:  # noqa: BLE001
            print(f"  HPO failed: {e!r}")
            result["hpo"] = "failed"
            result["hpo_error"] = repr(e)
            result["traceback_hpo"] = traceback.format_exc()
            # Continue — train can still run with pipeline defaults.
    _release_cuda()  # drop HPO trial residuals before full train

    # 2. Train ───────────────────────────────────────────────────────────────
    try:
        ckpt = train_specialist(
            condition,
            resume=resume,
            overrides=train_overrides or {},
            arch=arch,
            weight_override=f"{arch}.pt",
        )
        result["ckpt"] = str(ckpt)
        result["train"] = "ok"
    except Exception as e:  # noqa: BLE001
        if _is_skip_architecture(e):
            # Adapter declined this (arch, cond) pair (e.g. DETR + small dataset).
            print(f"  SKIP: {e}")
            result["train"] = "skipped"
            result["skip_reason"] = str(e)
            result["elapsed_min"] = (time.time() - t0) / 60
            return result
        print(f"  train failed: {e!r}")
        result["train"] = "failed"
        result["train_error"] = repr(e)
        result["traceback_train"] = traceback.format_exc()
        result["elapsed_min"] = (time.time() - t0) / 60
        return result  # without a checkpoint, threshold/gate can't run
    _release_cuda()  # drop trainer + optimizer + dataloader residuals

    # 3. Threshold calibration on val ────────────────────────────────────────
    try:
        th = find_specialist_threshold(condition, arch=arch)
        result["threshold"] = "ok"
        result["threshold_passed"] = bool(th.get("passed"))
        result["calibrated_conf"] = th.get("threshold")
    except Exception as e:  # noqa: BLE001
        print(f"  threshold finder failed: {e!r}")
        result["threshold"] = "failed"
        result["threshold_error"] = repr(e)
    _release_cuda()  # drop predict_batch residuals before test eval

    # 4. KPI gate on test ────────────────────────────────────────────────────
    try:
        gate = evaluate_specialist(condition, arch=arch)
        write_manifest(condition, gate, arch=arch)
        result["kpi"] = "ok"
        result["kpi_passed"] = bool(gate.get("passed"))
        result["metrics"] = gate.get("metrics", {})
        result["gates"] = gate.get("gates", {})
    except Exception as e:  # noqa: BLE001
        print(f"  KPI gate failed: {e!r}")
        result["kpi"] = "failed"
        result["kpi_error"] = repr(e)

    result["elapsed_min"] = (time.time() - t0) / 60
    return result


# ── Sweep driver ────────────────────────────────────────────────────────────


def _save_results(results: list[dict]) -> Path:
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(
        {"results": results, "saved_at": time.time()}, indent=2,
    ))
    return RESULTS_JSON


def _pair_result_path(arch: str, condition: str) -> Path:
    return PAIR_RESULTS_DIR / f"{arch}_{condition}.json"


def _build_single_pair_cmd(
    arch: str,
    condition: str,
    *,
    skip_hpo: bool,
    hpo_trials: int,
    hpo_epochs: int,
    hpo_fraction: float,
    hpo_seed: int,
    resume: bool,
    train_overrides: dict | None,
) -> list[str]:
    """Build the argv to re-invoke this script in single-pair mode."""
    cmd: list[str] = [
        sys.executable, str(Path(__file__).resolve()),
        "--single-pair",
        "--archs", arch,
        "--conditions", condition,
        "--hpo-trials", str(hpo_trials),
        "--hpo-epochs", str(hpo_epochs),
        "--hpo-fraction", str(hpo_fraction),
        "--hpo-seed", str(hpo_seed),
    ]
    if skip_hpo:
        cmd.append("--skip-hpo")
    if resume:
        cmd.append("--resume")
    if train_overrides:
        if "epochs" in train_overrides:
            cmd += ["--epochs", str(train_overrides["epochs"])]
        if "batch" in train_overrides:
            cmd += ["--batch", str(train_overrides["batch"])]
    return cmd


def _run_pair_subprocess(
    arch: str,
    condition: str,
    **kwargs,
) -> dict:
    """Spawn this script as a subprocess for one (arch, condition) pair.

    Each pair gets a fresh Python interpreter → fresh CUDA context → zero
    chance of cross-pair VRAM fragmentation. The pair writes its result JSON
    to PAIR_RESULTS_DIR; we read it after the subprocess exits.
    """
    out_path = _pair_result_path(arch, condition)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    cmd = _build_single_pair_cmd(arch, condition, **kwargs)
    t0 = time.time()
    try:
        # No capture — let the subprocess's stdout/stderr stream live to the
        # parent terminal so the user can watch progress in real time.
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # Subprocess crashed hard (CUDA driver crash, segfault, etc).
        elapsed = (time.time() - t0) / 60
        if out_path.exists():
            # Pair logged its own failure before crashing — use that.
            r = json.loads(out_path.read_text(encoding="utf-8"))
            r["subprocess"] = "crashed_after_log"
            r["subprocess_returncode"] = e.returncode
            return r
        return {
            "arch": arch,
            "condition": condition,
            "subprocess": "crashed",
            "subprocess_returncode": e.returncode,
            "elapsed_min": elapsed,
        }
    except KeyboardInterrupt:
        raise

    if not out_path.exists():
        return {
            "arch": arch,
            "condition": condition,
            "subprocess": "no_output",
            "elapsed_min": (time.time() - t0) / 60,
        }
    return json.loads(out_path.read_text(encoding="utf-8"))


def run_sweep(
    archs: list[str],
    conditions: list[str],
    *,
    skip_hpo: bool = False,
    hpo_trials: int = 20,
    hpo_epochs: int = 8,
    hpo_fraction: float = 0.10,
    hpo_seed: int = 42,
    resume: bool = False,
    train_overrides: dict | None = None,
) -> dict:
    print(f"\n{'█' * 60}")
    print("  SPECIALIST ARCHITECTURE SWEEP — subprocess-per-pair")
    print(f"  archs:      {archs}")
    print(f"  conditions: {conditions}")
    print(f"  pairs:      {len(archs) * len(conditions)}")
    print(f"  HPO:        {'skipped' if skip_hpo else f'{hpo_trials} trials × {hpo_epochs}ep × {int(hpo_fraction*100)}% subset'}")
    print(f"{'█' * 60}")

    all_results: list[dict] = []
    t_total = time.time()

    # Outer loop is conditions, inner is archs — easier to inspect partial
    # results because you see each condition fully evaluated before moving on.
    for cond in conditions:
        for arch in archs:
            print(f"\n{'=' * 60}")
            print(f"  PAIR: arch={arch}  condition={cond}  (subprocess)")
            print(f"{'=' * 60}")
            r = _run_pair_subprocess(
                arch, cond,
                skip_hpo=skip_hpo,
                hpo_trials=hpo_trials,
                hpo_epochs=hpo_epochs,
                hpo_fraction=hpo_fraction,
                hpo_seed=hpo_seed,
                resume=resume,
                train_overrides=train_overrides,
            )
            all_results.append(r)
            _save_results(all_results)
            elapsed_min = r.get("elapsed_min", 0)
            print(f"  pair done in {elapsed_min:.1f} min  "
                  f"(running total: {(time.time() - t_total) / 3600:.2f} h)")

    elapsed_h = (time.time() - t_total) / 3600
    print(f"\n{'=' * 60}")
    print(f"  Sweep complete in {elapsed_h:.2f} hours.")
    print(f"  → {RESULTS_JSON}")
    print(f"{'=' * 60}")
    return {"results": all_results}


def _single_pair_main(args, train_overrides: dict | None) -> int:
    """Entry point for --single-pair. Runs one (arch, condition) and writes
    its result JSON to PAIR_RESULTS_DIR. Always exits 0 on graceful failure
    (errors are recorded in the JSON); only crashes hard for unhandleable
    issues (segfault, CUDA driver crash, KeyboardInterrupt)."""
    if len(args.archs) != 1 or len(args.conditions) != 1:
        print("--single-pair requires exactly one --archs and one --conditions value")
        return 2
    arch, cond = args.archs[0], args.conditions[0]
    r = sweep_one(
        arch, cond,
        skip_hpo=args.skip_hpo,
        hpo_trials=args.hpo_trials,
        hpo_epochs=args.hpo_epochs,
        hpo_fraction=args.hpo_fraction,
        hpo_seed=args.hpo_seed,
        resume=args.resume,
        train_overrides=train_overrides,
    )
    out = _pair_result_path(arch, cond)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"\n  → wrote {out}")
    return 0


def _summary_table(results: list[dict]) -> str:
    lines = ["\n  arch         condition       hpo  train  thr  kpi  mAP50    P      R"]
    lines.append("  " + "-" * 75)
    for r in results:
        arch = r.get("arch", "?")[:12].ljust(12)
        cond = r.get("condition", "?")[:13].ljust(13)
        h = (r.get("hpo") or "?")[:6].ljust(4)
        t = (r.get("train") or "?")[:6].ljust(5)
        th = (r.get("threshold") or "?")[:4].ljust(3)
        if r.get("train") == "skipped":
            kpi_verdict = "SKIP"
        elif r.get("kpi_passed"):
            kpi_verdict = "PASS"
        elif r.get("kpi") == "ok":
            kpi_verdict = "FAIL"
        else:
            kpi_verdict = "·"
        m = r.get("metrics") or {}
        map50 = f"{m.get('mAP50', 0):.3f}" if m else "  -  "
        p = f"{m.get('test_precision', 0):.3f}" if m else "  -  "
        rc = f"{m.get('test_recall', 0):.3f}" if m else "  -  "
        lines.append(
            f"  {arch} {cond}  {h}  {t}  {th}  {kpi_verdict:<4}  "
            f"{map50}  {p}  {rc}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="+", default=ALL_ARCHS,
                        help=f"architecture slugs to sweep (default: {ALL_ARCHS})")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS,
                        choices=CONDITIONS,
                        help="conditions to sweep (default: all 6)")
    parser.add_argument("--skip-hpo", action="store_true",
                        help="skip HPO mini-runs; reuse existing HPO json if "
                             "present, otherwise use pipeline.yaml defaults")
    parser.add_argument("--hpo-trials", type=int, default=20)
    parser.add_argument("--hpo-epochs", type=int, default=8,
                        help="epochs per HPO trial (plan §5.1: 5-10)")
    parser.add_argument("--hpo-fraction", type=float, default=0.10)
    parser.add_argument("--hpo-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="resume any in-progress training within a pair")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override train.epochs in pipeline.yaml (useful "
                             "for smoke tests; e.g. --epochs 3)")
    parser.add_argument("--batch", type=int, default=None,
                        help="override train.batch in pipeline.yaml")
    parser.add_argument("--single-pair", action="store_true",
                        help="internal: run exactly one (arch, condition) and "
                             "write its result JSON. The driver re-invokes "
                             "this script with --single-pair per pair so each "
                             "gets a fresh CUDA context.")
    args = parser.parse_args()

    train_overrides: dict = {}
    if args.epochs is not None:
        train_overrides["epochs"] = args.epochs
    if args.batch is not None:
        train_overrides["batch"] = args.batch

    if args.single_pair:
        sys.exit(_single_pair_main(args, train_overrides or None))

    out = run_sweep(
        args.archs,
        args.conditions,
        skip_hpo=args.skip_hpo,
        hpo_trials=args.hpo_trials,
        hpo_epochs=args.hpo_epochs,
        hpo_fraction=args.hpo_fraction,
        hpo_seed=args.hpo_seed,
        resume=args.resume,
        train_overrides=train_overrides or None,
    )
    print(_summary_table(out["results"]))
