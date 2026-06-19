"""
src/train/promote.py
────────────────────
Per-condition winner picker for the Phase-1 specialist sweep.

Reads runs/sweep/results.json. For each condition:
  - Filters to entries that passed the KPI gate (mAP@0.5, P, R all ≥ targets).
  - Picks the highest mAP@0.5 among the passers.
  - Copies that candidate's weights/best.pt, threshold.json, and kpi_gate.json
    into runs/specialists/specialist_<cond>/, the canonical "winner" location
    that downstream phases (pseudo, student) read from.

Per plan §2.4 / §3.3, a specialist that fails the KPI gate must NOT be used
to generate pseudo-labels — `--force-best` lets you override this for pipeline
smoke-testing only.

Writes a manifest to runs/sweep/promotions.json documenting what was promoted
(or why it wasn't).

Usage:
  python src/train/promote.py                 # promote passers only
  python src/train/promote.py --force-best    # promote best mAP even if KPI fails
  python src/train/promote.py --conditions recession    # restrict scope
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train.train_specialist import specialist_run_dir  # noqa: E402

SWEEP_RESULTS = ROOT / "runs" / "sweep" / "results.json"
PROMOTIONS_OUT = ROOT / "runs" / "sweep" / "promotions.json"
CONDITIONS = ["caries", "gingivitis", "plaque", "discoloration", "ulcer", "recession"]

# Files we copy from the winning candidate dir into the canonical dir.
PROMOTED_FILES = ("threshold.json", "kpi_gate.json")
PROMOTED_WEIGHTS = ("best.pt",)


def _load_results() -> list[dict]:
    if not SWEEP_RESULTS.exists():
        raise FileNotFoundError(
            f"No sweep results at {SWEEP_RESULTS}. Run src/train/sweep.py first."
        )
    data = json.loads(SWEEP_RESULTS.read_text())
    return data.get("results", [])


def _select_winner(
    candidates: list[dict], *, force_best: bool,
) -> tuple[dict | None, str]:
    """Return (winner, reason). Winner is None if nothing qualifies."""
    if not candidates:
        return None, "no candidates"

    passers = [c for c in candidates if c.get("kpi_passed")]
    if passers:
        winner = max(
            passers,
            key=lambda c: (c.get("metrics") or {}).get("mAP50", 0.0),
        )
        return winner, "kpi_passed; selected highest mAP50 among passers"

    if not force_best:
        return None, "no candidate passed KPI gate (use --force-best to override)"

    # Fall back to best mAP50 (or best of train_completed, then by mAP) — useful
    # for smoke-testing downstream phases before gates are fully met.
    train_ok = [c for c in candidates if c.get("train") == "ok"]
    pool = train_ok or candidates
    winner = max(
        pool,
        key=lambda c: (c.get("metrics") or {}).get("mAP50", 0.0),
    )
    return winner, "force_best fallback: no passers; selected highest mAP50"


def _promote_candidate(condition: str, candidate: dict) -> Path:
    """Copy weights/best.pt + threshold.json + kpi_gate.json from the
    candidate's sweep dir into runs/specialists/specialist_<cond>/."""
    arch = candidate.get("arch")
    if not arch:
        raise ValueError(f"candidate has no arch: {candidate}")

    src_dir = specialist_run_dir(condition, arch)
    dst_dir = specialist_run_dir(condition, None)
    (dst_dir / "weights").mkdir(parents=True, exist_ok=True)

    for fname in PROMOTED_WEIGHTS:
        src = src_dir / "weights" / fname
        if not src.exists():
            print(f"  WARN: missing {src}")
            continue
        shutil.copy2(src, dst_dir / "weights" / fname)

    for fname in PROMOTED_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)

    return dst_dir


def promote(
    conditions: list[str], *, force_best: bool = False,
) -> dict:
    results = _load_results()

    # Group by condition.
    by_cond: dict[str, list[dict]] = {c: [] for c in conditions}
    for r in results:
        cond = r.get("condition")
        if cond in by_cond:
            by_cond[cond].append(r)

    promotions: dict[str, dict] = {}
    for cond, cands in by_cond.items():
        winner, reason = _select_winner(cands, force_best=force_best)
        entry: dict = {
            "n_candidates": len(cands),
            "reason": reason,
            "force_best": force_best,
        }
        if winner is None:
            entry["promoted"] = False
            entry["skipped"] = True
            print(f"  [{cond}] SKIP — {reason}")
        else:
            dst = _promote_candidate(cond, winner)
            entry["promoted"] = True
            entry["arch"] = winner.get("arch")
            entry["from"] = str(specialist_run_dir(cond, winner.get("arch")))
            entry["to"] = str(dst)
            entry["metrics"] = winner.get("metrics", {})
            entry["kpi_passed"] = bool(winner.get("kpi_passed"))
            m = winner.get("metrics") or {}
            print(f"  [{cond}] PROMOTED arch={winner['arch']}  "
                  f"mAP50={m.get('mAP50', 0):.3f}  "
                  f"P={m.get('test_precision', 0):.3f}  "
                  f"R={m.get('test_recall', 0):.3f}  "
                  f"(kpi_passed={winner.get('kpi_passed')})")
        promotions[cond] = entry

    out = {"promotions": promotions, "saved_at": time.time()}
    PROMOTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROMOTIONS_OUT.write_text(json.dumps(out, indent=2))
    print(f"\n  → {PROMOTIONS_OUT}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS,
                        choices=CONDITIONS)
    parser.add_argument("--force-best", action="store_true",
                        help="promote highest-mAP candidate even if no "
                             "candidate passed the KPI gate (smoke-test only)")
    args = parser.parse_args()

    promote(args.conditions, force_best=args.force_best)
