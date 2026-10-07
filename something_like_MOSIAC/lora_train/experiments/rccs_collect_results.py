"""
rccs_collect_results.py

Parses ORCA .out files from a completed RCCS batch (run_accuracy_benchmark.py
--export-rccs-batch <dir>, submitted via rccs_submit_orca.sh --batch-dir, .out files retrieved
back next to their .inp files) into a results.json that run_accuracy_benchmark.py's
--rccs-results flag can merge back into the qcl-run job results.

Beyond plain ok/error, every wavefunction-correlated job also gets an aux_basis_outcome label --
whether the aux-basis mechanism itself (a real historical precedent, or ORCA's AutoAux fallback,
see classify_aux_basis_plan() in run_accuracy_benchmark.py) actually worked:

    not_applicable         -- not a WF-correlated method, no aux basis was needed
    precedent_ok            } aux basis was given (from the pool or AutoAux) and the job
    autoaux_fallback_ok     } terminated normally
    precedent_failed         } aux basis was given but the job still failed for an
    autoaux_fallback_failed  } aux-basis-related reason (see AUX_BASIS_ERROR_PATTERNS) --
                                "provided by the plan but failed"
    precedent_unrelated_error         } aux basis was given, job failed, but for a REAL,
    autoaux_fallback_unrelated_error  } unrelated reason (e.g. QCISD on a system with zero
                                         correlated electron pairs -- a genuine method/system
                                         mismatch, not an aux-basis problem)

Usage:
    python rccs_collect_results.py --dir rccs_batch_export/organic_general
    python rccs_collect_results.py --dir rccs_batch_export/organic_general --dir rccs_batch_export/metal_general
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_TERMINATED_OK_RE = re.compile(r"\*+ORCA TERMINATED NORMALLY\*+")

# Must match run_accuracy_benchmark.py's AUX_BASIS_ERROR_PATTERNS exactly -- kept as a separate
# copy rather than importing that module, since this script is meant to run standalone against a
# retrieved batch dir without pulling in the MCP/rag/RAG-index import chain at the top of that
# file (which needs a live .env and network-adjacent imports this script has no reason to need).
AUX_BASIS_ERROR_PATTERNS = [
    re.compile(r"please provide an AuxC basis", re.IGNORECASE),
    re.compile(r"need at least one auxiliary basis set", re.IGNORECASE),
    re.compile(r"not an appropriate auxiliary basis set for correlated methods", re.IGNORECASE),
]

# General ORCA error line -- used only to populate a human-readable `error` field, not for
# classification (AUX_BASIS_ERROR_PATTERNS above decides aux_basis_outcome on its own terms).
# Real formats seen: bare "Error (ORCA_MDCI): ..."; "[file orca_mdci/...]: Error (ORCA_MAIN): ...
# aborting the run"; and MDCI pair-index-prefixed "  0 Error (ORCA_MDCI): ..." (the leading
# integer is which pair MDCI was processing, not part of the message) -- found 2026-08-12, the
# naive "^\s*Error" version silently missed this last one and fell through to the "no error line
# matched" fallback despite the file clearly containing a real, classifiable error.
_ERROR_LINE_RE = re.compile(
    r"^\s*(?:\d+\s+)?(?:\[file .*?\]:\s*)?Error \([^)]*\):.*$", re.MULTILINE
)


def parse_orca_out(text: str) -> dict[str, Any]:
    if _TERMINATED_OK_RE.search(text):
        m = _ENERGY_RE.search(text)
        return {"status": "ok", "energy_eh": float(m.group(1)) if m else None, "error": None}
    m = _ERROR_LINE_RE.search(text)
    error = m.group(0).strip() if m else "ORCA did not terminate normally (no error line matched)"
    return {"status": "error", "energy_eh": None, "error": error}


def classify_outcome(aux_basis_plan: str, status: str, error: str | None) -> str:
    if aux_basis_plan == "not_applicable":
        return "not_applicable"
    if status == "ok":
        return f"{aux_basis_plan}_ok"
    is_aux_error = bool(error) and any(p.search(error) for p in AUX_BASIS_ERROR_PATTERNS)
    return f"{aux_basis_plan}_failed" if is_aux_error else f"{aux_basis_plan}_unrelated_error"


def collect_dir(batch_dir: Path) -> dict[str, dict]:
    meta_path = batch_dir / "_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found -- this dir wasn't produced by --export-rccs-batch, or "
            f"predates the _meta.json sidecar (re-export with the current script)."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    results: dict[str, dict] = {}
    for node_id, m in meta.items():
        # inp_stem may differ from node_id when the export deduplicated a case-insensitive
        # filesystem collision (see run_accuracy_benchmark.py's --export-rccs-batch, 2026-08-17)
        # -- .get() default keeps this compatible with older _meta.json files that predate the
        # field, where node_id was always the on-disk stem.
        stem = m.get("inp_stem", node_id)
        out_path = batch_dir / f"{stem}.out"
        if not out_path.exists():
            results[node_id] = {
                **m, "status": "missing", "energy_eh": None,
                "error": f"{out_path.name} not found -- job may still be queued/running, or "
                         f".out was never retrieved from RCCS",
                "aux_basis_outcome": "unknown",
            }
            continue
        parsed = parse_orca_out(out_path.read_text(encoding="utf-8", errors="replace"))
        outcome = classify_outcome(m["aux_basis_plan"], parsed["status"], parsed["error"])
        results[node_id] = {**m, **parsed, "aux_basis_outcome": outcome}
    return results


def print_summary(cell: str, results: dict[str, dict]) -> None:
    from collections import Counter

    n = len(results)
    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    outcome_counts = Counter(r["aux_basis_outcome"] for r in results.values())
    print(f"\n=== {cell}: {n_ok}/{n} ok ===")
    for outcome, count in sorted(outcome_counts.items()):
        print(f"  {outcome:32s} {count}")

    provided_but_failed = {
        nid: r for nid, r in results.items()
        if r["aux_basis_outcome"] in ("precedent_failed", "autoaux_fallback_failed")
    }
    if provided_but_failed:
        print(f"  -- {len(provided_but_failed)} job(s) had an aux basis PROVIDED but still failed:")
        for nid, r in provided_but_failed.items():
            print(f"     {nid}: {r['functional']}/{r['basis']} ({r['aux_basis_plan']}) -- {r['error']}")

    missing = {nid: r for nid, r in results.items() if r["status"] == "missing"}
    if missing:
        print(f"  -- {len(missing)} job(s) have no .out file yet (not retrieved / still running)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, action="append", required=True,
                   help="One per-cell batch dir (containing _meta.json + .inp/.out files). "
                        "Repeat for multiple cells. results.json is written inside each dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for batch_dir in args.dir:
        cell = batch_dir.name
        results = collect_dir(batch_dir)
        out_path = batch_dir / "results.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print_summary(cell, results)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
