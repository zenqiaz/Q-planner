#!/usr/bin/env python3
"""retrieval_diff.py -- compare the current text-scraping status logic against OPI grepper recipes.

Read-only. Walks a tree of archived ORCA ``.out`` files and, for each one, evaluates both
retrieval paths, then reports every disagreement.

Why this exists
---------------
Result retrieval is the agent's observation space, so swapping its source is an instrument
change rather than a refactor. Two semantic differences are visible in the source before any
file is read, and this script measures how often they actually bite:

* ``has_terminated_normally`` greps ``"****ORCA TERMINATED NORMALLY****"`` (with asterisks);
  the current check greps the bare ``"ORCA TERMINATED NORMALLY"``. OPI is the stricter of the two.
* ``has_scf_converged`` greps ``"SCF CONVERGED AFTER"``, so a job that runs no SCF at all
  reports as *not converged* rather than as *not applicable*.

A third difference cannot be read off the source: whether OPI's Grepper raises on files the
current reader swallows (it reads with ``errors="ignore"``). A path that raises where the other
returns a plausible value turns a silent-wrong result into a loud one, which matters more here
than it looks -- so exceptions are recorded per file as an outcome, not as a crash.

Usage
-----
    python retrieval_diff.py                              # whole tree, from this directory
    python retrieval_diff.py --root ./something_like_MOSIAC --limit 200
    python retrieval_diff.py --out diff_rows.csv --examples 10
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

try:
    from opi.output.grepper.recipes import (
        has_aborted_run,
        has_geometry_optimization_converged,
        has_scf_converged,
        has_terminated_normally,
    )
except ImportError as exc:  # pragma: no cover - environment problem, not a data problem
    sys.exit(f"opi is not importable from this interpreter ({sys.executable}): {exc}")


# ---------------------------------------------------------------------------
# Current logic, copied verbatim from the server so the diff compares like with like.
# Sources: server_helpers.py:271 (_read_out_text), server_with_product.py:339
# (_classify_orca_error_code). SOURCE_LITERALS below guards against drift.
# ---------------------------------------------------------------------------

SOURCE_LITERALS = {
    "server_with_product.py": [
        '"SCF NOT CONVERGED"',
        '"***IMAGINARY MODE***"',
        '"CASSCF ORBITAL OPTIMIZATION DID NOT CONVERGE"',
    ],
}


def read_out_text(path: Path) -> str:
    """Mirror of _read_out_text's read: whole file, utf-8, errors ignored."""
    return path.read_text(encoding="utf-8", errors="ignore")


def current_terminated_normally(out_text: str) -> bool:
    """Mirror of the server's success test."""
    return "ORCA TERMINATED NORMALLY" in out_text


def current_error_code(out_text: str) -> Optional[str]:
    """Verbatim copy of _classify_orca_error_code."""
    upper = out_text.upper()
    if "SCF NOT CONVERGED" in upper or "FAILED TO CONVERGE" in upper:
        return "SCF_NOT_CONVERGED"
    if "***IMAGINARY MODE***" in upper:
        return "IMAG_FREQ"
    if (
        "THE CAS PROCEDURE HAS NOT CONVERGED" in upper
        or "CASSCF ORBITAL OPTIMIZATION DID NOT CONVERGE" in upper
        or "CAS PROCEDURE DID NOT CONVERGE" in upper
        or ("CASSCF" in upper and "NOT CONVERGED" in upper)
    ):
        return "CASSCF_NOT_CONVERGED"
    return None


def verify_source_literals(base: Path) -> list[str]:
    """Warn if the copied logic has drifted from the server it was copied from."""
    warnings = []
    for fname, literals in SOURCE_LITERALS.items():
        src = base / fname
        if not src.exists():
            warnings.append(f"{fname} not found next to this script -- drift unchecked")
            continue
        text = src.read_text(encoding="utf-8", errors="ignore")
        for lit in literals:
            if lit not in text:
                warnings.append(f"{fname} no longer contains {lit} -- copied logic may be stale")
    return warnings


# ---------------------------------------------------------------------------
# Per-file comparison
# ---------------------------------------------------------------------------

FIELDS = [
    "path", "size_bytes",
    "cur_ok", "opi_ok", "ok_agree",
    "cur_code", "opi_scf_converged", "scf_agree",
    "opi_aborted", "opi_opt_converged",
    "cur_exc", "opi_exc",
    "cur_ms", "opi_ms",
]


def compare_one(path: Path) -> dict:
    row = {k: "" for k in FIELDS}
    row["path"] = str(path)
    try:
        row["size_bytes"] = path.stat().st_size
    except OSError:
        row["size_bytes"] = ""

    # --- current path ---
    t0 = time.perf_counter()
    try:
        text = read_out_text(path)
        row["cur_ok"] = current_terminated_normally(text)
        row["cur_code"] = current_error_code(text) or ""
    except Exception as exc:
        row["cur_exc"] = f"{type(exc).__name__}: {exc}"
    row["cur_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # --- OPI path (each recipe opens the file itself) ---
    t0 = time.perf_counter()
    try:
        row["opi_ok"] = has_terminated_normally(path)
        row["opi_scf_converged"] = has_scf_converged(path)
        row["opi_aborted"] = has_aborted_run(path)
        row["opi_opt_converged"] = has_geometry_optimization_converged(path)
    except Exception as exc:
        row["opi_exc"] = f"{type(exc).__name__}: {exc}"
    row["opi_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if row["cur_exc"] == "" and row["opi_exc"] == "":
        row["ok_agree"] = row["cur_ok"] == row["opi_ok"]
        # The current path only ever asserts SCF *failure*; OPI asserts success.
        # They disagree when the current path says nothing is wrong and OPI says
        # no SCF converged, or vice versa.
        cur_scf_failed = row["cur_code"] == "SCF_NOT_CONVERGED"
        row["scf_agree"] = cur_scf_failed != bool(row["opi_scf_converged"])

    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="directory to walk (default: this directory)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files (0 = no limit)")
    ap.add_argument("--max-mb", type=float, default=20.0, help="skip files larger than this")
    ap.add_argument("--out", default="retrieval_diff_rows.csv", help="CSV of every row")
    ap.add_argument("--examples", type=int, default=5, help="example paths to print per disagreement class")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    for w in verify_source_literals(base):
        print(f"[warn] {w}")

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    paths = sorted(p for p in root.rglob("*.out") if p.is_file())
    max_bytes = int(args.max_mb * 1024 * 1024)
    skipped_big = []
    kept = []
    for p in paths:
        try:
            if p.stat().st_size > max_bytes:
                skipped_big.append(p)
                continue
        except OSError:
            continue
        kept.append(p)
        if args.limit and len(kept) >= args.limit:
            break

    print(f"root            : {root}")
    print(f"found .out      : {len(paths)}")
    print(f"compared        : {len(kept)}")
    if skipped_big:
        print(f"skipped (>{args.max_mb} MB): {len(skipped_big)}")
    if not kept:
        return 0

    rows = []
    t_start = time.perf_counter()
    for i, p in enumerate(kept, 1):
        rows.append(compare_one(p))
        if i % 250 == 0:
            print(f"  ... {i}/{len(kept)}", flush=True)
    wall = time.perf_counter() - t_start

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = base / out_path
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ---------------- report ----------------
    def section(title: str) -> None:
        print()
        print(title)
        print("-" * len(title))

    cur_exc = [r for r in rows if r["cur_exc"]]
    opi_exc = [r for r in rows if r["opi_exc"]]
    clean = [r for r in rows if not r["cur_exc"] and not r["opi_exc"]]

    section("Exceptions  (a path that raises where the other returns a value)")
    print(f"current raised : {len(cur_exc)}")
    print(f"OPI raised     : {len(opi_exc)}")
    for r in opi_exc[: args.examples]:
        print(f"    {r['opi_exc']}  <- {r['path']}")
    for r in cur_exc[: args.examples]:
        print(f"    [current] {r['cur_exc']}  <- {r['path']}")

    section("TERMINATED NORMALLY  (current bare substring  vs  OPI ****-delimited****)")
    cm = Counter((r["cur_ok"], r["opi_ok"]) for r in clean)
    print(f"  both ok        : {cm[(True, True)]}")
    print(f"  both not ok    : {cm[(False, False)]}")
    print(f"  current ok, OPI NOT ok : {cm[(True, False)]}   <-- OPI stricter; these runs change class")
    print(f"  current NOT ok, OPI ok : {cm[(False, True)]}   <-- unexpected; inspect")
    for cell, label in (((True, False), "cur_ok_opi_not"), ((False, True), "cur_not_opi_ok")):
        ex = [r for r in clean if (r["cur_ok"], r["opi_ok"]) == cell][: args.examples]
        for r in ex:
            print(f"    [{label}] {r['path']}")

    section("SCF  (current asserts failure  vs  OPI asserts success)")
    cm2 = Counter((r["cur_code"] == "SCF_NOT_CONVERGED", bool(r["opi_scf_converged"])) for r in clean)
    print(f"  current silent, OPI converged      : {cm2[(False, True)]}")
    print(f"  current SCF_NOT_CONVERGED, OPI not : {cm2[(True, False)]}")
    print(f"  current silent, OPI NOT converged  : {cm2[(False, False)]}   <-- includes jobs with no SCF at all")
    print(f"  current SCF_NOT_CONVERGED, OPI yes : {cm2[(True, True)]}   <-- both fired; e.g. retried SCF")
    for r in [r for r in clean if (r["cur_code"] == "SCF_NOT_CONVERGED") == bool(r["opi_scf_converged"])][: args.examples]:
        print(f"    [disagree] cur_code={r['cur_code'] or '-'} opi_scf={r['opi_scf_converged']}  {r['path']}")

    section("Current error codes")
    for code, n in Counter(r["cur_code"] or "(none)" for r in clean).most_common():
        print(f"  {code:<24} {n}")

    section("OPI-only signals  (not in the payload today -- adding them would be enrichment)")
    print(f"  aborted=True           : {sum(1 for r in clean if r['opi_aborted'])}")
    print(f"  opt_converged=True     : {sum(1 for r in clean if r['opi_opt_converged'])}")
    print(f"  aborted but current ok : {sum(1 for r in clean if r['opi_aborted'] and r['cur_ok'])}   <-- 'aborting' is matched case-sensitively, loosely")

    section("Cost")
    cur_tot = sum(r["cur_ms"] for r in rows if isinstance(r["cur_ms"], float))
    opi_tot = sum(r["opi_ms"] for r in rows if isinstance(r["opi_ms"], float))
    print(f"  current total : {cur_tot/1000:.1f} s")
    print(f"  OPI total     : {opi_tot/1000:.1f} s   (4 recipes = 4 separate reads per file)")
    print(f"  wall          : {wall:.1f} s")

    print()
    print(f"rows written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
