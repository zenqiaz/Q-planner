"""
build_no_orca_only_corpus.py

Step 1 of the "drop orca_only, does entropy blow up" check (user's request:
build the no-orca_only corpus first and inspect entropy before deciding whether
a full retrain is worth it -- if entropy is much higher, SFT may need much more
data to avoid a worse version of the organic_general memorization/generalization
gap already found, whereas RAG doesn't need to "learn" the mapping so should
degrade more gracefully).

Reimplements generate_sft.py's exact pipeline (same is_clean quality filter,
EXCLUDED_FUNCTIONALS, RAG_ONLY_CELLS exclusion, apply_upload_cap, stratified
split_by_upload, apply_caps) but:
  1. operating on rag.py's raw, unfiltered pool as the input record set (not a
     jsonl file -- verified elsewhere this session to have 100% entry_id
     coverage of the actual official sft_{cell}.jsonl/sft_val_{cell}.jsonl,
     i.e. it's a superset of whatever corpus generate_sft.py's original run
     actually drew from)
  2. with orca_only=False, so PM6/AM1 and other non-ORCA functionals are kept

Does NOT write jsonl output or retrain anything -- this is purely a corpus/
entropy inspection step, per the user's explicit "check entropy first" request.

Usage:
    python build_no_orca_only_corpus.py
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"D:\brick\D\202606\working\something_like MOSIAC")
sys.path.insert(0, r"D:\brick\D\20260217\working")

from generate_sft import (  # noqa: E402
    EXCLUDED_FUNCTIONALS, ORCA_VALID_FUNCTIONALS, RAG_ONLY_CELLS,
    DEFAULT_UPLOAD_CAP, DEFAULT_MAJORITY_RATIO,
    is_clean, apply_upload_cap, split_by_upload, apply_caps,
)
import rag  # noqa: E402

PHASE1_CELLS = ["organic_general", "metal_general"]


def load_records_in_memory(records: list[dict], quality: str, orca_only: bool) -> list[dict]:
    """Same filtering logic as generate_sft.load_records(), operating on an
    in-memory record list instead of reading a jsonl file."""
    out = []
    n_lda = n_orca = 0
    for r in records:
        cell = r.get("specialist_cell", "")
        if cell in RAG_ONLY_CELLS:
            continue
        func = (r.get("functional") or "").upper()
        if func in EXCLUDED_FUNCTIONALS:
            n_lda += 1
            continue
        if orca_only and func and func not in ORCA_VALID_FUNCTIONALS:
            n_orca += 1
            continue
        if is_clean(r, quality):
            out.append(r)
    print(f"  excluded {n_lda} LDA, {n_orca} non-ORCA (orca_only={orca_only})", file=sys.stderr)
    return out


def entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    h = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def summarize(name: str, records: list[dict]) -> None:
    by_cell: dict[str, list[dict]] = {}
    for r in records:
        by_cell.setdefault(r["specialist_cell"], []).append(r)
    print(f"\n--- {name} ---")
    for cell in PHASE1_CELLS:
        recs = by_cell.get(cell, [])
        if not recs:
            print(f"  {cell}: 0 records")
            continue
        pairs = Counter((r.get("functional"), r.get("basis")) for r in recs)
        funcs = Counter(r.get("functional") for r in recs)
        n_invalid = sum(1 for r in recs if (r.get("functional") or "").upper() not in ORCA_VALID_FUNCTIONALS)
        print(f"  {cell}: n={len(recs)}, unique (func,basis) pairs={len(pairs)}, "
              f"entropy={entropy_bits(pairs):.2f} bits, "
              f"non-ORCA-valid={n_invalid} ({n_invalid/len(recs):.1%})")
        print(f"    top functionals: {funcs.most_common(6)}")


def defalsify_semi_empirical_basis_warning(records: list[dict]) -> list[dict]:
    """Emulate the nomad_parse_gjf.py fix (basis-not-identified is a false
    positive for SEMI_EMPIRICAL records -- basis is genuinely absent for those
    methods, not unparsed) on already-parsed records, without a full re-parse
    of raw Gaussian logs. Only strips that one warning string; everything else
    about the record is untouched."""
    out = []
    n_fixed = 0
    for r in records:
        if (r.get("method_family") == "SEMI_EMPIRICAL"
                and "basis not identified" in (r.get("parse_warnings") or [])):
            r = dict(r)
            r["parse_warnings"] = [w for w in r["parse_warnings"] if w != "basis not identified"]
            n_fixed += 1
        out.append(r)
    print(f"[step1] de-falsified 'basis not identified' warning on {n_fixed} "
          f"semi-empirical records", file=sys.stderr)
    return out


def main():
    print("[step1] loading raw, unfiltered pool from rag.py...", file=sys.stderr)
    raw_pool = rag.load_pool(upload_cap=None, exclude_functionals=frozenset())
    print(f"[step1] raw pool: {len(raw_pool)} records", file=sys.stderr)
    raw_pool = defalsify_semi_empirical_basis_warning(raw_pool)

    for orca_only, label in [(True, "CURRENT (orca_only=True)"), (False, "PROPOSED (orca_only=False)")]:
        print(f"\n{'='*90}\n{label}\n{'='*90}", file=sys.stderr)
        records = load_records_in_memory(raw_pool, quality="strict", orca_only=orca_only)
        records = [r for r in records if r.get("specialist_cell") in PHASE1_CELLS]
        print(f"  {len(records)} records pass quality filter (phase1 cells only)", file=sys.stderr)

        records = apply_upload_cap(records, cap=DEFAULT_UPLOAD_CAP)
        train_raw, val_raw = split_by_upload(records)
        train = apply_caps(train_raw, ratio=DEFAULT_MAJORITY_RATIO)
        val = apply_caps(val_raw, ratio=DEFAULT_MAJORITY_RATIO)

        summarize(f"{label} -- TRAIN (post-cap, {len(train)} total)", train)
        summarize(f"{label} -- VAL (post-cap, {len(val)} total)", val)


if __name__ == "__main__":
    main()
