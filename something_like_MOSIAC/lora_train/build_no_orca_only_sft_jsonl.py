"""
build_no_orca_only_sft_jsonl.py

Writes the actual train/val JSONL files for the orca_only=False corpus
(entropy-checked in build_no_orca_only_corpus.py, moderate not extreme
increase -- see project memory rag_sft_generalization_gap.md and the
chapter plan's insight #9). Same pipeline as that script, but this time
calls record_to_sft() and writes output, instead of just reporting stats.

Writes to a SEPARATE data directory (jsonl_no_orca/) so the existing
official sft_{cell}.jsonl / sft_val_{cell}.jsonl files (and the adapters
trained on them) are never touched.

Usage:
    python build_no_orca_only_sft_jsonl.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\brick\D\202606\working\something_like MOSIAC")
sys.path.insert(0, r"D:\brick\D\20260217\working")

from generate_sft import (  # noqa: E402
    EXCLUDED_FUNCTIONALS, ORCA_VALID_FUNCTIONALS, RAG_ONLY_CELLS,
    DEFAULT_UPLOAD_CAP, DEFAULT_MAJORITY_RATIO,
    is_clean, apply_upload_cap, split_by_upload, apply_caps, record_to_sft,
)
import rag  # noqa: E402

PHASE1_CELLS = ["organic_general", "metal_general"]
OUT_DIR = Path(__file__).resolve().parent / "jsonl_no_orca"


def defalsify_semi_empirical_basis_warning(records: list[dict]) -> list[dict]:
    out = []
    n_fixed = 0
    for r in records:
        if (r.get("method_family") == "SEMI_EMPIRICAL"
                and "basis not identified" in (r.get("parse_warnings") or [])):
            r = dict(r)
            r["parse_warnings"] = [w for w in r["parse_warnings"] if w != "basis not identified"]
            n_fixed += 1
        out.append(r)
    print(f"[build] de-falsified 'basis not identified' warning on {n_fixed} "
          f"semi-empirical records", file=sys.stderr)
    return out


def load_records_in_memory(records: list[dict], quality: str, orca_only: bool) -> list[dict]:
    out = []
    n_lda = n_orca = 0
    for r in records:
        if r.get("specialist_cell", "") in RAG_ONLY_CELLS:
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
    print(f"[build] excluded {n_lda} LDA, {n_orca} non-ORCA (orca_only={orca_only})",
          file=sys.stderr)
    return out


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(record_to_sft(r)) + "\n")


def main():
    print("[build] loading raw, unfiltered pool from rag.py...", file=sys.stderr)
    raw_pool = rag.load_pool(upload_cap=None, exclude_functionals=frozenset())
    raw_pool = defalsify_semi_empirical_basis_warning(raw_pool)
    print(f"[build] raw pool: {len(raw_pool)} records", file=sys.stderr)

    records = load_records_in_memory(raw_pool, "strict", orca_only=False)
    records = [r for r in records if r.get("specialist_cell") in PHASE1_CELLS]
    print(f"[build] {len(records)} records pass quality filter (phase1 cells only)",
          file=sys.stderr)

    records = apply_upload_cap(records, cap=DEFAULT_UPLOAD_CAP)
    train_raw, val_raw = split_by_upload(records)
    train = apply_caps(train_raw, ratio=DEFAULT_MAJORITY_RATIO)
    val = apply_caps(val_raw, ratio=DEFAULT_MAJORITY_RATIO)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cell in PHASE1_CELLS:
        cell_train = [r for r in train if r.get("specialist_cell") == cell]
        cell_val = [r for r in val if r.get("specialist_cell") == cell]
        write_jsonl(cell_train, OUT_DIR / f"sft_{cell}.jsonl")
        write_jsonl(cell_val, OUT_DIR / f"sft_val_{cell}.jsonl")
        print(f"[build] {cell}: wrote {len(cell_train)} train / {len(cell_val)} val "
              f"-> {OUT_DIR}", file=sys.stderr)

    print(f"\n[build] done -- data dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
