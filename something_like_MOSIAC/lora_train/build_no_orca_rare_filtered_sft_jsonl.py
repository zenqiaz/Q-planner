"""
build_no_orca_rare_filtered_sft_jsonl.py

Follow-up to build_no_orca_only_sft_jsonl.py's orca_only=False retrain (job 9903),
which regressed organic_general to 74.5% both-match (from the official 90.0%).
Root-cause breakdown (job 14046) found PM6 itself is learned PERFECTLY (100%,
n=60) -- the regression is a long-tail class-imbalance problem: every functional
with <20 training examples scored 0% val accuracy, because upload-level train/val
splitting left some of them with as few as 1 training example (e.g. MPW1PW91: 1
train example, 13 val instances -- never really seen during training).

RARE_FUNCTIONALS below is the exact set with <20 examples in organic_general's
orca_only=False train pool (confirmed via direct count, user-approved threshold).
Excluding them BEFORE the train/val split (not after) keeps train and val
consistent -- a functional dropped from train is also dropped from val, so we
never score against a functional the model had ~0 chance to learn.

PM6 (410 train examples) is unaffected by this filter -- it stays in. Only
organic_general needs this; metal_general's entropy barely moved under
orca_only=False and its retrain already passed (84.3%), so it is not touched.

Writes to a THIRD, separate data directory (jsonl_no_orca_rare_filtered/) --
neither the official jsonl/ nor the first no-orca jsonl_no_orca/ are touched.

Usage:
    python build_no_orca_rare_filtered_sft_jsonl.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\brick\D\202606\working\something_like MOSIAC")
sys.path.insert(0, r"D:\brick\D\20260217\working")

from generate_sft import (  # noqa: E402
    DEFAULT_UPLOAD_CAP, DEFAULT_MAJORITY_RATIO,
    apply_upload_cap, split_by_upload, apply_caps, record_to_sft,
)
import rag  # noqa: E402
from build_no_orca_only_corpus import (  # noqa: E402
    defalsify_semi_empirical_basis_warning, load_records_in_memory,
)

CELL = "organic_general"
OUT_DIR = Path(__file__).resolve().parent / "jsonl_no_orca_rare_filtered"

# Exact set with <20 training examples in the orca_only=False organic_general
# train pool, confirmed by direct count (2026-08-04): AM1(1), B1B95(4),
# B2PLYP(9), B3PW91(3), BE1PBE(2), CAM-B3LYP(5), CCSD(T)(1), HF(7), HSE06(5),
# LC-WPBE(2), M06(17), M062X(18), MP2(3), MPW1PW91(1), WB97X-D(1).
RARE_FUNCTIONALS = frozenset({
    "AM1", "B1B95", "B2PLYP", "B3PW91", "BE1PBE", "CAM-B3LYP", "CCSD(T)",
    "HF", "HSE06", "LC-WPBE", "M06", "M062X", "MP2", "MPW1PW91", "WB97X-D",
})


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
    records = [r for r in records if r.get("specialist_cell") == CELL]
    print(f"[build] {len(records)} records pass quality filter (orca_only=False, {CELL} only)",
          file=sys.stderr)

    before = len(records)
    records = [r for r in records
               if (r.get("functional") or "").upper() not in RARE_FUNCTIONALS]
    print(f"[build] rare-functional filter: dropped {before - len(records)} records "
          f"({', '.join(sorted(RARE_FUNCTIONALS))})", file=sys.stderr)

    records = apply_upload_cap(records, cap=DEFAULT_UPLOAD_CAP)
    train_raw, val_raw = split_by_upload(records)
    train = apply_caps(train_raw, ratio=DEFAULT_MAJORITY_RATIO)
    val = apply_caps(val_raw, ratio=DEFAULT_MAJORITY_RATIO)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, OUT_DIR / f"sft_{CELL}.jsonl")
    write_jsonl(val, OUT_DIR / f"sft_val_{CELL}.jsonl")
    print(f"[build] {CELL}: wrote {len(train)} train / {len(val)} val -> {OUT_DIR}",
          file=sys.stderr)

    print(f"\n[build] done -- data dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
