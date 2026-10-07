"""
build_rag_sft_experiment_inputs_v2.py

Fixes the dataset-composition confound found in the v1 ablation experiment
(see project memory rag_sft_generalization_gap.md, "Bigger finding" section):
v1 sourced its query/train pool from rag.py's load_pool() (only LDA excluded,
50/upload cap), which has materially different label composition than the
generate_sft.py pipeline the deployed adapters were actually trained on (full
ORCA_VALID_FUNCTIONALS whitelist, 400/upload/cell cap) -- e.g. organic_general's
PBE0 share was 5.3% in v1's val vs 75.8% in the official split.

v2 fixes this by reconstructing the EXACT official train/val split with full
canonical fields (elements, n_atoms, charge, multiplicity, solvent, task_type --
needed for RAG querying but stripped out of the sft_*.jsonl files themselves):
join sft_{cell}.jsonl / sft_val_{cell}.jsonl's entry_ids against rag.py's raw,
unfiltered pool (load_pool(upload_cap=None, exclude_functionals=frozenset())),
which has 100% entry_id coverage of both official files (verified before writing
this script). This gives the TRUE train set the adapters were fine-tuned on and
the TRUE held-out val set they were validated on, with orca_only=True consistent
with how the current adapters were actually trained -- no retraining involved,
this only fixes which records define "true" labels for the ablation comparison.

Usage:
    python build_rag_sft_experiment_inputs_v2.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RAG_REPO = Path(r"D:\brick\D\20260217\working")
sys.path.insert(0, str(RAG_REPO))

import rag  # noqa: E402

SYSTEM_PROMPT = (
    "You are a QC parameter specialist. Given a molecule description and task, "
    "output ONLY a JSON object with the ORCA calculation parameters."
)

PARAM_FIELDS = [
    "functional", "basis", "ri_approx", "aux_basis",
    "grid_level", "final_grid_level", "dispersion", "scf_convergence",
    "relativistic", "solvent_model", "solvent",
    "nroots", "casscf_nel", "casscf_norb", "cbs_scheme",
]

PHASE1_CELLS = ["organic_general", "metal_general"]

JSONL_DIR = Path(__file__).resolve().parent / "jsonl"
OUT_DIR = Path(__file__).resolve().parent / "experiments"


def user_content(r: dict) -> str:
    """Exact user-message format from generate_sft.py's record_to_sft()."""
    return (
        f"Molecule: {r.get('formula') or 'unknown'}, {r.get('n_atoms') or '?'} atoms, "
        f"{r.get('system_type')}, charge={r.get('charge')}, mult={r.get('multiplicity')}\n"
        f"Task: {r.get('task_type')}"
    )


def gold_params(r: dict) -> dict:
    return {k: r[k] for k in PARAM_FIELDS if r.get(k) is not None}


def query_features(r: dict) -> dict:
    return {
        "elements": r.get("elements") or [],
        "n_atoms": r.get("n_atoms"),
        "charge": r.get("charge"),
        "multiplicity": r.get("multiplicity"),
        "solvent": r.get("solvent"),
        "task_type": r.get("task_type"),
        "system_type": r.get("system_type"),
        "specialist_cell": r.get("specialist_cell"),
    }


def official_entry_ids(cell: str) -> tuple[set[str], set[str]]:
    """Entry IDs of the ACTUAL train/val split generate_sft.py produced for `cell`."""
    def ids(path: Path) -> set[str]:
        out = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out.add(r["metadata"]["entry_id"])
        return out

    train_ids = ids(JSONL_DIR / f"sft_{cell}.jsonl")
    val_ids = ids(JSONL_DIR / f"sft_val_{cell}.jsonl")
    return train_ids, val_ids


def main():
    print("[build v2] loading raw, unfiltered pool for entry_id lookup...", file=sys.stderr)
    raw_pool = rag.load_pool(upload_cap=None, exclude_functionals=frozenset())
    by_id = {r["entry_id"]: r for r in raw_pool if r.get("entry_id")}
    print(f"[build v2] raw pool: {len(raw_pool)} records, {len(by_id)} unique entry_ids",
          file=sys.stderr)

    train_records: list[dict] = []
    val_records: list[dict] = []
    for cell in PHASE1_CELLS:
        train_ids, val_ids = official_entry_ids(cell)
        missing_train = [i for i in train_ids if i not in by_id]
        missing_val = [i for i in val_ids if i not in by_id]
        if missing_train or missing_val:
            raise RuntimeError(
                f"{cell}: {len(missing_train)} train / {len(missing_val)} val entry_ids "
                "not found in raw pool -- lookup assumption broken, do not proceed silently"
            )
        # Force specialist_cell to the cell this entry_id's official file says it
        # belongs to -- do NOT trust raw_pool's own specialist_cell field, which
        # can be stale relative to classify_cell() (same bug class as
        # rag_eval_holdout.py's _reclassify(), found independently here: 6
        # organic_general entries carried a drifted "highlevel_SP" label from the
        # raw pool on first run of this script).
        for i in train_ids:
            rec = dict(by_id[i])
            rec["specialist_cell"] = cell
            train_records.append(rec)
        for i in val_ids:
            rec = dict(by_id[i])
            rec["specialist_cell"] = cell
            val_records.append(rec)
        print(f"[build v2] {cell}: official train={len(train_ids)} val={len(val_ids)}, "
              f"100% resolved against raw pool", file=sys.stderr)

    train_idx = rag.build_index(train_records)
    print(f"[build v2] RAG index built from {len(train_records)} official-train records "
          f"(exact set the current adapters were fine-tuned on)", file=sys.stderr)

    experiment_records = []
    n_by_cell: dict[str, int] = {}

    for rec in val_records:
        cell = rec.get("specialist_cell")
        n_by_cell[cell] = n_by_cell.get(cell, 0) + 1

        features = query_features(rec)
        hits = rag.query(train_idx, features, k=5, query_smiles=rec.get("smiles"))

        gold = gold_params(rec)
        base_user = user_content(rec)
        rag_block = rag.format_examples(hits) if hits else ""

        sft_alone_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": base_user},
        ]
        sft_rag_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + ([{"role": "system", "content": rag_block}] if rag_block else []) + [
            {"role": "user", "content": base_user},
        ]

        rag_alone_top1 = hits[0] if hits else None
        rag_alone_correct = bool(
            rag_alone_top1
            and rag_alone_top1.get("functional") == rec.get("functional")
            and rag_alone_top1.get("basis") == rec.get("basis")
        )

        experiment_records.append({
            "entry_id": rec.get("entry_id"),
            "specialist_cell": cell,
            "gold": gold,
            "true_functional": rec.get("functional"),
            "true_basis": rec.get("basis"),
            "n_retrieved": len(hits),
            "rag_alone": {
                "top1_functional": rag_alone_top1.get("functional") if rag_alone_top1 else None,
                "top1_basis": rag_alone_top1.get("basis") if rag_alone_top1 else None,
                "top1_exact_correct": rag_alone_correct,
            },
            "sft_alone_messages": sft_alone_messages,
            "sft_rag_messages": sft_rag_messages,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = OUT_DIR / f"rag_sft_experiment_inputs_v2_{ts}.json"
    payload = {
        "type": "rag_sft_experiment_inputs_v2",
        "note": "query/train pool = exact official generate_sft.py split (orca_only=True), "
                "joined via entry_id against rag.py's raw pool -- fixes v1's dataset-"
                "composition confound (see rag_sft_generalization_gap.md)",
        "timestamp_utc": ts,
        "config": {"k": 5, "cells": PHASE1_CELLS},
        "train_size": len(train_records),
        "val_size_total": len(val_records),
        "n_by_cell": n_by_cell,
        "records": experiment_records,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(f"\n[build v2] wrote {len(experiment_records)} experiment records -> {out_path}")
    for cell, n in n_by_cell.items():
        print(f"  {cell}: {n}")

    for cell in PHASE1_CELLS:
        crecs = [r for r in experiment_records if r["specialist_cell"] == cell]
        if not crecs:
            continue
        acc = 100 * sum(r["rag_alone"]["top1_exact_correct"] for r in crecs) / len(crecs)
        print(f"  {cell}: RAG-alone top1_exact (functional+basis) = {acc:.1f}% (n={len(crecs)})")


if __name__ == "__main__":
    main()
