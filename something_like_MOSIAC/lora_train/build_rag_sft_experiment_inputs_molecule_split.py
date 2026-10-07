"""
build_rag_sft_experiment_inputs_molecule_split.py

Same design as build_rag_sft_experiment_inputs_v2.py (join sft_{cell}.jsonl /
sft_val_{cell}.jsonl entry_ids against rag.py's raw, unfiltered pool to
recover full structured fields for RAG querying), but pointed at the
molecule-identity-level split (jsonl_molecule_split/, see project memory
rag_sft_smiles_novelty_confound.md) instead of the official upload-level
split. The RAG index is built from THIS split's train records, so retrieval
is scoped to the same corpus the molecule-split adapters were fine-tuned on.

Also emits a `train_sample_records` list: a seeded random sample of TRAIN
records (same size as that cell's val set, capped) with sft_alone_messages
+ gold, for the memorization-vs-generalization check. Because this split is
zero-leakage by construction, "trained on this exact record" is well-defined
here (unlike the old upload-level split, where near-duplicate prompts across
different entry_ids made "memorized" ambiguous) -- these ARE the exact
records the adapter was fine-tuned on, so SFT-alone accuracy on this sample
is the memorized-subset number, directly comparable to accuracy on
`records` (val, entirely novel molecules by construction).

Usage:
    python build_rag_sft_experiment_inputs_molecule_split.py
"""
from __future__ import annotations

import json
import random
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
TRAIN_SAMPLE_SEED = 42
TRAIN_SAMPLE_CAP = 300  # keep GPU time reasonable; val sets are 208/295 anyway

JSONL_DIR = Path(__file__).resolve().parent / "jsonl_molecule_split"
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
    """Entry IDs of the molecule-split train/val files for `cell`."""
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
    print("[build molsplit] loading raw, unfiltered pool for entry_id lookup...", file=sys.stderr)
    raw_pool = rag.load_pool(upload_cap=None, exclude_functionals=frozenset())
    by_id = {r["entry_id"]: r for r in raw_pool if r.get("entry_id")}
    print(f"[build molsplit] raw pool: {len(raw_pool)} records, {len(by_id)} unique entry_ids",
          file=sys.stderr)

    train_records: list[dict] = []
    val_records: list[dict] = []
    train_ids_by_cell: dict[str, list[str]] = {}
    for cell in PHASE1_CELLS:
        train_ids, val_ids = official_entry_ids(cell)
        missing_train = [i for i in train_ids if i not in by_id]
        missing_val = [i for i in val_ids if i not in by_id]
        if missing_train or missing_val:
            raise RuntimeError(
                f"{cell}: {len(missing_train)} train / {len(missing_val)} val entry_ids "
                "not found in raw pool -- lookup assumption broken, do not proceed silently"
            )
        train_ids_by_cell[cell] = sorted(train_ids)
        for i in train_ids:
            rec = dict(by_id[i])
            rec["specialist_cell"] = cell
            train_records.append(rec)
        for i in val_ids:
            rec = dict(by_id[i])
            rec["specialist_cell"] = cell
            val_records.append(rec)
        print(f"[build molsplit] {cell}: molecule-split train={len(train_ids)} val={len(val_ids)}, "
              f"100% resolved against raw pool", file=sys.stderr)

    train_idx = rag.build_index(train_records)
    print(f"[build molsplit] RAG index built from {len(train_records)} molecule-split-train records "
          f"(exact set the molecule-split adapters were fine-tuned on)", file=sys.stderr)

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

    # --- memorization check: seeded sample of TRAIN records per cell ---
    rng = random.Random(TRAIN_SAMPLE_SEED)
    train_sample_records = []
    train_sample_n_by_cell: dict[str, int] = {}
    by_id_and_cell = {(r["entry_id"], r["specialist_cell"]): r for r in train_records}
    for cell in PHASE1_CELLS:
        ids = list(train_ids_by_cell[cell])
        rng.shuffle(ids)
        sample_n = min(TRAIN_SAMPLE_CAP, n_by_cell.get(cell, 0) or TRAIN_SAMPLE_CAP, len(ids))
        sampled_ids = ids[:sample_n]
        train_sample_n_by_cell[cell] = len(sampled_ids)
        for i in sampled_ids:
            rec = by_id_and_cell[(i, cell)]
            base_user = user_content(rec)
            train_sample_records.append({
                "entry_id": rec.get("entry_id"),
                "specialist_cell": cell,
                "gold": gold_params(rec),
                "true_functional": rec.get("functional"),
                "true_basis": rec.get("basis"),
                "sft_alone_messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": base_user},
                ],
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = OUT_DIR / f"rag_sft_experiment_inputs_molecule_split_{ts}.json"
    payload = {
        "type": "rag_sft_experiment_inputs_molecule_split",
        "note": "query/train pool = molecule-identity-level split (jsonl_molecule_split/, "
                "zero SMILES leakage by construction), joined via entry_id against rag.py's "
                "raw pool -- mirrors build_rag_sft_experiment_inputs_v2.py's methodology "
                "applied to the new split. train_sample_records is a seeded sample of TRAIN "
                "records (memorized-subset check) directly comparable to `records` (val, "
                "entirely novel by construction).",
        "timestamp_utc": ts,
        "config": {"k": 5, "cells": PHASE1_CELLS, "train_sample_seed": TRAIN_SAMPLE_SEED,
                   "train_sample_cap": TRAIN_SAMPLE_CAP},
        "train_size": len(train_records),
        "val_size_total": len(val_records),
        "n_by_cell": n_by_cell,
        "train_sample_n_by_cell": train_sample_n_by_cell,
        "records": experiment_records,
        "train_sample_records": train_sample_records,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(f"\n[build molsplit] wrote {len(experiment_records)} val records + "
          f"{len(train_sample_records)} train-sample records -> {out_path}")
    for cell, n in n_by_cell.items():
        print(f"  {cell}: val={n}  train_sample={train_sample_n_by_cell.get(cell)}")

    for cell in PHASE1_CELLS:
        crecs = [r for r in experiment_records if r["specialist_cell"] == cell]
        if not crecs:
            continue
        acc = 100 * sum(r["rag_alone"]["top1_exact_correct"] for r in crecs) / len(crecs)
        print(f"  {cell}: RAG-alone top1_exact (functional+basis) = {acc:.1f}% (n={len(crecs)})")


if __name__ == "__main__":
    main()
