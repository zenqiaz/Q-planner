"""
build_rag_sft_experiment_inputs.py

GPU-free step 1 of the "does RAG add lift on top of the fine-tuned LoRA
specialist" experiment (organic_general + metal_general, phase1 cells).
Builds the full experiment INPUT set locally -- query records, their RAG
retrievals, and both prompt variants (SFT-alone / SFT+RAG) -- and writes it
to a local JSON file before any generation call runs on the cluster. This is
the "keep experiment data in a local file before we start" step; the actual
LoRA generation (needs GPU) is a separate script that reads this file.

Reuses the exact SFT prompt template from generate_sft.py (SYSTEM_PROMPT,
PARAM_FIELDS, record_to_sft's user-message format) and the exact RAG
retrieval + train/val split logic from this repo's rag.py /
rag_eval_holdout.py (both already fixed for hash-seed non-determinism and
unstratified-split bugs this session -- see project memory
rag_holdout_eval_and_tmqm_fix.md). Using the SAME pool/split for all three
conditions (RAG-alone, SFT-alone, SFT+RAG) is deliberate: it's what makes
the three numbers comparable to each other. The existing SFT-alone numbers
(90.0%/83.9% both-match) were measured on a DIFFERENT val sample
(sft_val_{cell}.jsonl, from generate_sft.py's own split) and are NOT directly
comparable to this experiment's SFT-alone condition -- this script
re-generates its own SFT-alone prompts on this experiment's shared val split
specifically so all three conditions are apples-to-apples.

RAG examples are injected as an ADDITIONAL system message (between the
original SYSTEM_PROMPT and the user's Molecule/Task line), matching the real
agent's planner message-stack convention (SKILL blocks are also separate
system messages, see CLAUDE.md's Pre-Planning Pipeline) rather than mixing
retrieved context into the user turn.

Usage:
    python build_rag_sft_experiment_inputs.py
    python build_rag_sft_experiment_inputs.py --k 5 --val-frac 0.10 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAG_REPO = Path(r"D:\brick\D\20260217\working")
sys.path.insert(0, str(RAG_REPO))

import rag  # noqa: E402
from rag_eval_holdout import _reclassify, split_by_upload  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = _reclassify(rag.load_pool())
    train_records, val_records = split_by_upload(pool, val_frac=args.val_frac, seed=args.seed)
    train_idx = rag.build_index(train_records)
    print(f"[build] pool={len(pool)} train={len(train_records)} val={len(val_records)}", file=sys.stderr)

    experiment_records = []
    n_by_cell = {}

    for rec in val_records:
        cell = rec.get("specialist_cell")
        if cell not in PHASE1_CELLS:
            continue
        n_by_cell[cell] = n_by_cell.get(cell, 0) + 1

        features = query_features(rec)
        hits = rag.query(train_idx, features, k=args.k, query_smiles=rec.get("smiles"))

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
    out_path = OUT_DIR / f"rag_sft_experiment_inputs_{ts}.json"
    payload = {
        "type": "rag_sft_experiment_inputs",
        "timestamp_utc": ts,
        "config": {"k": args.k, "val_frac": args.val_frac, "seed": args.seed,
                    "cells": PHASE1_CELLS},
        "pool_size": len(pool),
        "train_size": len(train_records),
        "val_size_total": len(val_records),
        "n_by_cell": n_by_cell,
        "records": experiment_records,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(f"\n[build] wrote {len(experiment_records)} experiment records -> {out_path}")
    for cell, n in n_by_cell.items():
        print(f"  {cell}: {n}")

    # Quick local sanity check: RAG-alone accuracy on this exact query set,
    # so the number we compare SFT-alone/SFT+RAG against is visible right away.
    for cell in PHASE1_CELLS:
        crecs = [r for r in experiment_records if r["specialist_cell"] == cell]
        if not crecs:
            continue
        acc = 100 * sum(r["rag_alone"]["top1_exact_correct"] for r in crecs) / len(crecs)
        print(f"  {cell}: RAG-alone top1_exact (functional+basis) = {acc:.1f}% (n={len(crecs)})")


if __name__ == "__main__":
    main()
