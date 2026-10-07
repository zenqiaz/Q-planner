"""
check_molecule_novelty_split.py

Re-derives the "memorized vs. novel" accuracy split (Chapter 5 / insight #7,
job 9692) using molecule identity (canonical SMILES) instead of exact
(prompt, functional, basis) string matching.

Why: the SFT prompt itself is coarse -- "Molecule: {formula}, {n_atoms} atoms,
{system_type}, charge={charge}, mult={multiplicity}\\nTask: {task_type}" -- it
contains no SMILES/InChI. The original "memorized" check (does this exact
prompt+label triple recur in train under a different entry_id) can misclassify
two different molecules that happen to share formula/n_atoms/charge/mult/task
as "the same", and can also undercount leakage: a val molecule whose SMILES
DOES appear in train but under a different true label was previously counted
as "novel" even though the model may have already seen that exact structure
during training.

This script instead:
  1. Reconstructs the exact official train/val split (same join as
     build_rag_sft_experiment_inputs_v2.py: sft_{cell}.jsonl entry_ids against
     rag.py's raw, unfiltered pool, which carries the `smiles` field).
  2. Canonicalizes SMILES with RDKit and builds, per cell, the set of
     canonical SMILES appearing anywhere in train.
  3. Classifies each val record as:
       - "seen"    -- canonical SMILES appears in train (any label)
       - "novel"   -- canonical SMILES does not appear in train
       - "no_id"   -- SMILES missing or RDKit-unparseable (can't judge)
  4. Joins against the v2 ablation results
     (rag_sft_experiment_results_20260803T075244Z.json) by entry_id to pull
     sft_alone / rag_alone accuracy, and reports accuracy by the new split.
  5. Cross-tabulates against the OLD prompt+label "memorized" definition to
     show exactly how much the two definitions disagree.

Usage:
    python check_molecule_novelty_split.py
    python check_molecule_novelty_split.py --results <results.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAG_REPO = Path(r"D:\brick\D\20260217\working")
sys.path.insert(0, str(RAG_REPO))

import rag  # noqa: E402

try:
    from rdkit import Chem
except ImportError:
    Chem = None

PHASE1_CELLS = ["organic_general", "metal_general"]
JSONL_DIR = Path(__file__).resolve().parent / "jsonl"
EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"
DEFAULT_RESULTS = EXPERIMENTS_DIR / "rag_sft_experiment_results_20260803T075244Z.json"


def official_entry_ids(cell: str) -> tuple[set[str], set[str]]:
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


def canonical_smiles(smiles: str | None) -> str | None:
    if not smiles or Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def old_memorized_key(r: dict) -> tuple:
    """Same (prompt, functional, basis) triple the v2 experiment used."""
    return (
        r.get("formula"), r.get("n_atoms"), r.get("system_type"),
        r.get("charge"), r.get("multiplicity"), r.get("task_type"),
        r.get("functional"), r.get("basis"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args()

    if Chem is None:
        raise RuntimeError("rdkit not installed -- pip install rdkit")

    print("[novelty-split] loading raw, unfiltered pool for entry_id lookup...", file=sys.stderr)
    raw_pool = rag.load_pool(upload_cap=None, exclude_functionals=frozenset())
    by_id = {r["entry_id"]: r for r in raw_pool if r.get("entry_id")}
    print(f"[novelty-split] raw pool: {len(raw_pool)} records, {len(by_id)} unique entry_ids",
          file=sys.stderr)

    print(f"[novelty-split] loading ablation results from {args.results}", file=sys.stderr)
    with args.results.open(encoding="utf-8") as f:
        results_payload = json.load(f)
    results_by_id = {r["entry_id"]: r for r in results_payload["results"]}

    overall_report: dict[str, dict] = {}

    for cell in PHASE1_CELLS:
        train_ids, val_ids = official_entry_ids(cell)
        missing = [i for i in (train_ids | val_ids) if i not in by_id]
        if missing:
            raise RuntimeError(f"{cell}: {len(missing)} entry_ids missing from raw pool")

        train_records = [by_id[i] for i in train_ids]
        val_records = [by_id[i] for i in val_ids]

        # canonical SMILES coverage + train identity set
        train_smiles_coverage = 0
        train_smiles_set: set[str] = set()
        for r in train_records:
            cs = canonical_smiles(r.get("smiles"))
            if cs:
                train_smiles_coverage += 1
                train_smiles_set.add(cs)

        val_smiles_coverage = 0
        buckets: dict[str, list[dict]] = defaultdict(list)  # "seen"/"novel"/"no_id"
        old_key_in_train: set[tuple] = {old_memorized_key(r) for r in train_records}

        cross_tab: Counter[tuple[str, str]] = Counter()  # (new_bucket, old_label)

        for r in val_records:
            cs = canonical_smiles(r.get("smiles"))
            old_memorized = old_memorized_key(r) in old_key_in_train
            old_label = "old_memorized" if old_memorized else "old_novel"

            if cs is None:
                bucket = "no_id"
            elif cs in train_smiles_set:
                bucket = "seen"
                val_smiles_coverage += 1
            else:
                bucket = "novel"
                val_smiles_coverage += 1

            buckets[bucket].append(r)
            cross_tab[(bucket, old_label)] += 1

        # accuracy per bucket, joined against ablation results
        bucket_stats = {}
        for bucket, recs in buckets.items():
            joined = []
            for r in recs:
                res = results_by_id.get(r["entry_id"])
                if res is None:
                    continue
                joined.append(res)
            n = len(joined)
            if n == 0:
                bucket_stats[bucket] = {"n": 0}
                continue
            sft_alone_acc = 100 * sum(
                1 for j in joined if j["sft_alone"]["both_match"]
            ) / n
            rag_alone_acc = 100 * sum(
                1 for j in joined if j["rag_alone"]["top1_exact_correct"]
            ) / n
            bucket_stats[bucket] = {
                "n": n,
                "sft_alone_both_match_pct": round(sft_alone_acc, 1),
                "rag_alone_top1_pct": round(rag_alone_acc, 1),
            }

        print(f"\n=== {cell} ===")
        print(f"  train SMILES coverage: {train_smiles_coverage}/{len(train_records)} "
              f"({100*train_smiles_coverage/len(train_records):.1f}%), "
              f"{len(train_smiles_set)} unique canonical structures")
        print(f"  val SMILES coverage:   {val_smiles_coverage}/{len(val_records)} "
              f"({100*val_smiles_coverage/len(val_records):.1f}%)")
        print(f"  new split (by molecule identity):")
        for bucket in ("seen", "novel", "no_id"):
            st = bucket_stats.get(bucket, {"n": 0})
            print(f"    {bucket:8s} n={st['n']:4d}  "
                  f"sft_alone={st.get('sft_alone_both_match_pct', '--')}%  "
                  f"rag_alone={st.get('rag_alone_top1_pct', '--')}%")
        print(f"  cross-tab vs. old (prompt+label) memorized/novel definition:")
        for bucket in ("seen", "novel", "no_id"):
            for old_label in ("old_memorized", "old_novel"):
                c = cross_tab.get((bucket, old_label), 0)
                if c:
                    print(f"    new={bucket:8s} old={old_label:14s}  n={c}")

        overall_report[cell] = {
            "train_smiles_coverage": train_smiles_coverage,
            "train_total": len(train_records),
            "val_smiles_coverage": val_smiles_coverage,
            "val_total": len(val_records),
            "bucket_stats": bucket_stats,
            "cross_tab": {f"{k[0]}|{k[1]}": v for k, v in cross_tab.items()},
        }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENTS_DIR / "molecule_novelty_split_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(overall_report, f, indent=2)
    print(f"\n[novelty-split] wrote {out_path}")


if __name__ == "__main__":
    main()
