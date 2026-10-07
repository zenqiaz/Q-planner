"""
build_molecule_split.py

Rebuilds the organic_general / metal_general train/val split at MOLECULE
granularity instead of upload granularity, to fix a leakage mode the old
split allowed: split_by_upload() (generate_sft.py) keeps correlated batches
together, but a molecule reported by two DIFFERENT uploads (independent
NOMAD depositors studying the same well-known compound) could still land on
opposite sides of the split. That's exactly what powered the old
"memorized vs. novel" confusion (see project memory
rag_sft_smiles_novelty_confound.md) -- and a naive post-hoc SMILES-identity
re-check of that same confusion turned out to just reselect two known
corpus monocultures as "novel" (org_general's PBE0 mega-batch, metal_general's
tmQM/TPSSH batch) rather than fixing anything, because SMILES coverage
correlates with source, not because the split changed.

User's actual request (2026-08-05): don't cherry-pick a definition after the
fact -- rebuild the split itself so no molecule spans train/val, reshuffle,
and inspect label distribution BEFORE retraining (retraining is cheap; a
degenerate split is not obviously safe to train on without checking first).

Grouping key per record (see molecule_group_key() for the full rationale):
    - canonical RDKit SMILES, if the record has a parseable `smiles` field
      (this is the real fix -- ties every occurrence of the same molecule,
      across every upload/depositor, to one side of the split)
    - else a per-record singleton key, for the ~1/4-1/3 of records with no
      usable SMILES (mostly Gaussian) -- no duplicate-detection is possible
      for these without structural info, same residual risk profile the
      original official split had everywhere; documented, not hidden

Stratification: same as generate_sft.py's split_by_upload -- each group's
stratum is (specialist_cell, its most common functional), so every
functional with >=2 groups gets a proportional train/val split instead of
depending on which groups a shuffle happens to select.

This script does NOT retrain -- it only builds the new jsonl files and
prints a label-distribution report for review before that step.

Usage:
    python build_molecule_split.py [--seed N] [--out-dir jsonl_molecule_split]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAG_REPO = Path(r"D:\brick\D\20260217\working")
GENERATE_SFT_DIR = Path(r"D:\brick\D\202606\working\something_like MOSIAC")

# GENERATE_SFT_DIR has its own (older, shadowing) rag.py -- import rag from
# RAG_REPO first, then add GENERATE_SFT_DIR only for generate_sft itself.
sys.path.insert(0, str(RAG_REPO))
import rag  # noqa: E402

sys.path.insert(0, str(GENERATE_SFT_DIR))
import generate_sft as gs  # noqa: E402

try:
    from rdkit import Chem
except ImportError:
    Chem = None

PHASE1_CELLS = ["organic_general", "metal_general"]
OUT_BASE = Path(__file__).resolve().parent


def canonical_smiles(smiles: str | None) -> str | None:
    if not smiles or Chem is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def molecule_group_key(r: dict) -> str:
    """Molecule-identity grouping key: canonical SMILES when available (this
    is what actually prevents the same structure from spanning train/val).

    Records with no usable SMILES (mostly Gaussian, ~25-33% of the pool --
    see rag_smiles_fingerprint.md) fall back to a per-RECORD singleton key,
    not per-upload: grouping them by upload_id was tried first and found to
    make train/val proportions coarse and unpredictable (whole large Gaussian
    uploads swinging a whole functional's val share by 10+ points, since one
    "upload" group is a single atomic, indivisible unit for the split). Since
    there's no structural signal to detect true duplicates in this subset
    anyway, per-record grouping is not giving up any real protection --
    upload-level grouping wasn't detecting cross-upload duplicates either
    (that was exactly the original memorization-confusion mechanism this
    whole investigation started from). It does mean these specific records
    have no leakage guarantee, same residual risk profile the ORIGINAL
    official split had for its entire pool -- an honest, not a hidden,
    trade-off.
    """
    cs = canonical_smiles(r.get("smiles"))
    if cs:
        return f"smiles:{cs}"
    return f"record:{r.get('entry_id', '')}"


def split_by_molecule(records: list[dict], val_frac: float = 0.10, seed: int = 42
                       ) -> tuple[list[dict], list[dict]]:
    """Stratified by (specialist_cell, dominant functional), grouped by
    molecule identity (see module docstring) instead of upload_id alone.

    Record-count-weighted, not group-count-weighted: a plain "10% of groups
    per stratum" draw (the original version of this function, and
    generate_sft.py's split_by_upload()) can leave train and val with very
    different marginal functional proportions when group sizes vary a lot
    within a stratum (found 2026-08-05: organic_general train came out
    PBE0-majority, 48%, while val came out QCISD-majority, 53% -- both
    "healthy" by entropy, but not comparable to each other). Fixed by
    greedily filling each stratum's val allocation by RECORD count until it
    reaches ~val_frac of that stratum's total records, so val's per-functional
    share tracks train's, not just group presence/absence.
    """
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[molecule_group_key(r)].append(r)

    group_stratum: dict[str, tuple[str, str]] = {}
    for gid, recs in by_group.items():
        cell = Counter(r["specialist_cell"] for r in recs).most_common(1)[0][0]
        func = Counter(r.get("functional") for r in recs).most_common(1)[0][0]
        group_stratum[gid] = (cell, func)

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for gid, stratum in group_stratum.items():
        strata[stratum].append(gid)

    import random
    rng = random.Random(seed)
    val_set: set[str] = set()
    for stratum, gids in strata.items():
        gids = sorted(gids)
        rng.shuffle(gids)
        sizes = [len(by_group[g]) for g in gids]
        total = sum(sizes)
        if len(gids) < 2 or total < 2:
            continue
        target = val_frac * total

        chosen: list[str] = []
        cum = 0
        for gid, sz in zip(gids, sizes):
            if chosen and cum >= target:
                break
            # stop BEFORE adding a group if doing so overshoots the target
            # by more than skipping it would undershoot -- keeps val's
            # record share close to val_frac instead of always rounding up
            if chosen and abs(cum + sz - target) > abs(cum - target):
                break
            chosen.append(gid)
            cum += sz
        if len(chosen) >= len(gids):
            chosen = chosen[:-1]  # always leave >=1 group in train
        val_set.update(chosen)

    train_set = set(by_group) - val_set
    train = [r for gid in train_set for r in by_group[gid]]
    val = [r for gid in val_set for r in by_group[gid]]
    return train, val


def label_report(name: str, records: list[dict]) -> None:
    n = len(records)
    funcs = Counter(r.get("functional") for r in records)
    probs = [c / n for c in funcs.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    top5 = ", ".join(f"{f}={c}" for f, c in funcs.most_common(5))
    print(f"    {name:6s} n={n:5d}  unique_functionals={len(funcs):3d}  "
          f"entropy={entropy:.2f} bits  top5=[{top5}]")


def downsample_functional(records: list[dict], cell: str, functional: str,
                           keep_frac: float, seed: int) -> list[dict]:
    """Randomly drop a fraction of one (cell, functional)'s records before
    grouping/splitting -- used to de-dominate a single monoculture functional
    (e.g. metal_general's tmQM-sourced TPSSH, 81-82% of the pool) so it
    doesn't swamp label entropy and the eventual 'novel' bucket. Applied
    pre-split so the reduced count feeds group-level stratification too, not
    just a post-hoc reweighting."""
    import random
    rng = random.Random(seed)
    keep: list[dict] = []
    dropped = 0
    for r in records:
        if r["specialist_cell"] == cell and (r.get("functional") or "").upper() == functional:
            if rng.random() < keep_frac:
                keep.append(r)
            else:
                dropped += 1
        else:
            keep.append(r)
    print(f"[molecule-split] downsampled {cell}/{functional} to {keep_frac:.0%}: "
          f"dropped {dropped} records", file=sys.stderr)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="jsonl_molecule_split")
    ap.add_argument("--tpssh-keep-frac", type=float, default=0.5,
                     help="Fraction of metal_general TPSSH records to keep "
                          "before grouping/splitting (de-dominate the tmQM "
                          "monoculture). Set to 1.0 to disable.")
    ap.add_argument("--pbe0-keep-frac", type=float, default=0.5,
                     help="Fraction of organic_general PBE0 records to keep "
                          "(de-dominate the ORCA PBE0 mega-batch, mirrors "
                          "--tpssh-keep-frac). Set to 1.0 to disable.")
    ap.add_argument("--cot", action="store_true",
                     help="Chain-of-thought variant (insight #16, 2026-08-14): write CoT-"
                          "formatted records (cot_templates.py reasoning span + JSON, under "
                          "gs.COT_SYSTEM_PROMPT) instead of bare JSON -- same molecule-identity "
                          "split as the production dataset, varying only CoT vs. not, so a "
                          "later accuracy comparison isn't confounded by split quality. Records "
                          "with no matching cot_templates reasoning are dropped (see "
                          "gs.record_to_sft_cot()'s docstring). If --out-dir is left at its "
                          "default, switches it to jsonl_molecule_split_cot so the production "
                          "split is never overwritten.")
    args = ap.parse_args()
    if args.cot and args.out_dir == "jsonl_molecule_split":
        args.out_dir = "jsonl_molecule_split_cot"

    if Chem is None:
        raise RuntimeError("rdkit not installed -- pip install rdkit")

    print("[molecule-split] loading raw pool...", file=sys.stderr)
    raw_pool = rag.load_pool(upload_cap=None, exclude_functionals=frozenset())
    print(f"[molecule-split] raw pool: {len(raw_pool)} records", file=sys.stderr)

    # Same filter chain as generate_sft.load_records(), applied in-memory
    # instead of re-reading from a merged jsonl file.
    records = []
    n_excluded_lda = n_excluded_orca = n_unclean = n_rag_only = 0
    for r in raw_pool:
        cell = r.get("specialist_cell", "")
        if cell in gs.RAG_ONLY_CELLS:
            n_rag_only += 1
            continue
        func = (r.get("functional") or "").upper()
        if func in gs.EXCLUDED_FUNCTIONALS:
            n_excluded_lda += 1
            continue
        if func and func not in gs.ORCA_VALID_FUNCTIONALS:
            n_excluded_orca += 1
            continue
        if not gs.is_clean(r, "strict"):
            n_unclean += 1
            continue
        records.append(r)
    print(f"[molecule-split] after filters: {len(records)} records "
          f"(dropped {n_rag_only} rag-only, {n_excluded_lda} LDA, "
          f"{n_excluded_orca} non-ORCA-functional, {n_unclean} unclean)",
          file=sys.stderr)

    records = gs.apply_upload_cap(records, cap=gs.DEFAULT_UPLOAD_CAP, seed=args.seed)
    print(f"[molecule-split] after upload cap: {len(records)} records", file=sys.stderr)

    # restrict to the 2 phase-1 cells only, matching official scope
    records = [r for r in records if r["specialist_cell"] in PHASE1_CELLS]

    train_raw, val_raw = split_by_molecule(records, val_frac=0.10, seed=args.seed)
    print(f"[molecule-split] pre-balance split: {len(train_raw)} train / {len(val_raw)} val",
          file=sys.stderr)

    train = gs.apply_caps(train_raw, ratio=gs.DEFAULT_MAJORITY_RATIO, seed=args.seed)
    val = gs.apply_caps(val_raw, ratio=gs.DEFAULT_MAJORITY_RATIO, seed=args.seed)

    if args.tpssh_keep_frac < 1.0:
        # Applied AFTER apply_caps(), directly on the final train/val record
        # lists -- doing this pre-split instead gets diluted by apply_caps()'s
        # own independent re-sampling of the majority task_type (TPSSH is
        # metal_general's majority-task/SP-task functional too, so its budget
        # backfills from whatever TPSSH supply remains regardless of how much
        # was pre-filtered). Post-hoc, exact-fraction removal is what "remove
        # about half of TPSSH entries" actually means. Safe w.r.t. the
        # molecule-group leakage invariant: tmQM/TPSSH groups are ~1 record
        # each (each compound reported once), so dropping individual records
        # here doesn't split any multi-record group across train/val.
        train = downsample_functional(train, "metal_general", "TPSSH",
                                       args.tpssh_keep_frac, args.seed)
        val = downsample_functional(val, "metal_general", "TPSSH",
                                     args.tpssh_keep_frac, args.seed + 1)

    if args.pbe0_keep_frac < 1.0:
        train = downsample_functional(train, "organic_general", "PBE0",
                                       args.pbe0_keep_frac, args.seed + 2)
        val = downsample_functional(val, "organic_general", "PBE0",
                                     args.pbe0_keep_frac, args.seed + 3)

    out_dir = OUT_BASE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # cross-check: verify zero molecule-identity leakage for SMILES-covered records
    train_smiles = {canonical_smiles(r.get("smiles")) for r in train} - {None}
    leak = [r for r in val if canonical_smiles(r.get("smiles")) in train_smiles
            and canonical_smiles(r.get("smiles")) is not None]

    print(f"\n[molecule-split] leakage check: {len(leak)} val records share a canonical "
          f"SMILES with train (should be 0)")

    print(f"\n[molecule-split] label distribution report:")
    for cell in PHASE1_CELLS:
        print(f"\n  === {cell} ===")
        ctrain = [r for r in train if r["specialist_cell"] == cell]
        cval = [r for r in val if r["specialist_cell"] == cell]
        label_report("train", ctrain)
        label_report("val", cval)

        train_path = out_dir / f"sft_{cell}.jsonl"
        val_path = out_dir / f"sft_val_{cell}.jsonl"
        n_train_written = gs.write_jsonl(ctrain, train_path, cot=args.cot)
        n_val_written = gs.write_jsonl(cval, val_path, cot=args.cot)
        drop_note = (f" ({len(ctrain) - n_train_written} dropped, no cot_templates match)"
                     if args.cot and n_train_written < len(ctrain) else "")
        val_drop_note = (f" ({len(cval) - n_val_written} dropped)"
                          if args.cot and n_val_written < len(cval) else "")
        print(f"    wrote {train_path} ({n_train_written} of {len(ctrain)}{drop_note}) + "
              f"{val_path} ({n_val_written} of {len(cval)}{val_drop_note})")


if __name__ == "__main__":
    main()
