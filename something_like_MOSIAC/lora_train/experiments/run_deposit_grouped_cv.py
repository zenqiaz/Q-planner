"""Deposit-grouped versus random cross-validation for the precedent-matching task.

Diagnostic for: does a classical selector's advantage over a constant survive when no deposit
(upload_id) appears on both sides of the split?

Design notes, all of which exist because an earlier version got them wrong:

  * MATCHED POPULATIONS. The split effect must be measured on the same rows. The grouped run has
    to drop rows with a blank upload_id, so a random run is reported on that same eligible subset
    alongside the all-rows random run. Comparing all-rows-random against eligible-rows-grouped
    conflates a change of split with a change of population.
  * Blank upload_id is NOT a deposit. An earlier version mapped every blank to "NA", fusing 124
    metal rows into one fabricated deposit. All 124 of those rows carry the malformed label
    B3LYP/b3lyp (a functional token in the basis field), so the primary treatment excludes them;
    a singleton sensitivity is also reported, and it admits those malformed labels.
  * FEATURE FIDELITY with run_classical_baseline_molecule_split.py: missing atom count buckets to
    "unknown" (not "S" -- that bug moved 353 organic rows), category vocabularies are fitted on
    the TRAINING fold only, and k-NN is given StandardScaler-scaled features fitted on the
    training fold, as the original baseline does. Trees are scale-invariant so RF uses raw values.
  * Label support is reported separately from accuracy: a closed-set classifier cannot emit a
    label absent from its training fold, so micro accuracy conflates "could not possibly be right"
    with "was wrong". NOTE this ceiling binds RF and k-NN; a generative selector is not subject
    to it.
  * The constant comparator is chosen on the TRAINING fold and scored on the test fold.

Scope: the sampled, lightly filtered corpus (LDA and null functional/basis removed only). This
does NOT apply the production SFT loader's ORCA-functional, placeholder-basis, element or quality
filters, so malformed labels such as a basis of "7=3)" survive in organic_general.

An upload is not a study: several uploads can belong to one deposit, and the merged parse splits
tmQM's single deposit into 50 synthetic upload IDs. Upload grouping is therefore a weaker control
than an independent-study holdout.

CPU only. No GPU, no DFT.  Usage:  python run_deposit_grouped_cv.py [--pool PATH] [--out PATH]
"""
from __future__ import annotations

import argparse, collections, hashlib, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DEFAULT_POOLS = [
    Path(r"D:\brick\D\202606\working\something_like MOSIAC\jsonl\methods.jsonl"),
    Path(r"D:\brick\D\20260217\working\something_like_MOSIAC\jsonl\methods.jsonl"),
]
SEED = 42
N_SPLITS = 5
EXCLUDED_FUNCTIONALS = {"LDA"}
ORGANIC_ELEMENTS = ["C", "H", "N", "O", "F"]
CELLS = ["organic_general", "metal_general"]
US = chr(31)
BUCKETS = ["S", "M", "L", "XL", "unknown"]


def scale_bucket(n_atoms):
    """Matches run_classical_baseline_molecule_split.py exactly: falsy -> 'unknown'."""
    if not n_atoms:
        return "unknown"
    if n_atoms <= 10:
        return "S"
    if n_atoms <= 30:
        return "M"
    if n_atoms <= 100:
        return "L"
    return "XL"


def fit_vocab(cell, train_records):
    """Matches FeatureBuilder.fit: organic uses a fixed element list, metal derives from train."""
    if cell == "organic_general":
        ev = list(ORGANIC_ELEMENTS)
    else:
        ev = sorted({e for r in train_records for e in (r.get("elements") or [])})
    tv = sorted({r.get("task_type") or "UNKNOWN" for r in train_records})
    return ev, tv


def transform(records, ev, tv):
    """Matches FeatureBuilder.transform."""
    rows = []
    for r in records:
        n_atoms = r.get("n_atoms") or 0
        elements = set(r.get("elements") or [])
        task = r.get("task_type") or "UNKNOWN"
        row = [np.log1p(max(n_atoms, 0)), float(r.get("charge") or 0),
               1.0 if (r.get("multiplicity") or 1) != 1 else 0.0,
               1.0 if (r.get("solvent") or "").strip() else 0.0]
        row += [1.0 if e in elements else 0.0 for e in ev]
        row += [1.0 if task == t else 0.0 for t in tv]
        b = scale_bucket(r.get("n_atoms"))
        row += [1.0 if b == x else 0.0 for x in BUCKETS]
        rows.append(row)
    return np.array(rows, dtype=np.float64)


def evaluate(cell, records, y, groups, splits, label):
    """Per-fold vocabulary fitting; RF on raw features, k-NN on train-fitted scaled features."""
    per_fold, agg = [], collections.Counter()
    for i, (tr, te) in enumerate(splits):
        if len(set(y[tr])) < 2 or len(te) == 0:
            continue
        rtr = [records[j] for j in tr]
        rte = [records[j] for j in te]
        ev, tv = fit_vocab(cell, rtr)
        Xtr, Xte = transform(rtr, ev, tv), transform(rte, ev, tv)
        sc = StandardScaler().fit(Xtr)
        seen = set(y[tr])
        covered = np.array([v in seen for v in y[te]])
        row = {"fold": i, "n_train": int(len(tr)), "n_test": int(len(te)),
               "n_features": int(Xtr.shape[1]),
               "n_train_groups": int(len(set(groups[tr]))) if groups is not None else None,
               "coverage": float(covered.mean())}
        src = collections.Counter((records[j].get("source_code") or "?") for j in te)
        row["test_sources"] = dict(src.most_common())
        for nm in ("rf", "knn"):
            if nm == "rf":
                m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
                a, b = Xtr, Xte
            else:
                m = KNeighborsClassifier(n_neighbors=1)
                a, b = sc.transform(Xtr), sc.transform(Xte)
            m.fit(a, y[tr])
            ok = m.predict(b) == y[te]
            row[nm] = float(ok.mean())
            row[nm + "_given_seen"] = float(ok[covered].mean()) if covered.any() else None
            agg[nm] += int(ok.sum())
            agg[nm + "_seen_ok"] += int(ok[covered].sum())
        const = collections.Counter(y[tr]).most_common(1)[0][0]
        ok_c = (y[te] == const)
        row["constant"] = float(ok_c.mean())
        row["constant_label"] = const.replace(US, "/")
        agg["constant"] += int(ok_c.sum())
        agg["n"] += int(len(te))
        agg["covered"] += int(covered.sum())
        per_fold.append(row)
    n, cov = agg["n"], agg["covered"]
    out = {"label": label, "n_scored": n, "n_folds": len(per_fold), "coverage": cov / n,
           "rf": 100 * agg["rf"] / n, "knn": 100 * agg["knn"] / n,
           "constant": 100 * agg["constant"] / n,
           "rf_given_seen": (100 * agg["rf_seen_ok"] / cov) if cov else None,
           "knn_given_seen": (100 * agg["knn_seen_ok"] / cov) if cov else None,
           "per_fold": per_fold}
    gs = out["rf_given_seen"]
    print(f"  {label:38s} n={n:5d}  RF {out['rf']:5.1f}%  1NN {out['knn']:5.1f}%  "
          f"const {out['constant']:5.1f}%  | cover {100*out['coverage']:5.1f}%  "
          f"RF|seen {gs:5.1f}%" if gs is not None
          else f"  {label:38s} n={n:5d}  RF {out['rf']:5.1f}%")
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None, help="explicit path to methods.jsonl")
    ap.add_argument("--out", default="deposit_grouped_cv.json")
    args = ap.parse_args()

    if args.pool:
        pool = Path(args.pool)
        if not pool.exists():
            print(f"ERROR: --pool not found: {pool}", file=sys.stderr); return 1
    else:
        pool = next((p for p in DEFAULT_POOLS if p.exists()), None)
        if pool is None:
            print("ERROR: no methods.jsonl found; pass --pool", file=sys.stderr); return 1
        print("NOTE: --pool not given; using the first default that exists.")
    rows = [json.loads(l) for l in pool.open(encoding="utf-8") if l.strip()]
    me = Path(__file__).resolve()
    print(f"pool: {pool}\n  {len(rows):,} records   sha256={sha256(pool)[:16]}...")
    print(f"script sha256={sha256(me)[:16]}...\n")

    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "pool": str(pool), "pool_sha256": sha256(pool), "pool_rows": len(rows),
              "script": str(me), "script_sha256": sha256(me),
              "seed": SEED, "n_splits": N_SPLITS,
              "excluded_functionals": sorted(EXCLUDED_FUNCTIONALS),
              "scope_note": "sampled corpus; LDA and null functional/basis removed only; "
                            "production SFT filters NOT applied",
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "sklearn": sklearn.__version__},
              "cells": {}}

    for cell in CELLS:
        f = {"in_cell": 0, "dropped_lda": 0, "dropped_null_label": 0, "blank_upload": 0}
        kept = []
        for r in rows:
            if r.get("specialist_cell") != cell:
                continue
            f["in_cell"] += 1
            if str(r.get("functional") or "").upper() in EXCLUDED_FUNCTIONALS:
                f["dropped_lda"] += 1; continue
            if not r.get("functional") or not r.get("basis"):
                f["dropped_null_label"] += 1; continue
            if not (r.get("upload_id") or "").strip():
                f["blank_upload"] += 1
            kept.append(r)

        y = np.array([f"{r.get('functional')}{US}{r.get('basis')}" for r in kept])
        up = np.array([(r.get("upload_id") or "").strip() for r in kept])
        pc = collections.Counter(y)
        p = np.array([v / len(y) for v in pc.values()])
        print("=" * 104)
        print(f"{cell}: in cell {f['in_cell']:,} -> -LDA {f['dropped_lda']:,} "
              f"-null {f['dropped_null_label']:,} -> kept {len(kept):,}   "
              f"blank upload_id {f['blank_upload']:,}")
        print(f"  distinct pairs {len(pc)}  modal {pc.most_common(1)[0][0].replace(US,'/')} "
              f"{100*pc.most_common(1)[0][1]/len(y):.1f}%  inverse-Simpson {1/(p**2).sum():.2f}")
        if f["blank_upload"]:
            bl = collections.Counter((r["functional"], r["basis"]) for r in kept
                                     if not (r.get("upload_id") or "").strip())
            print(f"  blank-upload labels: {[(a+'/'+b, n) for (a, b), n in bl.most_common(3)]}")

        res = {"filters": f, "n_kept": len(kept), "n_distinct_pairs": len(pc),
               "modal_pair": pc.most_common(1)[0][0].replace(US, "/"),
               "modal_share_pct": 100 * pc.most_common(1)[0][1] / len(y),
               "inverse_simpson": float(1 / (p ** 2).sum()), "runs": {}}

        idx = np.arange(len(kept))
        elig = idx[up != ""]

        res["runs"]["random_all_rows"] = evaluate(
            cell, kept, y, None,
            list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(idx)),
            "random 5-fold, all kept rows")

        # MATCHED: random split on exactly the group-eligible rows
        ke, ye, ue = [kept[j] for j in elig], y[elig], up[elig]
        res["runs"]["random_eligible_rows"] = evaluate(
            cell, ke, ye, None,
            list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(np.arange(len(ke)))),
            "random 5-fold, eligible rows [MATCHED]")

        k = min(N_SPLITS, len(set(ue)))
        r2 = evaluate(cell, ke, ye, ue,
                      list(GroupKFold(n_splits=k).split(np.arange(len(ke)), ye, groups=ue)),
                      "upload-grouped, blanks excluded")
        r2["n_groups"] = len(set(ue))
        r2["n_blank_rows_excluded"] = int(len(kept) - len(ke))
        res["runs"]["grouped_blanks_excluded"] = r2

        g2 = np.array([u if u else "__singleton_%d" % i for i, u in enumerate(up)])
        k2 = min(N_SPLITS, len(set(g2)))
        r3 = evaluate(cell, kept, y, g2,
                      list(GroupKFold(n_splits=k2).split(idx, y, groups=g2)),
                      "upload-grouped, blanks singleton")
        r3["n_groups"] = len(set(g2))
        r3["caveat"] = "admits the malformed blank-upload labels"
        res["runs"]["grouped_blanks_singleton"] = r3

        report["cells"][cell] = res
        print()

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"per-fold results + hashes written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
