"""
freeze_three_way_router.py

Fits the three-way router (PBE0 / QCISD / MP2) on ALL 102 organic development reactions and
writes its parameters to disk. Run ONCE, BEFORE any held-out errors exist.

This exists so the held-out evaluation is a genuine single-shot test rather than a re-fit. The
frozen artifact records the feature list, scaler statistics, coefficients, class order and a hash,
so `evaluate_frozen_router.py` cannot silently alter the model.

Design fixed in advance (review follow-up plan S1):
  * candidate set : PBE0/def2-TZVP, QCISD/6-311G(2df,2p), MP2/def2-TZVP
  * features      : degree_unsaturation, n_atoms  (formula-derivable, no calculation needed)
  * model         : multinomial logistic regression on standardized features
  * label         : argmin per-reaction absolute error against the benchmark reference
  * training data : the 102 organic_general reactions used throughout development
  * comparator    : fixed MP2 (the best single method on the development set)
  * endpoint      : paired MAE difference, bootstrap 95% CI
"""
from __future__ import annotations
import glob, hashlib, json, os
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from boundary_analysis import load_from_pair_log, to_matrix

FEATURES = ["degree_unsaturation", "n_atoms"]
CLASSES = ["PBE0", "QCISD", "MP2"]
BASE = Path(__file__).parent
OUT = BASE / "accuracy_benchmark_logs" / "frozen_three_way_router.json"


def main() -> None:
    p1 = BASE / "accuracy_benchmark_logs/specialist_pair_organic_general_rccs.json"
    p2 = Path(sorted(glob.glob(str(BASE / "accuracy_benchmark_logs/specialist_pair_organic_general_MP2*.json")),
                     key=os.path.getmtime)[-1])
    rows, ids, _, ea, eb, cell = load_from_pair_log(p1)
    mp2 = {r["reaction_id"]: r["pair_a_abs_error_kcal_mol"]
           for r in json.loads(p2.read_text(encoding="utf-8"))["per_reaction"]
           if r.get("pair_a_abs_error_kcal_mol") is not None}
    keep = [i for i, rid in enumerate(ids) if rid in mp2]
    X, used = to_matrix([rows[i] for i in keep], FEATURES)
    err = np.vstack([np.array(ea)[keep], np.array(eb)[keep],
                     np.array([mp2[ids[i]] for i in keep])])
    y = np.argmin(err, axis=0)

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=0))
    pipe.fit(X, y)
    sc, clf = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]

    art = {
        "created": "2026-09-28",
        "purpose": "frozen for single-shot held-out evaluation (review follow-up plan S1)",
        "cell": cell,
        "features": used,
        "classes": CLASSES,
        "class_order_in_model": [int(c) for c in clf.classes_],
        "scaler_mean": sc.mean_.tolist(),
        "scaler_scale": sc.scale_.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "n_train_reactions": int(len(y)),
        "train_label_counts": {CLASSES[i]: int((y == i).sum()) for i in range(3)},
        "train_fixed_mae": {CLASSES[i]: float(err[i].mean()) for i in range(3)},
        "train_best_fixed": CLASSES[int(np.argmin(err.mean(axis=1)))],
        "train_oracle_mae": float(err.min(axis=0).mean()),
        "comparator_for_heldout": "fixed MP2 (best single method on development set)",
        "primary_endpoint": "paired MAE difference (fixed MP2 - routed), bootstrap 95% CI",
    }
    art["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in art.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest()[:16]
    OUT.write_text(json.dumps(art, indent=2), encoding="utf-8")

    print(f"FROZEN on {art['n_train_reactions']} development reactions")
    print(f"  features   : {used}")
    print(f"  labels     : {art['train_label_counts']}")
    print(f"  fixed MAE  : {  {k: round(v,2) for k,v in art['train_fixed_mae'].items()} }")
    print(f"  best fixed : {art['train_best_fixed']}  (comparator for the held-out test)")
    print(f"  sha256[:16]: {art['sha256']}")
    print(f"\nwrote {OUT}")
    print("\nThis file must NOT be regenerated after held-out errors are known.")


if __name__ == "__main__":
    main()
