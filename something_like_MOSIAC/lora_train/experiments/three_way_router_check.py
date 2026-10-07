"""
three_way_router_check.py

FEASIBILITY CHECK, not a validation. Prompted by the external review's §0 finding
(`section_6_7_review.md`): the PBE0/QCISD router (MAE 10.62) does NOT beat a third fixed method,
plain MP2 (10.58), on the identical 102 organic reactions. The two-way router beat the two
methods it chose between, but not the best fixed choice available.

That failure is about WHICH methods were routed between, not necessarily about routing: the
3-way oracle over {PBE0, QCISD, MP2} reaches 5.89 vs fixed MP2's 10.58 -- 44.3% headroom, with
per-reaction winners spread 21/40/39%. This script asks whether a low-capacity router can capture
enough of that to beat fixed MP2.

HONEST STATUS -- read before quoting any number from this:
  * Fitted and evaluated on the SAME 102 reactions used throughout development. LOOCV makes each
    prediction out-of-sample, but the candidate set, features and model family were all chosen
    with knowledge of these data. This is hypothesis-generating.
  * MP2 entered the candidate pool because pair 2 was selected AFTER pair 1 succeeded, so its
    inclusion is itself outcome-guided.
  * A claim of practical value needs the review's §1 design: freeze this router, then evaluate
    once on reactions never used here.
"""
from __future__ import annotations
import argparse, json, glob, os
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut

from boundary_analysis import load_from_pair_log, to_matrix

METHODS = ("PBE0", "QCISD", "MP2")


def loocv_multiclass(X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    preds = np.empty_like(y)
    for tr, te in LeaveOneOut().split(X):
        if len(np.unique(y[tr])) < 2:
            preds[te] = np.bincount(y[tr]).argmax()
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, random_state=seed))
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])[0]
    return preds


def boot_ci(d: np.ndarray, rng, n: int = 10000):
    v = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(v, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-permutations", type=int, default=10000)
    ap.add_argument("--features", default="degree_unsaturation,n_atoms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    base = Path(__file__).parent
    p1 = base / "accuracy_benchmark_logs/specialist_pair_organic_general_rccs.json"
    p2 = Path(sorted(glob.glob(str(base / "accuracy_benchmark_logs/specialist_pair_organic_general_MP2*.json")),
                     key=os.path.getmtime)[-1])

    rows, ids, _, ea, eb, cell = load_from_pair_log(p1)
    m2 = {r["reaction_id"]: r["pair_a_abs_error_kcal_mol"]
          for r in json.loads(p2.read_text(encoding="utf-8"))["per_reaction"]
          if r.get("pair_a_abs_error_kcal_mol") is not None}
    keep = [i for i, rid in enumerate(ids) if rid in m2]
    X, used = to_matrix([rows[i] for i in keep], args.features.split(","))
    err = np.vstack([np.array(ea)[keep], np.array(eb)[keep],
                     np.array([m2[ids[i]] for i in keep])])          # 3 x n
    n = err.shape[1]
    y = np.argmin(err, axis=0)

    fixed = err.mean(axis=1)
    best_i = int(np.argmin(fixed)); best_fixed = err[best_i]
    oracle = err.min(axis=0)
    worst = err.max(axis=0)

    preds = loocv_multiclass(X, y, args.seed)
    routed = err[preds, np.arange(n)]
    acc = float((preds == y).mean())
    floor = float(np.bincount(y).max() / n)

    d = best_fixed - routed
    lo, hi = boot_ci(d, rng, args.n_permutations)
    be = (worst.mean() - best_fixed.mean()) / (worst.mean() - oracle.mean())

    print(f"cell={cell}  n={n}  features={used}  classes={METHODS}")
    for i, m in enumerate(METHODS):
        print(f"  fixed {m:<6} MAE {fixed[i]:6.2f}   wins {int((y==i).sum()):>3}/{n} ({100*(y==i).mean():.0f}%)")
    print(f"\n  best fixed            = {METHODS[best_i]} at {fixed[best_i]:.2f}   <- the bar")
    print(f"  3-way ROUTED (LOOCV)  = {routed.mean():.2f}")
    print(f"  3-way oracle          = {oracle.mean():.2f}  (headroom {100*(fixed[best_i]-oracle.mean())/fixed[best_i]:.1f}%)")
    print(f"\n  gain over best fixed  = {d.mean():+.2f} kcal/mol   95% CI [{lo:+.2f},{hi:+.2f}]")
    print(f"  routing accuracy      = {100*acc:.1f}%  (majority floor {100*floor:.1f}%)")
    print(f"  break-even payoff-weighted requirement = {100*be:.1f}%")
    verdict = ("BEATS best fixed" if lo > 0 else
               "WORSE than best fixed" if hi < 0 else
               "NOT distinguishable from best fixed")
    print(f"  => {verdict}")

    # permutation null on the gain itself
    null = np.empty(args.n_permutations)
    for k in range(args.n_permutations):
        pe = rng.permutation(preds)
        null[k] = (best_fixed - err[pe, np.arange(n)]).mean()
    p = float((null >= d.mean()).mean())
    print(f"  permutation p (gain vs shuffled routing) = {p:.4f}")

    print("\n  per-class capture:")
    for i, m in enumerate(METHODS):
        sel = preds == i
        if sel.sum():
            print(f"    routed to {m:<6} n={int(sel.sum()):>3}  actual-best-here {int((y[sel]==i).sum()):>3}"
                  f"  MAE {routed[sel].mean():6.2f}  (fixed {METHODS[best_i]} on same: {best_fixed[sel].mean():6.2f})")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "cell": cell, "n": n, "features": used, "methods": list(METHODS),
            "fixed_mae": {m: float(fixed[i]) for i, m in enumerate(METHODS)},
            "best_fixed": METHODS[best_i], "routed_mae": float(routed.mean()),
            "oracle_mae": float(oracle.mean()), "gain": float(d.mean()),
            "ci95": [float(lo), float(hi)], "accuracy": acc, "majority_floor": floor,
            "break_even_payoff_weighted": float(be), "permutation_p": p,
            "status": "FEASIBILITY CHECK on development data - not validation",
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
