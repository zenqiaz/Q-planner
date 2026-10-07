"""
routing_value_analysis.py

Tests the THREE-WAY claim, which is stronger than boundary_analysis.py's two-way one:

    boundary_analysis.py  : "is there a border separating A-wins from B-wins?"
    this script           : "does ROUTING between A and B beat the best SINGLE fixed choice --
                             i.e. do we actually need experts, and can a learnable rule capture
                             enough of the gain to be worth it?"

A border can be real and still worthless: if A and B are both bad, or if the router gets the
high-stakes reactions wrong, routing buys nothing over just picking the better functional and
using it everywhere. This script measures that directly.

Reported quantities
-------------------
MAE_A, MAE_B        : each functional used everywhere (the no-routing options)
MAE_oracle          : per-reaction best (perfect routing -- an UPPER BOUND, not achievable)
MAE_routed          : the LOOCV-honest learned router's actual cost
break-even accuracy : routing accuracy at which MAE_routed == best fixed
paired test         : is MAE_routed < best-fixed by more than chance? (bootstrap + permutation)

The oracle gain is positive by construction and proves nothing on its own. The claim that
matters is MAE_routed < min(MAE_A, MAE_B), with the gap surviving a paired test.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut

from boundary_analysis import load_from_pair_log, to_matrix


def loocv_predictions(X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Honest per-sample predictions: each point predicted by a model never trained on it."""
    preds = np.empty_like(y)
    for tr, te in LeaveOneOut().split(X):
        if len(np.unique(y[tr])) < 2:          # degenerate fold -> fall back to majority
            preds[te] = np.bincount(y[tr]).argmax()
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, random_state=seed))
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])[0]
    return preds


def paired_tests(routed: np.ndarray, fixed: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """Bootstrap CI on the mean paired difference, plus a sign-flip permutation p-value."""
    rng = np.random.default_rng(seed)
    diff = fixed - routed                       # positive = routing is better
    n = len(diff)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    # sign-flip null: the per-reaction improvement is symmetric about zero
    flips = np.array([(diff * rng.choice([-1.0, 1.0], n)).mean() for _ in range(n_boot)])
    p = float((np.abs(flips) >= abs(diff.mean())).mean())
    return boot, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair-log", type=Path, required=True)
    ap.add_argument("--features", default="degree_unsaturation,n_atoms")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    rows, ids, labels, errs_a, errs_b, cell = load_from_pair_log(args.pair_log)
    feats = args.features.split(",") if args.features else None
    X, used = to_matrix(rows, feats)
    y = np.array(labels)
    a, b = np.array(errs_a), np.array(errs_b)

    mae_a, mae_b = a.mean(), b.mean()
    best_fixed_name = "A" if mae_a <= mae_b else "B"
    best_fixed_err = a if mae_a <= mae_b else b
    oracle_err = np.minimum(a, b)
    worst_err = np.maximum(a, b)

    preds = loocv_predictions(X, y, args.seed)
    # label convention from boundary_analysis.load_from_pair_log: y==1 means pair_A wins
    # (verified 2026-09-28: choosing by y reproduces min(a,b) exactly, and y.sum()==a_wins)
    routed_err = np.where(preds == 1, a, b)
    acc = float((preds == y).mean())

    mn, mx = oracle_err.mean(), worst_err.mean()
    break_even = (mx - best_fixed_err.mean()) / (mx - mn)
    captured = (best_fixed_err.mean() - routed_err.mean()) / (best_fixed_err.mean() - mn)

    boot, p = paired_tests(routed_err, best_fixed_err, args.n_boot, args.seed)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"cell={cell}  n={len(y)}  features={used}")
    print(f"  MAE A (everywhere)      : {mae_a:.2f}")
    print(f"  MAE B (everywhere)      : {mae_b:.2f}")
    print(f"  MAE best fixed ({best_fixed_name})       : {best_fixed_err.mean():.2f}   <- the bar routing must clear")
    print(f"  MAE ROUTED (LOOCV)      : {routed_err.mean():.2f}")
    print(f"  MAE oracle (upper bnd)  : {mn:.2f}")
    print()
    print(f"  routing accuracy (LOOCV): {100*acc:.1f}%   break-even: {100*break_even:.1f}%")
    print(f"  improvement over best fixed: {best_fixed_err.mean()-routed_err.mean():+.2f} kcal/mol "
          f"({100*(best_fixed_err.mean()-routed_err.mean())/best_fixed_err.mean():+.1f}%)")
    print(f"  fraction of oracle headroom captured: {100*captured:.1f}%")
    print(f"  bootstrap 95% CI on improvement: [{lo:+.2f}, {hi:+.2f}]  sign-flip p={p:.4f}")
    verdict = ("ROUTING BEATS BEST FIXED" if lo > 0 else
               "improvement NOT significant (CI includes 0)")
    print(f"  => {verdict}")

    # regional breakdown: does each predicted region prefer its own specialist?
    print("\n  per-predicted-region (does each side's specialist actually win there?):")
    for r, nm in ((1, "region predicted A"), (0, "region predicted B")):
        m = preds == r
        if m.sum():
            print(f"    {nm:<20} n={m.sum():3d}  MAE_A={a[m].mean():6.2f}  MAE_B={b[m].mean():6.2f}  "
                  f"-> {'A' if a[m].mean()<b[m].mean() else 'B'} better by {abs(a[m].mean()-b[m].mean()):.2f}")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "cell": cell, "n": int(len(y)), "features": used,
            "mae_a": float(mae_a), "mae_b": float(mae_b),
            "mae_best_fixed": float(best_fixed_err.mean()), "best_fixed": best_fixed_name,
            "mae_routed": float(routed_err.mean()), "mae_oracle": float(mn),
            "routing_accuracy": acc, "break_even_accuracy": float(break_even),
            "improvement_kcal_mol": float(best_fixed_err.mean() - routed_err.mean()),
            "headroom_captured_frac": float(captured),
            "bootstrap_ci95": [float(lo), float(hi)], "signflip_p": p,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
