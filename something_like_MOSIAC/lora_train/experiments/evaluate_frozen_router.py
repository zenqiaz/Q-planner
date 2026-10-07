"""
evaluate_frozen_router.py

SINGLE-SHOT evaluation of the frozen three-way router on held-out GMTKN55 reactions
(FH51 + TAUT15, C/H/N/O/F, <=16 atoms, neutral closed-shell) that were never used in any prior
analysis in this project.

Everything that could be tuned was fixed in advance and is READ, not chosen, here:
  * router parameters  -> accuracy_benchmark_logs/frozen_three_way_router.json (sha256 verified)
  * features           -> whatever that file says (degree_unsaturation, n_atoms)
  * comparator         -> whatever that file says (fixed MP2)
  * primary endpoint   -> paired MAE difference vs comparator, bootstrap 95% CI

Feature construction deliberately mirrors run_accuracy_benchmark.representative_species /
reaction_profile: the representative species is the LARGEST species in the reaction, and
degree_unsaturation = (2*nC + 2 + nN - nH - nF)/2 from that species' element counts. Any deviation
would mean the frozen model is being applied to a different feature basis than it was fitted on.

Pre-registered caveats reported alongside the result:
  1. size extrapolation -- development spanned 1-11 atoms (median 4), this set is 2-16 (median 13),
     so the <=11-atom subset is reported separately;
  2. endpoint shift -- FH51/TAUT15 are BOTH reaction energies, and S6.7.6 already found the
     development gain reverses on reaction energies (G2RC n=15, -1.53). Expect failure; the value
     is whether that failure replicates at larger n.

Run once. Do not refit, do not adjust features, do not re-run with different options and report the
better outcome.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

HARTREE_TO_KCAL = 627.5094740631
BASE = Path(__file__).parent
FROZEN = BASE / "accuracy_benchmark_logs" / "frozen_three_way_router.json"
EXPECT_SHA = "965bd671569d5278"


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def element_counts(xyz: str) -> dict[str, int]:
    c: dict[str, int] = {}
    for line in xyz.strip().split("\n"):
        t = line.split()
        if t:
            c[t[0]] = c.get(t[0], 0) + 1
    return c


def final_energy(path: Path) -> float | None:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if "ORCA TERMINATED NORMALLY" not in txt:
        return None
    hits = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", txt)
    return float(hits[-1]) if hits else None


def boot_ci(d: np.ndarray, rng, n: int = 10000):
    v = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(v, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groundtruth", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    art = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert art["sha256"] == EXPECT_SHA, f"frozen router hash mismatch: {art['sha256']} != {EXPECT_SHA}"
    CLASSES = art["classes"]                       # ["PBE0","QCISD","MP2"]
    FEATS = art["features"]
    mean = np.array(art["scaler_mean"]); scale = np.array(art["scaler_scale"])
    coef = np.array(art["coef"]); inter = np.array(art["intercept"])
    order = art["class_order_in_model"]
    comparator = art["train_best_fixed"]
    print(f"frozen router loaded  sha256={art['sha256']}  features={FEATS}")
    print(f"  trained on {art['n_train_reactions']} reactions; comparator fixed in advance = {comparator}")
    print(f"  endpoint fixed in advance = {art['primary_endpoint']}\n")

    gt = json.loads(args.groundtruth.read_text(encoding="utf-8"))
    species, reactions = gt["species"], gt["reactions"]

    # --- energies ---
    E: dict[str, dict[str, float]] = {}
    missing = []
    for name in species:
        E[name] = {}
        for m in CLASSES:
            f = args.out_dir / f"sp_{sanitize(name)}_{m}.out"
            e = final_energy(f) if f.exists() else None
            if e is None:
                missing.append(f"{name}/{m}")
            else:
                E[name][m] = e
    print(f"energies: {len(species)} species x {len(CLASSES)} methods, missing {len(missing)}")
    if missing:
        print("  missing:", missing[:6])

    # --- reaction errors ---
    rows, errs = [], []
    skipped = 0
    for r in reactions:
        names, coeffs, ref = r["names"], r["coeffs"], r["ref"]
        if not all(n in E and all(m in E[n] for m in CLASSES) for n in names):
            skipped += 1
            continue
        err = []
        for m in CLASSES:
            calc = sum(c * E[n][m] for n, c in zip(names, coeffs)) * HARTREE_TO_KCAL
            err.append(abs(calc - ref))
        rep = max((species[n] for n in names), key=lambda s: s["n_atoms"])
        cnt = element_counts(rep["xyz"])
        nC, nH, nN, nF = (cnt.get(e, 0) for e in ("C", "H", "N", "F"))
        feat = {"n_atoms": float(rep["n_atoms"]),
                "degree_unsaturation": (2 * nC + 2 + nN - nH - nF) / 2.0}
        rows.append((r["rid"], [feat[f] for f in FEATS], float(rep["n_atoms"])))
        errs.append(err)
    print(f"reactions usable: {len(rows)}/{len(reactions)} (skipped {skipped})\n")

    X = np.array([r[1] for r in rows]); err = np.array(errs).T      # 3 x n
    nat = np.array([r[2] for r in rows]); n = X.shape[0]

    # --- apply the frozen model (no fitting) ---
    Z = (X - mean) / scale
    logits = Z @ coef.T + inter
    pred_pos = np.argmax(logits, axis=1)
    pred = np.array([order[p] for p in pred_pos])
    routed = err[pred, np.arange(n)]

    ci = CLASSES.index(comparator)
    comp = err[ci]
    oracle = err.min(axis=0)
    fixed = err.mean(axis=1)

    print("=== HELD-OUT RESULT (single shot) ===")
    for i, m in enumerate(CLASSES):
        tag = "  <- pre-registered comparator" if m == comparator else ""
        print(f"  fixed {m:<6} MAE {fixed[i]:6.2f}{tag}")
    print(f"  FROZEN ROUTER    MAE {routed.mean():6.2f}")
    print(f"  oracle           MAE {oracle.mean():6.2f}")
    d = comp - routed
    lo, hi = boot_ci(d, rng, args.n_boot)
    print(f"\n  PRIMARY ENDPOINT: paired gain vs fixed {comparator} = {d.mean():+.2f} kcal/mol")
    print(f"    95% CI [{lo:+.2f}, {hi:+.2f}]   "
          f"{'ROUTER BETTER' if lo>0 else 'ROUTER WORSE' if hi<0 else 'NOT DISTINGUISHABLE'}")
    print(f"  routing distribution: " +
          ", ".join(f"{m} {int((pred==i).sum())}" for i, m in enumerate(CLASSES)))
    print(f"  accuracy vs per-reaction best: {100*(pred==np.argmin(err,axis=0)).mean():.1f}%")

    print("\n=== pre-registered caveat 1: size subsets (development was 1-11 atoms) ===")
    for lab, m in (("<=11 atoms (in training range)", nat <= 11), (">11 atoms (extrapolation)", nat > 11)):
        if m.sum() >= 2:
            dd = (comp - routed)[m]; l2, h2 = boot_ci(dd, rng, 2000)
            print(f"  {lab:<32} n={int(m.sum()):>3}  gain {dd.mean():+.2f}  95% CI [{l2:+.2f},{h2:+.2f}]")
        elif m.sum():
            print(f"  {lab:<32} n={int(m.sum()):>3}  (too few for an interval)")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "frozen_sha256": art["sha256"], "comparator": comparator,
            "n_reactions": int(n), "fixed_mae": {m: float(fixed[i]) for i, m in enumerate(CLASSES)},
            "routed_mae": float(routed.mean()), "oracle_mae": float(oracle.mean()),
            "gain_vs_comparator": float(d.mean()), "ci95": [float(lo), float(hi)],
            "routing_distribution": {m: int((pred == i).sum()) for i, m in enumerate(CLASSES)},
            "endpoint_note": "all reactions are reaction energies (FH51+TAUT15)",
            "status": "single-shot held-out evaluation; router frozen before data existed",
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
