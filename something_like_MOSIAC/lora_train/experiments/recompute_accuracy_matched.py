"""
recompute_accuracy_matched.py

Tier-1 fix for the external review's critical item on Section 6.5.

The published table excludes a NON-LEARNED policy's reaction whenever that policy predicted the
same (functional, basis) as the retrieval selector, on the stated rationale that an identical
prediction should not be "double-counted as two independent data points". That rationale does not
hold for a *policy* comparison: if two policies choose the same method for a reaction they
genuinely incur the same error there, and shared cases are the basis of a paired comparison rather
than duplicate observations. Excluding them leaves each policy's MAE computed over a different
reaction set, and the exclusion is not random -- it removes exactly the reactions where the fixed
policy agreed with retrieval.

This script recomputes every condition's MAE over ALL reactions it can evaluate, counting ties for
both policies, and reports the counts the review asked to see separately (attempted / evaluable /
tie-excluded / not evaluable). It needs no new ORCA jobs: `scoring.per_reaction` in the benchmark
log already carries each condition's per-reaction absolute error and its tie flag.

Run:  python recompute_accuracy_matched.py [--log <benchmark json>] [--cell organic_general]
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent
CONDS = ["fixed_default", "literature_majority", "skills_heuristic", "random_floor",
         "selector_rag", "selector_rf", "selector_sft"]
NON_LEARNED = {"fixed_default", "literature_majority", "skills_heuristic", "random_floor"}


def boot_ci(d: np.ndarray, rng, n: int = 10000):
    v = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(v, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--cell", default="organic_general")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    log = args.log or sorted(
        glob.glob(str(BASE / "accuracy_benchmark_logs" / "accuracy_benchmark_*.json")),
        key=os.path.getmtime)[-1]
    d = json.load(open(log))
    if args.cell not in d.get("cells", {}):
        raise SystemExit(f"cell {args.cell} not in {os.path.basename(log)}")
    per = d["cells"][args.cell]["scoring"]["per_reaction"]
    print(f"log  : {os.path.basename(log)}")
    print(f"cell : {args.cell}   curated reactions: {len(per)}\n")

    # ---- per-condition accounting ------------------------------------------------
    err_all, err_nontie, counts = {}, {}, {}
    for cond in CONDS:
        a, nt, nmiss, ntie = [], [], 0, 0
        for r in per:
            cd = (r.get("conditions") or {}).get(cond)
            if not cd or cd.get("status") != "ok" or cd.get("abs_error_kcal_mol") is None:
                nmiss += 1
                continue
            e = float(cd["abs_error_kcal_mol"])
            a.append(e)
            if cd.get("tie_with_selector"):
                ntie += 1
            else:
                nt.append(e)
        err_all[cond] = np.array(a)
        err_nontie[cond] = np.array(nt)
        counts[cond] = dict(attempted=len(per), evaluable=len(a), tied=ntie, not_evaluable=nmiss)

    print("ACCOUNTING (the counts the review asked to be reported separately)")
    print(f"  {'condition':<22}{'attempted':>10}{'evaluable':>10}{'tied':>7}{'not eval':>10}")
    for cond in CONDS:
        c = counts[cond]
        print(f"  {cond:<22}{c['attempted']:>10}{c['evaluable']:>10}{c['tied']:>7}{c['not_evaluable']:>10}")

    print("\nMAE: AS PUBLISHED (ties dropped for non-learned) vs CORRECTED (all evaluable)")
    print(f"  {'condition':<22}{'n pub':>7}{'MAE pub':>10}{'n corr':>8}{'MAE corr':>10}{'shift':>9}")
    rows = {}
    for cond in CONDS:
        pub = err_nontie[cond] if cond in NON_LEARNED else err_all[cond]
        corr = err_all[cond]
        shift = corr.mean() - pub.mean() if len(pub) and len(corr) else float("nan")
        star = "  <-- changed" if cond in NON_LEARNED and abs(shift) > 0.005 else ""
        print(f"  {cond:<22}{len(pub):>7}{pub.mean():>10.2f}{len(corr):>8}{corr.mean():>10.2f}"
              f"{shift:>+9.2f}{star}")
        rows[cond] = dict(n_published=int(len(pub)), mae_published=float(pub.mean()),
                          n_corrected=int(len(corr)), mae_corrected=float(corr.mean()),
                          shift=float(shift), **counts[cond])

    # ---- paired comparisons on each pair's own common support ---------------------
    print("\nPAIRED fixed-vs-learned comparisons, each on its OWN matched reaction set")
    print("  (positive difference = the fixed policy is more accurate)")
    idx = {}
    for cond in CONDS:
        m = {}
        for r in per:
            cd = (r.get("conditions") or {}).get(cond)
            if cd and cd.get("status") == "ok" and cd.get("abs_error_kcal_mol") is not None:
                m[r["reaction_id"]] = float(cd["abs_error_kcal_mol"])
        idx[cond] = m
    pairs = []
    print(f"  {'fixed':<20}{'learned':<14}{'n':>5}{'fixed':>8}{'learned':>9}{'diff':>8}{'95% CI':>20}")
    for f in ["fixed_default", "literature_majority", "skills_heuristic"]:
        for l in ["selector_rag", "selector_rf", "selector_sft"]:
            common = sorted(set(idx[f]) & set(idx[l]))
            if len(common) < 5:
                continue
            fa = np.array([idx[f][k] for k in common])
            la = np.array([idx[l][k] for k in common])
            diff = la - fa
            lo, hi = boot_ci(diff, rng)
            flag = "" if lo > 0 else ("  n.s." if hi > 0 else "  learned better")
            print(f"  {f:<20}{l:<14}{len(common):>5}{fa.mean():>8.2f}{la.mean():>9.2f}"
                  f"{diff.mean():>+8.2f}   [{lo:+.2f},{hi:+.2f}]{flag}")
            pairs.append(dict(fixed=f, learned=l, n=len(common), mae_fixed=float(fa.mean()),
                              mae_learned=float(la.mean()), diff=float(diff.mean()),
                              ci95=[float(lo), float(hi)]))

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"log": os.path.basename(log), "cell": args.cell,
             "n_curated": len(per), "conditions": rows, "paired": pairs,
             "note": "MAE corrected = all evaluable reactions, ties counted for every policy. "
                     "The published figures dropped ties for the four non-learned conditions."},
            indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
