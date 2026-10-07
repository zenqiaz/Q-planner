"""
sweep_failure_mechanisms.py

Systematic sweep across EVERY selector_rag/selector_rf/selector_sft prediction in both cells'
real accuracy-benchmark scoring data -- not just the hand-picked worst cases used to originally
identify the three failure mechanisms (chapter_plan insight #15 / outline.md #6.6). For every
prediction, classifies it as:
  - at least as good as fixed_default (regret <= 0)
  - worse than fixed_default, and if so, which of the three mechanisms it matches:
      1. rare-precedent extrapolation: the predicted (functional, basis) pair occurs
         RARE_THRESHOLD times or fewer in the RAG pool for that cell
      2. popular-but-suboptimal convention: functional is wavefunction-correlated
         (CCSD/CCSD(T)/DLPNO-CCSD(T)/MP2/QCISD) and not already caught by mechanism 1
      3. common functional, systematic bias: functional is a pure GGA (PBE, BP86) and not
         already caught by mechanism 1
  - or unclassified-worse, if it's worse but matches none of the three (reported honestly,
    not forced into a bucket)

Classification is by objective, stated rules checked in priority order (1 -> 2 -> 3 ->
unclassified) -- not re-fit to make the three mechanisms look more complete than they are.

Usage:
    python sweep_failure_mechanisms.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"D:\brick\D\20260217\working")
import rag  # noqa: E402

RARE_THRESHOLD = 5
WF_CORRELATED = {"CCSD", "CCSD(T)", "DLPNO-CCSD(T)", "MP2", "QCISD"}
PURE_GGA = {"PBE", "BP86"}

REPORTS = [
    ("accuracy_benchmark_logs/accuracy_benchmark_20260812T104756Z.json", "organic_general"),
    ("accuracy_benchmark_logs/accuracy_benchmark_20260813T041341Z.json", "metal_general"),
]


def build_pool_frequency() -> dict[str, Counter]:
    pool = rag.load_pool()
    freq: dict[str, Counter] = {"organic_general": Counter(), "metal_general": Counter()}
    for r in pool:
        cell = r.get("specialist_cell")
        f, b = r.get("functional"), r.get("basis")
        if cell in freq and f and b:
            freq[cell][(f, b)] += 1
    return freq


def classify(setting: str, cell: str, freq: dict[str, Counter]) -> str:
    functional, basis = setting.split("/", 1)
    functional, basis = functional.strip(), basis.strip()
    n = freq[cell].get((functional, basis), 0)
    if n <= RARE_THRESHOLD:
        return f"mechanism_1_rare_precedent (n={n})"
    if functional in WF_CORRELATED:
        return "mechanism_2_popular_suboptimal_wf"
    if functional in PURE_GGA:
        return "mechanism_3_common_gga_bias"
    return "unclassified_worse"


def main() -> None:
    freq = build_pool_frequency()

    all_rows: list[dict] = []
    for path, cell in REPORTS:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in report["cells"][cell]["scoring"]["per_reaction"]:
            fd = row["conditions"].get("fixed_default")
            if not fd or fd["status"] != "ok":
                continue
            fd_err = fd["abs_error_kcal_mol"]
            for cname in ["selector_rag", "selector_rf", "selector_sft"]:
                c = row["conditions"].get(cname)
                if c and c["status"] == "ok":
                    regret = c["abs_error_kcal_mol"] - fd_err
                    all_rows.append({
                        "cell": cell, "reaction_id": row["reaction_id"], "condition": cname,
                        "setting": c["setting"], "error": c["abs_error_kcal_mol"],
                        "fd_error": fd_err, "regret": regret,
                    })

    n_total = len(all_rows)
    n_better_or_equal = sum(1 for r in all_rows if r["regret"] <= 0)
    n_worse = sum(1 for r in all_rows if r["regret"] > 0)
    n_meaningfully_worse = sum(1 for r in all_rows if r["regret"] > 5.0)

    print(f"Total selector_rag/rf/sft predictions with a valid fixed_default comparison: {n_total}")
    print(f"  at least as good as fixed_default (regret <= 0): {n_better_or_equal} "
          f"({100 * n_better_or_equal / n_total:.1f}%)")
    print(f"  worse than fixed_default (regret > 0):           {n_worse} "
          f"({100 * n_worse / n_total:.1f}%)")
    print(f"  meaningfully worse (regret > 5 kcal/mol):        {n_meaningfully_worse} "
          f"({100 * n_meaningfully_worse / n_total:.1f}%)")

    print("\nMechanism breakdown, among the meaningfully-worse cases (regret > 5 kcal/mol):")
    worse_rows = [r for r in all_rows if r["regret"] > 5.0]
    mech_counts = Counter()
    mech_examples: dict[str, list] = {}
    for r in worse_rows:
        mech = classify(r["setting"], r["cell"], freq)
        mech_base = mech.split(" (")[0]
        mech_counts[mech_base] += 1
        mech_examples.setdefault(mech_base, []).append(r)

    for mech, count in mech_counts.most_common():
        pct = 100 * count / len(worse_rows) if worse_rows else 0
        print(f"  {mech:40s} {count:4d}  ({pct:.1f}% of meaningfully-worse cases, "
              f"{100 * count / n_total:.1f}% of all predictions)")

    print("\nPer-condition breakdown (which selector contributes most to each mechanism):")
    for mech in mech_counts:
        by_cond = Counter(r["condition"] for r in mech_examples[mech])
        print(f"  {mech}: {dict(by_cond)}")

    print("\nUnclassified-worse examples (worse but none of the 3 mechanisms match) -- "
          "reported for honesty, not swept under the rug:")
    unclassified = [r for r in worse_rows if classify(r["setting"], r["cell"], freq).startswith("unclassified")]
    for r in unclassified[:15]:
        print(f"  {r['cell']:15s} {r['reaction_id']:22s} {r['condition']:15s} "
              f"{r['setting']:25s} regret={r['regret']:+.2f}")
    if len(unclassified) > 15:
        print(f"  ... and {len(unclassified) - 15} more")

    out = {
        "n_total": n_total, "n_better_or_equal": n_better_or_equal, "n_worse": n_worse,
        "n_meaningfully_worse": n_meaningfully_worse,
        "mechanism_counts": dict(mech_counts),
        "rare_threshold": RARE_THRESHOLD,
        "all_rows": all_rows,
    }
    out_path = Path("failure_mechanism_sweep_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
