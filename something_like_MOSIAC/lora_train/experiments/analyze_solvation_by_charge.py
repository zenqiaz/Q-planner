"""
analyze_solvation_by_charge.py

Model-side half of the solvation question (corpus-side half is memory
`project_solvation_zero_data_floor_2026-09`): DOES the fine-tuned selector emit solvation, and
is that emission CONDITIONED ON CHARGE?

Zero new compute -- reads `eval_report_full.json` reports already produced on NII by
`eval_full_schema.py`, whose `predictions` entries carry the prompt (containing `charge=`), the
gold label, and the parsed prediction.

Reports it cross-tabulates solvation emission against charge, and compares to the gold rate.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent / "solvation_test"
CHARGE_RE = re.compile(r"charge=(-?\d+)")
ORDER = ["<= -2", "-1", "0 (neutral)", "+1", ">= +2"]


def bucket(c: int) -> str:
    if c <= -2: return "<= -2"
    if c == -1: return "-1"
    if c == 0:  return "0 (neutral)"
    if c == 1:  return "+1"
    return ">= +2"


def has_solv(d) -> bool:
    return bool(isinstance(d, dict) and (d.get("solvent") or d.get("solvent_model")))


def main() -> None:
    grand = defaultdict(lambda: {"n": 0, "gold": 0, "pred": 0, "both": 0})
    for path in sorted(HERE.glob("*.json")):
        rep = json.loads(path.read_text(encoding="utf-8"))
        preds = rep.get("predictions") or []
        tab = defaultdict(lambda: {"n": 0, "gold": 0, "pred": 0, "both": 0})
        n_unparsed = 0
        for p in preds:
            m = CHARGE_RE.search(p.get("prompt", "") or "")
            if not m:
                continue
            if p.get("pred_parsed") is None:
                n_unparsed += 1
                continue
            b = bucket(int(m.group(1)))
            g, q = has_solv(p.get("gold")), has_solv(p.get("pred_parsed"))
            for t in (tab, grand):
                t[b]["n"] += 1
                t[b]["gold"] += int(g)
                t[b]["pred"] += int(q)
                t[b]["both"] += int(g and q)

        tot = sum(v["n"] for v in tab.values())
        print("=" * 78)
        print(f"{path.stem}   ({tot} parsed predictions, {n_unparsed} unparsed)")
        print(f"  {'charge':<14}{'n':>6}{'gold solv':>11}{'pred solv':>11}{'gold%':>8}{'pred%':>8}")
        for b in ORDER:
            v = tab[b]
            if v["n"]:
                print(f"  {b:<14}{v['n']:>6}{v['gold']:>11}{v['pred']:>11}"
                      f"{100*v['gold']/v['n']:>7.1f}%{100*v['pred']/v['n']:>7.1f}%")

    print("=" * 78)
    print("ALL REPORTS COMBINED")
    print(f"  {'charge':<14}{'n':>6}{'gold solv':>11}{'pred solv':>11}{'gold%':>8}{'pred%':>8}")
    for b in ORDER:
        v = grand[b]
        if v["n"]:
            print(f"  {b:<14}{v['n']:>6}{v['gold']:>11}{v['pred']:>11}"
                  f"{100*v['gold']/v['n']:>7.1f}%{100*v['pred']/v['n']:>7.1f}%")
    n = sum(v["n"] for v in grand.values())
    g = sum(v["gold"] for v in grand.values())
    q = sum(v["pred"] for v in grand.values())
    both = sum(v["both"] for v in grand.values())
    print(f"\n  overall: n={n}  gold-solvated={g} ({100*g/n:.1f}%)  "
          f"pred-solvated={q} ({100*q/n:.1f}%)  both={both}")
    charged = sum(v["n"] for b, v in grand.items() if b != "0 (neutral)")
    charged_pred = sum(v["pred"] for b, v in grand.items() if b != "0 (neutral)")
    neu = grand["0 (neutral)"]
    print(f"  charged species:  {charged_pred}/{charged} predicted-solvated "
          f"({100*charged_pred/max(1,charged):.1f}%)")
    print(f"  neutral species:  {neu['pred']}/{neu['n']} predicted-solvated "
          f"({100*neu['pred']/max(1,neu['n']):.1f}%)")


if __name__ == "__main__":
    main()
