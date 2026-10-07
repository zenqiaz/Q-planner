"""
score_specialist_pair_from_rccs.py

Scores a specialist-pair experiment purely from RCCS results, with NO qcl involvement.
Needed 2026-09-17 because qcl is unusable (pod venv interpreter is a dangling symlink; see
memory project_qcl_venv_python_missing_2026-09) and run_specialist_pair_experiment.py's own
scoring path runs the cheap half on qcl BEFORE merging RCCS results, so it cannot be used as-is.

The scoring loop below is a faithful reproduction of that script's own logic (same energy
bookkeeping, same "both settings + finite reference" usability rule, same output schema that
`boundary_analysis.py --pair-log` consumes), using `run_accuracy_benchmark.HARTREE_TO_KCAL`
unchanged. Nothing about the comparison methodology is altered here -- only where the energies
are read from.

--- One documented, physically exact correction -------------------------------------------------
QCISD on the hydrogen atom fails in ORCA with:
    Error (ORCA_MDCI): Number of processes (1) in parallel calculation exceeds number of pairs (0)
because a one-electron system has ZERO correlated electron pairs. This exact signature is already
documented in run_accuracy_benchmark.py as "a genuine method/system mismatch, not an aux-basis
problem". It is not a convergence failure or a bug.

For a one-electron system the QCISD energy IS the Hartree-Fock energy, exactly -- correlation
requires at least two electrons, so the correlation energy is identically zero by definition, not
approximately so. ORCA prints the value immediately before aborting ("Reference energy ...
-0.499809815"), and it sits correctly just above the exact non-relativistic -0.5 Eh for hydrogen.

Without this substitution, 67 of organic_general's 103 reactions (W4-11 atomization energies of
any H-containing molecule need the H atom) lose their pair_b energy and drop out, cutting the
comparison to n=36. With it, the full set is recoverable. The substitution is applied ONLY to
(species with 1 electron, correlated method) and is recorded explicitly in the output JSON under
"corrections" so it is visible to anyone reading the result, never silent.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import run_accuracy_benchmark as arb  # noqa: E402

LOG_DIR = Path(__file__).parent / "accuracy_benchmark_logs"

# (species_id, functional) -> (energy_eh, why). Exact HF = QCISD for a 1-electron system.
ZERO_PAIR_SUBSTITUTIONS = {
    ("w4_atom_H", "QCISD"): (
        -0.499809815,
        "H atom has 1 electron => 0 correlated pairs; QCISD energy is exactly the HF "
        "reference energy ORCA printed before aborting in MDCI",
    ),
}


def _parse_setting(s: str) -> tuple[str, str]:
    f, b = s.split("/", 1)
    return f.strip(), b.strip()


def load_energy_dump(path: Path) -> dict[str, float | None]:
    """dump lines: '<batchdir>|<inp_stem>|<energy or ERROR>'"""
    out: dict[str, float | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        _, stem, e = parts
        out[stem] = None if e == "ERROR" else float(e)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", required=True, choices=["organic_general", "metal_general"])
    p.add_argument("--pair-a", required=True)
    p.add_argument("--pair-b", required=True)
    p.add_argument("--energy-dump", type=Path, required=True)
    p.add_argument("--meta", type=Path, action="append", required=True,
                   help="_meta.json from each export dir (repeatable)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    pair_a, pair_b = _parse_setting(args.pair_a), _parse_setting(args.pair_b)
    gt = arb.load_ground_truth(args.cell, None, random.Random(0))
    species, reactions = gt["species"], gt["reactions"]

    energies_by_stem = load_energy_dump(args.energy_dump)

    # node_id -> (species_id, functional, basis, inp_stem), from the exporters' own manifests
    energy_cache: dict[tuple[str, str, str], float] = {}
    n_meta = n_missing = 0
    for meta_path in args.meta:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for _node_id, m in meta.items():
            n_meta += 1
            key = (m["species_id"], m["functional"], m["basis"])
            e = energies_by_stem.get(m["inp_stem"])
            if e is None:
                sub = ZERO_PAIR_SUBSTITUTIONS.get((m["species_id"], m["functional"]))
                if sub is not None:
                    energy_cache[key] = sub[0]
                    print(f"  [correction] {m['species_id']} / {m['functional']}: using "
                          f"{sub[0]} Eh -- {sub[1]}")
                    continue
                n_missing += 1
                continue
            energy_cache[key] = e
    print(f"  loaded {len(energy_cache)}/{n_meta} energies ({n_missing} genuinely missing)")

    per_reaction: list[dict[str, Any]] = []
    for r in reactions:
        row: dict[str, Any] = {
            "reaction_id": r["reaction_id"], "ref_kcal_mol": r["ref_kcal_mol"],
            "species_ids": r["species_ids"], "coeffs": r["coeffs"],
        }
        ok = True
        for label, setting in (("pair_a", pair_a), ("pair_b", pair_b)):
            f, b = setting
            tot = 0.0
            for sid, coeff in zip(r["species_ids"], r["coeffs"]):
                e = energy_cache.get((sid, f, b))
                if e is None:
                    ok = False
                    break
                tot += coeff * e
            if not ok:
                break
            row[f"{label}_kcal_mol"] = tot * arb.HARTREE_TO_KCAL
            row[f"{label}_abs_error_kcal_mol"] = abs(row[f"{label}_kcal_mol"] - r["ref_kcal_mol"])
        if ok and r["ref_kcal_mol"] is not None and arb.math.isfinite(r["ref_kcal_mol"]):
            per_reaction.append(row)

    a_wins = sum(1 for r in per_reaction if r["pair_a_abs_error_kcal_mol"] < r["pair_b_abs_error_kcal_mol"])
    b_wins = sum(1 for r in per_reaction if r["pair_b_abs_error_kcal_mol"] < r["pair_a_abs_error_kcal_mol"])
    mae_a = sum(r["pair_a_abs_error_kcal_mol"] for r in per_reaction) / max(1, len(per_reaction))
    mae_b = sum(r["pair_b_abs_error_kcal_mol"] for r in per_reaction) / max(1, len(per_reaction))

    print(f"  usable reactions (both settings + valid reference): {len(per_reaction)}/{len(reactions)}")
    print(f"  pair_a ({pair_a[0]}/{pair_a[1]}): wins={a_wins}  MAE={mae_a:.2f} kcal/mol")
    print(f"  pair_b ({pair_b[0]}/{pair_b[1]}): wins={b_wins}  MAE={mae_b:.2f} kcal/mol")

    out = {
        "cell": args.cell, "pair_a": list(pair_a), "pair_b": list(pair_b),
        "n_reactions_total": len(reactions), "n_reactions_usable": len(per_reaction),
        "a_wins": a_wins, "b_wins": b_wins,
        "mae_a_kcal_mol": mae_a, "mae_b_kcal_mol": mae_b,
        "backend": "RCCS only (qcl unavailable -- pod venv interpreter missing)",
        "corrections": [
            {"species_id": sid, "functional": f, "energy_eh": v[0], "reason": v[1]}
            for (sid, f), v in ZERO_PAIR_SUBSTITUTIONS.items()
            if (sid, f[0] if False else f) and any(
                k[0] == sid and k[1] == f for k in energy_cache)
        ],
        "per_reaction": per_reaction,
    }
    LOG_DIR.mkdir(exist_ok=True)
    out_path = args.out or (LOG_DIR / f"specialist_pair_{args.cell}_rccs.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
