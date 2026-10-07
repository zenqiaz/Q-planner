"""
export_specialist_pair_cheap_to_rccs.py

Exports the CHEAP-DFT half of a specialist-pair experiment as ORCA .inp files for RCCS batch
submission -- the exact complement of run_specialist_pair_experiment.py's own
`--export-rccs-batch`, which only exports the EXPENSIVE (wavefunction-correlated / large-basis)
half and sends cheap jobs to qcl.

Why this exists (2026-09-17): qcl became unusable -- the pod's venv interpreter
(`/data/zhang/ollama/QCagent_venv/bin/python3`) is a symlink to the container's
`/usr/bin/python3`, which no longer exists, so `start_mcp_server.sh` dies with exit 127 and every
MCP session closes immediately. `/data` is a persistent hostPath mount so the venv survived while
the container root filesystem lost system Python. Nothing is wrong with this project's code; the
168-job metal_general batch returned 0/168 in 0.0s with no compute spent. RCCS is unaffected and
already running the QCISD half, so routing the cheap half there too is the clean bypass.

This deliberately reuses run_accuracy_benchmark.py's OWN building blocks unchanged
(`collect_sp_jobs`, `is_cheap_dft`, `build_aux_basis_lookup`, `write_orca_inp`,
`sanitize_node_id`, `classify_aux_basis_plan`) so the exported inputs are byte-identical in
format to what the established pipeline produces -- `write_orca_inp`'s own docstring notes it
matches the main-line format `server_with_product.py` uses on qcl, precisely so an RCCS-computed
energy is a like-for-like comparison rather than merely a similar one. Output layout
(`<batch>/<cell>/*.inp` + `_meta.json`) mirrors the existing exporter so
`--rccs-results` / `rccs_collect_results.py` can collect it the same way.

Usage:
    python export_specialist_pair_cheap_to_rccs.py --cell metal_general \\
        --pair-a B3LYP/def2-TZVP --pair-b TPSSH/def2-TZVP \\
        --out-dir specialist_pair_batch_metal_cheap
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import run_accuracy_benchmark as arb  # noqa: E402


def _parse_setting(s: str) -> tuple[str, str]:
    if "/" not in s:
        raise SystemExit(f"--pair-a/--pair-b must be FUNCTIONAL/BASIS (got {s!r})")
    f, b = s.split("/", 1)
    return f.strip(), b.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", required=True, choices=["organic_general", "metal_general"])
    p.add_argument("--pair-a", required=True)
    p.add_argument("--pair-b", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--exclude-species", action="append", default=[])
    args = p.parse_args()

    pair_a, pair_b = _parse_setting(args.pair_a), _parse_setting(args.pair_b)

    print(f"Loading full curated ground truth for {args.cell} (no subsampling)...")
    gt = arb.load_ground_truth(args.cell, None, arb.random.Random(0))
    species, reactions = gt["species"], gt["reactions"]
    print(f"  species={len(species)} reactions={len(reactions)}")

    print("Loading RAG pool for aux-basis precedent lookup...")
    pool = arb.canonicalize_pool(arb.rag.load_pool())
    aux_lookup = arb.build_aux_basis_lookup(pool)

    conditions = {r["reaction_id"]: {"pair_a": pair_a, "pair_b": pair_b} for r in reactions}
    jobs = sorted(arb.collect_sp_jobs(reactions, conditions))
    if args.exclude_species:
        jobs = [j for j in jobs if j[0] not in args.exclude_species]

    cheap = [j for j in jobs if arb.is_cheap_dft(j[1], j[2])]
    expensive = [j for j in jobs if j not in cheap]
    print(f"  total SP jobs: {len(jobs)}  cheap: {len(cheap)}  expensive (NOT exported here): {len(expensive)}")

    cell_dir = args.out_dir / args.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    used_lower: set[str] = set()
    for sid, f, b in cheap:
        node_id = arb.sanitize_node_id(sid, f, b)
        inp_stem, lower, suffix = node_id, node_id.lower(), 2
        while lower in used_lower:          # case-insensitive filesystem guard, same as the
            inp_stem = f"{node_id}__dup{suffix}"   # original exporter (Windows export bug, 2026-08-17)
            lower = inp_stem.lower()
            suffix += 1
        used_lower.add(lower)
        (cell_dir / f"{inp_stem}.inp").write_text(
            arb.write_orca_inp(sid, f, b, species, aux_lookup=aux_lookup), encoding="utf-8")
        meta[node_id] = {
            "species_id": sid, "functional": f, "basis": b,
            "aux_basis_plan": arb.classify_aux_basis_plan(f, b, aux_lookup),
            "inp_stem": inp_stem,
        }
    (cell_dir / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  wrote {len(cheap)} .inp file(s) -> {cell_dir}")


if __name__ == "__main__":
    main()
