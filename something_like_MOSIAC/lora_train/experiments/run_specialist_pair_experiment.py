"""
run_specialist_pair_experiment.py

Companion to run_accuracy_benchmark.py, built 2026-09-08 for the "decision-boundary" experiment
proposed in paper discussion: instead of comparing this project's own selector conditions
(fixed_default/literature_majority/skills_heuristic/random_floor/selector_rag/rf/sft), this
script runs exactly TWO caller-specified (functional, basis) settings across every reaction in a
cell's full curated ground-truth pool, so every reaction gets a real head-to-head A-vs-B error
comparison -- the input `boundary_analysis.py` needs to test whether a linear decision boundary
in feature space separates "A wins" from "B wins".

Deliberately NOT a generic refactor of run_accuracy_benchmark.py's 7-condition machinery --
score_reactions() there hardcodes those 7 names and their tie-exclusion logic (which references
selector_rag specifically), neither of which applies here. Instead this script imports and
reuses the underlying, condition-agnostic building blocks directly:
  - load_ground_truth / GROUND_TRUTH_FILES -- full curated reaction+species pool per cell
  - is_cheap_dft / classify_aux_basis_plan / build_aux_basis_lookup / write_orca_inp /
    sanitize_node_id -- cheap-vs-expensive routing and RCCS .inp export, unchanged
  - run_batch_with_watchdog / kill_stuck_orca_processes / sweep_stale_job_dirs -- qcl execution,
    chunking, and the hard-won hang/disk-pressure mitigations, unchanged
  - extract_energy_cache / HARTREE_TO_KCAL -- energy bookkeeping, unchanged
None of run_accuracy_benchmark.py's own code is modified by this script.

Usage (mirrors run_accuracy_benchmark.py's own CLI where the concepts overlap):
    # scope only, no ORCA calls -- get job counts / RCCS export before spending real compute
    python run_specialist_pair_experiment.py --cell organic_general \\
        --pair-a PBE0/def2-TZVP --pair-b "QCISD/6-311G(2df,2p)" \\
        --dry-run --export-rccs-batch specialist_pair_batch

    # real run, once the pair is confirmed
    python run_specialist_pair_experiment.py --cell organic_general \\
        --pair-a PBE0/def2-TZVP --pair-b "QCISD/6-311G(2df,2p)" \\
        --export-rccs-batch specialist_pair_batch --qcl-session-chunk-size 30

    # after the RCCS batch is submitted (rccs_submit_orca.sh) and collected
    # (rccs_collect_results.py), merge and score:
    python run_specialist_pair_experiment.py --cell organic_general \\
        --pair-a PBE0/def2-TZVP --pair-b "QCISD/6-311G(2df,2p)" \\
        --export-rccs-batch specialist_pair_batch --rccs-results specialist_pair_batch \\
        --qcl-session-chunk-size 30

Output: {cell}_specialist_pair_{pair_a_slug}_vs_{pair_b_slug}.json, shaped for
boundary_analysis.py --pair-log <this file>: one entry per reaction with ref_kcal_mol,
species_ids/coeffs (for feature building), and abs_error_kcal_mol for both "pair_a"/"pair_b".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import run_accuracy_benchmark as arb  # noqa: E402 -- reuse its module-level setup (dotenv, RAG, SSH)

LOG_DIR = Path(__file__).parent / "accuracy_benchmark_logs"


def _parse_setting(s: str) -> tuple[str, str]:
    if "/" not in s:
        raise SystemExit(f"--pair-a/--pair-b must be FUNCTIONAL/BASIS (got {s!r})")
    f, b = s.split("/", 1)
    return f.strip(), b.strip()


def _slug(setting: tuple[str, str]) -> str:
    raw = f"{setting[0]}_{setting[1]}"
    return re.sub(r"[^A-Za-z0-9]+", "", raw)[:24]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", required=True, choices=["organic_general", "metal_general"])
    parser.add_argument("--pair-a", required=True, help="FUNCTIONAL/BASIS, e.g. PBE0/def2-TZVP")
    parser.add_argument("--pair-b", required=True, help="FUNCTIONAL/BASIS, e.g. QCISD/6-311G(2df,2p)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=arb.DEFAULT_CONCURRENCY)
    parser.add_argument("--allow-expensive", action="store_true",
                         help="Same meaning as run_accuracy_benchmark.py's flag -- off by default, "
                              "since these two settings will very often include one expensive one.")
    parser.add_argument("--export-rccs-batch", type=Path, default=None)
    parser.add_argument("--rccs-results", type=Path, default=None)
    parser.add_argument("--qcl-session-chunk-size", type=int, default=None)
    parser.add_argument("--exclude-species", action="append", default=[])
    args = parser.parse_args()

    pair_a = _parse_setting(args.pair_a)
    pair_b = _parse_setting(args.pair_b)

    print(f"Loading full curated ground truth for {args.cell} (no subsampling)...")
    gt = arb.load_ground_truth(args.cell, None, arb.random.Random(0))  # rng unused when n_target=None
    species, reactions = gt["species"], gt["reactions"]
    print(f"  species={len(species)} reactions={len(reactions)}")

    print("Loading RAG pool for aux-basis precedent lookup...")
    pool = arb.canonicalize_pool(arb.rag.load_pool())
    aux_lookup = arb.build_aux_basis_lookup(pool)

    conditions_by_reaction = {r["reaction_id"]: {"pair_a": pair_a, "pair_b": pair_b} for r in reactions}
    jobs = sorted(arb.collect_sp_jobs(reactions, conditions_by_reaction))
    if args.exclude_species:
        jobs = [j for j in jobs if j[0] not in args.exclude_species]

    if not args.allow_expensive:
        cheap_jobs = [j for j in jobs if arb.is_cheap_dft(j[1], j[2])]
        expensive_jobs = [j for j in jobs if j not in cheap_jobs]
        if expensive_jobs:
            dest = " -- exporting to RCCS batch dir" if args.export_rccs_batch else \
                " (pass --allow-expensive to include them on qcl, or --export-rccs-batch to route them to RCCS)"
            print(f"  cheap-DFT filter: {len(expensive_jobs)} job(s) need a correlated wavefunction "
                  f"method or large basis{dest}")
        if args.export_rccs_batch:
            cell_dir = args.export_rccs_batch / args.cell
            cell_dir.mkdir(parents=True, exist_ok=True)
            meta: dict[str, dict] = {}
            used_lower: set[str] = set()
            for sid, f, b in expensive_jobs:
                node_id = arb.sanitize_node_id(sid, f, b)
                inp_stem, lower, suffix = node_id, node_id.lower(), 2
                while lower in used_lower:
                    inp_stem = f"{node_id}__dup{suffix}"
                    lower = inp_stem.lower()
                    suffix += 1
                used_lower.add(lower)
                inp_text = arb.write_orca_inp(sid, f, b, species, aux_lookup=aux_lookup)
                (cell_dir / f"{inp_stem}.inp").write_text(inp_text, encoding="utf-8")
                meta[node_id] = {
                    "species_id": sid, "functional": f, "basis": b,
                    "aux_basis_plan": arb.classify_aux_basis_plan(f, b, aux_lookup),
                    "inp_stem": inp_stem,
                }
            (cell_dir / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"  wrote {len(expensive_jobs)} .inp file(s) -> {cell_dir}")
        jobs = cheap_jobs

    est_wall_s = (len(jobs) * 20) / max(1, args.concurrency)
    print(f"  distinct SP jobs needed on qcl: {len(jobs)} (~{est_wall_s/60:.1f} min estimate)")

    if args.dry_run:
        print("  --dry-run: skipping ORCA execution.")
        return

    print("Sweeping stale job dirs from any previous crashed run...")
    arb.asyncio.run(arb.sweep_stale_job_dirs())

    if args.qcl_session_chunk_size and len(jobs) > args.qcl_session_chunk_size:
        n_chunks = -(-len(jobs) // args.qcl_session_chunk_size)
        print(f"  submitting {len(jobs)} SP jobs to qcl across {n_chunks} chunk(s) of up to "
              f"{args.qcl_session_chunk_size} (fresh MCP session per chunk)...")
        job_results: dict[tuple[str, str, str], dict] = {}
        total_duration_ms = 0
        for ci in range(0, len(jobs), args.qcl_session_chunk_size):
            chunk = jobs[ci:ci + args.qcl_session_chunk_size]
            chunk_est_s = (len(chunk) * 20) / max(1, args.concurrency)
            cbatch = arb.run_batch_with_watchdog(chunk, species, args.concurrency, max(300.0, chunk_est_s * 3))
            total_duration_ms += cbatch["duration_ms"]
            n_ok = sum(1 for p in cbatch["job_results"].values() if p.get("status") == "ok")
            status = "HUNG (watchdog fired, 0 results)" if cbatch.get("watchdog_timeout") else f"{n_ok}/{len(chunk)} ok"
            print(f"    chunk {ci // args.qcl_session_chunk_size}: {len(chunk)} jobs -- {status}")
            job_results.update(cbatch["job_results"])
            if cbatch.get("watchdog_timeout"):
                arb.kill_stuck_orca_processes()
        batch = {"job_results": job_results, "duration_ms": total_duration_ms}
    else:
        print(f"  submitting {len(jobs)} concurrent SP jobs (concurrency={args.concurrency}) to qcl...")
        batch = arb.run_batch_with_watchdog(jobs, species, args.concurrency, max(600.0, est_wall_s * 3))

    job_results = batch["job_results"]
    for payload in job_results.values():
        payload.setdefault("aux_basis_plan", "not_applicable")
        payload.setdefault("aux_basis_outcome", "not_applicable")

    n_rccs_loaded = 0
    if args.rccs_results:
        rccs_path = args.rccs_results / args.cell / "results.json"
        if rccs_path.exists():
            rccs_raw = json.loads(rccs_path.read_text(encoding="utf-8"))
            for node_id, r in rccs_raw.items():
                key = (r["species_id"], r["functional"], r["basis"])
                job_results[key] = {"status": r["status"], "energy_eh": r.get("energy_eh"),
                                     "error": r.get("error")}
            n_rccs_loaded = len(rccs_raw)
            print(f"  merged {n_rccs_loaded} RCCS result(s) from {rccs_path}")
        else:
            print(f"  [warn] --rccs-results given but {rccs_path} does not exist")

    n_ok = sum(1 for p in job_results.values() if p.get("status") == "ok")
    print(f"  batch done in {batch['duration_ms']/1000:.1f}s: {n_ok}/{len(jobs)+n_rccs_loaded} jobs ok")

    energy_cache = arb.extract_energy_cache(job_results)

    per_reaction = []
    n_both = 0
    for r in reactions:
        row: dict[str, Any] = {
            "reaction_id": r["reaction_id"], "ref_kcal_mol": r["ref_kcal_mol"],
            "species_ids": r["species_ids"], "coeffs": r["coeffs"],
        }
        ok = True
        for label, setting in (("pair_a", pair_a), ("pair_b", pair_b)):
            f, b = setting
            energies = []
            for sid, coeff in zip(r["species_ids"], r["coeffs"]):
                e = energy_cache.get((sid, f, b))
                if e is None:
                    ok = False
                    break
                energies.append(coeff * e)
            if not ok:
                break
            row[f"{label}_kcal_mol"] = sum(energies) * arb.HARTREE_TO_KCAL
            row[f"{label}_abs_error_kcal_mol"] = abs(row[f"{label}_kcal_mol"] - r["ref_kcal_mol"])
        if ok and r["ref_kcal_mol"] is not None and arb.math.isfinite(r["ref_kcal_mol"]):
            n_both += 1
            per_reaction.append(row)

    a_wins = sum(1 for r in per_reaction if r["pair_a_abs_error_kcal_mol"] < r["pair_b_abs_error_kcal_mol"])
    b_wins = sum(1 for r in per_reaction if r["pair_b_abs_error_kcal_mol"] < r["pair_a_abs_error_kcal_mol"])
    print(f"  usable reactions (both settings + valid reference): {n_both}/{len(reactions)}")
    print(f"  pair_a ({pair_a[0]}/{pair_a[1]}) wins: {a_wins}   pair_b ({pair_b[0]}/{pair_b[1]}) wins: {b_wins}")

    out = {
        "cell": args.cell, "pair_a": list(pair_a), "pair_b": list(pair_b),
        "n_reactions_total": len(reactions), "n_reactions_usable": n_both,
        "a_wins": a_wins, "b_wins": b_wins, "per_reaction": per_reaction,
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    LOG_DIR.mkdir(exist_ok=True)
    out_path = LOG_DIR / f"specialist_pair_{args.cell}_{_slug(pair_a)}_vs_{_slug(pair_b)}_{timestamp}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
