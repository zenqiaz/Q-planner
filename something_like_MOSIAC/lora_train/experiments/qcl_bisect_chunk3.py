"""One-off: sub-bisect the 30 jobs in metal_general's hung chunk 3 to find the exact culprit."""
from __future__ import annotations

import random
import time

from run_accuracy_benchmark import load_ground_truth, run_batch_with_watchdog

CHUNK3_JOBS = [
    ('ED13', 'PBE0', 'def2-TZVP'), ('ED13', 'TPSSH', 'def2-SVP'),
    ('ED14', 'B3LYP', 'GEN'), ('ED14', 'B3LYP', 'def2-SVP'), ('ED14', 'PBE', 'def2-TZVP'),
    ('ED14', 'PBE0', 'def2-TZVP'), ('ED14', 'TPSSH', 'def2-SVP'),
    ('ED15', 'B3LYP', 'GEN'), ('ED15', 'B3LYP', 'def2-SVP'), ('ED15', 'BP86', 'def2-tzvp'),
    ('ED15', 'PBE', 'def2-TZVP'), ('ED15', 'PBE0', 'def2-TZVP'), ('ED15', 'TPSSH', 'def2-SVP'),
    ('ED21', 'B3LYP', 'DEF2-TZVP'), ('ED21', 'B3LYP', 'GEN'), ('ED21', 'B3LYP', 'def2-SVP'),
    ('ED21', 'PBE0', 'def2-TZVP'), ('ED21', 'TPSSH', 'def2-SVP'),
    ('ED32', 'B3LYP', 'DEF2-TZVP'), ('ED32', 'B3LYP', 'def2-SVP'), ('ED32', 'PBE', 'def2-TZVP'),
    ('ED32', 'PBE0', 'def2-TZVP'), ('ED32', 'TPSSH', 'def2-SVP'),
    ('ED36', 'B3LYP', '6-31G(2df,p)'), ('ED36', 'B3LYP', '6-31G*'), ('ED36', 'B3LYP', 'def2-SVP'),
    ('ED36', 'PBE0', 'def2-TZVP'), ('ED36', 'TPSSH', 'def2-SVP'), ('ED36', 'WB97X-D3', 'def2-TZVP'),
    ('ED37', 'B3LYP', '6-31G*'),
]


def main() -> None:
    gt = load_ground_truth("metal_general", None, random.Random(0))
    species = gt["species"]
    have = set(species.keys())
    needed = {sid for sid, _, _ in CHUNK3_JOBS}
    print(f"species available for: {needed & have}, MISSING: {needed - have}")

    sub_chunk_size = 5
    subchunks = [CHUNK3_JOBS[i:i + sub_chunk_size] for i in range(0, len(CHUNK3_JOBS), sub_chunk_size)]
    for si, sub in enumerate(subchunks):
        t0 = time.monotonic()
        print(f"[sub {si}] {sub}")
        batch = run_batch_with_watchdog(sub, species, 6, 120.0)
        elapsed = time.monotonic() - t0
        job_results = batch["job_results"]
        n_ok = sum(1 for p in job_results.values() if p.get("status") == "ok")
        if batch.get("watchdog_timeout"):
            print(f"[sub {si}] HUNG after {elapsed:.0f}s -- culprit in {sub}\n")
        else:
            failed = [k for k, p in job_results.items() if p.get("status") != "ok"]
            print(f"[sub {si}] OK in {elapsed:.0f}s: {n_ok}/{len(sub)} ok"
                  f"{f', failed: {failed}' if failed else ''}\n")


if __name__ == "__main__":
    main()
