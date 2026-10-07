"""One-off: sub-bisect the 30 jobs in metal_general's hung chunk 2 to find the exact culprit."""
from __future__ import annotations

import time

from run_accuracy_benchmark import load_ground_truth, run_batch_with_watchdog

CHUNK2_JOBS = [
    ('ED03', 'B3LYP', 'GEN'), ('ED03', 'B3LYP', 'def2-SVP'), ('ED03', 'PBE', 'def2-TZVP'),
    ('ED03', 'PBE0', 'def2-TZVP'), ('ED03', 'TPSSH', 'def2-SVP'),
    ('ED04', 'B3LYP', 'def2-SVP'), ('ED04', 'PBE', 'def2-TZVP'), ('ED04', 'PBE0', 'def2-TZVP'),
    ('ED04', 'TPSSH', 'def2-SVP'), ('ED04', 'TPSSH', 'def2-TZVP'), ('ED04', 'WB97X-D3', 'def2-SVP'),
    ('ED05', 'B3LYP', 'def2-SVP'), ('ED05', 'M06', 'def2-TZVP'), ('ED05', 'PBE0', 'def2-TZVP'),
    ('ED05', 'PBE0', 'lanl2dz'), ('ED05', 'TPSSH', 'def2-SVP'),
    ('ED09', 'B3LYP', '6-311G**'), ('ED09', 'B3LYP', 'DEF2-TZVP'), ('ED09', 'B3LYP', 'def2-SVP'),
    ('ED09', 'PBE0', 'def2-TZVP'), ('ED09', 'TPSSH', 'def2-SVP'),
    ('ED10', 'B3LYP', 'DEF2-TZVP'), ('ED10', 'B3LYP', 'def2-SVP'), ('ED10', 'CAM-B3LYP', 'def2-TZVP'),
    ('ED10', 'PBE', 'def2-TZVP'), ('ED10', 'PBE0', 'def2-TZVP'), ('ED10', 'TPSSH', 'def2-SVP'),
    ('ED13', 'B3LYP', 'DEF2-TZVP'), ('ED13', 'B3LYP', 'GEN'), ('ED13', 'B3LYP', 'def2-SVP'),
]


def main() -> None:
    import random
    gt = load_ground_truth("metal_general", None, random.Random(0))
    species = gt["species"]
    have = set(species.keys())
    needed = {sid for sid, _, _ in CHUNK2_JOBS}
    print(f"species available for: {needed & have}, MISSING: {needed - have}")

    sub_chunk_size = 5
    subchunks = [CHUNK2_JOBS[i:i + sub_chunk_size] for i in range(0, len(CHUNK2_JOBS), sub_chunk_size)]
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
