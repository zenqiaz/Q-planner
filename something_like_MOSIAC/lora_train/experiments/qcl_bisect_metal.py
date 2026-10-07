"""
qcl_bisect_metal.py

One-off diagnostic: metal_general's 493-job cheap-DFT qcl batch has hung for the FULL watchdog
window (never producing a single result) on 3 consecutive attempts (concurrency=12 twice,
concurrency=6 once) while organic_general's 240-job batch completes cleanly every time. Lower
concurrency made it WORSE (proportionally longer wait, same zero-results outcome), ruling out a
general load/instability explanation -- this points at one specific stuck job that blocks a
semaphore slot forever, eventually starving the whole batch once every slot is occupied by a
hung job.

Reuses run_accuracy_benchmark.py's own job-collection + qcl execution machinery (same reactions,
same seed, same cheap-DFT filter) so the job list here is IDENTICAL to what the real benchmark
run submits -- just split into small chunks with a short per-chunk watchdog, so a hang shows up
in minutes instead of 80.

Usage:
    python qcl_bisect_metal.py --chunk-size 30
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from run_accuracy_benchmark import (
    RANDOM_SEED, canonicalize_pool, canonicalize_sft_predictions, compute_majority_by_cell,
    build_vocab_by_cell, collect_conditions, collect_sp_jobs, is_cheap_dft,
    run_batch_with_watchdog, load_ground_truth, train_rf_by_cell,
)
import rag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-size", type=int, default=30)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--per-chunk-timeout-s", type=float, default=300.0)
    p.add_argument("--start-chunk", type=int, default=0,
                    help="Skip earlier chunks already confirmed healthy in a prior run.")
    p.add_argument("--sft-predictions", type=Path,
                   default=Path("../benchmark_sft_predictions_retrained.json"),
                   help="Same file the real benchmark run uses -- needed for exact job-set parity.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading RAG pool and building the same job list run_accuracy_benchmark.py would...")
    pool = canonicalize_pool(rag.load_pool())
    idx = rag.build_index(pool)
    majority = compute_majority_by_cell(pool)
    vocab = build_vocab_by_cell(pool)
    rng = random.Random(RANDOM_SEED)
    rf_models = train_rf_by_cell(pool)

    sft_predictions = None
    if args.sft_predictions.exists():
        raw_sft = json.loads(args.sft_predictions.read_text(encoding="utf-8"))
        sft_predictions = canonicalize_sft_predictions(raw_sft).get("metal_general")
        print(f"Loaded SFT predictions from {args.sft_predictions} for exact job-set parity.")

    gt = load_ground_truth("metal_general", 35, rng)
    species, reactions = gt["species"], gt["reactions"]
    conditions_by_reaction = collect_conditions(
        reactions, species, "metal_general", majority, vocab, idx, rng, rf_models=rf_models,
        sft_predictions=sft_predictions,
    )
    jobs = sorted(collect_sp_jobs(reactions, conditions_by_reaction))
    cheap_jobs = [j for j in jobs if is_cheap_dft(j[1], j[2])]
    print(f"Reproduced {len(cheap_jobs)} cheap-DFT job(s) for metal_general "
          f"(matches the real benchmark run's job set exactly, same seed/filters).")

    chunks = [cheap_jobs[i:i + args.chunk_size] for i in range(0, len(cheap_jobs), args.chunk_size)]
    print(f"Split into {len(chunks)} chunk(s) of up to {args.chunk_size} jobs each.\n")

    for ci, chunk in enumerate(chunks):
        if ci < args.start_chunk:
            print(f"[chunk {ci}] skipped (--start-chunk={args.start_chunk})")
            continue
        t0 = time.monotonic()
        print(f"[chunk {ci}] {len(chunk)} jobs: {chunk}")
        batch = run_batch_with_watchdog(chunk, species, args.concurrency, args.per_chunk_timeout_s)
        elapsed = time.monotonic() - t0
        job_results = batch["job_results"]
        n_ok = sum(1 for p in job_results.values() if p.get("status") == "ok")
        if batch.get("watchdog_timeout"):
            print(f"[chunk {ci}] HUNG -- watchdog fired after {elapsed:.0f}s, 0 results gathered. "
                  f"Culprit is somewhere in this chunk's {len(chunk)} jobs.\n")
        else:
            failed = [k for k, p in job_results.items() if p.get("status") != "ok"]
            print(f"[chunk {ci}] OK in {elapsed:.0f}s: {n_ok}/{len(chunk)} ok"
                  f"{f', failed (non-hang): {failed}' if failed else ''}\n")


if __name__ == "__main__":
    main()
