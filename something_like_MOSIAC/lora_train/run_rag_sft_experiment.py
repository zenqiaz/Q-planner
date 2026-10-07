"""
run_rag_sft_experiment.py

GPU step 2 of the "does RAG add lift on top of the fine-tuned LoRA
specialist" experiment. Reads the local experiment-input file produced by
build_rag_sft_experiment_inputs.py (query records + RAG-alone predictions +
both prompt variants, already fixed before this script runs) and generates
FOUR conditions per record:
  - vanilla:   base model, NO adapter, NO RAG examples (sft_alone_messages)
  - rag_alone: retrieval only, no generation (scored from the input file)
  - sft_alone: fine-tuned adapter, no RAG examples
  - sft_rag:   fine-tuned adapter + RAG examples injected

vanilla isolates how much of sft_alone's accuracy comes from the base
model's own pretrained chemistry knowledge vs. what the LoRA adapter
actually learned -- this was a real gap in the original 3-condition design
(the RAG_CLAUDE.md handoff's planned comparison always included a zero-shot
baseline; this script had never actually run it). Generated once per
record, BEFORE any adapter is attached to the base model, since PeftModel
wrapping changes what a plain model.generate() call sees.

Reuses the multi-adapter loading pattern from eval_full_schema.py (base
model loaded once, adapters attached/switched via PeftModel) and
_extract_json from train_lora_sft.py.

Usage:
    python run_rag_sft_experiment.py --input experiments/rag_sft_experiment_inputs_<ts>.json
    python run_rag_sft_experiment.py   # auto-picks the newest experiments/*.json
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from train_lora_sft import DEFAULT_OUTPUT_DIR, _extract_json

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def latest_input_file() -> Path:
    candidates = sorted(glob.glob(str(EXPERIMENTS_DIR / "rag_sft_experiment_inputs_*.json")))
    if not candidates:
        raise FileNotFoundError(f"No experiment-input files in {EXPERIMENTS_DIR}")
    return Path(candidates[-1])


@torch.no_grad()
def generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> tuple[str, dict | None]:
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return completion, _extract_json(completion)


def score(pred: dict | None, rec: dict) -> dict:
    func_match = bool(pred and pred.get("functional") == rec["true_functional"])
    basis_match = bool(pred and pred.get("basis") == rec["true_basis"])
    return {
        "parsed": pred is not None,
        "func_match": func_match,
        "basis_match": basis_match,
        "both_match": func_match and basis_match,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    in_path = args.input or latest_input_file()
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    records = payload["records"]
    print(f"Loaded {len(records)} experiment records from {in_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    base_model.eval()

    t0 = time.monotonic()

    print(f"\n{'=' * 90}\nvanilla (base model, no adapter, no RAG): {len(records)} records\n{'=' * 90}")
    vanilla_by_id: dict[str, dict] = {}
    for i, rec in enumerate(records):
        completion, pred = generate(base_model, tokenizer, rec["sft_alone_messages"], args.max_new_tokens)
        vanilla_by_id[rec["entry_id"]] = {
            "completion": completion, "pred": pred, **score(pred, rec),
        }
        if (i + 1) % 25 == 0:
            print(f"  [vanilla] {i + 1}/{len(records)}")

    model = None
    results = []

    by_cell: dict[str, list[dict]] = {}
    for rec in records:
        by_cell.setdefault(rec["specialist_cell"], []).append(rec)

    for cell, cell_records in by_cell.items():
        adapter_dir = args.output_dir / cell
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"  [skip] no adapter at {adapter_dir}")
            continue

        if model is None:
            model = PeftModel.from_pretrained(base_model, str(adapter_dir), adapter_name=cell)
        else:
            model.load_adapter(str(adapter_dir), adapter_name=cell)
        model.set_adapter(cell)
        model.eval()

        print(f"\n{'=' * 90}\n{cell}: {len(cell_records)} records\n{'=' * 90}")
        for i, rec in enumerate(cell_records):
            sft_completion, sft_pred = generate(model, tokenizer, rec["sft_alone_messages"], args.max_new_tokens)
            rag_completion, rag_pred = generate(model, tokenizer, rec["sft_rag_messages"], args.max_new_tokens)

            sft_score = score(sft_pred, rec)
            rag_score = score(rag_pred, rec)

            results.append({
                "entry_id": rec["entry_id"],
                "specialist_cell": cell,
                "true_functional": rec["true_functional"],
                "true_basis": rec["true_basis"],
                "vanilla": vanilla_by_id[rec["entry_id"]],
                "rag_alone": rec["rag_alone"],
                "sft_alone": {"completion": sft_completion, "pred": sft_pred, **sft_score},
                "sft_rag": {"completion": rag_completion, "pred": rag_pred, **rag_score},
            })

            if (i + 1) % 25 == 0:
                print(f"  [{cell}] {i + 1}/{len(cell_records)}")

    duration_s = time.monotonic() - t0

    summary_by_cell = {}
    for cell, cell_records in by_cell.items():
        crows = [r for r in results if r["specialist_cell"] == cell]
        if not crows:
            continue
        n = len(crows)
        summary_by_cell[cell] = {
            "n": n,
            "vanilla_both_match_pct": 100 * sum(r["vanilla"]["both_match"] for r in crows) / n,
            "rag_alone_top1_exact_pct": 100 * sum(r["rag_alone"]["top1_exact_correct"] for r in crows) / n,
            "sft_alone_both_match_pct": 100 * sum(r["sft_alone"]["both_match"] for r in crows) / n,
            "sft_rag_both_match_pct": 100 * sum(r["sft_rag"]["both_match"] for r in crows) / n,
            "vanilla_parse_rate_pct": 100 * sum(r["vanilla"]["parsed"] for r in crows) / n,
            "sft_alone_parse_rate_pct": 100 * sum(r["sft_alone"]["parsed"] for r in crows) / n,
            "sft_rag_parse_rate_pct": 100 * sum(r["sft_rag"]["parsed"] for r in crows) / n,
        }

    print(f"\n{'=' * 90}\nSUMMARY (duration {duration_s:.1f}s)\n{'=' * 90}")
    print(f"{'cell':<20} {'n':>5} {'Vanilla':>9} {'RAG-alone':>11} {'SFT-alone':>11} {'SFT+RAG':>11}")
    for cell, s in summary_by_cell.items():
        print(f"{cell:<20} {s['n']:>5} {s['vanilla_both_match_pct']:>8.1f}% "
              f"{s['rag_alone_top1_exact_pct']:>10.1f}% "
              f"{s['sft_alone_both_match_pct']:>10.1f}% {s['sft_rag_both_match_pct']:>10.1f}%")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = EXPERIMENTS_DIR / f"rag_sft_experiment_results_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "type": "rag_sft_experiment_results",
            "timestamp_utc": ts,
            "input_file": str(in_path),
            "base_model": args.base_model,
            "duration_s": duration_s,
            "summary_by_cell": summary_by_cell,
            "results": results,
        }, f, indent=2, ensure_ascii=True)
    print(f"\nSaved full results -> {out_path}")


if __name__ == "__main__":
    main()
