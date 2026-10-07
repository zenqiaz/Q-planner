"""
run_pm6_breakdown_eval.py

The orca_only=False retrain of organic_general (job 9903, lora_output_no_orca/
organic_general) regressed to 74.5% both-match on its own val set, down from the
official orca_only=True adapter's 90.0% -- see chapter plan insight #11. Open
question: is that regression concentrated in the newly-added PM6/semi-empirical
population, or spread across the whole (now more diverse) label space?

Re-runs evaluate_cell()'s generation loop (same messages/scoring logic as
train_lora_sft.py, but keeping every per-record prediction instead of just the
first 5 samples) against jsonl_no_orca/sft_val_organic_general.jsonl, then
splits accuracy by whether the gold functional is PM6 vs everything else.

Usage:
    python run_pm6_breakdown_eval.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora_sft import _extract_json, load_jsonl, DEFAULT_OUTPUT_DIR

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "jsonl_no_orca"
OUTPUT_DIR = BASE_DIR / "lora_output_no_orca"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
CELL = "organic_general"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 128


@torch.no_grad()
def generate(model, tokenizer, messages: list[dict]) -> tuple[str, dict | None]:
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return completion, _extract_json(completion)


def main():
    val_path = DATA_DIR / f"sft_val_{CELL}.jsonl"
    val_records = load_jsonl(val_path)
    print(f"Loaded {len(val_records)} val records from {val_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    adapter_dir = OUTPUT_DIR / CELL
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), adapter_name=CELL)
    model.set_adapter(CELL)
    model.eval()

    t0 = time.monotonic()
    results = []
    for i, rec in enumerate(val_records):
        messages = rec["messages"]
        gold = _extract_json(messages[-1]["content"]) or {}
        prompt_messages = messages[:-1]

        completion, pred = generate(model, tokenizer, prompt_messages)
        func_match = bool(pred and pred.get("functional") == gold.get("functional"))
        basis_match = bool(pred and pred.get("basis") == gold.get("basis"))
        results.append({
            "entry_id": rec.get("metadata", {}).get("entry_id"),
            "prompt": prompt_messages[-1]["content"],
            "gold": gold,
            "pred": pred,
            "parsed": pred is not None,
            "func_match": func_match,
            "basis_match": basis_match,
            "both_match": func_match and basis_match,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(val_records)}")

    duration_s = time.monotonic() - t0

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "parse_rate_pct": 100 * sum(r["parsed"] for r in rows) / n,
            "func_match_pct": 100 * sum(r["func_match"] for r in rows) / n,
            "basis_match_pct": 100 * sum(r["basis_match"] for r in rows) / n,
            "both_match_pct": 100 * sum(r["both_match"] for r in rows) / n,
        }

    pm6_rows = [r for r in results if r["gold"].get("functional") == "PM6"]
    non_pm6_rows = [r for r in results if r["gold"].get("functional") != "PM6"]

    pm6_summary = summarize(pm6_rows)
    non_pm6_summary = summarize(non_pm6_rows)
    overall_summary = summarize(results)

    import collections
    pm6_pred_funcs = collections.Counter(r["pred"].get("functional") if r["pred"] else None for r in pm6_rows)

    print(f"\n{'=' * 90}\nSUMMARY (duration {duration_s:.1f}s)\n{'=' * 90}")
    print(f"overall:  n={overall_summary['n']}  both_match={overall_summary['both_match_pct']:.1f}%")
    print(f"PM6 true: n={pm6_summary['n']}  both_match={pm6_summary['both_match_pct']:.1f}%  "
          f"func_match={pm6_summary['func_match_pct']:.1f}%")
    print(f"non-PM6:  n={non_pm6_summary['n']}  both_match={non_pm6_summary['both_match_pct']:.1f}%  "
          f"func_match={non_pm6_summary['func_match_pct']:.1f}%")
    print(f"\nPredicted functional for PM6-true rows:")
    for func, c in pm6_pred_funcs.most_common(10):
        print(f"  {func!r}: {c} ({100 * c / max(pm6_summary['n'], 1):.1f}%)")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = EXPERIMENTS_DIR / f"pm6_breakdown_eval_{ts}.json"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "type": "pm6_breakdown_eval",
            "timestamp_utc": ts,
            "cell": CELL,
            "adapter_dir": str(adapter_dir),
            "val_path": str(val_path),
            "duration_s": duration_s,
            "overall": overall_summary,
            "pm6_true": pm6_summary,
            "non_pm6_true": non_pm6_summary,
            "pm6_true_predicted_functional_counts": dict(pm6_pred_funcs),
            "results": results,
        }, f, indent=2, ensure_ascii=True)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
