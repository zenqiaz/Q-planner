"""
run_rare_filtered_breakdown_eval.py

Job 14122 (organic_general retrained on jsonl_no_orca_rare_filtered/, dropping the
15 <20-example functionals) only reached 75.4% both-match -- barely above the
original orca_only=False retrain's 74.5% (job 9903), despite removing exactly the
classes that scored 0% there. Expected result if the regression were purely a
long-tail class-imbalance problem: ~81-82% (weighted average of PM6's 100% and the
ORCA-valid non-PM6 subset's 80.2%, see chapter plan insight #11). The actual 75.4%
suggests the BLYP/B3LYP basis-selection-specific degradation found in job 14046's
breakdown (basis_match 83.6% vs 93.2% official for those classes) persisted even
after removing the rare tail -- this script gets the full per-functional-class
breakdown to confirm.

Usage:
    python run_rare_filtered_breakdown_eval.py
"""
from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora_sft import _extract_json, load_jsonl

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "jsonl_no_orca_rare_filtered"
OUTPUT_DIR = BASE_DIR / "lora_output_no_orca_rare_filtered"
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

    by_func = collections.defaultdict(list)
    for r in results:
        by_func[r["gold"].get("functional")].append(r)

    n = len(results)
    overall_func = 100 * sum(r["func_match"] for r in results) / n
    overall_basis = 100 * sum(r["basis_match"] for r in results) / n
    overall_both = 100 * sum(r["both_match"] for r in results) / n

    print(f"\n{'=' * 90}\nSUMMARY (duration {duration_s:.1f}s)\n{'=' * 90}")
    print(f"overall: n={n}  func_match={overall_func:.1f}%  basis_match={overall_basis:.1f}%  both_match={overall_both:.1f}%")
    print()
    print(f"{'functional':<12}{'n':>6}{'func_acc':>11}{'basis_acc':>12}{'both_acc':>11}")
    for func, rows in sorted(by_func.items(), key=lambda kv: -len(kv[1])):
        fn = len(rows)
        fa = 100 * sum(r["func_match"] for r in rows) / fn
        ba = 100 * sum(r["basis_match"] for r in rows) / fn
        ba2 = 100 * sum(r["both_match"] for r in rows) / fn
        print(f"{str(func):<12}{fn:>6}{fa:>10.1f}%{ba:>11.1f}%{ba2:>10.1f}%")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = EXPERIMENTS_DIR / f"rare_filtered_breakdown_eval_{ts}.json"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "type": "rare_filtered_breakdown_eval",
            "timestamp_utc": ts,
            "cell": CELL,
            "adapter_dir": str(adapter_dir),
            "val_path": str(val_path),
            "duration_s": duration_s,
            "overall": {"n": n, "func_match_pct": overall_func, "basis_match_pct": overall_basis, "both_match_pct": overall_both},
            "results": results,
        }, f, indent=2, ensure_ascii=True)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
