"""
eval_full_schema.py

Inference-only, extended re-eval of an already-trained LoRA adapter from
train_lora_sft.py. The training script's own evaluate_cell() only scores
functional+basis exact match and logs 5 sample predictions. This script:

  - scores every field in PARAM_FIELDS (not just functional/basis)
  - reports per-field accuracy (given the field is present in gold),
    missing-field rate, and hallucinated-field rate
  - reports whole-record schema-shape match (pred key set == gold key set)
  - reports whole-record exact match (pred dict == gold dict, all fields)
  - flags any predicted key that isn't even in PARAM_FIELDS at all
  - dumps full predictions for the ENTIRE val set (not capped at 5)

No retraining involved -- loads the base model once and the saved LoRA
adapter(s) as named PEFT adapters, switching between them with set_adapter()
so the ~16GB base model is only loaded once even for multiple cells.

Usage:
    python eval_full_schema.py --cell phase1
    python eval_full_schema.py --cell organic_general
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from train_lora_sft import (
    PARAM_FIELDS,
    PHASE1_CELLS,
    ALL_CELLS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    load_jsonl,
    _extract_json,
)

# 2026-08-15: same two prompts as generate_sft.py / sft_infer_for_benchmark.py, copied locally
# (both files already do this rather than cross-import across the two working directories).
# Used by --prompt-mode to test the CoT-trained adapter under the prompt it was NOT trained
# under -- does the reasoning benefit survive, or is it prompt-triggered only (same open
# question the catastrophic-forgetting probe raised for the JSON-only adapter, mirrored here).
PLAIN_SYSTEM_PROMPT = (
    "You are a QC parameter specialist. Given a molecule description and task, "
    "output ONLY a JSON object with the ORCA calculation parameters."
)
COT_SYSTEM_PROMPT = (
    "You are a QC parameter specialist. Given a molecule description and task, "
    "first give a brief 1-3 sentence chemistry-based justification for your method choice, "
    "then on a new line output ONLY a JSON object with the ORCA calculation parameters. "
    "The following methods require an explicit auxiliary basis (aux_basis field): CCSD, "
    "CCSD(T), MP2, MP3, MP4, QCISD, CISD, CEPA, CASPT2, NEVPT2, and any DLPNO-* method. "
    "For these, set aux_basis to the orbital basis with a '/C' suffix "
    "(e.g. basis 'def2-TZVP' -> aux_basis 'def2-TZVP/C')."
)


@torch.no_grad()
def evaluate_full(model, tokenizer, val_records: list[dict], max_new_tokens: int,
                   system_prompt_override: str | None = None) -> dict:
    model.eval()
    n = len(val_records)
    n_parsed = 0
    n_exact_full = 0
    n_keyset_match = 0
    n_records_with_unknown_field = 0
    unknown_fields_seen = Counter()

    field_stats = {
        f: {"gold_has": 0, "pred_has": 0, "match": 0, "missing": 0, "hallucinated": 0}
        for f in PARAM_FIELDS
    }
    predictions = []

    for rec in val_records:
        messages = rec["messages"]
        gold = _extract_json(messages[-1]["content"]) or {}
        prompt_messages = messages[:-1]
        if system_prompt_override is not None:
            prompt_messages = [
                {"role": "system", "content": system_prompt_override}
                if m["role"] == "system" else m
                for m in prompt_messages
            ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = _extract_json(completion)

        predictions.append({
            "prompt": prompt_messages[-1]["content"],
            "gold": gold,
            "pred_raw": completion,
            "pred_parsed": pred,
        })

        if pred is None:
            continue
        n_parsed += 1

        gold_keys = set(gold.keys())
        pred_keys = set(pred.keys())

        if pred_keys == gold_keys:
            n_keyset_match += 1
        if pred == gold:
            n_exact_full += 1

        unknown_here = pred_keys - set(PARAM_FIELDS)
        if unknown_here:
            n_records_with_unknown_field += 1
            for k in unknown_here:
                unknown_fields_seen[k] += 1

        for f in PARAM_FIELDS:
            g_has = f in gold
            p_has = f in pred
            if g_has:
                field_stats[f]["gold_has"] += 1
            if p_has:
                field_stats[f]["pred_has"] += 1
            if g_has and p_has and gold[f] == pred[f]:
                field_stats[f]["match"] += 1
            if g_has and not p_has:
                field_stats[f]["missing"] += 1
            if p_has and not g_has:
                field_stats[f]["hallucinated"] += 1

    field_report = {}
    for f, s in field_stats.items():
        if s["gold_has"] == 0:
            continue
        field_report[f] = dict(
            s,
            accuracy_given_present=round(s["match"] / s["gold_has"], 4),
        )

    return {
        "n_val": n,
        "n_parsed": n_parsed,
        "parse_rate": round(n_parsed / n, 4) if n else 0.0,
        "full_exact_match_rate": round(n_exact_full / n, 4) if n else 0.0,
        "keyset_shape_match_rate": round(n_keyset_match / n, 4) if n else 0.0,
        "records_with_unknown_field": n_records_with_unknown_field,
        "unknown_fields_seen": dict(unknown_fields_seen),
        "field_accuracy": field_report,
        "predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended full-schema re-eval of saved LoRA adapters")
    p.add_argument("--cell", default="phase1", choices=ALL_CELLS + ["all", "main", "phase1"])
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--eval-max-new-tokens", type=int, default=128)
    p.add_argument("--prompt-mode", choices=["native", "plain", "cot"], default="native",
                   help="native (default): use each val record's own system message as-is "
                        "(the CoT val set's own COT_SYSTEM_PROMPT, if evaluating a CoT-trained "
                        "adapter). plain: force every prompt to PLAIN_SYSTEM_PROMPT (no "
                        "reasoning instruction) regardless of what the val set was built with -- "
                        "tests whether a CoT-trained adapter's benefit is prompt-triggered only. "
                        "cot: force COT_SYSTEM_PROMPT -- for evaluating a non-CoT adapter under "
                        "the reasoning-eliciting prompt it never trained on.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cell == "phase1":
        cells = PHASE1_CELLS
    elif args.cell in ("all", "main"):
        cells = PHASE1_CELLS
    else:
        cells = [args.cell]

    print(f"Base model: {args.base_model}")
    print(f"Cells: {cells}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    prompt_override = {"native": None, "plain": PLAIN_SYSTEM_PROMPT, "cot": COT_SYSTEM_PROMPT}[args.prompt_mode]
    tag = "" if args.prompt_mode == "native" else f"_{args.prompt_mode}prompt"
    print(f"prompt_mode: {args.prompt_mode}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    model = None
    for cell in cells:
        adapter_dir = args.output_dir / cell
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"  [skip] no adapter at {adapter_dir}")
            continue

        val_path = args.data_dir / f"sft_val_{cell}.jsonl"
        val_records = load_jsonl(val_path)
        if not val_records:
            print(f"  [skip] no val records for {cell} ({val_path})")
            continue

        if model is None:
            model = PeftModel.from_pretrained(base_model, str(adapter_dir), adapter_name=cell)
        else:
            model.load_adapter(str(adapter_dir), adapter_name=cell)
        model.set_adapter(cell)

        print(f"\n{'=' * 90}\nFull-schema eval: {cell} ({len(val_records)} val records) "
              f"[prompt_mode={args.prompt_mode}]\n{'=' * 90}")
        report = evaluate_full(model, tokenizer, val_records, args.eval_max_new_tokens,
                                system_prompt_override=prompt_override)

        print(f"  parse_rate={report['parse_rate']:.1%}  "
              f"full_exact_match={report['full_exact_match_rate']:.1%}  "
              f"keyset_shape_match={report['keyset_shape_match_rate']:.1%}  "
              f"records_with_unknown_field={report['records_with_unknown_field']}")
        if report["unknown_fields_seen"]:
            print(f"  unknown fields seen: {report['unknown_fields_seen']}")
        for f, s in sorted(report["field_accuracy"].items(), key=lambda kv: -kv[1]["gold_has"]):
            print(f"    {f:16s} gold_has={s['gold_has']:4d}  "
                  f"acc_given_present={s['accuracy_given_present']:.1%}  "
                  f"missing={s['missing']:3d}  hallucinated={s['hallucinated']:3d}")

        out_path = adapter_dir / f"eval_report_full{tag}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
