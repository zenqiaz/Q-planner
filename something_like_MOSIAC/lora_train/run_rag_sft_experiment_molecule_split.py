"""
run_rag_sft_experiment_molecule_split.py

GPU inference for the molecule-identity-level split's ablation
(vanilla / rag_alone / sft_alone / sft_rag, scored on `records` = val, all
genuinely novel molecules by construction) AND the memorization-vs-
generalization check (sft_alone scored on `train_sample_records`, a seeded
sample of the exact records the adapter was fine-tuned on -- directly
comparable to sft_alone's val accuracy, since this split has zero SMILES
leakage between the two).

Mirrors run_rag_sft_experiment.py's structure (same generate()/score()
helpers, same multi-adapter loading pattern) but reads the molecule-split
input file (build_rag_sft_experiment_inputs_molecule_split.py) and points
--output-dir at lora_output_molecule_split by default.

Usage:
    python run_rag_sft_experiment_molecule_split.py --input experiments/rag_sft_experiment_inputs_molecule_split_<ts>.json
    python run_rag_sft_experiment_molecule_split.py   # auto-picks the newest matching file
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

from train_lora_sft import _extract_json

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
DEFAULT_OUTPUT_DIR = BASE_DIR / "lora_output_molecule_split"


def latest_input_file() -> Path:
    candidates = sorted(glob.glob(str(EXPERIMENTS_DIR / "rag_sft_experiment_inputs_molecule_split_*.json")))
    if not candidates:
        raise FileNotFoundError(f"No molecule-split experiment-input files in {EXPERIMENTS_DIR}")
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
    val_records = payload["records"]
    train_sample_records = payload.get("train_sample_records", [])
    print(f"Loaded {len(val_records)} val records + {len(train_sample_records)} "
          f"train-sample records from {in_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    base_model.eval()

    t0 = time.monotonic()

    print(f"\n{'=' * 90}\nvanilla (base model, no adapter, no RAG): {len(val_records)} val records\n{'=' * 90}")
    vanilla_by_id: dict[str, dict] = {}
    for i, rec in enumerate(val_records):
        completion, pred = generate(base_model, tokenizer, rec["sft_alone_messages"], args.max_new_tokens)
        vanilla_by_id[rec["entry_id"]] = {
            "completion": completion, "pred": pred, **score(pred, rec),
        }
        if (i + 1) % 25 == 0:
            print(f"  [vanilla] {i + 1}/{len(val_records)}")

    model = None
    val_results = []
    train_results = []

    val_by_cell: dict[str, list[dict]] = {}
    for rec in val_records:
        val_by_cell.setdefault(rec["specialist_cell"], []).append(rec)

    train_by_cell: dict[str, list[dict]] = {}
    for rec in train_sample_records:
        train_by_cell.setdefault(rec["specialist_cell"], []).append(rec)

    cells = sorted(set(val_by_cell) | set(train_by_cell))
    for cell in cells:
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

        cell_val_records = val_by_cell.get(cell, [])
        print(f"\n{'=' * 90}\n{cell}: {len(cell_val_records)} val records (ablation)\n{'=' * 90}")
        for i, rec in enumerate(cell_val_records):
            sft_completion, sft_pred = generate(model, tokenizer, rec["sft_alone_messages"], args.max_new_tokens)
            rag_completion, rag_pred = generate(model, tokenizer, rec["sft_rag_messages"], args.max_new_tokens)

            sft_score = score(sft_pred, rec)
            rag_score = score(rag_pred, rec)

            val_results.append({
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
                print(f"  [{cell}] {i + 1}/{len(cell_val_records)}")

        cell_train_records = train_by_cell.get(cell, [])
        print(f"\n{'=' * 90}\n{cell}: {len(cell_train_records)} train-sample records "
              f"(memorization check, sft_alone only)\n{'=' * 90}")
        for i, rec in enumerate(cell_train_records):
            sft_completion, sft_pred = generate(model, tokenizer, rec["sft_alone_messages"], args.max_new_tokens)
            sft_score = score(sft_pred, rec)
            train_results.append({
                "entry_id": rec["entry_id"],
                "specialist_cell": cell,
                "true_functional": rec["true_functional"],
                "true_basis": rec["true_basis"],
                "sft_alone": {"completion": sft_completion, "pred": sft_pred, **sft_score},
            })
            if (i + 1) % 25 == 0:
                print(f"  [{cell} train-sample] {i + 1}/{len(cell_train_records)}")

    duration_s = time.monotonic() - t0

    summary_by_cell = {}
    for cell in cells:
        crows = [r for r in val_results if r["specialist_cell"] == cell]
        trows = [r for r in train_results if r["specialist_cell"] == cell]
        if not crows:
            continue
        n = len(crows)
        entry = {
            "n_val": n,
            "vanilla_both_match_pct": 100 * sum(r["vanilla"]["both_match"] for r in crows) / n,
            "rag_alone_top1_exact_pct": 100 * sum(r["rag_alone"]["top1_exact_correct"] for r in crows) / n,
            "sft_alone_both_match_pct": 100 * sum(r["sft_alone"]["both_match"] for r in crows) / n,
            "sft_rag_both_match_pct": 100 * sum(r["sft_rag"]["both_match"] for r in crows) / n,
            "vanilla_parse_rate_pct": 100 * sum(r["vanilla"]["parsed"] for r in crows) / n,
            "sft_alone_parse_rate_pct": 100 * sum(r["sft_alone"]["parsed"] for r in crows) / n,
            "sft_rag_parse_rate_pct": 100 * sum(r["sft_rag"]["parsed"] for r in crows) / n,
        }
        if trows:
            nt = len(trows)
            entry["n_train_sample"] = nt
            entry["memorized_sft_alone_both_match_pct"] = 100 * sum(
                r["sft_alone"]["both_match"] for r in trows) / nt
        summary_by_cell[cell] = entry

    print(f"\n{'=' * 90}\nSUMMARY (duration {duration_s:.1f}s)\n{'=' * 90}")
    print(f"{'cell':<18} {'n':>5} {'Vanilla':>9} {'RAG-alone':>11} {'SFT-alone':>11} "
          f"{'SFT+RAG':>11} {'Memorized(train)':>18}")
    for cell, s in summary_by_cell.items():
        mem = s.get("memorized_sft_alone_both_match_pct")
        mem_str = f"{mem:.1f}%" if mem is not None else "n/a"
        print(f"{cell:<18} {s['n_val']:>5} {s['vanilla_both_match_pct']:>8.1f}% "
              f"{s['rag_alone_top1_exact_pct']:>10.1f}% "
              f"{s['sft_alone_both_match_pct']:>10.1f}% {s['sft_rag_both_match_pct']:>10.1f}% "
              f"{mem_str:>18}")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = EXPERIMENTS_DIR / f"rag_sft_experiment_results_molecule_split_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "type": "rag_sft_experiment_results_molecule_split",
            "timestamp_utc": ts,
            "input_file": str(in_path),
            "base_model": args.base_model,
            "duration_s": duration_s,
            "summary_by_cell": summary_by_cell,
            "val_results": val_results,
            "train_sample_results": train_results,
        }, f, indent=2, ensure_ascii=True)
    print(f"\nSaved full results -> {out_path}")


if __name__ == "__main__":
    main()
