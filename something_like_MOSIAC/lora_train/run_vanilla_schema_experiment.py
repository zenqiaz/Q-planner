"""
run_vanilla_schema_experiment.py

Adds a fifth condition to the RAG/SFT ablation: "vanilla_schema" -- base model,
no adapter, no RAG examples, but WITH a system-prompt block that tells it the
exact output schema (functional/basis keys) and the in-corpus vocabulary of
valid functionals/bases for that cell.

Motivation (see project chapter plan, insight #10): job 9898's true zero-shot
"vanilla" condition scored 0.0% on both cells, but that's not a clean chemistry-
knowledge floor -- the base model reliably produced valid JSON, just with a
different key schema (`"method"` + a verbose ORCA-input-shaped block) rather
than the flat `{"functional": ..., "basis": ...}` schema SFT teaches, and it was
never told which functionals/bases are even in scope. This condition isolates
"does the base model know good DFT practice, given the output contract" from
"does the base model know the output contract at all."

Reuses the same 912-record v2 input file and the existing job 9898 results file
(adds "vanilla_schema" into each record and re-derives summary_by_cell) so the
final report has all five columns (Vanilla / Vanilla+schema / RAG-alone /
SFT-alone / SFT+RAG) in one place, without re-running the more expensive
sft_alone/sft_rag/rag_alone conditions.

Usage:
    python run_vanilla_schema_experiment.py --input experiments/rag_sft_experiment_inputs_v2_<ts>.json \
        --results experiments/rag_sft_experiment_results_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora_sft import _extract_json

BASE_DIR = Path(__file__).resolve().parent
JSONL_DIR = BASE_DIR / "jsonl"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

SCHEMA_PROMPT_TEMPLATE = (
    "You are a QC parameter specialist. Given a molecule description and task, "
    "output ONLY a JSON object with these exact keys: \"functional\" (the exchange-"
    "correlation functional or wavefunction method) and \"basis\" (the basis set). "
    "Valid functionals for this class of system: {functionals}. "
    "Valid basis sets for this class of system: {bases}. "
    "Choose the single best functional and basis set for the given molecule and task "
    "from those lists."
)


def load_vocab(cell: str) -> tuple[list[str], list[str]]:
    funcs: set[str] = set()
    bases: set[str] = set()
    with (JSONL_DIR / f"sft_{cell}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                params = json.loads(rec["messages"][2]["content"])
            except Exception:
                continue
            func = params.get("functional")
            basis = params.get("basis")
            if func and "=" not in func:
                funcs.add(func)
            if basis and "=" not in basis:
                bases.add(basis)
    return sorted(funcs), sorted(bases)


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


def score(pred: dict | None, true_functional: str | None, true_basis: str | None) -> dict:
    func_match = bool(pred and pred.get("functional") == true_functional)
    basis_match = bool(pred and pred.get("basis") == true_basis)
    return {
        "parsed": pred is not None,
        "func_match": func_match,
        "basis_match": basis_match,
        "both_match": func_match and basis_match,
    }


def latest_file(pattern: str) -> Path:
    candidates = sorted(EXPERIMENTS_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No files matching {pattern} in {EXPERIMENTS_DIR}")
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None,
                     help="rag_sft_experiment_inputs_v2_*.json (user_content source)")
    ap.add_argument("--results", type=Path, default=None,
                     help="rag_sft_experiment_results_*.json (job 9898 output, to extend)")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    in_path = args.input or latest_file("rag_sft_experiment_inputs_v2_*.json")
    results_path = args.results or latest_file("rag_sft_experiment_results_*.json")

    in_payload = json.loads(in_path.read_text(encoding="utf-8"))
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))

    in_records_by_id = {r["entry_id"]: r for r in in_payload["records"]}
    print(f"Loaded {len(in_records_by_id)} input records from {in_path}")
    print(f"Loaded {len(results_payload['results'])} existing results from {results_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    cells = sorted({r["specialist_cell"] for r in results_payload["results"]})
    vocab_by_cell = {cell: load_vocab(cell) for cell in cells}
    for cell, (funcs, bases) in vocab_by_cell.items():
        print(f"  [{cell}] vocab: {len(funcs)} functionals, {len(bases)} bases")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    t0 = time.monotonic()
    n = len(results_payload["results"])
    print(f"\n{'=' * 90}\nvanilla_schema (base model, no adapter, schema+vocab primed): {n} records\n{'=' * 90}")

    for i, res_rec in enumerate(results_payload["results"]):
        entry_id = res_rec["entry_id"]
        cell = res_rec["specialist_cell"]
        in_rec = in_records_by_id[entry_id]
        base_user = in_rec["sft_alone_messages"][1]["content"]

        funcs, bases = vocab_by_cell[cell]
        schema_prompt = SCHEMA_PROMPT_TEMPLATE.format(
            functionals=", ".join(funcs), bases=", ".join(bases),
        )
        messages = [
            {"role": "system", "content": schema_prompt},
            {"role": "user", "content": base_user},
        ]

        completion, pred = generate(model, tokenizer, messages, args.max_new_tokens)
        res_rec["vanilla_schema"] = {
            "completion": completion, "pred": pred,
            **score(pred, res_rec["true_functional"], res_rec["true_basis"]),
        }

        if (i + 1) % 25 == 0:
            print(f"  [vanilla_schema] {i + 1}/{n}")

    duration_s = time.monotonic() - t0

    for cell in cells:
        crows = [r for r in results_payload["results"] if r["specialist_cell"] == cell]
        cn = len(crows)
        summary = results_payload["summary_by_cell"].setdefault(cell, {})
        summary["vanilla_schema_both_match_pct"] = 100 * sum(r["vanilla_schema"]["both_match"] for r in crows) / cn
        summary["vanilla_schema_func_match_pct"] = 100 * sum(r["vanilla_schema"]["func_match"] for r in crows) / cn
        summary["vanilla_schema_parse_rate_pct"] = 100 * sum(r["vanilla_schema"]["parsed"] for r in crows) / cn

    print(f"\n{'=' * 90}\nSUMMARY (duration {duration_s:.1f}s)\n{'=' * 90}")
    print(f"{'cell':<20} {'n':>5} {'Vanilla':>9} {'+Schema':>9} {'RAG-alone':>11} {'SFT-alone':>11} {'SFT+RAG':>11}")
    for cell in cells:
        s = results_payload["summary_by_cell"][cell]
        print(f"{cell:<20} {s['n']:>5} {s.get('vanilla_both_match_pct', 0):>8.1f}% "
              f"{s['vanilla_schema_both_match_pct']:>8.1f}% "
              f"{s['rag_alone_top1_exact_pct']:>10.1f}% "
              f"{s['sft_alone_both_match_pct']:>10.1f}% {s['sft_rag_both_match_pct']:>10.1f}%")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = EXPERIMENTS_DIR / f"rag_sft_experiment_results_with_schema_{ts}.json"
    results_payload["vanilla_schema_duration_s"] = duration_s
    results_payload["vanilla_schema_input_file"] = str(in_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, ensure_ascii=True)
    print(f"\nSaved extended results -> {out_path}")


if __name__ == "__main__":
    main()
