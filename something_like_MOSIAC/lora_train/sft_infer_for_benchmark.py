"""
sft_infer_for_benchmark.py

Inference-only: runs the trained LoRA adapters (organic_general, metal_general) against the
exact reaction species used by run_accuracy_benchmark.py's accuracy experiment, producing a
selector_sft prediction for each reaction. Not eval against the corpus's own held-out val set
(that's eval_full_schema.py's job) -- this scores against GMTKN55/W4-11/MOR41/TMC151 species,
an entirely different, out-of-corpus input source.

Prompt format matches generate_sft.py's record_to_sft() EXACTLY (same SYSTEM_PROMPT, same user
message template) so the model sees the same distribution it was trained on, just with input
values (formula/n_atoms/system_type/charge/multiplicity/task_type) drawn from the accuracy
benchmark's ground-truth species instead of the NOMAD corpus.

Input:  JSON produced by run_accuracy_benchmark.py --export-sft-input, shape
        {cell: [{reaction_id, formula, n_atoms, system_type, charge, multiplicity, task_type}, ...]}
Output: --output: JSON {cell: {reaction_id: [functional, basis]}} -- feed straight back into
        run_accuracy_benchmark.py --sft-predictions.
        --full-output: JSON {cell: {reaction_id: {full predicted PARAM_FIELDS dict}}} -- the
        model's complete prediction (aux_basis, dispersion, grid_level, ...), not just
        functional/basis. 2026-08-11: aux_basis IS one of the trained PARAM_FIELDS (see
        train_lora_sft.py), so the already-trained adapter should already be capable of
        predicting it -- no retraining needed to check this, just re-inference that actually
        keeps the full output instead of discarding everything but functional/basis.

Usage (on NII, inside an sbatch job -- see run_benchmark_sft_infer.sbatch):
    python sft_infer_for_benchmark.py --input benchmark_sft_input.json --output benchmark_sft_predictions.json

2026-08-17: default system prompt switched to COT_SYSTEM_PROMPT. The prompt-dependence test
(job 27082, eval_full_schema.py --prompt-mode plain vs. native) found the CoT-trained adapter's
accuracy gain is prompt-triggered, not weight-baked in -- querying it under the OLD plain
SYSTEM_PROMPT collapsed metal_general's functional accuracy to 13.6% (below the pre-CoT
baseline's 77.6%), while basis stayed unaffected. Since lora_output_molecule_split_cot is the
adapter this script is now expected to be pointed at by default, COT_SYSTEM_PROMPT is the
default too -- use --plain-prompt to query an OLDER, non-CoT adapter
(lora_output_molecule_split / lora_output_molecule_split_auxhint), which was never trained on
the reasoning-eliciting prompt and should NOT receive it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from train_lora_sft import DEFAULT_OUTPUT_DIR, _extract_json

# Must match generate_sft.py's SYSTEM_PROMPT exactly -- this is what the OLDER (non-CoT)
# adapters were trained on. Use --plain-prompt to select this for those adapters.
SYSTEM_PROMPT = (
    "You are a QC parameter specialist. Given a molecule description and task, "
    "output ONLY a JSON object with the ORCA calculation parameters."
)

# Must match generate_sft.py's COT_SYSTEM_PROMPT exactly (added 2026-08-14, see cot_templates.py
# / generate_sft.py --cot) -- this is what lora_output_molecule_split_cot was trained on, and is
# now the DEFAULT prompt below (see module docstring: querying that adapter under the plain
# prompt above collapsed metal_general functional accuracy to 13.6%, job 27082).
COT_SYSTEM_PROMPT = (
    "You are a QC parameter specialist. Given a molecule description and task, "
    "first give a brief 1-3 sentence chemistry-based justification for your method choice, "
    "then on a new line output ONLY a JSON object with the ORCA calculation parameters. "
    "The following methods require an explicit auxiliary basis (aux_basis field): CCSD, "
    "CCSD(T), MP2, MP3, MP4, QCISD, CISD, CEPA, CASPT2, NEVPT2, and any DLPNO-* method. "
    "For these, set aux_basis to the orbital basis with a '/C' suffix "
    "(e.g. basis 'def2-TZVP' -> aux_basis 'def2-TZVP/C')."
)

# 2026-08-12: originally an inference-time-only test against the OLD adapter (came back
# negative -- no retraining, no effect, see aux_basis_sft_generalization_finding_2026-08-11.md).
# generate_sft.py's canonical SYSTEM_PROMPT now includes this same text verbatim (concatenated,
# character-for-character identical to SYSTEM_PROMPT + AUX_BASIS_HINT here), so the NEW
# lora_output_molecule_split_auxhint adapter was actually trained on this exact prompt --
# --aux-hint is now the correct way to query THAT adapter, not just a one-off test. Querying the
# OLD lora_output_molecule_split adapter must still omit --aux-hint (it never saw this text).
# Method list matches run_accuracy_benchmark.py's _WF_CORRELATED_PATTERN exactly.
AUX_BASIS_HINT = (
    " The following methods require an explicit auxiliary basis (aux_basis field): CCSD, "
    "CCSD(T), MP2, MP3, MP4, QCISD, CISD, CEPA, CASPT2, NEVPT2, and any DLPNO-* method. "
    "For these, set aux_basis to the orbital basis with a '/C' suffix "
    "(e.g. basis 'def2-TZVP' -> aux_basis 'def2-TZVP/C')."
)

# 2026-08-12: cheap test #2 -- a rule STATEMENT (AUX_BASIS_HINT above) plus full retraining on
# it still didn't work, root-caused to a total absence of ANY training example pairing QCISD (or
# any wavefunction-correlated method, in organic_general specifically) with a real aux_basis --
# the model had nothing to imitate even once told the rule. A full worked EXAMPLE (not just a
# rule) is a stronger, different kind of signal: standard few-shot in-context learning, tested
# here at INFERENCE time only (no retraining) before considering baking a synthetic example into
# training data. Real record from the pool (specialist_cell=highlevel_SP, the Zn-enzyme DLPNO-
# CCSD(T)/MP2 benchmark -- not organic_general/metal_general, but real, not fabricated data) --
# format matches build_user_message()/generate_sft.py's record_to_sft() target exactly.
FEWSHOT_USER = "Molecule: Zn, 1 atoms, TM_closed, charge=0, mult=1\nTask: SP"
FEWSHOT_ASSISTANT = '{"functional": "DLPNO-CCSD(T)", "basis": "def2-SVP", "aux_basis": "def2-SVP/C"}'


def build_user_message(row: dict) -> str:
    """Must match generate_sft.py's record_to_sft() user-message template exactly."""
    return (
        f"Molecule: {row['formula'] or 'unknown'}, {row['n_atoms'] or '?'} atoms, "
        f"{row['system_type']}, charge={row['charge']}, mult={row['multiplicity']}\n"
        f"Task: {row['task_type']}"
    )


@torch.no_grad()
def infer_cell(model, tokenizer, rows: list[dict], max_new_tokens: int,
                system_prompt: str = SYSTEM_PROMPT, fewshot: bool = False,
                ) -> tuple[dict[str, list], dict[str, dict]]:
    model.eval()
    out: dict[str, list] = {}
    full_out: dict[str, dict] = {}
    for row in rows:
        messages = [{"role": "system", "content": system_prompt}]
        if fewshot:
            messages.append({"role": "user", "content": FEWSHOT_USER})
            messages.append({"role": "assistant", "content": FEWSHOT_ASSISTANT})
        messages.append({"role": "user", "content": build_user_message(row)})
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion = tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = _extract_json(completion)
        f = pred.get("functional") if pred else None
        b = pred.get("basis") if pred else None
        if f and b:
            out[row["reaction_id"]] = [f, b]
            full_out[row["reaction_id"]] = pred
        else:
            print(f"  [warn] {row['reaction_id']}: no parseable (functional,basis) -- raw: {completion[:200]!r}")
    return out, full_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT inference for the accuracy benchmark's reactions")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--full-output", type=Path, default=None,
                    help="Also write the complete predicted PARAM_FIELDS dict per reaction "
                         "(aux_basis, dispersion, grid_level, ...), not just functional/basis. "
                         "Default: --output with '_full' inserted before the extension.")
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-new-tokens", type=int, default=200,
                    help="Default raised from 128 to 200 (2026-08-17): COT_SYSTEM_PROMPT "
                         "completions include a reasoning span before the JSON and run longer "
                         "than the old JSON-only completions.")
    p.add_argument("--plain-prompt", action="store_true",
                    help="Use SYSTEM_PROMPT (old JSON-only instruction, no reasoning span) "
                         "instead of the new default COT_SYSTEM_PROMPT. Required when querying "
                         "an OLDER, non-CoT adapter (lora_output_molecule_split / "
                         "lora_output_molecule_split_auxhint) -- those were never trained on "
                         "COT_SYSTEM_PROMPT and job 27082 showed querying a CoT adapter under "
                         "THIS plain prompt instead collapses metal_general functional accuracy "
                         "to 13.6%%, so get this flag right for whichever adapter --output-dir "
                         "actually points at.")
    p.add_argument("--aux-hint", action="store_true",
                    help="Append AUX_BASIS_HINT to the system prompt at INFERENCE time only "
                         "(no retraining) -- cheap test of whether the already-trained adapter "
                         "can use an explicit rule it never saw during training. Only meaningful "
                         "with --plain-prompt: COT_SYSTEM_PROMPT already includes this exact "
                         "hint text, so it is ignored (not double-appended) without --plain-prompt.")
    p.add_argument("--fewshot", action="store_true",
                    help="Insert one real worked example (FEWSHOT_USER/FEWSHOT_ASSISTANT, a "
                         "real DLPNO-CCSD(T) pool record) before the actual query, at INFERENCE "
                         "time only -- cheap test of few-shot in-context learning vs. a "
                         "declarative rule statement alone.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cells = list(payload.keys())

    if args.plain_prompt:
        system_prompt = SYSTEM_PROMPT + AUX_BASIS_HINT if args.aux_hint else SYSTEM_PROMPT
    else:
        if args.aux_hint:
            print("[warn] --aux-hint ignored: COT_SYSTEM_PROMPT already includes that hint "
                  "text verbatim -- pass --plain-prompt too if you actually meant the old "
                  "plain-prompt + aux-hint combination.")
        system_prompt = COT_SYSTEM_PROMPT

    print(f"Base model: {args.base_model}")
    print(f"Adapter dir: {args.output_dir}")
    print(f"Cells: {cells}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"prompt: {'plain' if args.plain_prompt else 'cot'}"
          f"{'+aux_hint' if args.plain_prompt and args.aux_hint else ''}")
    print(f"fewshot: {args.fewshot}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    full_output_path = args.full_output or args.output.with_stem(args.output.stem + "_full")

    model = None
    results: dict[str, dict] = {}
    full_results: dict[str, dict] = {}
    for cell in cells:
        rows = payload[cell]
        adapter_dir = args.output_dir / cell
        if not (adapter_dir / "adapter_config.json").exists():
            print(f"  [skip] no adapter at {adapter_dir}")
            continue

        if model is None:
            model = PeftModel.from_pretrained(base_model, str(adapter_dir), adapter_name=cell)
        else:
            model.load_adapter(str(adapter_dir), adapter_name=cell)
        model.set_adapter(cell)

        print(f"\n{'=' * 80}\nInferring {cell} ({len(rows)} reactions)\n{'=' * 80}")
        cell_out, cell_full_out = infer_cell(model, tokenizer, rows, args.max_new_tokens,
                                              system_prompt, args.fewshot)
        print(f"  {len(cell_out)}/{len(rows)} parsed successfully")
        n_aux = sum(1 for p in cell_full_out.values() if p.get("aux_basis"))
        print(f"  {n_aux}/{len(cell_full_out)} predictions include a non-null aux_basis")
        results[cell] = cell_out
        full_results[cell] = cell_full_out

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")

    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output_path.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
    print(f"Wrote {full_output_path}")


if __name__ == "__main__":
    main()
