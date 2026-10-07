"""
probe_catastrophic_forgetting.py

Tests whether LoRA SFT fine-tuning degraded the BASE model's latent chemistry-reasoning
capability, as opposed to the base model simply never having it. Motivated by a direct question
during discussion of the three failure-mode mechanisms (chapter_plan insight #15): all three
show the trained selectors pattern-matching on corpus co-occurrence rather than reasoning from
chemistry -- but is that because Llama-3.1-8B-Instruct never had this reasoning, or because
narrow JSON-schema fine-tuning suppressed/overwrote it (catastrophic forgetting)?

Design: ask the SAME open-ended chemistry-reasoning question (NOT the narrow method-selection
JSON task) to (a) the raw base model, no adapter, and (b) the base model + the SFT adapter that
actually produced the accuracy-benchmark's selector_sft predictions
(lora_output_molecule_split_auxhint). Uses a NEUTRAL system prompt for both, deliberately
different from the training system prompt (which instructs "output ONLY a JSON object") --
using the training prompt would confound "lost the reasoning" with "trained to always emit
JSON regardless of the question," which is a related but distinct effect also worth checking
separately (see the --training-prompt-too flag).

One question per failure mechanism (chapter_plan insight #15):
  Q1 (mechanism 3): GGA delocalization error for atomization energies
  Q2 (mechanism 2): why QCISD needs an adequate basis to pay off
  Q3 (mechanism 1): why HF is especially bad for multiply-bonded systems

Usage (on NII, inside an sbatch job):
    python probe_catastrophic_forgetting.py --output forgetting_probe_results.json
    python probe_catastrophic_forgetting.py --output forgetting_probe_results.json --training-prompt-too
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

NEUTRAL_SYSTEM_PROMPT = "You are a helpful computational chemistry assistant."

TRAINING_SYSTEM_PROMPT = (
    "You are a QC parameter specialist. Given a molecule description and task, "
    "output ONLY a JSON object with the ORCA calculation parameters. "
    "The following methods require an explicit auxiliary basis (aux_basis field): CCSD, "
    "CCSD(T), MP2, MP3, MP4, QCISD, CISD, CEPA, CASPT2, NEVPT2, and any DLPNO-* method. "
    "For these, set aux_basis to the orbital basis with a '/C' suffix "
    "(e.g. basis 'def2-TZVP' -> aux_basis 'def2-TZVP/C')."
)

QUESTIONS = {
    "mechanism_3_gga_bias": (
        "In density functional theory, why might a GGA functional like PBE give a "
        "systematically less accurate atomization energy than a hybrid functional like B3LYP? "
        "Answer in 3-4 sentences."
    ),
    "mechanism_2_wf_basis": (
        "If I run a QCISD calculation with a small basis set like 6-31G*, why might the result "
        "fail to show QCISD's usual accuracy advantage over a cheaper method? Answer in 3-4 "
        "sentences."
    ),
    "mechanism_1_hf_multiple_bonds": (
        "Why might Hartree-Fock theory be a particularly poor choice for computing the "
        "atomization energy of a molecule with multiple triple bonds, like cyanogen (N#C-C#N)? "
        "Answer in 3-4 sentences."
    ),
}


@torch.no_grad()
def ask(model, tokenizer, system_prompt: str, question: str, max_new_tokens: int = 220) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    gen = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapter-dir", type=Path,
                    default=Path("lora_output_molecule_split_auxhint/organic_general"),
                    help="The adapter that actually produced selector_sft's real predictions.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-prompt-too", action="store_true",
                    help="Also ask the fine-tuned model under its OWN training system prompt, "
                         "to check for mode-collapse (always emits JSON regardless of question) "
                         "as a distinct effect from lost reasoning under a neutral prompt.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    base_model.eval()

    results: dict[str, dict] = {}

    print("\n=== BASE MODEL (no adapter), neutral system prompt ===")
    for key, q in QUESTIONS.items():
        ans = ask(base_model, tokenizer, NEUTRAL_SYSTEM_PROMPT, q)
        results.setdefault(key, {})["base_neutral"] = ans
        print(f"\n[{key}]\n{ans}\n")

    print("\nLoading SFT adapter...")
    ft_model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    ft_model.eval()

    print("\n=== FINE-TUNED MODEL (SFT adapter), neutral system prompt ===")
    for key, q in QUESTIONS.items():
        ans = ask(ft_model, tokenizer, NEUTRAL_SYSTEM_PROMPT, q)
        results.setdefault(key, {})["finetuned_neutral"] = ans
        print(f"\n[{key}]\n{ans}\n")

    if args.training_prompt_too:
        print("\n=== FINE-TUNED MODEL (SFT adapter), TRAINING system prompt (mode-collapse check) ===")
        for key, q in QUESTIONS.items():
            ans = ask(ft_model, tokenizer, TRAINING_SYSTEM_PROMPT, q)
            results.setdefault(key, {})["finetuned_training_prompt"] = ans
            print(f"\n[{key}]\n{ans}\n")

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
