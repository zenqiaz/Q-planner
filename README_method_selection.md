# Route, Then Learn — code and artifacts for the DFT method-selection paper

This branch (`rag-retrieval`) of the **Q-planner** repository holds the code and artifacts behind
the paper *"Route, Then Learn: A General Recipe for Automating DFT Method Selection from Literature
Precedent"* (J. Chem. Inf. Model., submitted).

It is a branch rather than a separate repository because the two projects are the same system seen
from two sides, and separating them would hide that relationship:

| | what it is | where it is documented |
|---|---|---|
| **Q-planner** | the execution harness — turns a natural-language request into a deterministic ORCA workflow | [`readme.md`](readme.md) |
| **this paper** | how the harness decides *which method* to run, and whether learning that from literature precedent works | this file |

The paper's central result is about the method-selection layer; Q-planner is what executes the
chosen method. The harness is cited in the paper as a separate preprint.

> **Citing this code.** Do not cite a branch name — branches are mutable. Cite the archived
> release DOI for the tagged commit that accompanies the paper. See
> [Release and archiving](#release-and-archiving).

---

## Scope: what is and is not here

**Here (this branch):** the method-selection backends and the evaluation that the paper reports —
corpus construction, LoRA fine-tuning, BM25+Tanimoto retrieval, the classical baselines, the
executed-accuracy benchmark against real ORCA calculations, and the routing/boundary analysis.

**Not here:** the NOMAD download and parsing pipeline, which lives in a separate working tree
(`nomad_build_index.py`, `nomad_download_inp.py`, `nomad_parse_inp.py`, `nomad_parse_gjf.py`).
Those produce `methods.jsonl`, which is this branch's starting point. Also not here: ORCA itself,
benchmark reference values, and base-model weights — each remains under its own upstream terms.

---

## Repository layout

```
.                                  Q-planner harness (planner, executor, MCP tool server)
├── rag.py                         retrieval backend: BM25 + SMILES-Tanimoto rerank
├── canonical_ir.py                classify_cell() — the outer chemistry/task routing grid
├── sample_methods.py              stratified corpus sampling (upload + element caps)
└── something_like_MOSIAC/
    └── lora_train/
        ├── generate_sft.py        corpus construction: filters, caps, stratified split
        ├── train_lora_sft.py      LoRA SFT training (also defines load_jsonl, _extract_json)
        ├── run_rag_sft_experiment.py       the 5-condition ablation driver
        ├── run_classical_baseline_molecule_split.py   random forest / k-NN baselines
        ├── run_pm6_breakdown_eval.py       PM6-vs-non-PM6 diagnostic
        ├── run_rare_filtered_breakdown_eval.py   per-functional-class breakdown
        └── experiments/
            ├── run_accuracy_benchmark.py        executed-accuracy benchmark (real ORCA runs)
            ├── recompute_accuracy_matched.py    paired, tie-inclusive recomputation
            ├── boundary_analysis.py             LOOCV + label-permutation boundary test
            ├── routing_value_analysis.py        routed vs best-fixed comparison
            ├── three_way_router_check.py        3-candidate router feasibility
            ├── freeze_three_way_router.py       writes the pre-committed router artifact
            ├── evaluate_frozen_router.py        single-shot held-out evaluation
            ├── run_specialist_pair_experiment.py / score_specialist_pair_from_rccs.py
            ├── sweep_failure_mechanisms.py      the three verified failure mechanisms
            └── accuracy_benchmark_logs/         result JSONs (see below)
```

---

## Which script produces which paper result

| Paper item | Script | Artifact |
|---|---|---|
| Table 1 (precedent-match accuracy) | `run_rag_sft_experiment.py` | `rag_sft_experiment_results_20260803T075244Z.json` |
| Table 4 (executed accuracy, per-condition *n*) | `run_accuracy_benchmark.py` → `recompute_accuracy_matched.py` | `accuracy_matched_{organic,metal}_20261001.json` |
| Table 5 (paired fixed-vs-learned, both cells) | `recompute_accuracy_matched.py` | same as above, `paired` key |
| Table 6 (four method pairs) | `run_specialist_pair_experiment.py`, `boundary_analysis.py` | per-pair logs in `accuracy_benchmark_logs/` |
| Table 7 (by endpoint type) | `routing_value_analysis.py` | — |
| Table 8 (held-out, frozen router) | `evaluate_frozen_router.py` | `heldout_frozen_router_20260930.json` |
| SI per-reaction held-out errors | `evaluate_frozen_router.py` | `heldout_per_reaction_20260930.json` |
| SI classical baselines | `run_classical_baseline_molecule_split.py` | — |
| SI failure mechanisms | `sweep_failure_mechanisms.py` | — |

---

## The frozen router artifact

`experiments/accuracy_benchmark_logs/frozen_three_way_router.json` is the pre-committed model used
for the paper's one prospective test. It was written **before any held-out calculation was run or
scored**, and `evaluate_frozen_router.py` verifies it by digest before applying it, so a silently
re-fitted model cannot be scored.

Two digests identify it and they are **not interchangeable**:

- **Content digest** (what the evaluator checks; also stored in the file's own `sha256` field):
  SHA-256 of `json.dumps({k:v for k,v in artifact if k != "sha256"}, sort_keys=True)` =
  `965bd671569d5278205a37b00f5e8cb3fe5ddd8f148cbe373e57fa869abf4056`
- **File digest** of the artifact as written:
  `16622e36cdbe3201c6788715959a0d360b5929c20db07f6fbb4d41fd62414c74`

A file cannot contain its own file hash, which is why the stored field is a content digest. The
per-reaction results file has file digest
`549f1f53f7740c869f2541b99e106ed96c40d91a066ed106c34c1cb5edf5982d`.

> ### ⚠ The 32 held-out reactions are a spent test set
> FH51 and TAUT15 were used **once**, to evaluate a model fixed in advance. Re-fitting any selector
> on them and reporting the result would destroy the one property that makes that test
> informative. Treat them as exhausted; use a fresh benchmark subset instead.

---

## Reproducing the analysis

Ordering matters — several steps consume the previous step's output.

```bash
pip install -r requirements.txt            # harness deps
# plus: torch, peft, transformers, scikit-learn, rank_bm25, rdkit, numpy

# 1. corpus -> per-cell SFT splits  (input: methods.jsonl from the NOMAD pipeline)
python something_like_MOSIAC/lora_train/generate_sft.py --input jsonl/methods.jsonl

# 2. train a specialist  (SLURM; see the matching .sbatch for the cluster form)
python something_like_MOSIAC/lora_train/train_lora_sft.py --cell organic_general

# 3. the 5-condition ablation: vanilla / +schema / RAG / SFT / SFT+RAG
python something_like_MOSIAC/lora_train/run_rag_sft_experiment.py

# 4. executed accuracy — requires a working ORCA installation
python something_like_MOSIAC/lora_train/experiments/run_accuracy_benchmark.py
python something_like_MOSIAC/lora_train/experiments/recompute_accuracy_matched.py --cell organic_general
python something_like_MOSIAC/lora_train/experiments/recompute_accuracy_matched.py --cell metal_general

# 5. boundary / routing analysis
python something_like_MOSIAC/lora_train/experiments/boundary_analysis.py
python something_like_MOSIAC/lora_train/experiments/routing_value_analysis.py

# 6. the prospective test (single-shot; verifies the digest first)
python something_like_MOSIAC/lora_train/experiments/evaluate_frozen_router.py
```

Steps 1–3 need a GPU; step 4 needs ORCA and is the long pole (the paper's benchmark campaign was
984 ORCA jobs). Steps 5–6 are CPU-only and run in seconds from the stored JSONs.

### Two conventions that are easy to get wrong

- **Label polarity in pair logs.** In `boundary_analysis.load_from_pair_log`, `y == 1` means
  **pair A wins**. Choosing by label must reproduce `min(err_A, err_B)` exactly — a self-check
  worth running after any change, because inverting it silently produces a "router" that is worse
  than either member.
- **Feature construction.** The representative species is the **largest** species in the reaction,
  and `degree_unsaturation = (2*nC + 2 + nN - nH - nF)/2` from its element counts. Any deviation
  applies the frozen model to a different feature basis than it was fitted on.

---

## Release and archiving

Before the paper's camera-ready version:

1. Tag the commit that matches the submitted manuscript (e.g. `git tag -a jcim-submission -m ...`).
2. Archive that tag to Zenodo to mint a DOI pinned to one immutable commit.
3. Cite the **DOI** in the paper's Data and Software Availability statement — not this branch name.

## Licence

**Not yet licensed.** Reuse terms are pending confirmation of institutional copyright. NII is an
institute under the Research Organization of Information and Systems (ROIS), whose published
regulations on the handling of copyrighted works expressly cover computer programs, and which
ownership category applies to this code has not been established. Until that is settled no licence
is granted, which under default copyright means the code may be read but not reused. A licence will
be added before the archived release is tagged.

Third-party components are separate and remain under their own terms: ORCA, benchmark reference
data, NOMAD input files and base-model weights.
