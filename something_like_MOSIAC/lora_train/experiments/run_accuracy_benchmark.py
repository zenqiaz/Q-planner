"""
Method-selection accuracy benchmark experiment runner.

Implements the Procedure section of selected_method_accuracy_benchmark_design.md:
for every in-distribution ground-truth reaction (organic_general: G2RC + W4-11;
metal_general: MOR41 + TMC151), get a (functional, basis) prediction from four
baseline tiers plus the RAG-based learned selector, run real single-point ORCA
energies via the Q-Planner execution path (qcl pod), and score each condition's
reaction energy against the benchmark's ab-initio reference value.

Deliberate scope choices vs. the design doc's original language:
- Runs single-point energies AT THE BENCHMARK'S OWN REFERENCE GEOMETRY, not a
  re-optimized one. This is the standard GMTKN55/MOR41/W4-11 convention for
  screening electronic-structure methods -- it isolates the (functional, basis)
  choice's error from geometry-optimization error, and is far cheaper (SP median
  ~12s vs OPT median ~216s per this project's own runtime_reports history).
- `build_graph_from_plan`'s default node routing is SEQUENTIAL (each node's
  `goto` resolves to "next node in list order" unless a plan explicitly wires
  parallel branches) -- verified both by reading the executor's routing code
  and by re-checking a historical "run in parallel" plan's own timestamps,
  which summed sequentially. So this script does NOT use build_graph_from_plan
  for the SP batch -- it calls `session.call_tool("run_sp_energy", ...)`
  directly, `asyncio.gather`'d under a semaphore (default 12, just under the
  qcl pod's own `cpu: 16` resource limit). Real concurrency over one MCP
  session was empirically verified (`_concurrency_probe.py`, 2026-08-07):
  server-side ORCA runs via `asyncio.to_thread()` (server_helpers.py:270), so
  the server's event loop stays free to accept and dispatch further concurrent
  tool calls while earlier ones are still running -- confirmed via 3 concurrent
  SP jobs completing in ~half the sequential time.
- Reactions are stratified-subsampled to `--n-per-cell` (default 35, the
  design's own N=20-50 midpoint) because the curated ground truth (103
  organic_general + 35 metal_general reactions) is larger than that scope.

Usage:
    python run_accuracy_benchmark.py --dry-run          # plan only, no ORCA
    python run_accuracy_benchmark.py                    # full run, both cells
    python run_accuracy_benchmark.py --cell organic_general
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(r"D:\brick\D\20260217\working")
MOSAIC_ROOT = REPO_ROOT / "something_like_MOSIAC"
GROUND_TRUTH_DIR = Path(__file__).parent / "ground_truth_benchmarks"
LOG_DIR = Path(__file__).parent / "accuracy_benchmark_logs"

sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.environ.get("ENV_FILE", str(REPO_ROOT / ".env")))

# agent.py's own MCP_SSH_KEY (qclab_auto) does not exist on disk; ssh still connects by
# falling through to Windows OpenSSH's native Pageant-agent detection (which finds the
# qclab_zhang key currently loaded in Pageant), as long as SSH_AUTH_SOCK is NOT set to a
# stale/dead named pipe -- verified 2026-09-08 after a previously-hardcoded pipe path here
# (tied to an old Pageant process instance) started shadowing that native detection and
# broke every qcl connection with "Permission denied (publickey)". Do not reintroduce a
# hardcoded SSH_AUTH_SOCK override; native detection is what actually works right now.
# See project memory: qc_agent_core_infra (Pageant/SSH_AUTH_SOCK section).

import rag  # noqa: E402
from canonical_ir import classify_system, hill_formula  # noqa: E402

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

# Bound on concurrent in-flight run_sp_energy calls over one MCP session. Kept
# just under the qcl pod's own `cpu: 16` resource limit (each ORCA job runs at
# ncores=1 per this project's hard invariant) to leave a little headroom.
DEFAULT_CONCURRENCY = 12


def _build_mcp_server_params() -> StdioServerParameters:
    """Inlined from agent.py -- avoids importing that module (it reads
    openai_tools_geom.json via a CWD-relative path at import time)."""
    def _require(name: str) -> str:
        val = os.getenv(name, "").strip()
        if not val:
            raise RuntimeError(f"{name} is not set. Define it in your env file.")
        return val

    mode = os.getenv("MCP_MODE", "ssh").strip().lower()
    if mode == "local":
        cmd = os.getenv("MCP_SERVER_CMD", "python server_with_product.py")
        parts = cmd.split()
        return StdioServerParameters(command=parts[0], args=parts[1:], env=dict(os.environ))
    ssh_bin = os.getenv("MCP_SSH_BIN", "ssh")
    ssh_key = _require("MCP_SSH_KEY")
    ssh_host = _require("MCP_SSH_HOST")
    ssh_cmd = _require("MCP_SERVER_CMD")
    return StdioServerParameters(
        command=ssh_bin,
        args=["-i", ssh_key, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", ssh_host, ssh_cmd],
        env=dict(os.environ),
    )

HARTREE_TO_KCAL = 627.509474

GROUND_TRUTH_FILES = {
    "organic_general": [
        GROUND_TRUTH_DIR / "organic_general_gmtkn55_g2rc.json",
        GROUND_TRUTH_DIR / "organic_general_w4-11.json",
    ],
    "metal_general": [
        GROUND_TRUTH_DIR / "metal_general_mor41.json",
        GROUND_TRUTH_DIR / "metal_general_tmc151.json",
    ],
}

# Baseline tier 1: fixed universal "blind" default (design doc "always B3LYP" strawman) --
# the naive, system-unaware guess to contrast against the "proposed method" (selector_rag).
# Basis is def2-SVP, not 6-31G: empirically verified 2026-08-08 that ORCA's 6-31G library has
# no parameters for most transition metals ("basis set was either not assigned or not
# available"), which silently failed 37/39 of metal_general's blind-baseline jobs in the first
# contrast run. def2-SVP is confirmed valid on both a light-element probe and a real TM species
# (Ni-carbonyl ED03, _tm_basis_probe.py) -- covers the whole periodic table, keeping this a
# single universal blind default rather than a cell-specific special case.
FIXED_DEFAULT = ("B3LYP", "def2-SVP")

# Baseline tier 3: standalone replica of skills.py's MethodSelectionSkill advisory
# text for a plain SP energy job. The skill's own prose is ambiguous between
# B3LYP/def2-TZVP and PBE0/def2-TZVP for SP (skills.py:64) -- documented tie-break:
# prefer PBE0/def2-TZVP, since the skill lists it second/more-accurate for SP.
SKILLS_HEURISTIC_SP = ("PBE0", "def2-TZVP")

RANDOM_SEED = 20260807


def load_ground_truth(cell: str, n_target: int | None, rng: random.Random) -> dict[str, Any]:
    """Merge every ground-truth source file for one cell into one species/reactions pool,
    then stratified-subsample reactions down to n_target (design doc's own N=20-50/cell scope --
    the curated ground truth ended up larger than that during sourcing, so this brings actual
    execution back in line with what was approved, keeping proportional representation from
    every source file rather than letting one dominate)."""
    species: dict[str, dict] = {}
    by_source: dict[str, list[dict]] = {}
    for path in GROUND_TRUTH_FILES[cell]:
        data = json.loads(path.read_text(encoding="utf-8"))
        for sid, s in data["species"].items():
            species[sid] = s
        rxns = []
        for r in data["reactions"]:
            r = dict(r)
            r["source_file"] = path.name
            rxns.append(r)
        by_source[path.name] = rxns

    total = sum(len(v) for v in by_source.values())
    if n_target is None or total <= n_target:
        reactions = [r for rxns in by_source.values() for r in rxns]
    else:
        reactions = []
        for name, rxns in by_source.items():
            share = max(1, round(n_target * len(rxns) / total))
            reactions.extend(rng.sample(rxns, min(share, len(rxns))))
        rng.shuffle(reactions)
        reactions = reactions[:n_target]

    used_species = {sid for r in reactions for sid in r["species_ids"]}
    species = {sid: s for sid, s in species.items() if sid in used_species}
    return {"species": species, "reactions": reactions}


def representative_species(reaction: dict, species: dict[str, dict]) -> dict:
    """The largest species in a reaction stands in for the reaction's method-selection profile,
    matching how a chemist frames the calculation around the target compound, not the reagents."""
    sids = reaction["species_ids"]
    return max((species[sid] for sid in sids), key=lambda s: s["n_atoms"])


def reaction_profile(reaction: dict, species: dict[str, dict], cell: str) -> dict:
    rep = representative_species(reaction, species)
    sys_info = classify_system(rep["elements"], rep["multiplicity"])
    return {
        "elements": rep["elements"],
        "n_atoms": rep["n_atoms"],
        "charge": rep["charge"],
        "multiplicity": rep["multiplicity"],
        "task_type": "SP",
        "system_type": sys_info["system_type"],
        "specialist_cell": cell,
    }


# Three-tier functional canonicalization, built from an empirical probe (2026-08-08,
# _functional_validity_probe.py) of every distinct functional string in the pool against the
# real ORCA 6.1.1 build on qcl: submitted a trivial H-atom SP job per functional and recorded
# which keywords ORCA's parser actually accepts. 516/4,980 pool records (10.4%) used a
# functional string ORCA rejects outright with "UNRECOGNIZED ... KEYWORD".
#
# Tier 1 -- free spelling fixes (same functional, corpus and ORCA just disagree on hyphenation):
_FUNCTIONAL_SPELLING_FIX = {
    "M06-2X": "M062X",
    "WB97X-D": "WB97X-D3",
}
#
# Tier 2 -- exclude outright, do not substitute: PM6 is semi-empirical (NDDO), not a DFT
# functional -- it was never going to run as an ORCA DFT job, and relabeling a PM6 precedent
# as evidence for some DFT functional would misrepresent what the corpus record actually shows.
# This is 400 of the 516 bad-keyword records (78%) -- by far the dominant term.
_FUNCTIONAL_EXCLUDE = frozenset({"PM6"})
#
# Tier 3 -- canonical family map for the remaining true DFT orphans (functional names ORCA
# doesn't recognize under that exact spelling, with no free spelling fix available). Mapped to
# the nearest chemically-related functional confirmed valid by the same probe, not blanket
# B3LYP -- e.g. LC-WPBE (range-separated GGA) maps to CAM-B3LYP (range-separated hybrid), not
# to a family it has nothing to do with. BEPBE maps to B3LYP per explicit direction (2026-08-08).
_FUNCTIONAL_FAMILY_MAP = {
    "BEPBE": "B3LYP",       # Becke88+PBE correlation (GGA) -> nearest common hybrid, as directed
    "M06-L": "M06",         # Minnesota meta-GGA -> Minnesota hybrid, same family
    "MPW1PW91": "PBE0",     # modified-PW91 hybrid -> PBE-lineage hybrid
    "B1B95": "TPSSH",       # Becke hybrid + B95 meta-GGA correlation -> meta-GGA hybrid
    "LC-WPBE": "CAM-B3LYP", # range-separated GGA -> range-separated hybrid
    "MP4": "MP2",           # 4th-order Moller-Plesset -> 2nd-order, same wavefunction family
    "HSE06": "PBE0",        # screened hybrid -> unscreened PBE-lineage hybrid analog
    "BE1PBE": "PBE0",       # Becke 1-param + PBE correlation -> PBE-lineage hybrid
}


def canonicalize_functional(f: str | None) -> str | None:
    """Apply the three-tier fix to a single functional string. Returns None if excluded
    (PM6) or the input was falsy -- callers must treat None as 'drop this prediction'."""
    if not f:
        return None
    f = _FUNCTIONAL_SPELLING_FIX.get(f, f)
    if f in _FUNCTIONAL_EXCLUDE:
        return None
    return _FUNCTIONAL_FAMILY_MAP.get(f, f)


def canonicalize_pool(pool: list[dict]) -> list[dict]:
    """Apply the three-tier functional fix to every pool record's functional field, dropping
    PM6 records outright. Applied once to the whole pool before it feeds compute_majority_by_cell,
    build_vocab_by_cell, and rag.build_index -- so literature_majority, random_floor, and
    selector_rag all inherit clean, ORCA-executable functional names for free, with no
    per-condition special-casing needed downstream."""
    out = []
    for r in pool:
        raw_f = r.get("functional")
        if not raw_f:
            out.append(r)
            continue
        f = canonicalize_functional(raw_f)
        if f is None:
            continue
        if f != raw_f:
            r = dict(r, functional=f)
        out.append(r)
    return out


def canonicalize_sft_predictions(
    sft_predictions: dict[str, dict[str, list]],
) -> dict[str, dict[str, tuple[str, str]]]:
    """Same three-tier fix, applied to the external SFT-inference JSON -- those predictions
    never pass through canonicalize_pool() since they come from a separate NII round-trip, not
    the RAG/RF pool. A prediction whose functional canonicalizes to None (PM6) is dropped
    entirely rather than substituted, same policy as the pool."""
    out: dict[str, dict[str, tuple[str, str]]] = {}
    for cell, preds in sft_predictions.items():
        cell_out = {}
        for rid, (f, b) in preds.items():
            cf = canonicalize_functional(f)
            if cf is not None and b:
                cell_out[rid] = (cf, b)
        out[cell] = cell_out
    return out


def compute_majority_by_cell(pool: list[dict]) -> dict[str, tuple[str, str]]:
    """Baseline tier 2: literature-majority (functional, basis) pair per cell, from the live pool."""
    by_cell: dict[str, Counter] = {}
    for r in pool:
        c = r.get("specialist_cell")
        f, b = r.get("functional"), r.get("basis")
        if not c or not f or not b:
            continue
        by_cell.setdefault(c, Counter())[(f, b)] += 1
    return {c: counts.most_common(1)[0][0] for c, counts in by_cell.items() if counts}


def build_vocab_by_cell(pool: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """Baseline tier 4: distinct (functional, basis) pairs actually seen per cell (random draw pool)."""
    vocab: dict[str, set] = {}
    for r in pool:
        c = r.get("specialist_cell")
        f, b = r.get("functional"), r.get("basis")
        if c and f and b:
            vocab.setdefault(c, set()).add((f, b))
    return {c: sorted(v) for c, v in vocab.items()}


def get_selector_prediction(idx, profile: dict) -> tuple[str, str] | None:
    """Baseline "learned selector": RAG-alone top-1 (project precedent already established
    RAG-alone's precedent-matching accuracy in prior sessions; SFT inference on NII was judged
    out of scope for a locally-runnable test script -- see design doc's Procedure step 2 note)."""
    features = {
        "elements": profile["elements"],
        "n_atoms": profile["n_atoms"],
        "charge": profile["charge"],
        "multiplicity": profile["multiplicity"],
        "solvent": None,
        "task_type": profile["task_type"],
        "system_type": profile["system_type"],
        "specialist_cell": profile["specialist_cell"],
    }
    hits = rag.query(idx, features, k=1)
    if not hits:
        return None
    top = hits[0]
    f, b = top.get("functional"), top.get("basis")
    if not f or not b:
        return None
    return (f, b)


# --- Classical baseline: Random Forest, same feature scheme as
# run_classical_baseline_molecule_split.py's FeatureBuilder (kept local/self-contained here
# rather than importing that script, since its module-level SPLIT_DIR/ABLATION_INPUT_FILE
# constants point at files unrelated to this benchmark's own reaction set). Trained on the
# WHOLE canonicalized pool per cell (not a held-out split) -- this benchmark's ground-truth
# reactions (GMTKN55/W4-11/MOR41/TMC151) are an entirely different corpus from the NOMAD pool,
# so there is no train/test leakage concern the way there was for the pool's own held-out val set.
class RFFeatureBuilder:
    def __init__(self, cell: str):
        self.cell = cell
        self.element_vocab: list[str] = []
        self.task_vocab: list[str] = []

    def fit(self, records: list[dict]) -> None:
        elems = set()
        for r in records:
            elems.update(r.get("elements") or [])
        self.element_vocab = sorted(elems)
        self.task_vocab = sorted({r.get("task_type") or "UNKNOWN" for r in records})

    def transform(self, records: list[dict]):
        import numpy as np
        rows = []
        for r in records:
            n_atoms = r.get("n_atoms") or 0
            charge = r.get("charge") or 0
            mult = r.get("multiplicity") or 1
            elements = set(r.get("elements") or [])
            task = r.get("task_type") or "UNKNOWN"
            solvent = 1.0 if (r.get("solvent") or "").strip() else 0.0
            row = [np.log1p(max(n_atoms, 0)), float(charge), 1.0 if mult != 1 else 0.0, solvent]
            row += [1.0 if e in elements else 0.0 for e in self.element_vocab]
            row += [1.0 if task == t else 0.0 for t in self.task_vocab]
            rows.append(row)
        return np.array(rows, dtype=np.float64)


def train_rf_by_cell(pool: list[dict]) -> dict[str, tuple]:
    """Returns {cell: (RandomForestClassifier, LabelEncoder, RFFeatureBuilder)}."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    by_cell: dict[str, list[dict]] = {}
    for r in pool:
        c, f, b = r.get("specialist_cell"), r.get("functional"), r.get("basis")
        if c and f and b:
            by_cell.setdefault(c, []).append(r)

    models: dict[str, tuple] = {}
    for cell, records in by_cell.items():
        fb = RFFeatureBuilder(cell)
        fb.fit(records)
        X = fb.transform(records)
        y_joint = [f"{r.get('functional')}\x1f{r.get('basis')}" for r in records]
        le = LabelEncoder()
        y = le.fit_transform(y_joint)
        model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
        model.fit(X, y)
        models[cell] = (model, le, fb)
    return models


def get_rf_prediction(rf_models: dict[str, tuple], profile: dict) -> tuple[str, str] | None:
    entry = rf_models.get(profile["specialist_cell"])
    if entry is None:
        return None
    model, le, fb = entry
    X = fb.transform([profile])
    pred_idx = model.predict(X)[0]
    f, b = le.inverse_transform([pred_idx])[0].split("\x1f", 1)
    return (f, b)


def sanitize_node_id(species_id: str, functional: str, basis: str) -> str:
    """Must exactly match server_helpers.sanitize_label()'s transform of the same job_label
    string, since the server independently re-sanitizes whatever job_label it's given to name
    the job's working directory -- if the two diverge, cleanup targets a path that was never
    created (rm -rf on it silently no-ops) and the real directory is never removed. Found
    2026-08-07 after a fix attempt that didn't actually delete anything: server collapses
    consecutive underscores and strips leading/trailing ones, this didn't."""
    raw = f"sp_{species_id}_{functional}_{basis}"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "job"


# RCCS %maxcore setting for batch-mode jobs: RCCS's own standard allocation is 1.875 GB/core
# (see rccs_ims_access memory) -- 1700 MB leaves headroom under that per worker slot rather than
# claiming the theoretical max, since batch mode runs `concurrency` single-core ORCA processes
# at once on one node and a single job spiking to the full per-core allocation could starve a
# neighbor sharing the same node's total memory.
RCCS_MAXCORE_MB = 1700


# Found 2026-08-10, running the first real RCCS batch: RIJCOSX (qcl's use_ri=True default, added
# unconditionally by server_with_product.py's _build_calc() for every method) makes ORCA's MDCI
# (post-HF correlated) module fail with "Please provide an AuxC basis for RCSinglesFock" --
# RIJCOSX auto-provides an aux basis for the SCF/DFT Fock matrix, but NOT for the correlated
# module's own RI needs, which requires an explicit AuxC (or dropping RIJCOSX entirely). This is
# a LATENT bug in qcl's own server too (same unconditional RIJCOSX), just never triggered there
# because this project's cheap-DFT filter has always kept every CCSD/QCISD/MP2/DLPNO job off
# qcl -- these RCCS-only jobs are the first time any of this project's own code has ever actually
# run a wavefunction-correlated SP job. Confirmed via 10/39 real failures in the first batch, all
# identical MDCI errors. Fix: drop RIJCOSX for these methods (canonical, non-RI correlation --
# slower but needs no AuxC, and correctness matters more than speed for the already-small
# expensive-tier batch). Plain HF/RHF/UHF/ROHF and CASSCF are NOT included here -- they use
# RIJCOSX for the Fock matrix only, no MDCI module, confirmed fine in the earlier functional
# validity probe (_functional_validity_probe.py, 2026-08-08).
_WF_CORRELATED_PATTERN = re.compile(
    r"(CCSD|MP[234]|QCISD|CISD|CEPA|CASPT2|NEVPT2|DLPNO)",
    re.IGNORECASE,
)


def build_aux_basis_lookup(pool: list[dict]) -> dict[tuple[str, str], str]:
    """{(functional, basis): most-common real aux_basis} from pool records that actually recorded
    one -- 431/5082 records have a non-null aux_basis (90 of those wavefunction-correlated),
    mostly following ORCA's standard "{basis}/C" convention for def2-* families (e.g.
    def2-TZVPP -> def2-TZVPP/C). Reusing a real historical choice is more faithful than always
    auto-generating one, when precedent actually exists for the exact (functional, basis) pair --
    checked 2026-08-10, no such precedent exists yet for cc-pVTZ/cc-pCV5Z specifically (this
    project's currently pending RCCS retries), so those still correctly fall back to AutoAux."""
    counts: dict[tuple[str, str], Counter] = {}
    for r in pool:
        f, b, ab = r.get("functional"), r.get("basis"), r.get("aux_basis")
        if f and b and ab:
            counts.setdefault((f, b), Counter())[ab] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}


def classify_aux_basis_plan(
    functional: str, basis: str, aux_lookup: dict[tuple[str, str], str] | None = None,
) -> str:
    """Which of the three aux-basis handling paths write_orca_inp() will take for this
    (functional, basis) pair, WITHOUT generating the .inp text -- shared by write_orca_inp()
    and the export-time metadata sidecar so the label always matches what was actually written.
    One of:
      "not_applicable"    -- not a wavefunction-correlated method, no aux basis needed at all
      "precedent"         -- WF-correlated, a real historical aux_basis exists for this exact
                              (functional, basis) pair, reused via an explicit %basis AuxC block
      "autoaux_fallback"  -- WF-correlated, no precedent, ORCA's own AutoAux keyword used instead
    """
    if not _WF_CORRELATED_PATTERN.search(functional):
        return "not_applicable"
    if (aux_lookup or {}).get((functional, basis)):
        return "precedent"
    return "autoaux_fallback"


def write_orca_inp(
    sid: str, functional: str, basis: str, species: dict[str, dict],
    aux_lookup: dict[tuple[str, str], str] | None = None,
) -> str:
    """Same main-line format server_with_product.py's _build_calc() actually uses for
    run_sp_energy (SP RIJCOSX, use_ri=True is qcl's default) -- so an RCCS-computed energy is a
    like-for-like comparison with qcl's, not just chemically similar -- EXCEPT RIJCOSX is
    dropped for wavefunction-correlated methods (see _WF_CORRELATED_PATTERN above). Found
    2026-08-10, second real-batch failure round: DLPNO-family methods need an explicit auxiliary
    basis unconditionally (independent of RIJCOSX) for their domain/PNO construction --
    "For DLPNO calculations we need at least one auxiliary basis set". Preference order: (1) a
    real historical aux_basis from the pool for this exact (functional, basis) pair, via an
    explicit %basis AuxC block -- most faithful, reuses what an actual practitioner used; (2)
    AutoAux otherwise -- ORCA's own auto-generation, a no-op for canonical correlated methods
    that don't reference an aux basis at all, so safe to apply uniformly when no precedent
    exists. See classify_aux_basis_plan() for the label attached to each of these three paths."""
    s = species[sid]
    plan = classify_aux_basis_plan(functional, basis, aux_lookup)
    aux_block = ""
    if plan == "precedent":
        real_aux = aux_lookup[(functional, basis)]
        ri_kw = ""
        aux_block = f'%basis\n  AuxC "{real_aux}"\nend\n'
    elif plan == "autoaux_fallback":
        ri_kw = " AutoAux"
    else:
        ri_kw = " RIJCOSX"
    main_line = f"! {functional} {basis} SP{ri_kw}"
    return (
        f"{main_line}\n"
        f"{aux_block}"
        f"%maxcore {RCCS_MAXCORE_MB}\n\n"
        f"* xyz {s['charge']} {s['multiplicity']}\n"
        f"{s['xyz_angstrom']}\n"
        f"*\n"
    )


# Real ORCA error signatures observed across this project's own RCCS batches (2026-08-10/12) for
# jobs that DID get an aux basis (precedent or AutoAux) but failed anyway, vs. jobs that fail for
# an unrelated reason (e.g. "Number of processes (1) in parallel calculation exceeds number of
# pairs (0)" -- QCISD applied to a doublet H atom, which has zero correlated electron pairs; a
# genuine method/system mismatch, not an aux-basis problem). Order matters only for readability;
# any one match is sufficient. See rccs_collect_results.py, which uses this same list.
AUX_BASIS_ERROR_PATTERNS = [
    re.compile(r"please provide an AuxC basis", re.IGNORECASE),
    re.compile(r"need at least one auxiliary basis set", re.IGNORECASE),
    re.compile(r"not an appropriate auxiliary basis set for correlated methods", re.IGNORECASE),
]


def collect_conditions(
    reactions: list[dict],
    species: dict[str, dict],
    cell: str,
    majority: dict[str, tuple[str, str]],
    vocab: dict[str, list[tuple[str, str]]],
    idx,
    rng: random.Random,
    rf_models: dict[str, tuple] | None = None,
    sft_predictions: dict[str, tuple[str, str]] | None = None,
) -> dict[str, dict[str, tuple[str, str] | None]]:
    """For every reaction, compute the (functional, basis) each condition prescribes.
    rf_models (from train_rf_by_cell) and sft_predictions (reaction_id -> (functional, basis),
    from the NII inference round-trip) are optional -- omitted entirely (selector_rf/selector_sft
    stay None) unless explicitly built/loaded by the caller."""
    per_reaction: dict[str, dict] = {}
    cell_vocab = vocab.get(cell) or [FIXED_DEFAULT]
    cell_majority = majority.get(cell, FIXED_DEFAULT)
    for r in reactions:
        profile = reaction_profile(r, species, cell)
        selector = get_selector_prediction(idx, profile)
        rf_pred = get_rf_prediction(rf_models, profile) if rf_models else None
        sft_pred = sft_predictions.get(r["reaction_id"]) if sft_predictions else None
        per_reaction[r["reaction_id"]] = {
            "fixed_default": FIXED_DEFAULT,
            "literature_majority": cell_majority,
            "skills_heuristic": SKILLS_HEURISTIC_SP,
            "random_floor": rng.choice(cell_vocab),
            "selector_rag": selector,
            "selector_rf": rf_pred,
            "selector_sft": sft_pred,
        }
    return per_reaction


def collect_sp_jobs(
    reactions: list[dict],
    conditions_by_reaction: dict[str, dict],
) -> set[tuple[str, str, str]]:
    """Distinct (species_id, functional, basis) SP jobs actually needed, deduped globally --
    many reactions share reagent/atom species, and conditions often agree (ties)."""
    jobs: set[tuple[str, str, str]] = set()
    for r in reactions:
        conds = conditions_by_reaction[r["reaction_id"]]
        settings = {v for v in conds.values() if v is not None}
        for sid in r["species_ids"]:
            for (f, b) in settings:
                jobs.add((sid, f, b))
    return jobs


# Cheap-DFT filter: excludes correlated wavefunction methods (CCSD(T)/MP2/QCISD/CASSCF/HF-family)
# and large-zeta basis sets (5Z/QZ tiers). These generate scratch files an order of magnitude
# bigger than routine DFT/def2-* -- two CCSD(T)/cc-pCV5Z single-point jobs alone left ~30GB of
# orphaned scratch on qcl's root disk on 2026-08-07 and re-triggered node disk-pressure eviction.
# literature_majority/random_floor/selector_rag can all legitimately predict this tier (it's real
# precedent in the NOMAD-CCCBDB corpus for small reference molecules), so the filter drops the
# JOB, not the prediction -- the condition just scores as "missing_energy" for that reaction,
# same as any other unavailable energy, until it's run on a backend with real scratch headroom
# (RCCS, or qcl once ORCA_JOBS_DIR points at /ssd -- see ollama-pod-cpu-only.yml).
_EXPENSIVE_FUNCTIONAL_PATTERN = re.compile(
    r"(CCSD|MP[234]|QCISD|CISD|CEPA|CASSCF|CASPT2|NEVPT2|\bHF\b|\bRHF\b|\bUHF\b|\bROHF\b)",
    re.IGNORECASE,
)
_EXPENSIVE_BASIS_PATTERN = re.compile(
    r"(pV5Z|pCV5Z|pCVQZ|pVQZ|QZVPP|QZVP\b|aug-cc-pV[Q56]Z)",
    re.IGNORECASE,
)


def is_cheap_dft(functional: str | None, basis: str | None) -> bool:
    if not functional or not basis:
        return False
    if _EXPENSIVE_FUNCTIONAL_PATTERN.search(functional):
        return False
    if _EXPENSIVE_BASIS_PATTERN.search(basis):
        return False
    return True


# Empirically found (2026-08-07): a single run_sp_energy job leaves ~150MB of scratch/output
# behind in its jobs/ working directory on the pod's host disk (much more than expected for
# small-molecule SP jobs) -- 193 jobs alone filled 29.8GB and re-triggered node disk-pressure
# eviction mid-run, even after freeing 30GB beforehand.
#
# First fix attempt (one `ssh ... kubectl exec ... rm -rf <dir>` subprocess per completed job,
# serialized) still wasn't enough -- disk pressure recurred again after ~7 min even though a
# 90s-in spot check showed cleanup keeping pace (4 dirs, matching concurrency). Root cause:
# each cleanup call pays a full SSH handshake + kubectl exec startup (1-3s), which can't keep up
# with jobs completing every few seconds at concurrency=4 -- a backlog compounds silently until
# it's large enough to fill the freed disk headroom.
#
# Real fix: batch cleanup. Completed job ids are queued (no SSH call on the per-job path at
# all), and a background flusher issues ONE `rm -rf dir1 dir2 ... dirN` call every few seconds
# for whatever's accumulated, cutting SSH connection overhead by ~10x instead of paying it once
# per job.
class _CleanupQueue:
    def __init__(self) -> None:
        self.pending: list[str] = []
        self.lock = asyncio.Lock()

    async def add(self, node_id: str) -> None:
        async with self.lock:
            self.pending.append(node_id)

    async def flush(self) -> None:
        async with self.lock:
            batch, self.pending = self.pending, []
        if not batch:
            return
        await _cleanup_job_dirs(batch)


# Trash, not delete: found 2026-08-08 that the two biggest orphaned jobs from the 2026-08-07
# disk-pressure incident (sp_39_CCSD_T_cc-pCV5Z 19.7G, sp_w4_f2_CCSD_T_cc-pCV5Z 9.8G) were
# `rm -rf`'d during cleanup WITHOUT anyone checking whether they'd actually finished -- a
# completed CCSD(T)/cc-pCV5Z energy is expensive to redo and may have been thrown away unread.
# Both jobs move (not copy -- keeps this cheap even for multi-GB scratch) into a timestamped
# trash dir on /ssd (1.5TB free, vs. ~100G on root) instead of being destroyed outright. Requires
# the pod to have /ssd mounted at /ssd_scratch (see ollama-pod-cpu-only.yml's ORCA_JOBS_DIR /
# ssd-scratch volume) -- if that mount isn't present yet, `mv` still succeeds but lands inside
# the container's own ephemeral layer, which does NOT relieve root-disk pressure. Trash is not
# auto-emptied by this script; inspect and clear /ssd/zhang/orca_jobs_trash manually.
_TRASH_ROOT = "/ssd_scratch/orca_jobs_trash"

# Must match ORCA_JOBS_DIR in ollama-pod-cpu-only.yml (2026-08-08: redirected off root disk
# onto /ssd) -- these two hardcoded paths only agree with the server by convention, nothing
# enforces it. Found the hard way: the first real validation run after this redirect left all
# 17 job dirs uncleaned because this constant still pointed at the pre-redirect /data path.
JOBS_SOURCE_DIR = "/ssd_scratch/orca_jobs"


async def _cleanup_job_dirs(node_ids: list[str]) -> None:
    ssh_key = os.environ.get("MCP_SSH_KEY", "")
    ssh_host = os.environ.get("MCP_SSH_HOST", "")
    if not ssh_host or not node_ids:
        return
    batch_dir = f"{_TRASH_ROOT}/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    src_paths = " ".join(f"{JOBS_SOURCE_DIR}/{nid}" for nid in node_ids)
    remote_cmd = f"mkdir -p {batch_dir} && mv {src_paths} {batch_dir}/ 2>/dev/null; true"
    args = [os.getenv("MCP_SSH_BIN", "ssh")]
    if ssh_key:
        args += ["-i", ssh_key]
    args += ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", ssh_host,
              "kubectl", "exec", "-n", "ns-general", "zhang-ollama", "--", "sh", "-c",
              shlex.quote(remote_cmd)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            env=dict(os.environ),
        )
        await asyncio.wait_for(proc.wait(), timeout=60)
    except Exception:  # noqa: BLE001 -- cleanup best-effort, never abort the batch over it
        pass


def kill_stuck_orca_processes() -> None:
    """Found 2026-08-12/13: when a batch's watchdog fires, the LOCAL script gives up waiting --
    it never kills the remote ORCA process, which just keeps running server-side. Across several
    consecutive hung attempts at metal_general's job set, these piled up (54 concurrent single-
    core ORCA processes counted directly on the pod at one point, most already zombied but many
    still live and burning CPU), which measurably degrades every subsequent job's wall-clock time
    -- confirmed by `ps aux`/`top` on the pod showing 100% idle CPU immediately after clearing
    them, vs. a load average >8 before. Called after any chunk hits its watchdog timeout so later
    chunks in the same run aren't handicapped by a stuck chunk's leftover processes. Best-effort,
    synchronous (unlike _cleanup_job_dirs/sweep_stale_job_dirs, this runs from run_accuracy_
    benchmark.py's own sync main() chunk loop, not from inside run_batch_concurrent's event loop)."""
    ssh_key = os.environ.get("MCP_SSH_KEY", "")
    ssh_host = os.environ.get("MCP_SSH_HOST", "")
    if not ssh_host:
        return
    args = [os.getenv("MCP_SSH_BIN", "ssh")]
    if ssh_key:
        args += ["-i", ssh_key]
    args += ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", ssh_host,
              "kubectl", "exec", "-n", "ns-general", "zhang-ollama", "--", "sh", "-c",
              "pkill -9 -f orca_leanscf; pkill -9 -f /orca/orca_6; true"]
    try:
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        env=dict(os.environ), timeout=30)
    except Exception:  # noqa: BLE001 -- best-effort; a failed cleanup shouldn't abort the run
        pass


async def sweep_stale_job_dirs() -> None:
    """Move the whole jobs/ dir into the /ssd trash before submitting anything new, rather than
    deleting it outright -- see _TRASH_ROOT's docstring: two orphaned CCSD(T)/cc-pCV5Z jobs
    (29.6GB combined) were `rm -rf`'d on 2026-08-08 without ever checking whether they'd
    completed, discarding what may have been finished, expensive-to-redo energies. The in-session
    cleanup queue (_CleanupQueue/_periodic_cleanup_flusher below) can only clean up a job whose
    run_one_sp_job() coroutine actually returns -- it queues cleanup AFTER `await
    session.call_tool(...)` completes. If the pod/connection dies while ORCA is still
    mid-calculation (exactly what happened to those jobs), that await never returns and the job
    is never queued -- no amount of batching downstream can recover it. This sweep closes the
    coverage gap unconditionally (whatever crashed last time, jobs/ starts empty for the new run)
    while still preserving the evidence in trash instead of destroying it blind."""
    ssh_key = os.environ.get("MCP_SSH_KEY", "")
    ssh_host = os.environ.get("MCP_SSH_HOST", "")
    if not ssh_host:
        return
    batch_dir = f"{_TRASH_ROOT}/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_presweep"
    remote_cmd = (
        f"if [ -n \"$(ls -A {JOBS_SOURCE_DIR} 2>/dev/null)\" ]; then "
        f"mkdir -p {batch_dir} && mv {JOBS_SOURCE_DIR}/* {batch_dir}/ 2>/dev/null; "
        f"fi; true"
    )
    args = [os.getenv("MCP_SSH_BIN", "ssh")]
    if ssh_key:
        args += ["-i", ssh_key]
    args += ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", ssh_host,
              "kubectl", "exec", "-n", "ns-general", "zhang-ollama", "--",
              "sh", "-c", shlex.quote(remote_cmd)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            env=dict(os.environ),
        )
        await asyncio.wait_for(proc.wait(), timeout=60)
    except Exception:  # noqa: BLE001 -- best-effort; a failed sweep shouldn't abort the run
        pass


async def _periodic_cleanup_flusher(queue: _CleanupQueue, stop_event: asyncio.Event, interval: float = 4.0) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
        await queue.flush()
    await queue.flush()


async def run_one_sp_job(
    session: ClientSession,
    sem: asyncio.Semaphore,
    sid: str,
    functional: str,
    basis: str,
    species: dict[str, dict],
    cleanup_queue: _CleanupQueue,
) -> tuple[tuple[str, str, str], dict]:
    s = species[sid]
    node_id = sanitize_node_id(sid, functional, basis)
    async with sem:
        t0 = time.monotonic()
        try:
            res = await session.call_tool("run_sp_energy", {
                "geometry_xyz": s["xyz_angstrom"],
                "charge": s["charge"],
                "multiplicity": s["multiplicity"],
                "method": functional,
                "basis": basis,
                "job_label": node_id,
                "wall_timeout_seconds": 600,
            })
            text = res.content[0].text if res.content else "{}"
            payload = json.loads(text)
        except Exception as e:  # noqa: BLE001 -- record and continue, don't abort the whole batch
            payload = {"status": "error", "error": f"client_exception: {e!r}"}
        payload["_duration_ms"] = int((time.monotonic() - t0) * 1000)
    await cleanup_queue.add(node_id)
    return (sid, functional, basis), payload


async def run_batch_concurrent(
    jobs: list[tuple[str, str, str]],
    species: dict[str, dict],
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, Any]:
    """Every crash observed 2026-08-07 has the same shape: `asyncio.gather()` (the actual
    ORCA work) completes fine, but the `async with stdio_client(...)` / `async with
    ClientSession(...)` teardown then raises (`RuntimeError: dictionary changed size during
    iteration` in the mcp library's own concurrent receive loop, usually because the pod got
    disk-pressure-evicted right around when the batch finished and the SSH connection died
    mid-close). Without this, that teardown exception propagates past the `return` and every
    already-computed result is lost even though the actual chemistry succeeded. Fix: capture
    `results`/`duration_ms` in the outer scope before the `async with` exits, and catch the
    teardown exception so a good batch's data survives a bad connection close."""
    server_params = _build_mcp_server_params()
    sem = asyncio.Semaphore(concurrency)
    cleanup_queue = _CleanupQueue()
    stop_flusher = asyncio.Event()
    results: list | None = None
    duration_ms = 0
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                flusher_task = asyncio.create_task(_periodic_cleanup_flusher(cleanup_queue, stop_flusher))
                t0 = time.monotonic()
                try:
                    results = await asyncio.gather(*[
                        run_one_sp_job(session, sem, sid, f, b, species, cleanup_queue) for sid, f, b in jobs
                    ])
                finally:
                    stop_flusher.set()
                    await flusher_task  # runs one last flush before returning
                duration_ms = int((time.monotonic() - t0) * 1000)
    except Exception as e:  # noqa: BLE001 -- session teardown error; salvage whatever gather() produced
        print(f"  WARNING: MCP session teardown raised {e!r} -- "
              f"{'keeping the ' + str(len(results)) + ' results already gathered' if results is not None else 'no results were gathered before this'}")
    if results is None:
        return {"job_results": {}, "duration_ms": duration_ms, "session_error": True}
    job_results = dict(results)
    return {"job_results": job_results, "duration_ms": duration_ms}


# Found 2026-08-09: the teardown exception above (dead SSH connection mid-batch) gets caught and
# printed fine, but the process then hangs anyway and gets killed externally with no further
# progress -- e.g. metal_general's batch never even reaches its own summary print after the
# warning. Root cause: once the teardown exception has fired, `async with`'s __aexit__ has
# already run (unsuccessfully); what's left is a zombie ssh.exe subprocess from the dead
# connection that asyncio.run()'s own shutdown sequence (cancel remaining tasks, close the event
# loop) can hang on indefinitely trying to clean up -- a known pitfall with Windows'
# ProactorEventLoop and subprocess transports that don't terminate cleanly. Since that hang is
# inside asyncio.run()'s own machinery, nothing inside run_batch_concurrent() can bound it. Fix:
# run the whole asyncio.run() call in a daemon thread with a hard wall-clock timeout -- if
# cleanup hangs past the deadline, give up waiting on that thread (it can't block process exit,
# being a daemon) and treat the batch as failed rather than hanging until something external
# kills the entire script.
def run_batch_with_watchdog(
    jobs: list[tuple[str, str, str]],
    species: dict[str, dict],
    concurrency: int,
    timeout_s: float,
) -> dict[str, Any]:
    result_box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result_box["value"] = asyncio.run(run_batch_concurrent(jobs, species, concurrency))
        except Exception as e:  # noqa: BLE001 -- surface as a failed batch, don't crash the thread silently
            result_box["value"] = {"job_results": {}, "duration_ms": 0, "session_error": True,
                                    "watchdog_error": repr(e)}

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        print(f"  WARNING: batch did not finish within {timeout_s:.0f}s watchdog timeout "
              f"(likely a hung asyncio/subprocess teardown after a dropped qcl connection) -- "
              f"giving up on this cell's results rather than hanging indefinitely")
        return {"job_results": {}, "duration_ms": 0, "session_error": True, "watchdog_timeout": True}
    return result_box.get("value", {"job_results": {}, "duration_ms": 0, "session_error": True})


def extract_energy_cache(job_results: dict[tuple[str, str, str], dict]) -> dict[tuple[str, str, str], float | None]:
    cache: dict[tuple[str, str, str], float | None] = {}
    for key, payload in job_results.items():
        cache[key] = payload.get("energy_eh") if payload.get("status") == "ok" else None
    return cache


def score_reactions(
    reactions: list[dict],
    conditions_by_reaction: dict[str, dict],
    energy_cache: dict[tuple[str, str, str], float | None],
) -> dict[str, Any]:
    condition_names = ["fixed_default", "literature_majority", "skills_heuristic", "random_floor",
                        "selector_rag", "selector_rf", "selector_sft"]
    per_condition_errors: dict[str, list[float]] = {c: [] for c in condition_names}
    per_reaction_results = []

    for r in reactions:
        # Guard against non-finite ground-truth references -- found 2026-09-08 running the
        # full 103-reaction organic_general set for the first time: w4-11_TAE_t-hooo
        # (hydrotrioxy radical) has ref_kcal_mol=NaN in organic_general_w4-11.json (a genuine
        # gap in the source W4-11 table, not a parsing bug here -- never surfaced before since
        # the original 35-reaction subsample didn't happen to include it). One NaN abs_error
        # silently poisons an entire condition's mean via sum()/len() without this guard --
        # treat it the same as missing_energy rather than let it propagate.
        if r["ref_kcal_mol"] is None or not math.isfinite(r["ref_kcal_mol"]):
            per_reaction_results.append({
                "reaction_id": r["reaction_id"], "ref_kcal_mol": r["ref_kcal_mol"],
                "conditions": {}, "status": "invalid_reference",
            })
            continue
        conds = conditions_by_reaction[r["reaction_id"]]
        row = {"reaction_id": r["reaction_id"], "ref_kcal_mol": r["ref_kcal_mol"], "conditions": {}}
        selector_setting = conds.get("selector_rag")
        for cname in condition_names:
            setting = conds.get(cname)
            if setting is None:
                row["conditions"][cname] = {"setting": None, "status": "no_prediction"}
                continue
            f, b = setting
            energies = []
            missing = False
            for sid, coeff in zip(r["species_ids"], r["coeffs"]):
                e = energy_cache.get((sid, f, b))
                if e is None:
                    missing = True
                    break
                energies.append(coeff * e)
            is_tie = (cname != "selector_rag") and (selector_setting == setting)
            if missing:
                row["conditions"][cname] = {"setting": f"{f}/{b}", "status": "missing_energy", "tie_with_selector": is_tie}
                continue
            dE_kcal = sum(energies) * HARTREE_TO_KCAL
            abs_err = abs(dE_kcal - r["ref_kcal_mol"])
            row["conditions"][cname] = {
                "setting": f"{f}/{b}", "status": "ok",
                "computed_kcal_mol": dE_kcal, "abs_error_kcal_mol": abs_err,
                "tie_with_selector": is_tie,
            }
            if not is_tie or cname in ("selector_rag", "selector_rf", "selector_sft"):
                per_condition_errors[cname].append(abs_err)
        per_reaction_results.append(row)

    summary = {}
    for cname, errs in per_condition_errors.items():
        if errs:
            mae = sum(errs) / len(errs)
            rmse = (sum(e ** 2 for e in errs) / len(errs)) ** 0.5
            summary[cname] = {"n": len(errs), "mae_kcal_mol": mae, "rmse_kcal_mol": rmse}
        else:
            summary[cname] = {"n": 0, "mae_kcal_mol": None, "rmse_kcal_mol": None}

    return {"per_reaction": per_reaction_results, "summary_non_tie_only": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Method-selection accuracy benchmark")
    parser.add_argument("--cell", choices=["organic_general", "metal_general"], default=None,
                         help="Run only one cell; default runs both.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Compute predictions and job plan only; skip real ORCA execution.")
    parser.add_argument("--rag-k", type=int, default=1, help="RAG retrieval k (top-1 used for the prediction).")
    parser.add_argument("--n-per-cell", type=int, default=35,
                         help="Stratified-subsample reactions per cell to this count "
                              "(design doc's own N=20-50/cell scope; midpoint default). "
                              "Pass 0 to disable subsampling and run everything curated.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                         help="Max concurrent in-flight ORCA jobs over the MCP session.")
    parser.add_argument("--allow-expensive", action="store_true",
                         help="Disable the cheap-DFT filter and submit correlated-wavefunction "
                              "(CCSD(T)/MP2/QCISD/CASSCF/HF) or large-basis (5Z/QZ) jobs too. "
                              "Off by default -- these filled qcl's root disk on 2026-08-07.")
    parser.add_argument("--contrast-only", action="store_true",
                         help="Contrast experiment: only run the learned/informed conditions "
                              "(selector_rag, selector_rf, and selector_sft if --sft-predictions "
                              "is given) against the blind fixed baseline (fixed_default, "
                              "B3LYP/def2-SVP) -- skips literature_majority/skills_heuristic/"
                              "random_floor entirely, cutting job count.")
    parser.add_argument("--no-rf", action="store_true",
                         help="Skip training/using the Random Forest selector (selector_rf stays "
                              "None everywhere). RF training runs once at startup, ~seconds.")
    parser.add_argument("--sft-predictions", type=Path, default=None,
                         help="Path to a JSON file of {cell: {reaction_id: [functional, basis]}} "
                              "produced by the NII SFT-inference round-trip "
                              "(sft_infer_for_benchmark.py). Omit to leave selector_sft as None "
                              "everywhere -- this experiment can run/be reviewed without it.")
    parser.add_argument("--export-sft-input", type=Path, default=None,
                         help="Write {cell: [{reaction_id, formula, n_atoms, system_type, charge, "
                              "multiplicity, task_type}, ...]} to this path for the SAME "
                              "subsampled reactions this run would otherwise score, then continue "
                              "normally (works fine combined with --dry-run). Feed this file to "
                              "sft_infer_for_benchmark.py on NII to get matching SFT predictions.")
    parser.add_argument("--export-rccs-batch", type=Path, default=None,
                         help="Write one .inp file per job normally DROPPED by the cheap-DFT "
                              "filter (CCSD(T)/large-basis -- the 'expensive' tier qcl can't "
                              "safely run) to <dir>/<cell>/<node_id>.inp, matching qcl's own "
                              "SP RIJCOSX main-line format exactly. scp the per-cell dir to RCCS "
                              "and run rccs_submit_orca.sh --batch-dir on it. Cheap jobs still "
                              "go to qcl as normal -- this only captures what would otherwise be "
                              "silently skipped. Works with --dry-run (no ORCA calls needed to "
                              "export). Implies --allow-expensive is NOT required -- the cheap "
                              "filter still applies to the qcl-bound jobs.")
    parser.add_argument("--rccs-route-cheap", action="store_true",
                         help="Route the CHEAP-DFT tier to RCCS too (combined with "
                              "--export-rccs-batch), instead of submitting it to qcl at all. "
                              "Added 2026-08-13 after qcl proved unreliable for metal_general's "
                              "~493-job cheap batch (see qc_agent_qcl_session_degradation "
                              "memory) while RCCS handled its 105-job expensive tier with zero "
                              "hangs. With this flag, --export-rccs-batch's dir gets every job "
                              "for the cell, and nothing is submitted to qcl.")
    parser.add_argument("--qcl-session-chunk-size", type=int, default=None,
                         help="Split qcl submission into fresh-MCP-session chunks of this many "
                              "jobs each, run sequentially, results merged. Found 2026-08-12/13: "
                              "large single-session batches (metal_general's ~493 jobs) hung for "
                              "the full watchdog window every time, while the same jobs split "
                              "into <=30-job chunks completed fine. Omit for the old single-"
                              "session behavior (fine for smaller cells like organic_general).")
    parser.add_argument("--exclude-species", action="append", default=[],
                         help="Species ID to drop from qcl submission entirely (repeatable). "
                              "Found 2026-08-12 via qcl_bisect_metal.py/qcl_bisect_chunk2.py: "
                              "species ED04 (a CpCo-cyclobutadiene sandwich complex, MOR41 "
                              "ground truth) hangs ORCA's SCF for every functional/basis tried, "
                              "un-bounded by the server's own 600s wall_timeout_seconds -- this "
                              "starves the whole concurrent batch once every semaphore slot ends "
                              "up stuck on a copy of it. Reactions using an excluded species "
                              "score as missing_energy, same as any other unavailable energy.")
    parser.add_argument("--rccs-results", type=Path, default=None,
                         help="Directory of the form <dir>/<cell>/results.json (as produced by "
                              "rccs_collect_results.py against a --export-rccs-batch dir) -- "
                              "merges the RCCS-computed expensive-tier energies into this run's "
                              "job results before scoring, so reactions that need a "
                              "correlated-wavefunction/large-basis job aren't stuck at "
                              "missing_energy. Ignored under --dry-run.")
    args = parser.parse_args()

    cells = [args.cell] if args.cell else ["organic_general", "metal_general"]
    n_target = None if args.n_per_cell == 0 else args.n_per_cell

    if not args.dry_run:
        print("Sweeping stale job dirs from any previous crashed run...")
        asyncio.run(sweep_stale_job_dirs())

    print("Loading RAG pool and building index...")
    pool = canonicalize_pool(rag.load_pool())
    idx = rag.build_index(pool)
    majority = compute_majority_by_cell(pool)
    vocab = build_vocab_by_cell(pool)
    aux_lookup = build_aux_basis_lookup(pool)
    rng = random.Random(RANDOM_SEED)

    rf_models = None
    if not args.no_rf:
        print("Training Random Forest selector (classical baseline)...")
        rf_models = train_rf_by_cell(pool)

    sft_predictions = None
    if args.sft_predictions:
        raw_sft = json.loads(args.sft_predictions.read_text(encoding="utf-8"))
        sft_predictions = canonicalize_sft_predictions(raw_sft)
        n_raw = sum(len(v) for v in raw_sft.values())
        n_sft = sum(len(v) for v in sft_predictions.values())
        print(f"Loaded {n_raw} SFT predictions from {args.sft_predictions} "
              f"({n_sft} after canonicalization, {n_raw - n_sft} dropped as PM6)")

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    full_report: dict[str, Any] = {
        "type": "accuracy_benchmark_run",
        "timestamp_utc": timestamp,
        "dry_run": args.dry_run,
        "random_seed": RANDOM_SEED,
        "cells": {},
    }

    sft_export: dict[str, list[dict]] = {}

    for cell in cells:
        print(f"\n=== {cell} ===")
        gt = load_ground_truth(cell, n_target, rng)
        species, reactions = gt["species"], gt["reactions"]
        print(f"  species={len(species)} reactions={len(reactions)} (subsampled to n_per_cell={n_target})")

        if args.export_sft_input:
            rows = []
            for r in reactions:
                profile = reaction_profile(r, species, cell)
                rows.append({
                    "reaction_id": r["reaction_id"],
                    "formula": hill_formula(profile["elements"]),
                    "n_atoms": profile["n_atoms"],
                    "system_type": profile["system_type"],
                    "charge": profile["charge"],
                    "multiplicity": profile["multiplicity"],
                    "task_type": profile["task_type"],
                })
            sft_export[cell] = rows

        cell_sft_predictions = sft_predictions.get(cell) if sft_predictions else None
        conditions_by_reaction = collect_conditions(
            reactions, species, cell, majority, vocab, idx, rng,
            rf_models=rf_models, sft_predictions=cell_sft_predictions,
        )
        if args.contrast_only:
            keep = {"fixed_default", "selector_rag", "selector_rf", "selector_sft"}
            for conds in conditions_by_reaction.values():
                for cname in list(conds):
                    if cname not in keep:
                        conds[cname] = None
        jobs = sorted(collect_sp_jobs(reactions, conditions_by_reaction))
        if args.exclude_species:
            # Filtered here, before the cheap/expensive split and RCCS export -- previously this
            # only ran on the qcl-bound `jobs` list below, which is EMPTY under
            # --rccs-route-cheap, so an excluded species (e.g. ED04, known to hang ORCA's SCF
            # indefinitely regardless of qcl/RCCS) silently leaked into the RCCS .inp export.
            # Found 2026-08-17 reviewing a real export batch before submission.
            n_before = len(jobs)
            jobs = [j for j in jobs if j[0] not in args.exclude_species]
            n_dropped = n_before - len(jobs)
            if n_dropped:
                print(f"  --exclude-species: dropped {n_dropped} job(s) for "
                      f"{sorted(args.exclude_species)} -- these reactions score as missing_energy")
        if not args.allow_expensive:
            cheap_jobs = [j for j in jobs if is_cheap_dft(j[1], j[2])]
            expensive_jobs = [j for j in jobs if j not in cheap_jobs]
            n_skipped = len(expensive_jobs)
            if n_skipped:
                dest = " -- exporting to RCCS batch dir" if args.export_rccs_batch else " (pass --allow-expensive to include them on qcl, or --export-rccs-batch to route them to RCCS instead)"
                print(f"  cheap-DFT filter: skipping {n_skipped} job(s) needing a correlated "
                      f"wavefunction method or large basis{dest}")
            # --rccs-route-cheap: also send the CHEAP tier to RCCS instead of qcl. Added
            # 2026-08-13 after qcl proved unreliable for metal_general's ~493-job cheap-DFT
            # batch (see qc_agent_qcl_session_degradation memory) while RCCS ran its own
            # 105-job expensive tier with zero hangs -- routing the cheap tier there too sidesteps
            # the qcl session-degradation issue entirely rather than continuing to chase it.
            rccs_jobs = expensive_jobs + cheap_jobs if args.rccs_route_cheap else expensive_jobs
            if args.export_rccs_batch:
                cell_dir = args.export_rccs_batch / cell
                cell_dir.mkdir(parents=True, exist_ok=True)
                meta: dict[str, dict] = {}
                used_lower: set[str] = set()
                for sid, f, b in rccs_jobs:
                    node_id = sanitize_node_id(sid, f, b)
                    # Case-insensitive-filesystem guard (found 2026-08-17 exporting
                    # metal_general on Windows/NTFS): node_id is correctly case-sensitive
                    # (must match server_helpers.sanitize_label() exactly, see
                    # sanitize_node_id's own docstring), but two DIFFERENT node_ids that only
                    # differ by case (e.g. PBE_6-31G vs a model's own PBE_6-31g prediction)
                    # write to the SAME file on a case-insensitive filesystem, silently
                    # dropping one job. node_id (the meta.json KEY) stays exact; only the
                    # on-disk filename gets disambiguated, recorded in inp_stem so
                    # rccs_collect_results.py can find the right .out file back.
                    inp_stem, lower, suffix = node_id, node_id.lower(), 2
                    while lower in used_lower:
                        inp_stem = f"{node_id}__dup{suffix}"
                        lower = inp_stem.lower()
                        suffix += 1
                    used_lower.add(lower)
                    inp_text = write_orca_inp(sid, f, b, species, aux_lookup=aux_lookup)
                    (cell_dir / f"{inp_stem}.inp").write_text(inp_text, encoding="utf-8")
                    meta[node_id] = {
                        "species_id": sid, "functional": f, "basis": b,
                        "aux_basis_plan": classify_aux_basis_plan(f, b, aux_lookup),
                        "inp_stem": inp_stem,
                    }
                # Sidecar: rccs_collect_results.py needs the (species_id, functional, basis)
                # tuple and the aux_basis_plan label back from just a node_id filename --
                # sanitize_node_id() is a many-to-one transform, not reversible on its own.
                (cell_dir / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"  wrote {len(rccs_jobs)} .inp file(s) -> {cell_dir}"
                      f"{' (expensive + cheap, --rccs-route-cheap)' if args.rccs_route_cheap else ''}")
            jobs = [] if args.rccs_route_cheap else cheap_jobs
        est_wall_s = (len(jobs) * 20) / max(1, args.concurrency)  # ~20s/job (median-p90 midpoint), /concurrency
        print(f"  distinct SP jobs needed: {len(jobs)} "
              f"(~{est_wall_s/60:.0f} min wall-clock at concurrency={args.concurrency} estimate)")

        cell_report: dict[str, Any] = {
            "n_species": len(species), "n_reactions": len(reactions),
            "n_distinct_sp_jobs": len(jobs),
            "conditions_by_reaction": {
                rid: {k: (list(v) if v else None) for k, v in c.items()}
                for rid, c in conditions_by_reaction.items()
            },
        }

        if args.dry_run:
            print("  --dry-run: skipping ORCA execution.")
            cell_report["executed"] = False
            full_report["cells"][cell] = cell_report
            out_path = LOG_DIR / f"accuracy_benchmark_{timestamp}.json"
            out_path.write_text(json.dumps(full_report, indent=2, default=str), encoding="utf-8")
            continue

        # Found 2026-08-12/13: a single MCP session submitting metal_general's ~493-job batch
        # hung for the FULL watchdog window with zero results gathered, on FOUR consecutive
        # attempts (concurrency=12 twice, concurrency=6 once, and once more against a freshly
        # cleaned pod with zero leftover load) -- while organic_general's ~240-job single-session
        # batch completed cleanly every time, and metal_general itself completed fine when split
        # into <=30-job chunks each run through their OWN fresh session. This points at something
        # that degrades within one long-lived MCP session over many tool calls/hours (never
        # root-caused inside server_with_product.py itself -- out of scope for this run), not
        # chemistry or general qcl instability. Splitting into bounded per-session chunks is the
        # practical workaround: each chunk gets a fresh stdio_client/ClientSession, so nothing
        # ever accumulates past whatever threshold the single-session run was hitting.
        if args.qcl_session_chunk_size and len(jobs) > args.qcl_session_chunk_size:
            n_chunks = -(-len(jobs) // args.qcl_session_chunk_size)  # ceil div
            print(f"  submitting {len(jobs)} SP jobs to qcl across {n_chunks} chunk(s) of up to "
                  f"{args.qcl_session_chunk_size} (concurrency={args.concurrency} within each, "
                  f"fresh MCP session per chunk)...")
            job_results: dict[tuple[str, str, str], dict] = {}
            total_duration_ms = 0
            for ci in range(0, len(jobs), args.qcl_session_chunk_size):
                chunk = jobs[ci:ci + args.qcl_session_chunk_size]
                chunk_est_s = (len(chunk) * 20) / max(1, args.concurrency)
                chunk_watchdog_s = max(300.0, chunk_est_s * 3)
                cbatch = run_batch_with_watchdog(chunk, species, args.concurrency, chunk_watchdog_s)
                total_duration_ms += cbatch["duration_ms"]
                n_chunk_ok = sum(1 for p in cbatch["job_results"].values() if p.get("status") == "ok")
                status = "HUNG (watchdog fired, 0 results)" if cbatch.get("watchdog_timeout") else f"{n_chunk_ok}/{len(chunk)} ok"
                print(f"    chunk {ci // args.qcl_session_chunk_size}: {len(chunk)} jobs -- {status}")
                job_results.update(cbatch["job_results"])
                if cbatch.get("watchdog_timeout"):
                    print("      cleaning up any leftover ORCA processes before the next chunk...")
                    kill_stuck_orca_processes()
            batch = {"job_results": job_results, "duration_ms": total_duration_ms}
        else:
            print(f"  submitting {len(jobs)} concurrent SP jobs (concurrency={args.concurrency}) to qcl...")
            watchdog_timeout_s = max(600.0, est_wall_s * 3)
            batch = run_batch_with_watchdog(jobs, species, args.concurrency, watchdog_timeout_s)
        job_results = batch["job_results"]
        # Every job qcl itself ran passed the cheap-DFT filter, so it's never wavefunction-
        # correlated -- tag it "not_applicable" so the aux_basis_summary below covers all jobs
        # (qcl + RCCS) uniformly rather than silently omitting the qcl-run majority.
        for payload in job_results.values():
            payload.setdefault("aux_basis_plan", "not_applicable")
            payload.setdefault("aux_basis_outcome", "not_applicable")

        n_rccs_loaded = 0
        if args.rccs_results:
            rccs_path = args.rccs_results / cell / "results.json"
            if rccs_path.exists():
                rccs_raw = json.loads(rccs_path.read_text(encoding="utf-8"))
                for node_id, r in rccs_raw.items():
                    key = (r["species_id"], r["functional"], r["basis"])
                    job_results[key] = {
                        "status": r["status"], "energy_eh": r.get("energy_eh"),
                        "error": r.get("error"),
                        "aux_basis_plan": r["aux_basis_plan"],
                        "aux_basis_outcome": r["aux_basis_outcome"],
                    }
                n_rccs_loaded += len(rccs_raw)
                print(f"  merged {len(rccs_raw)} RCCS expensive-tier result(s) from {rccs_path}")
            else:
                print(f"  [warn] --rccs-results given but {rccs_path} does not exist -- "
                      f"expensive-tier reactions stay at missing_energy for this cell")

        n_ok = sum(1 for p in job_results.values() if p.get("status") == "ok")
        n_total = len(jobs) + n_rccs_loaded
        print(f"  ORCA batch done in {batch['duration_ms']/1000:.1f}s: {n_ok}/{n_total} jobs ok "
              f"({len(jobs)} qcl + {n_rccs_loaded} RCCS)")

        # Another target beside plain ok/error: for every WF-correlated job, did the aux-basis
        # mechanism itself work? "*_failed" means a real aux basis (precedent or AutoAux) was
        # given but the job still errored for an aux-basis-related reason (see
        # AUX_BASIS_ERROR_PATTERNS); "*_unrelated_error" means it errored for a different,
        # non-aux reason (e.g. QCISD on a system with zero correlated electron pairs).
        aux_basis_summary = Counter(p.get("aux_basis_outcome", "not_applicable") for p in job_results.values())
        print(f"  aux_basis_summary: {dict(aux_basis_summary)}")

        energy_cache = extract_energy_cache(job_results)
        scored = score_reactions(reactions, conditions_by_reaction, energy_cache)

        print("  Summary (non-tie MAE/RMSE, kcal/mol):")
        for cname, s in scored["summary_non_tie_only"].items():
            if s["n"]:
                print(f"    {cname:20s} n={s['n']:3d}  MAE={s['mae_kcal_mol']:.2f}  RMSE={s['rmse_kcal_mol']:.2f}")
            else:
                print(f"    {cname:20s} n=0 (no non-tie data)")

        cell_report["executed"] = True
        cell_report["batch_duration_ms"] = batch["duration_ms"]
        cell_report["n_jobs_ok"] = n_ok
        cell_report["n_jobs_total"] = n_total
        cell_report["n_rccs_jobs_merged"] = n_rccs_loaded
        cell_report["aux_basis_summary"] = dict(aux_basis_summary)
        cell_report["job_results"] = {
            sanitize_node_id(sid, f, b): payload for (sid, f, b), payload in job_results.items()
        }
        cell_report["scoring"] = scored
        full_report["cells"][cell] = cell_report

        # Write after every cell, not just at the end -- if the run gets interrupted (this
        # experiment has crashed on infra issues repeatedly today), whatever cell(s) already
        # finished are saved rather than lost.
        out_path = LOG_DIR / f"accuracy_benchmark_{timestamp}.json"
        out_path.write_text(json.dumps(full_report, indent=2, default=str), encoding="utf-8")
        print(f"  (partial results written to {out_path})")

    if args.export_sft_input:
        args.export_sft_input.parent.mkdir(parents=True, exist_ok=True)
        args.export_sft_input.write_text(json.dumps(sft_export, indent=2), encoding="utf-8")
        n_rows = sum(len(v) for v in sft_export.values())
        print(f"Wrote {n_rows} SFT-input rows -> {args.export_sft_input}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
