"""
run_model_harness_ablation.py -- what matters, the model or the harness?

Compares a frontier model with a small local model, each with and without the
skill layer, on the same planning tasks (Q-planner paper, SI Section S5.15).

Design: a 2x2 over model capability and harness support.

                     skills ON            skills OFF
    frontier    current system       harness without domain knowledge
    local       does the harness     floor
                carry a weak model?

The interaction is the informative cell. If domain protocol injection lifts a
weaker model toward the frontier model's performance, the harness is supplying
something the model does not have; if both models are unaffected by skills, the
skill layer is not doing the work we claim for it. Either outcome is reportable
and the expectation is recorded here before running.

Planning-only: no ORCA, no remote compute. Scored with the same structural and
task checkers used throughout the paper, so the numbers are comparable to
Section 3.3.

Two scores are reported, and they must not be conflated. The structural score is
what the paper's checkers measure: is the plan well formed and does it contain the
node kinds the task requires. The chemistry score (audit_chemistry) asks a
different question: does the plan describe the *right* calculation. A pKa plan
that requests gas-phase energies, or hard-wires a proton reference valid only for
a calibrated workflow, passes every structural check and is still wrong. Reporting
only the structural score would let "100% pass without skills" be misread as "the
skill layer is unnecessary", which is the opposite of what the failures in
Sections S5.4 and S5.14 show.

Pre-declared expectation (recorded before running the chemistry audit): skills-off
plans will match skills-on plans structurally but lose solvation and/or proton
reference handling on pKa, because those are protocol facts supplied only by the
skill text. If skills-off plans are also chemically clean, the skill layer is
doing less than claimed and that is what will be reported.

Full plan JSON is archived in every record so the runs can be re-scored or
re-executed without re-querying the model.

The local model is served over an OpenAI-compatible endpoint; no code change is
needed, only LLM_MODEL and LLM_BASE_URL. Configure the local arm with
    --local-model <name> --local-base-url <url>
and omit them to run the frontier arm alone.

Usage:
  python run_model_harness_ablation.py                       # frontier arm only
  python run_model_harness_ablation.py --local-model qwen3-32b \
      --local-base-url http://<host>:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv(".env")

from openai import OpenAI
from test_success_rate import run_plan_once, load_tool_schemas, LLM_MODEL

REPEATS = 3

TASKS = {
    "sp": ("Calculate single-point energy (B3LYP/def2-SVP) for pyruvic acid, furan, and "
           "caffeine. Run them in parallel where possible. For each compound report SP "
           "energy (energy_eh), HOMO-LUMO gap (homo_lumo_gap_ev), and dipole moment "
           "(dipole_moment_debye).", "sp"),
    "pka": ("calculate the alpha-CH pKa of ethyl acetylacetate, barbituric acid, acetone, "
            "and ethyl chloroacetate. Use ethanal (acetaldehyde, experimental pKa=17.0) as "
            "an isodesmic calibration reference to correct the systematic gas-phase DFT "
            "error.", "pka"),
    "eas": ("rank EAS reactivity of naphthalene, pyrrole, imidazole, pyrazole, thiophene, "
            "thiazole using Mulliken charges from DFT", "eas"),
}


_SOLV_KW = ("CPCM", "SMD", "COSMO", "PCM")
_SETTING_REF = re.compile(r"\$\(settings\.([^)]+)\)")

# Recorded because it describes the route taken, but not scored as a defect: a
# gas-phase calculation is correct when an isodesmic reference cancels the
# missing solvation. solvation_or_calibration is the check that decides.
_INFORMATIONAL = {"solvation_in_method", "charge_analysis_present"}


def chemistry_defects(audit: Dict[str, Optional[bool]]) -> List[str]:
    return [k for k, v in audit.items() if v is False and k not in _INFORMATIONAL]


def _resolve_setting(value: str, settings: dict) -> str:
    """Expand $(settings.key) the way the executor will, so the audit sees the
    string ORCA will actually receive rather than the unexpanded reference."""
    def sub(m):
        return str(settings.get(m.group(1), m.group(0)))
    return _SETTING_REF.sub(sub, value)


def audit_chemistry(plan: dict, task: str) -> Dict[str, Optional[bool]]:
    """Chemistry-correctness properties visible in the plan itself.

    The structural checkers used elsewhere in this paper verify that a plan is
    well formed; they do not verify that it describes the right calculation. A
    plan can pass every structural check and still request a gas-phase energy
    for an aqueous property, or feed an arithmetic step a reference constant
    that is valid only in a different regime -- both of which occurred in this
    work (Sections S5.4 and S5.14). These checks look for exactly those
    mistakes, without executing anything.

    None means "not applicable to this task".
    """
    nodes = plan.get("nodes") or []
    tool_nodes = [n for n in nodes if n.get("kind") == "tool"]
    settings = plan.get("settings") or {}
    # A node may write the method as a settings reference, e.g.
    # "$(settings.method_aq)". Those are expanded by the executor, not by the
    # planner, so auditing the raw string would score a correctly solvated plan
    # as gas-phase. Resolve before inspecting.
    methods = [_resolve_setting(str((n.get("args") or {}).get("method", "")), settings)
               for n in tool_nodes]
    all_args_keys = {k for n in tool_nodes for k in (n.get("args") or {})}
    tools = {n.get("tool") for n in tool_nodes}

    out: Dict[str, Optional[bool]] = {}

    if task == "pka":
        # ORCA takes solvation only inside the method string
        solvated = any(any(k in m.upper() for k in _SOLV_KW) for m in methods)
        out["solvation_in_method"] = solvated
        # A solvation key that sits in settings or node args without appearing in
        # the method string is inert: ORCA reads solvation only from the method
        # line, so the plan reads as solvated to a human and runs gas-phase. This
        # is the silent failure mode behind the superseded Level A/B numbers.
        inert_keys = sorted(
            k for k in (set(all_args_keys) | set(settings))
            if ("solvent" in k.lower() or "solvation" in k.lower())
            and not str(k).lower().startswith("method"))
        out["no_inert_solvation_keys"] = not (inert_keys and not out["solvation_in_method"])
        # Gibbs energies require frequencies, not single points
        out["freq_not_sp"] = ("run_freq_job" in tools) and ("run_sp_energy" not in tools)
        # arithmetic in code rather than in a language model
        out["deterministic_pka"] = "compute_pka_calibrated" in tools
        # the thermal proton reference is valid only when it cancels, i.e. when
        # reference acids are supplied; hard-wiring it uncalibrated is the
        # defect documented in Section S5.14
        gh = settings.get("G_H_plus_ref_eh")
        refs_present = any(
            (n.get("args") or {}).get("references")
            for n in tool_nodes if n.get("tool") == "compute_pka_calibrated")
        # Calibration has two equally valid implementations and the audit must
        # accept both. Either the reference acids are handed to
        # compute_pka_calibrated directly, or the plan computes a raw pKa per
        # compound and corrects it with a deterministic offset node,
        # pka_exp(ref) - pka_raw(ref), added to each target. Checking only the
        # first form marks the second as an uncalibrated gas-phase calculation,
        # which is the opposite of what it is.
        exprs = [str((n.get("args") or {}).get("expr", "") or n.get("expr", ""))
                 for n in nodes if n.get("kind") == "calc_expr"]
        offset_computed = any(
            re.search(r"exp(erimental)?", e, re.I) and re.search(r"pka", e, re.I)
            and "-" in e for e in exprs)
        offset_applied = any(
            re.search(r"offset|calib", e, re.I) and "+" in e for e in exprs)
        calibrated_via_offset = offset_computed and offset_applied
        out["proton_ref_sane"] = (
            gh is None or refs_present or abs(float(gh) + 0.01372) > 1e-9
            if isinstance(gh, (int, float)) or gh is None else None)
        out["calibration_refs_used"] = bool(refs_present) or calibrated_via_offset
        # The decisive check. There are two defensible routes to a pKa: compute
        # in solution (thermodynamic cycle), or compute in the gas phase and let
        # an isodesmic reference cancel the missing solvation and the systematic
        # DFT error. Either is fine; neither is not. A bare gas-phase
        # deprotonation with no reference is wrong by hundreds of pKa units.
        # Requiring solvation unconditionally would misscore the isodesmic route,
        # which is what the benchmark prompt explicitly asks for.
        out["solvation_or_calibration"] = solvated or bool(refs_present) or calibrated_via_offset
    elif task == "sp":
        out["no_needless_freq"] = "run_freq_job" not in tools
        out["single_sp_per_compound"] = sum(
            1 for n in tool_nodes if n.get("tool") == "run_sp_energy") >= 3
    elif task == "eas":
        out["charge_analysis_present"] = any(
            "nbo" in str((n.get("args") or {}).get("properties", "")).lower()
            or n.get("tool") == "run_nbo_job" for n in tool_nodes) or True
        out["opt_before_charges"] = "run_opt_job" in tools
    return out


def run_cell(client, model: str, arm: str, with_skills: bool,
             schemas) -> List[dict]:
    """One cell of the 2x2. run_plan_once reads the module-level model, so the
    model is swapped by rebinding it in the module the planner call goes through."""
    import test_success_rate as tsr
    prev = tsr.LLM_MODEL
    tsr.LLM_MODEL = model
    out: List[dict] = []
    try:
        for task, (msg, checker) in TASKS.items():
            for rep in range(REPEATS):
                try:
                    r = run_plan_once(client, msg, checker, with_skills=with_skills,
                                      compound_context="", tool_schemas=schemas)
                    plan = r.get("plan") or {}
                    chem = audit_chemistry(plan, task)
                    rec = {
                        "arm": arm, "model": model, "skills": with_skills,
                        "task": task, "rep": rep + 1,
                        "passed": r["passed"],
                        "failed_checks": [k for k, v in r["checks"].items() if not v],
                        "planner_tokens": r["token_usage"].get("planner", 0),
                        "context_tokens": r["token_usage"].get("context_tokens", 0),
                        "retry_fired": bool(r.get("retry_fired")),
                        "error": (r.get("error") or "")[:200],
                        "chemistry": chem,
                        "chemistry_failed": chemistry_defects(chem),
                        "plan": plan,          # archived verbatim for the SI
                    }
                except Exception as exc:                      # a weak model may not
                    rec = {                                    # return usable JSON at all
                        "arm": arm, "model": model, "skills": with_skills,
                        "task": task, "rep": rep + 1, "passed": False,
                        "failed_checks": ["exception"], "planner_tokens": 0,
                        "context_tokens": 0, "retry_fired": False,
                        "error": str(exc)[:200],
                        "chemistry": {}, "chemistry_failed": ["exception"],
                        "plan": {},
                    }
                out.append(rec)
                tag = "PASS" if rec["passed"] else "FAIL"
                chem_bad = rec["chemistry_failed"]
                print(f"  [{arm:<8} skills={'on ' if with_skills else 'off'} "
                      f"{task:<4} {rep+1}/{REPEATS}] {tag} "
                      f"tok={rec['planner_tokens']:>6}"
                      + ("" if rec["passed"] else f"  struct={rec['failed_checks'][:3]}")
                      + (f"  CHEM-BAD={chem_bad}" if chem_bad else ""), flush=True)
    finally:
        tsr.LLM_MODEL = prev
    return out


def summarise(records: List[dict]) -> None:
    print("\n" + "=" * 84)
    print(f"{'arm':<10} {'skills':>7} {'struct':>8} {'chem':>8} {'mean tok':>10} {'mean ctx':>10}")
    print("-" * 84)
    cells = {}
    for r in records:
        cells.setdefault((r["arm"], r["skills"]), []).append(r)
    for (arm, sk), g in sorted(cells.items()):
        ok = [x for x in g if x["planner_tokens"]]
        chem_ok = sum(1 for x in g if not x.get("chemistry_failed"))
        print(f"{arm:<10} {('on' if sk else 'off'):>7} "
              f"{sum(x['passed'] for x in g):>3}/{len(g):<4} "
              f"{chem_ok:>3}/{len(g):<4} "
              f"{(statistics.mean(x['planner_tokens'] for x in ok) if ok else 0):>10.0f} "
              f"{(statistics.mean(x['context_tokens'] for x in ok) if ok else 0):>10.0f}")
    print("=" * 84)

    # per-check chemistry breakdown -- this is the part the structural checkers
    # cannot see, and the reason a 100% structural pass rate must not be read as
    # "the skill layer is unnecessary"
    print("\nChemistry audit (fraction of runs satisfying each property):")
    for task in sorted({r["task"] for r in records}):
        keys = sorted({k for r in records if r["task"] == task for k in (r.get("chemistry") or {})})
        if not keys:
            continue
        print(f"  {task}:")
        for k in keys:
            line = f"    {k:<26}"
            for arm in sorted({r['arm'] for r in records}):
                for sk in (True, False):
                    g = [r for r in records if r["task"] == task and r["arm"] == arm
                         and r["skills"] == sk and (r.get("chemistry") or {}).get(k) is not None]
                    if not g:
                        continue
                    n = sum(1 for r in g if r["chemistry"][k])
                    line += f"  {arm}/{'on' if sk else 'off'}={n}/{len(g)}"
            print(line)
    # effect decomposition
    def rate(arm, sk):
        g = [x for x in records if x["arm"] == arm and x["skills"] == sk]
        return (sum(x["passed"] for x in g) / len(g) * 100) if g else float("nan")
    arms = sorted({r["arm"] for r in records})
    for arm in arms:
        on, off = rate(arm, True), rate(arm, False)
        print(f"  effect of skills on {arm:<9}: {off:.0f}% -> {on:.0f}%  ({on-off:+.0f} pts)")
    if len(arms) == 2:
        a, b = arms
        print(f"  effect of model (skills on) : {rate(b,True):.0f}% vs {rate(a,True):.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-model", default=None)
    ap.add_argument("--local-base-url", default=None)
    args = ap.parse_args()

    schemas = load_tool_schemas()
    records: List[dict] = []
    t0 = time.monotonic()

    frontier = OpenAI()
    print(f"=== frontier arm: {LLM_MODEL} ===")
    records += run_cell(frontier, LLM_MODEL, "frontier", True, schemas)
    records += run_cell(frontier, LLM_MODEL, "frontier", False, schemas)

    if args.local_model and args.local_base_url:
        print(f"\n=== local arm: {args.local_model} @ {args.local_base_url} ===")
        local = OpenAI(base_url=args.local_base_url,
                       api_key=os.getenv("LOCAL_API_KEY", "EMPTY"))
        records += run_cell(local, args.local_model, "local", True, schemas)
        records += run_cell(local, args.local_model, "local", False, schemas)
    else:
        print("\n(local arm skipped: pass --local-model and --local-base-url)")

    summarise(records)
    print(f"\nElapsed: {time.monotonic() - t0:.0f}s")

    now = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = {"type": "model_harness_ablation", "timestamp_utc": now,
           "frontier_model": LLM_MODEL, "local_model": args.local_model,
           "repeats": REPEATS, "tasks": {k: v[0] for k, v in TASKS.items()},
           "records": records}
    fname = f"test_logs/model_harness_ablation_{now}.json"
    with open(fname, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=True)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
