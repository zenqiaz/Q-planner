"""
cot_templates.py

Chain-of-thought reasoning templates for SFT training-data generation -- built in response to
insight #16 (chapter_plan_rag_sft_method_selection.md): the fine-tuned model's chemistry
reasoning survived fine-tuning intact, but the JSON-only output format never gives it room to
use it. This module generates a short, chemically-grounded reasoning span to prepend to each
training record's JSON target, keyed by method-family/basis-tier/task-type -- the same rule
categories already used informally in the three-mechanism failure analysis (insight #15) and,
before that, in skills.py's runtime domain rules.

Registry pattern deliberately mirrors skills.py's PlannerSkill/SKILL_REGISTRY design for
consistency with the rest of this codebase, even though the use case differs (skills.py injects
context into the LIVE planner prompt at inference time; this module generates TRAINING DATA
reasoning spans offline, for `generate_sft.py` to prepend to each record's JSON target).

**Explicit limitation, carried over from insight #16's own discussion -- do not let this module
overclaim what it can fix**: template selection is keyed to each record's EXISTING
(functional, basis) label. If that label itself reflects corpus convention rather than genuine
correctness (insight #15's own finding), the rendered reasoning will explain "why this choice
is defensible" convincingly regardless of whether it was actually the best choice -- this
module helps the ELICITATION problem (giving the model a reasoning step to output at all,
per insight #16) but does nothing by itself to fix mislabeled or merely-conventional training
labels. That is a separate, harder problem (see outline.md §7.7's outcome-linked-data
argument).

Usage:
    from cot_templates import build_reasoning
    reasoning = build_reasoning(record)   # record: same dict shape as generate_sft.py's input
    # -> str, 1-3 sentences, or "" if no template matched (record's own label context is too
    #    generic/unclassified -- generate_sft.py should fall back to no reasoning span in that
    #    case rather than force a template that doesn't apply)
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Shared classification sets -- kept consistent with run_accuracy_benchmark.py's
# _WF_CORRELATED_PATTERN / is_cheap_dft and the insight #15 mechanism definitions, so a
# template match means the same thing here as it does in the failure-mode analysis.
# ---------------------------------------------------------------------------

PURE_GGA = {"PBE", "BP86", "PW91", "PBESOL", "RPBE", "REVPBE", "BLYP"}
HYBRID = {"B3LYP", "PBE0", "TPSSH", "M06", "M06-2X", "M06L", "B3PW91", "MPW1PW91"}
RANGE_SEPARATED = {"WB97X-D3", "WB97X", "CAM-B3LYP", "LC-WPBE"}
WF_CORRELATED = {"CCSD", "CCSD(T)", "DLPNO-CCSD(T)", "MP2", "MP3", "MP4", "QCISD", "CISD", "CEPA"}
HF_FAMILY = {"HF", "RHF", "UHF", "ROHF"}
SEMI_EMPIRICAL = {"PM6", "PM3", "AM1", "PM7"}

MINIMAL_BASIS = {"STO-3G", "3-21G"}
DOUBLE_ZETA = {"6-31G", "6-31G*", "6-31G(D)", "6-31G(D,P)", "DEF2-SVP", "LANL2DZ"}
TRIPLE_ZETA_PLUS = {
    "6-311G", "6-311G*", "6-311+G(3DF,2P)", "6-311++G(D,P)", "DEF2-TZVP", "DEF2-TZVPP",
    "CC-PVTZ", "AUG-CC-PVTZ", "DEF2-QZVP",
}

ATOMIZATION_LIKE_TASK = {"SP", "TAE", "ATOMIZATION"}  # task_type values this applies most to


def _norm(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


class CoTTemplate:
    """Base class for one reasoning template. Mirrors skills.py's PlannerSkill shape."""
    name: str = ""
    priority: int = 50  # lower = considered first; build_reasoning uses the first match only

    def matches(self, record: dict[str, Any]) -> bool:
        raise NotImplementedError

    def render(self, record: dict[str, Any]) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Individual templates -- each one targets a specific method-family/task pattern.
# Chemistry content verified against the same sources already cited in this project's paper
# outline (Cohen/Mori-Sanchez/Yang 2008 for GGA delocalization error; Zhang & Musgrave 2007 for
# functional-dependent orbital-eigenvalue/gap accuracy; Reiher/Salomon/Hess 2001 for TM
# spin-state sensitivity) -- not invented for this module.
# ---------------------------------------------------------------------------

class GGABondEnergyTemplate(CoTTemplate):
    """Mechanism 3: pure GGA functional, atomization/bond-energy-flavored task."""
    name = "gga_bond_energy"
    priority = 10

    def matches(self, record: dict[str, Any]) -> bool:
        f = _norm(record.get("functional"))
        task = _norm(record.get("task_type"))
        return f in PURE_GGA and (task in ATOMIZATION_LIKE_TASK or "TAE" in task)

    def render(self, record: dict[str, Any]) -> str:
        f = record["functional"]
        return (
            f"{f} is a GGA functional without exact exchange, which carries known "
            f"delocalization error -- this systematically affects atomization and bond "
            f"energies, so results should be read with that bias in mind."
        )


class GGAGeneralTemplate(CoTTemplate):
    """Fallback for pure GGA functionals on tasks other than atomization/bond energies (OPT,
    FREQ, ...) -- found 2026-08-14 building the first real CoT dataset: 131 BLYP/organic_general
    records (task_type FREQ/OPT) fell through with no template at all, because
    GGABondEnergyTemplate is deliberately scoped to the specific task class its reasoning text
    is actually about. Lower priority than GGABondEnergyTemplate so the more specific,
    atomization-focused reasoning still wins when it applies -- this only fires for pure-GGA
    records that template didn't already catch."""
    name = "gga_general"
    priority = 15

    def matches(self, record: dict[str, Any]) -> bool:
        return _norm(record.get("functional")) in PURE_GGA

    def render(self, record: dict[str, Any]) -> str:
        f = record["functional"]
        return (
            f"{f} is a GGA functional without exact exchange, which carries known "
            f"delocalization error -- a reasonable, economical choice for many routine "
            f"calculations, but worth pairing with a hybrid cross-check where energetics "
            f"accuracy matters most."
        )


class WFCorrelatedBasisTemplate(CoTTemplate):
    """Mechanism 2: wavefunction-correlated method, basis tier determines the reasoning."""
    name = "wf_correlated_basis"
    priority = 10

    def matches(self, record: dict[str, Any]) -> bool:
        return _norm(record.get("functional")) in WF_CORRELATED

    def render(self, record: dict[str, Any]) -> str:
        f = record["functional"]
        b = _norm(record.get("basis"))
        if b in MINIMAL_BASIS or b in DOUBLE_ZETA:
            return (
                f"{f} is a wavefunction-correlated method, but its accuracy advantage depends "
                f"on the basis set being large enough to recover correlation energy -- a "
                f"double-zeta or smaller basis like {record.get('basis')} will leave much of "
                f"that advantage unrealized."
            )
        if b in TRIPLE_ZETA_PLUS:
            return (
                f"{f} paired with a triple-zeta or larger basis ({record.get('basis')}) gives "
                f"the correlation treatment enough flexibility to recover a meaningful fraction "
                f"of the correlation energy, which is what {f} is chosen for."
            )
        return (
            f"{f} is a wavefunction-correlated method -- its accuracy advantage over "
            f"cheaper methods depends on pairing it with a basis set large enough to actually "
            f"recover correlation energy."
        )


class HFCorrelationSensitiveTemplate(CoTTemplate):
    """Mechanism 1: HF (no correlation at all) on a system where correlation is likely to
    matter -- multiply-bonded small molecules are the clearest evidenced case (insight #15)."""
    name = "hf_correlation_sensitive"
    priority = 10

    def matches(self, record: dict[str, Any]) -> bool:
        return _norm(record.get("functional")) in HF_FAMILY

    def render(self, record: dict[str, Any]) -> str:
        return (
            "Hartree-Fock includes no electron correlation at all. Correlation energy missing "
            "from a bond scales with bond order, so HF is a particularly risky choice for "
            "systems with double or triple bonds, or generally wherever near-degenerate "
            "electronic structure is expected -- a correlated or hybrid method is safer there."
        )


class HybridGeneralPurposeTemplate(CoTTemplate):
    """Positive/reassuring case: hybrid functional, general purpose default."""
    name = "hybrid_general_purpose"
    priority = 20

    def matches(self, record: dict[str, Any]) -> bool:
        return _norm(record.get("functional")) in HYBRID

    def render(self, record: dict[str, Any]) -> str:
        f = record["functional"]
        return (
            f"{f} includes a fraction of exact exchange, which corrects a meaningful part of "
            f"the delocalization error pure GGA functionals have -- a reasonable general-"
            f"purpose default for routine thermochemistry."
        )


class RangeSeparatedTemplate(CoTTemplate):
    """Range-separated / long-range-corrected functionals -- charge-transfer, dispersion."""
    name = "range_separated"
    priority = 20

    def matches(self, record: dict[str, Any]) -> bool:
        return _norm(record.get("functional")) in RANGE_SEPARATED

    def render(self, record: dict[str, Any]) -> str:
        f = record["functional"]
        return (
            f"{f} applies a range-separated (or dispersion-corrected) treatment, which matters "
            f"most for charge-transfer states or non-covalent interactions that standard "
            f"hybrids can get wrong -- appropriate when either of those is plausibly at play."
        )


class TMSpinStateTemplate(CoTTemplate):
    """Transition-metal / open-shell systems: functional choice can flip the qualitative
    ground-state picture (Reiher, Salomon & Hess 2001), not just shift a number."""
    name = "tm_spin_state"
    priority = 5  # checked before the general functional-family templates -- TM context first

    def matches(self, record: dict[str, Any]) -> bool:
        elements = {_norm(e) for e in (record.get("elements") or [])}
        tm_like = elements & {
            "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "MO", "RU", "RH",
            "PD", "AG", "W", "OS", "IR", "PT", "AU",
        }
        return bool(tm_like) and _norm(record.get("multiplicity")) not in {"1", ""}

    def render(self, record: dict[str, Any]) -> str:
        return (
            "This is an open-shell transition-metal system -- functional choice can change "
            "which spin state comes out lowest in energy, not just the numerical accuracy "
            "(the well-documented B3LYP Fe(II) spin-state failure is the canonical example), "
            "so the functional here should be treated as a more consequential choice than for "
            "a routine closed-shell organic molecule."
        )


class SemiEmpiricalScreeningTemplate(CoTTemplate):
    """Semi-empirical methods: fast, not for quantitative accuracy."""
    name = "semi_empirical_screening"
    priority = 20

    def matches(self, record: dict[str, Any]) -> bool:
        return _norm(record.get("functional")) in SEMI_EMPIRICAL

    def render(self, record: dict[str, Any]) -> str:
        f = record["functional"]
        return (
            f"{f} is a semi-empirical method -- fast enough for geometry pre-optimization or "
            f"large-system screening, but not calibrated for quantitative energetics; if this "
            f"result needs to be accurate rather than just fast, a DFT or correlated follow-up "
            f"would be the safer choice."
        )


TEMPLATE_REGISTRY: list[CoTTemplate] = [
    TMSpinStateTemplate(),
    GGABondEnergyTemplate(),
    GGAGeneralTemplate(),
    WFCorrelatedBasisTemplate(),
    HFCorrelationSensitiveTemplate(),
    HybridGeneralPurposeTemplate(),
    RangeSeparatedTemplate(),
    SemiEmpiricalScreeningTemplate(),
]
TEMPLATE_REGISTRY.sort(key=lambda t: t.priority)


def build_reasoning(record: dict[str, Any]) -> str:
    """Returns a 1-3 sentence reasoning span for this record's (functional, basis, ...), or ""
    if nothing matched (generate_sft.py should skip the reasoning span entirely in that case,
    not force a generic filler sentence -- an empty match means this module doesn't have a
    verified-correct template for that case yet, which is more honest than a vague one)."""
    for template in TEMPLATE_REGISTRY:
        if template.matches(record):
            return template.render(record)
    return ""


if __name__ == "__main__":
    # Smoke test against a handful of real cases from this project's own failure-mode analysis
    # (insight #15) and worked examples, to confirm the templates fire on the records they're
    # actually meant for before this gets wired into generate_sft.py for real.
    demo_records = [
        {"functional": "PBE", "basis": "def2-TZVPP", "task_type": "SP", "elements": ["C", "H", "O"], "multiplicity": 1},
        {"functional": "QCISD", "basis": "6-31G*", "task_type": "SP", "elements": ["C", "N"], "multiplicity": 1},
        {"functional": "QCISD", "basis": "aug-cc-pVTZ", "task_type": "SP", "elements": ["C", "N"], "multiplicity": 1},
        {"functional": "HF", "basis": "3-21G", "task_type": "SP", "elements": ["C", "N"], "multiplicity": 1},
        {"functional": "PBE0", "basis": "def2-TZVP", "task_type": "SP", "elements": ["C", "H", "O"], "multiplicity": 1},
        {"functional": "B3LYP", "basis": "def2-SVP", "task_type": "SP", "elements": ["Fe", "C", "O"], "multiplicity": 5},
        {"functional": "PM6", "basis": None, "task_type": "OPT", "elements": ["C", "H"], "multiplicity": 1},
    ]
    for r in demo_records:
        print(f"{r['functional']}/{r['basis']}: {build_reasoning(r)}")
