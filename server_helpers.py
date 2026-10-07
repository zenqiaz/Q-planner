"""Shared helpers for ORCA OPI MCP server.

This module is intentionally *server-agnostic* (no FastMCP here) so it can be
imported by multiple server entrypoints.

Key robustness points
---------------------
- Defines `_status_summary` (the previous draft referenced it but didn't define it).
- Uses `extract_final_geometry_from_out` consistently.
- For SOLVATOR cluster extraction, prefers geometry files ORCA writes (*.xyz/*.trj)
  rather than assuming a Cartesian block exists in the .out.
- Disables ORCA JSON conversion by default (`jsonpropfile/jsongbwfile`), because
  in ORCA 6 the conversion step can invoke `orca_2json`, which may crash on some
  systems. You can re-enable it if you actually need the JSON artifacts.

All energies are returned in Eh (Hartree).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Optional

# ---- OPI imports ----
from opi.core import Calculator
from opi.input.blocks.block_geom import BlockGeom
from opi.input.blocks.block_scf import BlockScf
from opi.input.structures.structure import Structure


# -----------------------------
# Small text parsers
# -----------------------------

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

# Found 2026-08-10 (RCCS accuracy-benchmark run): RIJCOSX auto-provides an aux basis for the
# SCF/DFT Fock matrix, but NOT for ORCA's MDCI (post-HF correlated) module's own RI needs --
# adding it unconditionally (use_ri=True default) makes CCSD/CCSD(T)/QCISD/MP2/DLPNO-* jobs fail
# with "Please provide an AuxC basis for RCSinglesFock". Never caught before because
# run_accuracy_benchmark.py's cheap-DFT filter always kept these methods off qcl. Plain
# HF/RHF/UHF/ROHF and CASSCF are NOT excluded -- Fock-matrix-only RIJCOSX use, no MDCI module,
# confirmed fine empirically (_functional_validity_probe.py, 2026-08-08).
_WF_CORRELATED_PATTERN = re.compile(
    r"(CCSD|MP[234]|QCISD|CISD|CEPA|CASPT2|NEVPT2|DLPNO)",
    re.IGNORECASE,
)


def _last_float_in_line(line: str) -> Optional[float]:
    vals = _FLOAT_RE.findall(line)
    if not vals:
        return None
    try:
        return float(vals[-1])
    except Exception:
        return None


def extract_total_energy(output_text: str) -> Optional[float]:
    """Parse 'FINAL SINGLE POINT ENERGY' from ORCA output (Eh)."""
    for line in reversed(output_text.splitlines()):
        if "FINAL SINGLE POINT ENERGY" in line.upper():
            try:
                return float(line.split()[-1])
            except Exception:
                return None
    return None


def extract_total_enthalpy(output_text: str) -> Optional[float]:
    """Parse 'Total Enthalpy' from ORCA frequency output (Eh)."""
    for line in reversed(output_text.splitlines()):
        if "TOTAL ENTHALPY" in line.upper():
            v = _last_float_in_line(line)
            if v is not None:
                return v
    return None


def extract_gibbs_free_energy(output_text: str) -> Optional[float]:
    """Parse 'Final Gibbs free energy' from ORCA frequency output (Eh)."""
    for line in reversed(output_text.splitlines()):
        u = line.upper()
        if "FINAL GIBBS" in u and "FREE" in u:
            v = _last_float_in_line(line)
            if v is not None:
                return v
    return None


def extract_nbo_section(output_text: str) -> str:
    """Return a conservative chunk containing NBO/NPA-ish output (best-effort)."""
    lines = output_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        u = line.upper()
        if ("NBO" in u and "ANALYSIS" in u) or ("NATURAL POPULATIONS" in u) or ("NPA" in u and "NATURAL" in u):
            start = i
            break
    if start is None:
        return "\n".join(lines[-200:])
    return "\n".join(lines[start : min(len(lines), start + 400)])


def extract_final_geometry_from_out(output_text: str) -> str:
    """Parse the last 'CARTESIAN COORDINATES (ANGSTROEM)' block as XYZ (no header)."""
    lines = output_text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line.upper():
            start_idx = i

    if start_idx is None:
        return ""

    i = start_idx
    while i < len(lines) and "-----" not in lines[i]:
        i += 1
    i += 1  # first atom line

    xyz_lines: List[str] = []
    for j in range(i, len(lines)):
        parts = lines[j].split()
        if len(parts) < 4:
            break

        # "1  O  0.0  0.0  0.0" or "O  0.0  0.0  0.0"
        if len(parts) >= 5 and parts[0].lstrip("+-").isdigit():
            symbol = parts[1]
            x, y, z = parts[2], parts[3], parts[4]
        else:
            symbol = parts[0]
            x, y, z = parts[1], parts[2], parts[3]

        xyz_lines.append(f"{symbol} {x} {y} {z}")
    return "\n".join(xyz_lines)


# -----------------------------
# Misc helpers
# -----------------------------

def _tail_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if n > 0 else ""


def _read_text_safe(p: Path, max_chars: int = 300_000) -> str:
    if not p.exists():
        return ""
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return t[-max_chars:] if len(t) > max_chars else t


def _status_summary(result_like: SimpleNamespace) -> str:
    """Compact run summary for returning to the client."""
    label = getattr(result_like, "label", "")
    status = getattr(result_like, "status", "")
    rc = getattr(result_like, "return_code", "")
    cmd = getattr(result_like, "command", "")
    out_text = getattr(result_like, "output_text", "") or ""

    errish = []
    for ln in out_text.splitlines()[-800:]:
        u = ln.upper()
        if any(k in u for k in ("ERROR", "ABORT", "FATAL", "SEGMENTATION", "FAILED", "TERMINAT")):
            errish.append(ln)
    err_tail = "\n".join(errish[-30:]) if errish else _tail_lines(out_text, 40)
    return f"Label: {label}\nStatus: {status}\nReturn code: {rc}\nCommand: {cmd}\n--- Tail ---\n{err_tail}"


# -----------------------------
# OPI helpers
# -----------------------------

def _ensure_clean_dir(d: Path, clean: bool = True) -> None:
    if clean:
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)


def _xyz_with_header(geometry_xyz_no_header: str) -> str:
    geom_lines = [ln for ln in geometry_xyz_no_header.splitlines() if ln.strip()]
    nat = len(geom_lines)
    return f"{nat}\nOPI\n" + "\n".join(geom_lines) + "\n"


def _set_charge_mult(calc: Calculator, charge: int, multiplicity: int) -> None:
    """Set charge/multiplicity on calc.input and calc.structure in a version-tolerant way."""
    for attr in ("charge", "chg"):
        if hasattr(calc.input, attr):
            setattr(calc.input, attr, charge)
            break
    for attr in ("multiplicity", "mult", "spinmultiplicity"):
        if hasattr(calc.input, attr):
            setattr(calc.input, attr, multiplicity)
            break

    if hasattr(calc, "structure") and getattr(calc, "structure", None) is not None:
        for attr in ("charge", "chg"):
            if hasattr(calc.structure, attr):
                setattr(calc.structure, attr, charge)
                break
        for attr in ("multiplicity", "mult", "spinmultiplicity"):
            if hasattr(calc.structure, attr):
                setattr(calc.structure, attr, multiplicity)
                break


def _build_calc(
    label: str,
    workdir: Path,
    geometry_xyz: str,
    charge: int,
    multiplicity: int,
    method: str,
    basis: str,
    job_type: Literal["sp", "opt", "freq"],
    use_ri: bool,
    scf_max_iter: int,
    opt_max_iter: int,
    nbo: bool,
    ncores: int,
    clean_workdir: bool = True,
) -> Calculator:
    _ensure_clean_dir(workdir, clean=clean_workdir)

    xyz_path = workdir / "struc.xyz"
    xyz_path.write_text(_xyz_with_header(geometry_xyz), encoding="utf-8")

    structure = Structure.from_xyz(xyz_path)

    calc = Calculator(basename=label, working_dir=workdir)
    calc.structure = structure
    _set_charge_mult(calc, charge, multiplicity)

    task_kw = "OPT" if job_type == "opt" else ("FREQ" if job_type == "freq" else "SP")
    # AutoAux for WF-correlated methods, not just "no RIJCOSX": DLPNO-family methods need an
    # explicit auxiliary basis unconditionally (independent of RIJCOSX). Found 2026-08-10,
    # second real-batch failure round. No-op for canonical correlated methods.
    if _WF_CORRELATED_PATTERN.search(method):
        ri_kw = "AutoAux"
    else:
        ri_kw = "RIJCOSX" if use_ri else ""
    nbo_kw = "NBO" if nbo else ""

    main_line = f"! {method} {basis} {task_kw} {ri_kw} {nbo_kw}".strip()
    calc.input.add_arbitrary_string(main_line)

    calc.input.add_blocks(BlockScf(maxiter=scf_max_iter))

    if job_type == "opt":
        calc.input.add_blocks(BlockGeom(maxiter=opt_max_iter))

    if hasattr(calc.input, "ncores"):
        calc.input.ncores = int(ncores)

    return calc


def _read_out_text(workdir: Path, label: str) -> str:
    out_path = workdir / f"{label}.out"
    if out_path.exists():
        return out_path.read_text(encoding="utf-8", errors="ignore")
    outs = sorted(workdir.glob("*.out"))
    if outs:
        return outs[-1].read_text(encoding="utf-8", errors="ignore")
    return ""


async def _run_calc_with_timeout(calc: Calculator, wall_timeout_seconds: int):
    def _run_sync():
        calc.write_input()
        calc.run()
        return calc.get_output()

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run_sync), timeout=wall_timeout_seconds)
    except asyncio.TimeoutError:
        if hasattr(calc, "kill"):
            try:
                calc.kill()
            except Exception:
                pass
        raise


# -----------------------------
# XYZ/TRJ parsing helpers
# -----------------------------

def _last_xyz_frame_no_header(xyz_text: str) -> str:
    """Return last frame from an XYZ (possibly multi-frame) string, without header."""
    lines = [ln.rstrip("\n") for ln in xyz_text.splitlines()]
    i = 0
    frames: List[List[str]] = []

    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        try:
            nat = int(lines[i].strip())
        except Exception:
            atom_lines = [ln.strip() for ln in lines if ln.strip()]
            return "\n".join(atom_lines)
        i += 1
        if i < len(lines):
            i += 1  # comment line
        atom_block = []
        for _ in range(nat):
            if i >= len(lines):
                break
            ln = lines[i].strip()
            i += 1
            if ln:
                atom_block.append(ln)
        if atom_block:
            frames.append(atom_block)

    if frames:
        return "\n".join(frames[-1])
    atom_lines = [ln.strip() for ln in lines if ln.strip()]
    return "\n".join(atom_lines)


def _extract_cluster_geometry_from_workdir(workdir: Path, label: str) -> str:
    """Best-effort: find cluster geometry written by SOLVATOR."""
    candidates: List[Path] = []
    candidates += [workdir / f"{label}.xyz", workdir / f"{label}.trj"]
    candidates += sorted(workdir.glob("*.xyz"))
    candidates += sorted(workdir.glob("*.trj"))

    # Prefer: solvator output > exact label match > others
    # "solute.xyz" is the input; ".solvator.xyz" is the cluster output
    candidates = sorted(
        {p for p in candidates if p.exists()},
        key=lambda p: (
            "solvator" not in p.name,       # solvator files first
            p.name == "solute.xyz",          # input file last
            p.name != f"{label}.xyz",
            p.name != f"{label}.trj",
            len(p.name),
            p.name,
        ),
    )

    for p in candidates:
        txt = _read_text_safe(p)
        if txt.strip():
            return _last_xyz_frame_no_header(txt)
    return ""


# -----------------------------
# SOLVATOR runner
# -----------------------------

def _run_solvator_sync(
    geometry_xyz: str,
    charge: int,
    multiplicity: int,
    nsolv: int,
    job_label: str,
) -> dict:
    """Build an explicit-solvent cluster with SOLVATOR (water) using XTB+ALPB(WATER)."""
    if geometry_xyz is None or not str(geometry_xyz).strip():
        raise ValueError("geometry_xyz is empty")
    job_label = sanitize_label(job_label)
    jobs_root = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    jobs_root.mkdir(exist_ok=True)
    workdir = jobs_root / job_label
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)

    atom_lines = [ln for ln in str(geometry_xyz).splitlines() if ln.strip()]
    natoms = len(atom_lines)
    xyz_full = f"{natoms}\n\n" + "\n".join(atom_lines) + "\n"
    xyz_path = workdir / "solute.xyz"
    xyz_path.write_text(xyz_full, encoding="utf-8")

    structure = Structure.from_xyz(xyz_path)
    structure.charge = int(charge)
    structure.multiplicity = int(multiplicity)

    calc = Calculator(basename=job_label, working_dir=workdir)
    calc.structure = structure

    # Avoid optional JSON post-processing (orca_2json) unless requested.
    calc.input.add_arbitrary_string("%output\n  jsonpropfile false\n  jsongbwfile false\nend")

    calc.input.add_arbitrary_string("! XTB ALPB(WATER)")
    calc.input.add_arbitrary_string(
        f"%solvator\n  nsolv      {int(nsolv)}\n  clustermode docking\n  fixsolute  true\nend\n"
    )

    try:
        calc.write_input()
        calc.run()
    except Exception as e:
        tb = traceback.format_exc()
        out_text = _read_out_text(workdir, job_label)
        err_text = _read_text_safe(workdir / f"{job_label}.err")
        result_like = SimpleNamespace(
            label=job_label,
            command=["orca", f"{job_label}.inp"],
            status="error",
            return_code=1,
            output_text=out_text,
        )
        return {
            "status": "error",
            "label": job_label,
            "workdir": str(workdir),
            "error": str(e),
            "traceback": tb,
            "out_tail": _tail_lines(out_text, 120),
            "err_tail": _tail_lines(err_text, 120),
            "text": _status_summary(result_like),
        }

    out_text = _read_out_text(workdir, job_label)
    err_text = _read_text_safe(workdir / f"{job_label}.err")

    status = "ok" if "ORCA TERMINATED NORMALLY" in out_text else "error"
    return_code = 0 if status == "ok" else 1

    cluster_xyz = _extract_cluster_geometry_from_workdir(workdir, job_label)
    if not cluster_xyz:
        cluster_xyz = extract_final_geometry_from_out(out_text)

    result_like = SimpleNamespace(
        label=job_label,
        command=["orca", f"{job_label}.inp"],
        status=status,
        return_code=return_code,
        output_text=out_text,
    )

    return {
        "status": status,
        "label": job_label,
        "workdir": str(workdir),
        "cluster_geometry_xyz": cluster_xyz,
        "E_eh": extract_total_energy(out_text),
        "out_tail": _tail_lines(out_text, 120),
        "err_tail": _tail_lines(err_text, 120),
        "text": _status_summary(result_like),
    }

def sanitize_label(s: str) -> str:
    s = (s or "").strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "job"