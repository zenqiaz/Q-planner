import asyncio
import json
import os
import shutil
import sys
import re
from pathlib import Path
from typing import Literal, Optional, Dict, List

# Redirect stdout → stderr during imports so that any library startup messages
# don't corrupt the MCP stdio channel (critical in local mode).
_real_stdout = sys.stdout
sys.stdout = sys.stderr

from mcp.server.fastmcp import FastMCP
from pathlib import Path
import shutil
from types import SimpleNamespace



# ---- OPI imports ----
from opi.core import Calculator
from opi.input.structures.structure import Structure

from opi.input.blocks.block_scf import BlockScf
from opi.input.blocks.block_geom import BlockGeom

mcp = FastMCP("orca_tools_opi")

# ---- functions from helper ----

from server_helpers import (
    extract_total_energy,
    extract_nbo_section,
    extract_final_geometry_from_out,
    _ensure_clean_dir,
    _xyz_with_header,
    _set_charge_mult,
    _build_calc,
    _read_out_text,
    _run_calc_with_timeout,
    _run_solvator_sync,
    sanitize_label,
)

from geometry_helpers import (
    structure_proton_edit,
)

# Found 2026-08-10 (RCCS accuracy-benchmark run, the first time this project ever actually
# executed a wavefunction-correlated SP job): RIJCOSX auto-provides an aux basis for the SCF/DFT
# Fock matrix, but NOT for ORCA's MDCI (post-HF correlated) module's own RI needs -- adding it
# unconditionally (the use_ri=True default) makes CCSD/CCSD(T)/QCISD/MP2/DLPNO-* jobs fail with
# "Please provide an AuxC basis for RCSinglesFock". This was never caught before because
# run_accuracy_benchmark.py's cheap-DFT filter always kept these methods off qcl -- it's a real
# latent bug in this server, not something specific to the RCCS export script. Plain HF/RHF/UHF/
# ROHF and CASSCF are NOT excluded -- they use RIJCOSX for the Fock matrix only, no MDCI module,
# confirmed fine empirically (_functional_validity_probe.py, 2026-08-08).
_WF_CORRELATED_PATTERN = re.compile(
    r"(CCSD|MP[234]|QCISD|CISD|CEPA|CASPT2|NEVPT2|DLPNO)",
    re.IGNORECASE,
)

# -----------------------------
# Small text parsers (keep simple)
# -----------------------------
def extract_total_energy(output_text: str) -> Optional[float]:
    # ORCA usually prints: "FINAL SINGLE POINT ENERGY     -76.4..."
    for line in reversed(output_text.splitlines()):
        if "FINAL SINGLE POINT ENERGY" in line.upper():
            parts = line.split()
            try:
                return float(parts[-1])
            except Exception:
                return None
    return None




def extract_nbo_section(output_text: str) -> str:
    """Return the NPA/NBO section from ORCA output.

    Priority 1: "Summary of Natural Population Analysis:" block (has per-atom NPA charges).
    Priority 2: First "NBO ANALYSIS" / "NATURAL POPULATIONS" marker + 600 lines.
    Fallback:   Last 200 lines.
    """
    lines = output_text.splitlines()

    # Priority 1: find the NPA charges summary table
    npa_start = None
    for i, line in enumerate(lines):
        u = line.upper()
        if "SUMMARY OF NATURAL POPULATION ANALYSIS" in u or (
            "NATURAL POPULATION ANALYSIS" in u and "SUMMARY" in u
        ):
            npa_start = i
            break

    if npa_start is not None:
        # Include up to 150 lines — enough for the atom charges table
        return "\n".join(lines[npa_start : min(len(lines), npa_start + 150)])

    # Priority 2: first NBO/NPA section header
    start = None
    for i, line in enumerate(lines):
        u = line.upper()
        if ("NBO" in u and "ANALYSIS" in u) or ("NATURAL POPULATIONS" in u) or ("NPA" in u and "NATURAL" in u):
            start = i
            break
    if start is None:
        return "\n".join(lines[-200:])

    # Wider window to capture the NPA table that comes after the header
    return "\n".join(lines[start : min(len(lines), start + 800)])


def extract_final_geometry_from_out(output_text: str) -> str:
    """
    Parse a final CARTESIAN COORDINATES (ANGSTROEM) block.
    Works with both "index symbol x y z" and "symbol x y z".
    """
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
    # move to first atom line
    i += 1

    xyz_lines = []
    for j in range(i, len(lines)):
        parts = lines[j].split()
        if len(parts) < 4:
            break

        # case A: "1  O  0.0  0.0  0.0"
        if len(parts) >= 5 and parts[0].lstrip("+-").isdigit():
            symbol = parts[1]
            x, y, z = parts[2], parts[3], parts[4]
        # case B: "O  0.0  0.0  0.0"
        else:
            symbol = parts[0]
            x, y, z = parts[1], parts[2], parts[3]

        xyz_lines.append(f"{symbol} {x} {y} {z}")
    return "\n".join(xyz_lines)


# -----------------------------
# OPI helpers
# -----------------------------
def _ensure_clean_dir(d: Path, clean: bool = True):
    """Create workdir; optionally remove previous contents first."""
    if clean:
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)


def _xyz_with_header(geometry_xyz_no_header: str) -> str:
    geom_lines = [ln for ln in geometry_xyz_no_header.splitlines() if ln.strip()]
    nat = len(geom_lines)
    return f"{nat}\nOPI\n" + "\n".join(geom_lines) + "\n"


def _set_charge_mult(calc: Calculator, charge: int, multiplicity: int):
    """
    OPI stores charge/multiplicity on input/structure depending on version.
    We set what exists, safely.
    """
    for attr in ["charge", "chg"]:
        if hasattr(calc.input, attr):
            setattr(calc.input, attr, charge)
            break
    for attr in ["multiplicity", "mult", "spinmultiplicity"]:
        if hasattr(calc.input, attr):
            setattr(calc.input, attr, multiplicity)
            break

    if hasattr(calc, "structure"):
        for attr in ["charge", "chg"]:
            if hasattr(calc.structure, attr):
                setattr(calc.structure, attr, charge)
                break
        for attr in ["multiplicity", "mult", "spinmultiplicity"]:
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
    job_type: Literal["sp", "opt", "freq", "scan", "ts_opt", "casscf"],
    use_ri: bool,
    scf_max_iter: int,
    opt_max_iter: int,
    nbo: bool,
    ncores: int,
    clean_workdir: bool = True,
    raman: bool = False,
) -> Calculator:
    _ensure_clean_dir(workdir, clean=clean_workdir)

    xyz_path = workdir / "struc.xyz"
    xyz_path.write_text(_xyz_with_header(geometry_xyz), encoding="utf-8")

    structure = Structure.from_xyz(xyz_path)

    calc = Calculator(basename=label, working_dir=workdir)
    calc.structure = structure
    _set_charge_mult(calc, charge, multiplicity)

    # Keep your "same keywords" style: build one main line.
    # "scan" maps to OPT keyword (activates ORCA scan loop) but skips BlockGeom
    # to avoid a second %geom block conflicting with the %geom Scan block.
    # "ts_opt" uses OptTS keyword; caller adds its own %geom block.
    if job_type == "ts_opt":
        task_kw = "OptTS"
    elif job_type in ("opt", "scan"):
        task_kw = "OPT"
    elif job_type == "freq":
        task_kw = "FREQ"
    elif job_type == "casscf":
        task_kw = ""   # CASSCF keyword is the method itself; no separate task keyword
    else:
        task_kw = "SP"
    # AutoAux for WF-correlated methods, not just "no RIJCOSX": DLPNO-family methods need an
    # explicit auxiliary basis unconditionally (independent of RIJCOSX) for their domain/PNO
    # construction. Found 2026-08-10, second real-batch failure round (see _WF_CORRELATED_PATTERN
    # docstring above for the first). AutoAux is a no-op for canonical correlated methods that
    # don't reference an aux basis, so applying it uniformly to the whole WF-correlated group is
    # safe, not just to the DLPNO subset.
    if _WF_CORRELATED_PATTERN.search(method):
        ri_kw = "AutoAux"
    else:
        ri_kw = "RIJCOSX" if use_ri else ""
    nbo_kw = "NBO" if nbo else ""

    main_line = " ".join(p for p in [f"! {method}", basis, task_kw, ri_kw, nbo_kw] if p.strip())
    calc.input.add_arbitrary_string(main_line)

    # Raman: requires polarizability derivatives via %elprop Polar 1
    if raman:
        calc.input.add_arbitrary_string("%elprop\n  Polar 1\nend")

    # SCF control
    calc.input.add_blocks(BlockScf(maxiter=scf_max_iter))

    # OPT control (skip for "scan" and "ts_opt" — callers add their own %geom blocks)
    if job_type == "opt":
        calc.input.add_blocks(BlockGeom(maxiter=opt_max_iter))

    # cores
    if hasattr(calc.input, "ncores"):
        calc.input.ncores = int(ncores)

    return calc


def _read_out_text(workdir: Path, label: str) -> str:
    out_path = workdir / f"{label}.out"
    if out_path.exists():
        return out_path.read_text(encoding="utf-8", errors="ignore")
    # fallback: read any .out in folder
    outs = sorted(workdir.glob("*.out"))
    if outs:
        return outs[-1].read_text(encoding="utf-8", errors="ignore")
    return ""



def _tail_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if n > 0 else ""

def _read_text_safe(p: Path, max_chars: int = 200_000) -> str:
    if not p.exists():
        return ""
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return t[-max_chars:] if len(t) > max_chars else t

def _list_dir(workdir: Path, max_files: int = 80):
    try:
        items = sorted(workdir.iterdir(), key=lambda x: x.name)
    except Exception:
        return []
    out = []
    for p in items[:max_files]:
        try:
            st = p.stat()
            out.append({"name": p.name, "size": st.st_size, "is_dir": p.is_dir()})
        except Exception:
            out.append({"name": p.name, "size": None, "is_dir": p.is_dir()})
    return out

def _which(cmd: str) -> Optional[str]:
    import shutil as _sh
    return _sh.which(cmd)

def _collect_debug(workdir: Path, label: str, tail_lines: int = 120):
    out_path = workdir / f"{label}.out"
    inp_path = workdir / f"{label}.inp"
    err_path = workdir / f"{label}.err"
    return {
        "workdir": str(workdir),
        "orca_exe": _which("orca"),
        "files": _list_dir(workdir),
        "inp_tail": _tail_lines(_read_text_safe(inp_path), min(200, tail_lines)),
        "out_tail": _tail_lines(_read_text_safe(out_path), tail_lines),
        "err_tail": _tail_lines(_read_text_safe(err_path), tail_lines),
    }

def _error_summary(out_text: str) -> str:
    keys = ("ERROR", "ABORT", "FATAL", "TERMINAT", "did not converge", "SCF", "OPTIMIZATION")
    hits = []
    for ln in out_text.splitlines()[-1200:]:
        u = ln.upper()
        if any(k.upper() in u for k in keys):
            hits.append(ln)
    return "\n".join(hits[-60:])

def _classify_orca_error_code(out_text: str) -> Optional[str]:
    """Return a structured error code by scanning ORCA output text.

    Used to populate the 'code' field in error payloads so the client-side
    patch_and_retry logic can match on_error rules without text-parsing.
    """
    upper = out_text.upper()
    if "SCF NOT CONVERGED" in upper or "FAILED TO CONVERGE" in upper:
        return "SCF_NOT_CONVERGED"
    if "***IMAGINARY MODE***" in upper:
        return "IMAG_FREQ"
    if (
        "THE CAS PROCEDURE HAS NOT CONVERGED" in upper
        or "CASSCF ORBITAL OPTIMIZATION DID NOT CONVERGE" in upper
        or "CAS PROCEDURE DID NOT CONVERGE" in upper
        or ("CASSCF" in upper and "NOT CONVERGED" in upper)
    ):
        return "CASSCF_NOT_CONVERGED"
    return None


async def _run_calc_with_timeout(calc: Calculator, wall_timeout_seconds: int):
    def _run_sync():
        calc.write_input()
        calc.run()
        return calc.get_output()

    try:
        return await asyncio.wait_for(asyncio.to_thread(_run_sync), timeout=wall_timeout_seconds)
    except asyncio.TimeoutError:
        # Best-effort kill if supported
        if hasattr(calc, "kill"):
            try:
                calc.kill()
            except Exception:
                pass
        raise



# -----------------------------
# Thermochemistry parsers (E/H/G) from ORCA output
# -----------------------------
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def _last_float_in_line(line: str) -> Optional[float]:
    vals = _FLOAT_RE.findall(line)
    if not vals:
        return None
    try:
        return float(vals[-1])
    except Exception:
        return None

def extract_total_enthalpy(output_text: str) -> Optional[float]:
    # ORCA prints something like: "Total Enthalpy ...   -XXX.XXXX Eh"
    for line in reversed(output_text.splitlines()):
        u = line.upper()
        if "TOTAL ENTHALPY" in u:
            v = _last_float_in_line(line)
            if v is not None:
                return v
    return None

def extract_gibbs_free_energy(output_text: str) -> Optional[float]:
    # ORCA prints something like: "Final Gibbs free energy ...  -XXX.XXXX Eh"
    for line in reversed(output_text.splitlines()):
        u = line.upper()
        if "FINAL GIBBS" in u and "FREE" in u:
            v = _last_float_in_line(line)
            if v is not None:
                return v
    return None

def extract_casscf_root_energies(output_text: str) -> List[float]:
    """Extract per-root CASSCF energies from SA-CASSCF output.

    Handles two ORCA 6 formats:
      "E[SA-CASSCF] ROOT  N :  -XXX.XXXXXX Eh"
      "ROOT N: E = -XXX.XXXXXX Eh"
    Returns a list sorted by root index (empty if none found).
    """
    import re
    patterns = [
        re.compile(r"E\[SA-CASSCF\]\s+ROOT\s+(\d+)\s*:\s*(-\d+\.\d+)\s*Eh", re.IGNORECASE),
        re.compile(r"\bROOT\s+(\d+)\s*:\s*E\s*=\s*(-\d+\.\d+)\s*Eh", re.IGNORECASE),
    ]
    root_energies: Dict[int, float] = {}
    for line in output_text.splitlines():
        for pat in patterns:
            m = pat.search(line)
            if m:
                root_energies[int(m.group(1))] = float(m.group(2))
                break
    return [root_energies[i] for i in sorted(root_energies)]


def extract_dipole_moment(output_text: str) -> Optional[float]:
    """Parse dipole moment magnitude in Debye from ORCA output.

    Finds the last occurrence of:
        Magnitude (Debye)      :      2.089206051
    Returns the magnitude as a float, or None if not found.
    """
    pat = re.compile(r'Magnitude\s*\(Debye\)\s*:\s*([\d.]+)', re.IGNORECASE)
    result = None
    for m in pat.finditer(output_text):
        result = float(m.group(1))  # keep last occurrence
    return result


def extract_mulliken_charges(output_text: str) -> List[Dict]:
    """Parse MULLIKEN ATOMIC CHARGES block from ORCA output.

    Returns list of {atom_index, symbol, charge} dicts, one per heavy atom.
    Lines look like:    0 C  :    -0.037265
    """
    lines = output_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "MULLIKEN ATOMIC CHARGES" in line.upper():
            start = i
    if start is None:
        return []
    charges = []
    pat = re.compile(r'^\s*(\d+)\s+([A-Za-z]+)\s*:\s*([-\d.]+)')
    for line in lines[start + 2 :]:
        m = pat.match(line)
        if m:
            charges.append({
                "atom_index": int(m.group(1)),
                "symbol":     m.group(2),
                "charge":     float(m.group(3)),
            })
        elif line.strip() == "" or "SUM" in line.upper():
            break
    return charges


def extract_homo_lumo_gap(output_text: str) -> Optional[float]:
    """Parse HOMO and LUMO orbital energies from ORCA output and return the gap in eV.

    ORCA prints an ORBITAL ENERGIES block for all job types:

        ORBITAL ENERGIES
        ----------------
          NO   OCC          E(Eh)            E(eV)
           0   2.0000      -20.5680...      -559.8...
           ...
           4   2.0000       -0.4444...       -12.09...   <- HOMO (last OCC>0)
           5   0.0000        0.1764...         4.80...   <- LUMO (first OCC==0)

    Returns gap in eV, or None if block not found / fewer than 2 MOs parsed.
    """
    # Find the last ORBITAL ENERGIES block (TDDFT runs SCF first; take the last)
    lines = output_text.splitlines()
    block_start = None
    for i, line in enumerate(lines):
        if "ORBITAL ENERGIES" in line.upper():
            block_start = i

    if block_start is None:
        return None

    # Parse MO lines: index  occ  E(Eh)  E(eV)
    pat = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$')
    homo_ev = None
    lumo_ev = None
    for line in lines[block_start + 1: block_start + 500]:
        m = pat.match(line)
        if not m:
            continue
        occ = float(m.group(2))
        e_ev = float(m.group(4))
        if occ > 0.0:
            homo_ev = e_ev       # keep updating; last occupied = HOMO
        elif homo_ev is not None and lumo_ev is None:
            lumo_ev = e_ev       # first unoccupied after HOMO = LUMO
            break

    if homo_ev is None or lumo_ev is None:
        return None
    return round(lumo_ev - homo_ev, 6)


def extract_ir_spectrum(output_text: str) -> List[Dict]:
    """Parse IR spectrum from ORCA output, including imaginary modes.

    In ORCA 6, the IR SPECTRUM block only lists real (positive frequency) modes.
    Imaginary modes appear only in the VIBRATIONAL FREQUENCIES block with a
    '***imaginary mode***' marker.  This function combines both sources so that
    TS calculations report their imaginary mode(s) with negative freq_cm1 values.

    Returns [{"mode": int, "freq_cm1": float, "intensity_km_mol": float}, ...]
    sorted by mode number. Imaginary modes have intensity_km_mol = 0.0.
    Modes with |freq| <= 10 cm-1 (translations/rotations) are excluded.
    """
    lines = output_text.splitlines()
    result: List[Dict] = []

    # --- Step 1: Imaginary modes from VIBRATIONAL FREQUENCIES block ---
    # ORCA 6 format: "   6:      -205.12 cm**-1 ***imaginary mode***"
    imag_pat = re.compile(
        r'^\s+(\d+):\s+([-+]?\d+\.?\d*)\s+cm\*\*-1\s+\*+imaginary', re.IGNORECASE
    )
    for line in lines:
        m = imag_pat.match(line)
        if m:
            mode = int(m.group(1))
            freq = float(m.group(2))
            if abs(freq) > 10.0:
                result.append({"mode": mode, "freq_cm1": freq, "intensity_km_mol": 0.0})

    # --- Step 2: Real modes from IR SPECTRUM block (last occurrence) ---
    # ORCA format:
    #   Mode   freq       eps      Int      T**2   TX  TY  TZ
    #           cm**-1  L/(mol*cm)  km/mol   a.u.
    #   7:   948.32   0.000456   3.14  ...
    block_start = None
    for i in range(len(lines) - 1, -1, -1):
        if re.search(r'\bIR\s+SPECTRUM\b', lines[i], re.IGNORECASE):
            block_start = i
            break
    if block_start is not None:
        pat = re.compile(
            r'^\s*(\d+):\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)'
        )
        seen_modes = {r["mode"] for r in result}
        for line in lines[block_start + 1 : block_start + 200]:
            m = pat.match(line)
            if m:
                mode = int(m.group(1))
                freq = float(m.group(2))
                intensity = float(m.group(4))   # group(3)=eps, group(4)=Int km/mol
                if abs(freq) > 10.0 and mode not in seen_modes:
                    result.append({"mode": mode, "freq_cm1": freq,
                                   "intensity_km_mol": intensity})
                    seen_modes.add(mode)
            elif result and line.strip() and not line.strip().startswith(('-', '*')):
                if not re.match(r'\s*(mode|freq|cm\*\*)', line.strip(), re.IGNORECASE):
                    break

    return sorted(result, key=lambda x: x["mode"])


def extract_raman_spectrum(output_text: str) -> List[Dict]:
    """Parse RAMAN SPECTRUM block from ORCA output.

    Returns [{"mode": int, "freq_cm1": float, "activity": float, "depolarization": float}, ...]
    for real vibrational modes only (|freq| > 10 cm-1).
    Requires %elprop Polar 1 in the ORCA input.

    ORCA format:
      Mode    freq (cm**-1)   Activity   Depolarization
      6:      2078.52      8.498849      0.722060
    """
    lines = output_text.splitlines()
    block_start = None
    for i in range(len(lines) - 1, -1, -1):
        if re.search(r'\bRAMAN\s+SPECTRUM\b', lines[i], re.IGNORECASE):
            block_start = i
            break
    if block_start is None:
        return []

    pat = re.compile(r'^\s*(\d+):\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)')
    result = []
    for line in lines[block_start + 1: block_start + 100]:
        m = pat.match(line)
        if m:
            mode = int(m.group(1))
            freq = float(m.group(2))
            activity = float(m.group(3))
            depol = float(m.group(4))
            if abs(freq) > 10.0:
                result.append({"mode": mode, "freq_cm1": freq,
                                "activity": activity, "depolarization": depol})
        elif result and line.strip() and not line.strip().startswith(('-', '*')):
            if not re.match(r'\s*(mode|freq|activity|depol)', line.strip(), re.IGNORECASE):
                break
    return result


_EV_PER_CM1 = 1.0 / 8065.54439  # 1 cm⁻¹ in eV


def extract_excited_states(output_text: str) -> List[Dict]:
    """Parse TD-DFT excited states from ORCA absorption spectrum table.

    Finds the last occurrence of the
    "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" block.

    ORCA 6 format (columns: Transition | Energy(eV) | Energy(cm-1) | Wavelength(nm) | fosc(D2) | ...):
        0-1A  ->  1-1A    4.077777   32889.5   304.0   0.000000000   ...
        0-1A  ->  2-1A    8.200851   66144.3   151.2   0.170248747   ...

    Returns list of:
        {"state": int, "energy_ev": float, "wavelength_nm": float,
         "oscillator_strength": float}
    """
    lines = output_text.splitlines()
    block_start = None
    for i, line in enumerate(lines):
        if "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line.upper():
            block_start = i

    if block_start is None:
        return []

    # Match: "  0-1A  ->  N-XA    energy_ev   energy_cm1   wavelength_nm   fosc   ..."
    pat = re.compile(
        r'^\s+\d+-\S+\s*->\s*(\d+)-\S+\s+'  # transition label (group 1: target state #)
        r'([\d.]+)\s+'                         # energy_ev (group 2)
        r'[\d.]+\s+'                           # energy_cm1 (skip)
        r'([\d.]+)\s+'                         # wavelength_nm (group 3)
        r'([\d.]+(?:[eE][+-]?\d+)?)'           # fosc (group 4)
    )
    result = []
    for line in lines[block_start + 1: block_start + 200]:
        m = pat.match(line)
        if m:
            state = int(m.group(1))
            energy_ev = float(m.group(2))
            wavelength_nm = float(m.group(3))
            fosc = float(m.group(4))
            result.append({
                "state": state,
                "energy_ev": round(energy_ev, 4),
                "wavelength_nm": round(wavelength_nm, 2),
                "oscillator_strength": round(fosc, 6),
            })
        elif result and re.match(r'\s*-{10,}', line):
            break  # end-of-block separator
    return result


def _get_cluster_xyz_from_solvator_result(result: dict) -> Optional[str]:
    # Be defensive: different helper versions use different keys.
    for k in [
        "cluster_xyz",
        "cluster_geometry_xyz",
        "cluster_geometry",
        "geometry_xyz",
        "cluster",
        "xyz",
    ]:
        v = result.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


@mcp.tool()
async def run_solvator_cluster_thermo(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    nsolv: int = 3,
    thermo_engine: Literal["orca"] = "orca",
    # Reasonable cheap default for freq on a cluster:
    method: str = "r2scan-3c",
    basis: str = "",
    use_ri: bool = False,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 600,
    thermo_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """
    Build an explicit-solvent cluster with SOLVATOR and then compute E/H/G via an ORCA frequency job.

    Returns:
      - cluster geometry (XYZ, no header if that is what SOLVATOR returns)
      - E (FINAL SINGLE POINT ENERGY), H (Total Enthalpy), G (Final Gibbs free energy) in Eh
      - tails of logs for debugging
    """

    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"solvthermo_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    # 1) SOLVATOR cluster build (sync helper in a thread)
    solv_result = await asyncio.to_thread(
        _run_solvator_sync,
        geometry_xyz,
        charge,
        multiplicity,
        nsolv,
        job_label,
    )

    if not isinstance(solv_result, dict):
        return json.dumps({"status": "error", "label": job_label, "error": "SOLVATOR helper returned non-dict."})

    cluster_xyz_no_header = _get_cluster_xyz_from_solvator_result(solv_result) or ""

    # Normalize: accept either XYZ-with-header or atom-lines-only.
    lines = [ln.strip() for ln in cluster_xyz_no_header.splitlines() if ln.strip()]
    if lines:
        try:
            nat = int(lines[0])
            if len(lines) >= nat + 2:
                cluster_xyz_no_header = "\n".join(lines[2 : 2 + nat])
        except Exception:
            pass
    if not cluster_xyz_no_header:
        # Return solvator result as-is; thermo step skipped.
        solv_result.setdefault("status", "error")
        solv_result["label"] = job_label
        solv_result["thermo_status"] = "skipped"
        solv_result["thermo_error"] = "Could not find cluster XYZ in SOLVATOR result."
        return json.dumps(solv_result)

    # Validate: check solvator actually added water molecules
    input_lines = [ln for ln in str(geometry_xyz).splitlines() if ln.strip()]
    cluster_lines = [ln for ln in cluster_xyz_no_header.splitlines() if ln.strip()]
    if len(cluster_lines) <= len(input_lines):
        solv_result.setdefault("status", "error")
        solv_result["label"] = job_label
        solv_result["thermo_status"] = "skipped"
        solv_result["thermo_error"] = (
            f"SOLVATOR did not add solvent molecules: input has {len(input_lines)} atoms, "
            f"cluster has {len(cluster_lines)} atoms. Molecule may be too small for solvation."
        )
        solv_result["cluster_xyz"] = cluster_xyz_no_header
        return json.dumps(solv_result)

    # Guard: freq needs at least 2 atoms (no vibrations for a single atom)
    if len(cluster_lines) < 2:
        solv_result.setdefault("status", "error")
        solv_result["label"] = job_label
        solv_result["thermo_status"] = "skipped"
        solv_result["thermo_error"] = "Cannot run frequency calculation on a single atom."
        solv_result["cluster_xyz"] = cluster_xyz_no_header
        return json.dumps(solv_result)

    # 2) ORCA frequency job for thermochemistry
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    thermo_label = f"{job_label}_freq"
    thermo_workdir = jobs_dir / thermo_label

    # Write cluster xyz (with header) into thermo workdir
    _ensure_clean_dir(thermo_workdir)
    (thermo_workdir / "cluster.xyz").write_text(_xyz_with_header(cluster_xyz_no_header), encoding="utf-8")

    # Build ORCA calc with FREQ task
    calc = _build_calc(
        label=thermo_label,
        workdir=thermo_workdir,
        geometry_xyz=cluster_xyz_no_header,  # _build_calc writes struc.xyz; OK
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="freq",
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=1,
        nbo=False,
        ncores=ncores,
    )

    # Ensure the calc uses the cluster.xyz geometry we wrote (more explicit)
    xyz_path = thermo_workdir / "struc.xyz"
    xyz_path.write_text(_xyz_with_header(cluster_xyz_no_header), encoding="utf-8")
    calc.structure = Structure.from_xyz(xyz_path)
    _set_charge_mult(calc, charge, multiplicity)

    try:
        output = await _run_calc_with_timeout(calc, thermo_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(thermo_workdir, thermo_label)
        return json.dumps(
            {
                "status": solv_result.get("status", "ok"),
                "label": job_label,
                "cluster_xyz": cluster_xyz_no_header,
                "thermo_status": "timeout",
                "E_eh": extract_total_energy(out_text),
                "H_eh": extract_total_enthalpy(out_text),
                "G_eh": extract_gibbs_free_energy(out_text),
                "thermo_tail": "\n".join(out_text.splitlines()[-120:]),
                "solvator": solv_result,
                "product": ("E_eh", "H_eh", "G_eh")
                
            }
        )
    except Exception as e:
        out_text = _read_out_text(thermo_workdir, thermo_label)
        return json.dumps(
            {
                "status": solv_result.get("status", "ok"),
                "label": job_label,
                "cluster_xyz": cluster_xyz_no_header,
                "thermo_status": "error",
                "thermo_error": str(e),
                "thermo_tail": "\n".join(out_text.splitlines()[-120:]),
                "solvator": solv_result,
            }
        )

    ok = output.terminated_normally()
    out_text = _read_out_text(thermo_workdir, thermo_label)

    if not ok:
        return json.dumps(
            {
                "status": solv_result.get("status", "ok"),
                "label": job_label,
                "cluster_xyz": cluster_xyz_no_header,
                "thermo_status": "error",
                "thermo_tail": "\n".join(out_text.splitlines()[-150:]),
                "solvator": solv_result,
                "text": "ORCA freq did not terminate normally.",
            }
        )

    E = extract_total_energy(out_text)
    H = extract_total_enthalpy(out_text)
    G = extract_gibbs_free_energy(out_text)

    return json.dumps(
        {
            "status": "ok",
            "label": job_label,
            "cluster_xyz": cluster_xyz_no_header,
            "E_eh": E,
            "H_eh": H,
            "G_eh": G,
            "thermo_status": "ok",
            "thermo_label": thermo_label,
            "solvator": solv_result,
            "text": f"Status: OK\nE={E} Eh\nH={H} Eh\nG={G} Eh",
            "product": ("G_eh")
        }
    )

# -----------------------------
# Constraint helpers
# -----------------------------

def _constraint_auto_value(ctype: str, atoms: list, geometry_xyz: str) -> Optional[float]:
    """Compute the current value of a constraint from the geometry (bond length in Å)."""
    import math
    if ctype != "B" or len(atoms) < 2:
        return None
    coords = []
    for line in (geometry_xyz or "").strip().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                pass
    i, j = atoms[0], atoms[1]
    if i >= len(coords) or j >= len(coords):
        return None
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    dz = coords[i][2] - coords[j][2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def _build_constraint_line(c: dict, geometry_xyz: str = None) -> str:
    """Format one ORCA constraint entry: {B i j value C}.

    ORCA 6 %geom Constraints block requires explicit value + C modifier.
    If value is omitted, auto-computes from geometry_xyz (bond length only).
    """
    ctype = (c.get("type") or "B").upper()
    atoms = c.get("atoms") or []
    atom_str = " ".join(str(a) for a in atoms)
    value = c.get("value")
    if value is None:
        value = _constraint_auto_value(ctype, atoms, geometry_xyz)
    if value is None:
        raise ValueError(
            f"Constraint {ctype} {atoms}: no value provided and auto-computation failed. "
            "Supply an explicit 'value' in the constraint dict."
        )
    return f"    {{{ctype} {atom_str} {float(value):.4f} C}}"


def _build_constrained_geom_block(opt_max_iter: int, constraints: list,
                                   geometry_xyz: str = None) -> str:
    """Build a %geom block with maxiter and Constraints section."""
    lines = ["%geom", f"  maxiter {opt_max_iter}", "  Constraints"]
    lines.extend(_build_constraint_line(c, geometry_xyz) for c in constraints)
    lines += ["  end", "end"]
    return "\n".join(lines)


# -----------------------------
# Tools
# -----------------------------

@mcp.tool()
async def run_opt_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    opt_max_iter: int = 100,
    constraints: Optional[list] = None,
    wall_timeout_seconds: int = 1800,
    job_label: Optional[str] = None,
    ncores: int = 1,
    xtb_preopt: bool = True,
    xtb_preopt_timeout: int = 300,
    clean_workdir: bool = True,
    debug: bool = False,
    debug_tail_lines: int = 160,
) -> str:
    """Geometry optimization with optional debug bundle.

    Debug tips:
      - set debug=True to include inp/out tails, file listing, ORCA path
      - set clean_workdir=False to preserve an existing directory for post-mortem comparison
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"opt_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir = jobs_dir / job_label

    # --- optional xTB pre-optimisation ---
    # Run a cheap GFN2-xTB geometry optimisation first so B3LYP starts near the minimum.
    # Only activated when xtb_preopt=True (e.g. via on_error patch_and_retry).
    active_geometry = geometry_xyz
    if xtb_preopt:
        xtb_label = job_label + "_xtbpre"
        xtb_workdir = jobs_dir / xtb_label
        xtb_calc = _build_calc(
            label=xtb_label,
            workdir=xtb_workdir,
            geometry_xyz=geometry_xyz,
            charge=charge,
            multiplicity=multiplicity,
            method="XTB2",
            basis="",
            job_type="opt",
            use_ri=False,
            scf_max_iter=150,
            opt_max_iter=500,
            nbo=False,
            ncores=ncores,
            clean_workdir=True,
        )
        try:
            xtb_output = await _run_calc_with_timeout(xtb_calc, xtb_preopt_timeout)
            xtb_out_text = _read_out_text(xtb_workdir, xtb_label)
            xtb_geom = extract_final_geometry_from_out(xtb_out_text)
            if xtb_geom and xtb_geom.strip():
                active_geometry = xtb_geom
        except Exception:
            pass  # xTB failed — fall through to DFT with original geometry

    # When constraints are provided, use job_type="scan" so _build_calc adds the
    # OPT keyword but skips BlockGeom — we then add a single combined %geom block
    # containing both maxiter and the Constraints section.
    _job_type = "scan" if constraints else "opt"
    calc = _build_calc(
        label=job_label,
        workdir=workdir,
        geometry_xyz=active_geometry,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type=_job_type,
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=opt_max_iter,
        nbo=False,
        ncores=ncores,
        clean_workdir=clean_workdir,
    )
    if constraints:
        calc.input.add_arbitrary_string(
            _build_constrained_geom_block(opt_max_iter, constraints, geometry_xyz)
        )

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        resp = {
            "status": "timeout",
            "label": job_label,
            "workdir": str(workdir),
            "energy": extract_total_energy(out_text),
            "final_geometry_xyz": extract_final_geometry_from_out(out_text),
            "tail": _tail_lines(out_text, debug_tail_lines),
            "error_summary": _error_summary(out_text),
            "text": f"Status: TIMEOUT after {wall_timeout_seconds}s",
        }
        if debug:
            resp["debug"] = _collect_debug(workdir, job_label, tail_lines=debug_tail_lines)
        return json.dumps(resp)

    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        resp = {
            "status": "error",
            "label": job_label,
            "workdir": str(workdir),
            "error": str(e),
            "code": _classify_orca_error_code(out_text),
            "tail": _tail_lines(out_text, debug_tail_lines),
            "error_summary": _error_summary(out_text),
        }
        if debug:
            resp["debug"] = _collect_debug(workdir, job_label, tail_lines=debug_tail_lines)
        return json.dumps(resp)

    out_text = _read_out_text(workdir, job_label)
    ok = output.terminated_normally()

    # ORCA prints "*** THE OPTIMIZATION HAS CONVERGED ***" when converged.
    opt_converged = ("THE OPTIMIZATION HAS CONVERGED" in out_text.upper())

    energy = extract_total_energy(out_text)
    final_xyz = extract_final_geometry_from_out(out_text)

    if not ok:
        resp = {
            "status": "error",
            "label": job_label,
            "workdir": str(workdir),
            "opt_converged": opt_converged,
            "energy": energy,
            "final_geometry_xyz": final_xyz,
            "code": _classify_orca_error_code(out_text),
            "tail": _tail_lines(out_text, debug_tail_lines),
            "error_summary": _error_summary(out_text),
            "text": "ORCA did not terminate normally.",
        }
        if debug:
            resp["debug"] = _collect_debug(workdir, job_label, tail_lines=debug_tail_lines)
        return json.dumps(resp)

    # Terminated normally but may still hit maxoptiter (common case)
    status = "ok" if opt_converged else "not_converged"

    resp = resp = {
    "status": status,
    "label": job_label,
    "workdir": str(workdir),
    "opt_converged": opt_converged,
    "energy_eh": energy,  # rename is optional but nice
    "final_geometry_xyz": final_xyz,  # keep for backward compat
    "geometry_xyz": final_xyz,         # NEW: unified key for SessionState
    "provenance": {"geometry": "orca_opt"},
    "tail": _tail_lines(out_text, min(120, debug_tail_lines)),
    "text": f"Status: {status.upper()}\nFinal energy: {energy if energy is not None else 'N/A'} Eh",
    }
    if debug:
        resp["debug"] = _collect_debug(workdir, job_label, tail_lines=debug_tail_lines)
    return json.dumps(resp)




@mcp.tool()
async def run_nbo_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    job_type: Literal["sp", "opt"] = "sp",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    opt_max_iter: int = 50,
    wall_timeout_seconds: int = 600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"nbo_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir = jobs_dir / job_label

    calc = _build_calc(
        label=job_label,
        workdir=workdir,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type=job_type,
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=opt_max_iter,
        nbo=True,
        ncores=ncores,
    )

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps(
            {
                "status": "timeout",
                "label": job_label,
                "energy": extract_total_energy(out_text),
                "text": f"Status: TIMEOUT after {wall_timeout_seconds}s",
                "tail": "\n".join(out_text.splitlines()[-80:]),
            }
        )
    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps(
            {"status": "error", "label": job_label, "error": str(e), "tail": "\n".join(out_text.splitlines()[-80:])}
        )

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)

    if not ok:
        return json.dumps(
            {
                "status": "error",
                "label": job_label,
                "tail": "\n".join(out_text.splitlines()[-120:]),
                "text": "ORCA did not terminate normally.",
            }
        )

    energy = extract_total_energy(out_text)
    nbo_section = extract_nbo_section(out_text)

    return json.dumps(
        {
            "status": "ok",
            "label": job_label,
            "energy": energy,
            "nbo_section": nbo_section,
            "text": "Status: OK\n=== NBO / NPA Section (excerpt) ===\n" + nbo_section, 
            "product": ("energy", "nbo_section")
            
        }
    )


@mcp.tool()
async def run_sp_energy(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 600,
    job_label: Optional[str] = None,
    ncores: int = 1,
    properties: Optional[List[str]] = None,
    n_tddft_states: int = 5,
) -> str:
    """Single-point DFT job.

    properties: optional list of extra quantities to compute in the same ORCA run.
      "dipole"       — dipole moment magnitude (Debye); always returned, no extra cost
      "homo_lumo_gap"— KS orbital gap (eV); always returned, no extra cost
      "nbo"          — NBO/NPA analysis; adds NBO keyword to ORCA input
      "tddft"        — TD-DFT excited states; adds %tddft block (n_tddft_states roots)

    All of energy_eh, dipole_moment_debye, homo_lumo_gap_ev are always returned
    regardless of the properties list. nbo_section and excited_states are only
    returned when explicitly requested.
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    props = set(properties or [])

    if job_label is None:
        job_label = f"sp_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir = jobs_dir / job_label

    calc = _build_calc(
        label=job_label,
        workdir=workdir,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="sp",
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=50,
        nbo="nbo" in props,
        ncores=ncores,
    )

    if "tddft" in props:
        n_states = max(1, int(n_tddft_states))
        calc.input.add_arbitrary_string(f"%tddft\n  nroots {n_states}\nend")

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps(
            {"status": "timeout", "label": job_label, "text": f"Status: TIMEOUT after {wall_timeout_seconds}s"}
        )
    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({"status": "error", "label": job_label, "error": str(e),
                           "code": _classify_orca_error_code(out_text)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({"status": "error", "label": job_label,
                           "code": _classify_orca_error_code(out_text),
                           "tail": "\n".join(out_text.splitlines()[-120:])})

    result: Dict[str, Any] = {
        "status":             "ok",
        "label":              job_label,
        "energy_eh":          extract_total_energy(out_text),
        "dipole_moment_debye": extract_dipole_moment(out_text),
        "homo_lumo_gap_ev":   extract_homo_lumo_gap(out_text),
        "mulliken_charges":   extract_mulliken_charges(out_text),
        "product":            "energy_eh",
    }
    if "nbo" in props:
        result["nbo_section"] = extract_nbo_section(out_text)
    if "tddft" in props:
        result["excited_states"] = extract_excited_states(out_text)

    return json.dumps(result)



@mcp.tool()
async def run_freq_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run an ORCA frequency calculation. Returns E, H, G (Gibbs free energy) in Eh."""
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"freq_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir = jobs_dir / job_label

    calc = _build_calc(
        label=job_label,
        workdir=workdir,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="freq",
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=1,
        nbo=False,
        ncores=ncores,
    )

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        return json.dumps(
            {"status": "timeout", "label": job_label, "text": f"Status: TIMEOUT after {wall_timeout_seconds}s"}
        )
    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({"status": "error", "label": job_label, "error": str(e),
                           "code": _classify_orca_error_code(out_text)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({"status": "error", "label": job_label,
                           "code": _classify_orca_error_code(out_text),
                           "tail": "\n".join(out_text.splitlines()[-120:])})

    energy = extract_total_energy(out_text)
    enthalpy = extract_total_enthalpy(out_text)
    gibbs = extract_gibbs_free_energy(out_text)

    imag_code = _classify_orca_error_code(out_text)  # "IMAG_FREQ" or None
    ret: dict = {
        "status": "warning" if imag_code == "IMAG_FREQ" else "ok",
        "label": job_label,
        "energy_eh": energy,
        "enthalpy_eh": enthalpy,
        "gibbs_free_energy_eh": gibbs,
        "product": "gibbs_free_energy_eh",
    }
    if imag_code:
        ret["code"] = imag_code
    return json.dumps(ret)


@mcp.tool()
async def run_spectrum_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    spectrum_type: Literal["ir", "raman", "ir_raman"] = "ir",
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run an ORCA frequency calculation and return IR and/or Raman spectrum data.

    spectrum_type:
      "ir"       – IR spectrum only (always free with FREQ; default)
      "raman"    – IR + Raman spectrum (adds %elprop Polar 1; ~2x cost)
      "ir_raman" – same as "raman"

    Returns E/H/G (same as run_freq_job) plus:
      ir_spectrum:    [{mode, freq_cm1, intensity_km_mol}, ...]
      raman_spectrum: [{mode, freq_cm1, activity, depolarization}, ...]  (if requested)
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    do_raman = spectrum_type in ("raman", "ir_raman")

    if job_label is None:
        job_label = f"spec_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir = jobs_dir / job_label

    calc = _build_calc(
        label=job_label,
        workdir=workdir,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="freq",
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=1,
        nbo=False,
        ncores=ncores,
        raman=do_raman,
    )

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({
            "status": "timeout", "label": job_label,
            "text": f"Status: TIMEOUT after {wall_timeout_seconds}s",
            "ir_spectrum": extract_ir_spectrum(out_text),
        })
    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({"status": "error", "label": job_label, "error": str(e),
                           "code": _classify_orca_error_code(out_text)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({
            "status": "error", "label": job_label,
            "code": _classify_orca_error_code(out_text),
            "tail": "\n".join(out_text.splitlines()[-120:]),
        })

    energy = extract_total_energy(out_text)
    enthalpy = extract_total_enthalpy(out_text)
    gibbs = extract_gibbs_free_energy(out_text)
    ir_spec = extract_ir_spectrum(out_text)
    imag_modes = [e for e in ir_spec
                  if isinstance(e.get("freq_cm1"), (int, float)) and e["freq_cm1"] < -10]

    result = {
        "status": "warning" if imag_modes else "ok",
        "label": job_label,
        "energy_eh": energy,
        "enthalpy_eh": enthalpy,
        "gibbs_free_energy_eh": gibbs,
        "ir_spectrum": ir_spec,
        "spectrum_type": spectrum_type,
        "product": "gibbs_free_energy_eh",
    }
    if imag_modes:
        result["code"] = "IMAG_FREQ"
        result["imaginary_modes"] = imag_modes
    if do_raman:
        result["raman_spectrum"] = extract_raman_spectrum(out_text)

    return json.dumps(result)


@mcp.tool()
async def run_tddft_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    nroots: int = 5,
    use_ri: bool = True,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run an ORCA TD-DFT excited-state calculation on a pre-optimised geometry.

    Computes ground-state DFT energy and the lowest nroots singlet excited states.
    The input geometry must already be optimised (run run_opt_job first).

    Returns a dict with:
        energy_ground_state_eh: float  (ground-state DFT energy in Eh)
        excited_states: list of {state, energy_ev, wavelength_nm, oscillator_strength}

    Oscillator strength (fosc) interpretation:
        fosc >> 0  -> bright (electric-dipole-allowed) transition
        fosc ~  0  -> dark (forbidden) transition
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"tddft_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir = jobs_dir / job_label

    calc = _build_calc(
        label=job_label,
        workdir=workdir,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="sp",
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=1,
        nbo=False,
        ncores=ncores,
    )
    nroots = max(1, int(nroots))
    calc.input.add_arbitrary_string(f"%tddft\n  nroots {nroots}\nend")

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({
            "status": "timeout",
            "label": job_label,
            "excited_states": extract_excited_states(out_text),
            "text": f"Status: TIMEOUT after {wall_timeout_seconds}s",
        })
    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({"status": "error", "label": job_label, "error": str(e),
                           "code": _classify_orca_error_code(out_text)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({
            "status": "error",
            "label": job_label,
            "code": _classify_orca_error_code(out_text),
            "tail": "\n".join(out_text.splitlines()[-120:]),
        })

    energy = extract_total_energy(out_text)
    excited_states = extract_excited_states(out_text)
    gap_ev = extract_homo_lumo_gap(out_text)

    return json.dumps({
        "status": "ok",
        "label": job_label,
        "energy_ground_state_eh": energy,
        "homo_lumo_gap_ev": gap_ev,
        "excited_states": excited_states,
        "product": "excited_states",
        "text": (
            f"Status: OK\nGround state energy: {energy} Eh\n"
            f"HOMO-LUMO gap: {gap_ev} eV\n"
            f"Excited states found: {len(excited_states)}"
        ),
    })


# ─── Rigid surface scan ────────────────────────────────────────────────────────

def _build_scan_block(scan_coords: List[Dict]) -> str:
    """Build ORCA %geom Scan block for a rigid scan.

    scan_coords: [{"type": "B"|"A"|"D",
                   "atoms": [i, j, ...],   # 0-based ORCA indices
                   "start": float,         # Å for B; degrees for A/D
                   "end":   float,
                   "n_points": int}]       # number of calculation points
    """
    lines = ["%geom", "  Scan"]
    for c in scan_coords:
        coord_type = str(c["type"]).upper()
        atoms = " ".join(str(int(a)) for a in c["atoms"])
        start = float(c["start"])
        end   = float(c["end"])
        n     = int(c["n_points"])
        lines.append(f"    {coord_type} {atoms} = {start:.6f}, {end:.6f}, {n}")
    lines += ["  end", "End"]
    return "\n".join(lines)


def _extract_scan_results(out_text: str, scan_coords: List[Dict]) -> List[Dict]:
    """Parse ORCA scan output (relaxed or rigid), returning [{step, value, energy_eh}, ...].

    Stage 1 — summary table: ORCA end-of-scan summary (relaxed or rigid).
    Stage 2 — per-step markers: "RELAXED SURFACE SCAN STEP N" with last energy per step.
    Stage 3 — fallback: all FINAL SINGLE POINT ENERGY lines with computed values.
    """
    lines = out_text.splitlines()
    energy_pat = re.compile(
        r'FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-+]?\d+\.\d+)', re.IGNORECASE
    )

    def _computed_values(n_steps: int) -> List[float]:
        c0 = scan_coords[0] if scan_coords else None
        if c0 and int(c0.get("n_points", 0)) > 1:
            n = int(c0["n_points"])
            return [
                float(c0["start"]) + i * (float(c0["end"]) - float(c0["start"])) / (n - 1)
                for i in range(n_steps)
            ]
        return [float(i) for i in range(n_steps)]

    def _parse_summary_table(start_idx: int) -> List[Dict]:
        """Parse a scan summary table starting at start_idx."""
        # Rows: "  1   1.0000   -115.70900" or "  1  B(0,1): 1.0000  -115.70900"
        pat = re.compile(
            r'^\s+(\d+)\s+(?:\S+:\s*)?([-+]?\d+\.?\d*)\s+([-+]?\d+\.\d+)'
        )
        results = []
        for ln in lines[start_idx + 1 : start_idx + 400]:
            m = pat.match(ln)
            if m:
                results.append({
                    "step":      int(m.group(1)),
                    "value":     round(float(m.group(2)), 6),
                    "energy_eh": round(float(m.group(3)), 8),
                })
            elif results and re.match(r'\s*-{20,}', ln):
                break
        return results

    # Stage 1: find any scan summary table (relaxed or rigid, either keyword order)
    summary_patterns = [
        r'(RELAXED|RIGID)\s+SURFACE\s+SCAN\s+(SUMMARY|RESULTS)',
        r'(SUMMARY|RESULTS)\s+OF\s+(THE\s+)?(RELAXED|RIGID)\s+SURFACE\s+SCAN',
        r'THE\s+(RELAXED|RIGID)\s+SURFACE\s+SCAN\s+RESULTS',
    ]
    for spat in summary_patterns:
        for i, ln in enumerate(lines):
            if re.search(spat, ln, re.IGNORECASE):
                results = _parse_summary_table(i)
                if results:
                    return results

    # Stage 2: per-step parsing via "RELAXED SURFACE SCAN STEP N" markers
    # Each step ends just before the next step header (or EOF).
    step_pat = re.compile(r'RELAXED\s+SURFACE\s+SCAN\s+STEP\s+(\d+)', re.IGNORECASE)
    step_indices: List[tuple] = []  # (line_index, step_number)
    for i, ln in enumerate(lines):
        m = step_pat.search(ln)
        if m:
            step_indices.append((i, int(m.group(1))))

    if step_indices:
        results = []
        for k, (step_line_idx, step_num) in enumerate(step_indices):
            next_idx = step_indices[k + 1][0] if k + 1 < len(step_indices) else len(lines)
            # Collect the LAST FINAL SINGLE POINT ENERGY in this step's range
            step_energies = [
                float(m.group(1))
                for ln in lines[step_line_idx:next_idx]
                for m in [energy_pat.search(ln)] if m
            ]
            if step_energies:
                results.append({"step": step_num, "energy_eh": round(step_energies[-1], 8)})
        if results:
            vals = _computed_values(len(results))
            for k, r in enumerate(results):
                r["value"] = round(vals[k], 6)
            return results

    # Stage 3: fallback — all FINAL SINGLE POINT ENERGY lines with computed values
    energies = [float(m.group(1)) for m in energy_pat.finditer(out_text)]
    if not energies:
        return []
    vals = _computed_values(len(energies))
    return [
        {"step": i + 1, "value": round(vals[i], 6), "energy_eh": round(e, 8)}
        for i, e in enumerate(energies)
    ]


def _extract_scan_geometries(out_text: str, n_steps: int) -> List[str]:
    """Extract per-step optimised geometries from a relaxed scan output.

    Splits the output by "RELAXED SURFACE SCAN STEP N" markers and calls
    extract_final_geometry_from_out on each chunk.
    Returns a list of XYZ strings (atom-lines only, no header), indexed by (step-1).
    Steps with no parseable geometry return an empty string.
    """
    step_pat = re.compile(r'RELAXED\s+SURFACE\s+SCAN\s+STEP\s+(\d+)', re.IGNORECASE)
    lines = out_text.splitlines()

    step_indices: List[tuple] = []  # (line_idx, step_num)
    for i, ln in enumerate(lines):
        m = step_pat.search(ln)
        if m:
            step_indices.append((i, int(m.group(1))))

    if not step_indices:
        return []

    geometries = []
    for k, (start_idx, _step_num) in enumerate(step_indices):
        end_idx = step_indices[k + 1][0] if k + 1 < len(step_indices) else len(lines)
        chunk = "\n".join(lines[start_idx:end_idx])
        geom = extract_final_geometry_from_out(chunk)
        geometries.append(geom)
    return geometries


@mcp.tool()
async def run_scan_job(
    geometry_xyz: str,
    scan_coords: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    wall_timeout_seconds: int = 7200,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run a relaxed ORCA surface scan (geometry optimization at each scan point).

    scan_coords: JSON string with a list of coordinate dicts, e.g.:
        '[{"type":"B","atoms":[0,1],"start":0.8,"end":1.8,"n_points":11}]'
        type: "B" = bond (Å), "A" = angle (degrees), "D" = dihedral (degrees)
        atoms: 0-based ORCA atom indices (2 for B, 3 for A, 4 for D)
        n_points: total number of calculation points (inclusive)

    Returns:
        scan_results: [{step, value, energy_eh}, ...]
        min_energy_eh: float   (lowest energy found)
        min_value:     float   (coordinate value at minimum)
        n_points:      int
        scan_coords:   list    (echo of input coords)
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    try:
        coords = json.loads(scan_coords)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"status": "error", "error": f"scan_coords JSON parse failed: {e}"})
    if not isinstance(coords, list) or not coords:
        return json.dumps({"status": "error", "error": "scan_coords must be a non-empty JSON array"})

    if job_label is None:
        job_label = f"scan_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir  = jobs_dir / job_label

    calc = _build_calc(
        label=job_label,
        workdir=workdir,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="scan",   # OPT keyword activates scan loop; no extra %geom block
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=1,
        nbo=False,
        ncores=ncores,
    )
    calc.input.add_arbitrary_string(_build_scan_block(coords))

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        partial = _extract_scan_results(out_text, coords)
        return json.dumps({
            "status":      "timeout",
            "label":       job_label,
            "scan_results": partial,
            "text":        f"Status: TIMEOUT after {wall_timeout_seconds}s ({len(partial)} steps collected)",
        })
    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({"status": "error", "label": job_label, "error": str(e),
                           "code": _classify_orca_error_code(out_text)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        return json.dumps({
            "status": "error",
            "label":  job_label,
            "code":   _classify_orca_error_code(out_text),
            "tail":   "\n".join(out_text.splitlines()[-120:]),
        })

    scan_results = _extract_scan_results(out_text, coords)

    min_energy = min((p["energy_eh"] for p in scan_results), default=None)
    min_value  = next(
        (p["value"] for p in scan_results if p["energy_eh"] == min_energy), None
    ) if min_energy is not None else None

    max_energy = max((p["energy_eh"] for p in scan_results), default=None)
    max_value  = next(
        (p["value"] for p in scan_results if p["energy_eh"] == max_energy), None
    ) if max_energy is not None else None

    # Extract geometry at the PES maximum (for TS candidate seeding)
    max_geom = ""
    if scan_results and max_energy is not None:
        step_geoms = _extract_scan_geometries(out_text, len(scan_results))
        max_step_idx = next(
            (k for k, r in enumerate(scan_results) if r["energy_eh"] == max_energy), None
        )
        if max_step_idx is not None and max_step_idx < len(step_geoms):
            max_geom = step_geoms[max_step_idx]

    return json.dumps({
        "status":           "ok",
        "label":            job_label,
        "scan_results":     scan_results,
        "n_points":         len(scan_results),
        "min_energy_eh":    min_energy,
        "min_value":        min_value,
        "max_energy_eh":    max_energy,
        "max_value":        max_value,
        "geometry_xyz":     max_geom,       # PES maximum geometry (TS candidate)
        "max_geometry_xyz": max_geom,       # alias
        "scan_coords":      coords,
        "product":          "scan_results",
        "text": (
            f"Status: OK\n{len(scan_results)} scan points\n"
            f"Min energy: {min_energy} Eh at value {min_value}\n"
            f"Max energy: {max_energy} Eh at value {max_value}"
        ),
    })


@mcp.tool()
async def run_ts_opt_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    use_ri: bool = True,
    scf_max_iter: int = 150,
    opt_max_iter: int = 100,
    calc_hess: bool = True,
    wall_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run an ORCA transition-state optimisation (OptTS).

    Requires a starting geometry near the transition state (e.g., the PES maximum
    from run_scan_job). calc_hess=True (default) adds Calc_Hess true to the %geom
    block, which is strongly recommended for reliable TS optimisation.

    Returns:
        status:       "ok" | "not_converged" | "error" | "timeout"
        geometry_xyz: optimised TS geometry (atom lines, no header)
        energy_eh:    final energy in Hartree
        ts_converged: bool — whether ORCA reported a converged optimisation
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")

    if job_label is None:
        job_label = f"tsopt_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir = jobs_dir / job_label

    calc = _build_calc(
        label=job_label,
        workdir=workdir,
        geometry_xyz=geometry_xyz,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job_type="ts_opt",
        use_ri=use_ri,
        scf_max_iter=scf_max_iter,
        opt_max_iter=opt_max_iter,
        nbo=False,
        ncores=ncores,
    )

    # Build a single %geom block with maxiter and optional Calc_Hess
    geom_lines = ["%geom", f"  maxiter {opt_max_iter}"]
    if calc_hess:
        geom_lines.append("  Calc_Hess true")
    geom_lines.append("end")
    calc.input.add_arbitrary_string("\n".join(geom_lines))

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({
            "status":       "timeout",
            "label":        job_label,
            "energy_eh":    extract_total_energy(out_text),
            "geometry_xyz": extract_final_geometry_from_out(out_text),
            "text":         f"Status: TIMEOUT after {wall_timeout_seconds}s",
        })
    except Exception as e:
        out_text = _read_out_text(workdir, job_label)
        return json.dumps({
            "status": "error",
            "label":  job_label,
            "error":  str(e),
            "code":   _classify_orca_error_code(out_text),
            "tail":   "\n".join(out_text.splitlines()[-80:]),
        })

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    ts_converged = "THE OPTIMIZATION HAS CONVERGED" in out_text.upper()
    energy = extract_total_energy(out_text)
    final_xyz = extract_final_geometry_from_out(out_text)

    if not ok:
        return json.dumps({
            "status":       "error",
            "label":        job_label,
            "ts_converged": ts_converged,
            "energy_eh":    energy,
            "geometry_xyz": final_xyz,
            "code":         _classify_orca_error_code(out_text),
            "tail":         "\n".join(out_text.splitlines()[-120:]),
            "text":         "ORCA did not terminate normally.",
        })

    status = "ok" if ts_converged else "not_converged"
    return json.dumps({
        "status":       status,
        "label":        job_label,
        "ts_converged": ts_converged,
        "energy_eh":    energy,
        "geometry_xyz": final_xyz,
        "provenance":   {"geometry": "orca_tsopt"},
        "tail":         "\n".join(out_text.splitlines()[-60:]),
        "text":         f"Status: {status.upper()}\nTS energy: {energy} Eh",
    })


@mcp.tool()
async def run_casscf_job(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    nel: int = 2,
    norb: int = 2,
    nroots: int = 1,
    basis: str = "def2-SVP",
    maxiter: int = 300,
    scf_max_iter: int = 200,
    avas_variant: Optional[str] = None,
    wall_timeout_seconds: int = 3600,
    job_label: Optional[str] = None,
    ncores: int = 1,
) -> str:
    """Run a CASSCF single-point calculation with ORCA.

    Requires active space: nel (active electrons) and norb (active orbitals).
    Set nroots > 1 for state-averaged SA-CASSCF.
    Set avas_variant to activate AVAS orbital selection (ORCA 6 simple-keyword
    interface): "VALENCE-D" (d-block metals), "DOUBLE-D" (3d+4d shells),
    "VALENCE-DS" (d+s), "DOUBLE-DS", "VALENCE-F" (f-block), "DOUBLE-F".
    Returns energy_eh (total or SA-average) and energies_eh list (nroots > 1).
    """
    if not geometry_xyz.strip():
        raise ValueError("geometry_xyz is empty")
    if job_label is None:
        job_label = f"casscf_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    workdir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs")) / job_label

    # In ORCA 6, AVAS is a simple keyword: ! CASSCF basis AVAS(VALENCE-D)
    # No %avas block, no avas_* keywords inside %casscf.
    avas_kw = f" AVAS({avas_variant})" if avas_variant else ""
    method_kw = f"CASSCF{avas_kw}"

    calc = _build_calc(
        label=job_label, workdir=workdir, geometry_xyz=geometry_xyz,
        charge=charge, multiplicity=multiplicity,
        method=method_kw, basis=basis,
        job_type="casscf", use_ri=False,
        scf_max_iter=scf_max_iter, opt_max_iter=1,
        nbo=False, ncores=ncores,
    )

    casscf_lines = [
        f"%casscf",
        f"  nel    {nel}",
        f"  norb   {norb}",
        f"  nroots {nroots}",
        f"  maxiter {maxiter}",
        f"end",
    ]
    calc.input.add_arbitrary_string("\n".join(casscf_lines))

    try:
        output = await _run_calc_with_timeout(calc, wall_timeout_seconds)
    except asyncio.TimeoutError:
        return json.dumps({"status": "timeout", "label": job_label})
    except Exception as e:
        return json.dumps({"status": "error", "label": job_label, "error": str(e)})

    ok = output.terminated_normally()
    out_text = _read_out_text(workdir, job_label)
    if not ok:
        code = _classify_orca_error_code(out_text)
        result: dict = {
            "status": "error",
            "label": job_label,
            "tail": "\n".join(out_text.splitlines()[-120:]),
        }
        if code:
            result["code"] = code
        return json.dumps(result)

    energy = extract_total_energy(out_text)
    root_energies = extract_casscf_root_energies(out_text)
    ret: dict = {
        "status": "ok",
        "label": job_label,
        "energy_eh": energy,
    }
    if root_energies:
        ret["energies_eh"] = root_energies
    return json.dumps(ret)


# Ligand field strength for spin-state heuristic.
# Strong-field ligands stabilise low spin; weak-field ligands favour high spin.
_STRONG_FIELD_LIGS = {
    "cn", "cyanide", "co", "carbonyl", "no",
    "bipy", "phen", "terpy",
    "en", "edta", "cyclam", "cyclen",
    "acac", "acetylacetone",
    "nh3", "ammonia",
    "dppe", "dmpe", "dmf", "dmi",
}
_WEAK_FIELD_LIGS = {
    "cl", "chloride", "br", "bromide", "f", "fluoride", "i", "iodide",
    "water", "h2o", "oh", "hydroxide",
    "ncs", "thiocyanate", "acetate", "formate", "ox", "oxalate",
    "no2", "nitrite", "azide",
}


def _ligand_field(ligands: list) -> str:
    """Return 'strong' or 'weak' based on majority of recognisable ligands."""
    n_strong = sum(1 for l in ligands if l.lower() in _STRONG_FIELD_LIGS)
    n_weak   = sum(1 for l in ligands if l.lower() in _WEAK_FIELD_LIGS)
    return "strong" if n_strong >= n_weak else "weak"


def _auto_spin(metal: str, oxidation_state: str, geometry: str,
               ligands: Optional[list] = None) -> Optional[int]:
    """Return automatic spin multiplicity, or None if the case is ambiguous.

    Deterministic rules (geometry + d-count):
      d10 (Cu+, Zn2+, Ag+, Au+, Cd2+, Hg2+)  → 1
      d9  (Cu2+)                               → 2
      d8  sqp (Ni2+, Pd2+, Pt2+)              → 1  (always diamagnetic)
      d8  thd (Ni2+, Pd2+, Pt2+)              → 3  (triplet, rare)
      d3  oct (Cr3+, Mo3+)                    → 4  (always HS, only t2g3)

    Ligand-field heuristic for octahedral Fe/Co/Mn/Ni (strong vs weak field):
      Fe(III) d5:  strong→2  (t2g5, S=1/2),  weak→6  (t2g3eg2, S=5/2)
      Fe(II)  d6:  strong→1  (t2g6, S=0),    weak→5  (t2g4eg2, S=2)
      Co(III) d6:  strong→1  (t2g6, S=0),    weak→5  (t2g4eg2, S=2)
      Co(II)  d7:  strong→2  (t2g6eg1, S=1/2), weak→4 (t2g5eg2, S=3/2)
      Mn(II)  d5:  strong→2  (S=1/2),        weak→6  (S=5/2)
      Mn(III) d4:  strong→3  (t2g4, S=1),    weak→5  (t2g3eg1, S=2)
      Ni(II)  d8 oct: strong→1 (S=0, rare),  weak→3  (t2g6eg2, S=1)
    """
    m = metal.lower()
    ox_str = str(oxidation_state).strip().upper()
    _roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
    ox = _roman.get(ox_str)
    if ox is None:
        try:
            ox = int(ox_str)
        except ValueError:
            return None

    _group_e = {
        "cr": 6, "mn": 7, "fe": 8, "co": 9, "ni": 10, "cu": 11, "zn": 12,
        "mo": 6, "tc": 7, "ru": 8, "rh": 9, "pd": 10, "ag": 11, "cd": 12,
        "w":  6, "re": 7, "os": 8, "ir": 9, "pt": 10, "au": 11, "hg": 12,
    }
    if m not in _group_e:
        return None
    d = _group_e[m] - ox
    g = geometry.lower()
    ligs = ligands or []

    # ── Deterministic rules ──────────────────────────────────────────────────
    if d == 10:  return 1                   # d10: always S=0
    if d == 9:   return 2                   # d9:  always S=1/2
    if d == 8:
        if g == "sqp":  return 1            # d8 square planar: always S=0
        if g == "thd":  return 3            # d8 tetrahedral:   S=1 (triplet)
    if d == 3 and g == "oct":  return 4     # d3 oct: always t2g3, S=3/2

    # ── Ligand-field heuristic for oct Fe/Co/Mn/Ni ──────────────────────────
    if g == "oct":
        field = _ligand_field(ligs)
        strong = (field == "strong")
        if d == 5:   return 2 if strong else 6   # Fe(III), Mn(II)
        if d == 6:   return 1 if strong else 5   # Fe(II), Co(III)
        if d == 7:   return 2 if strong else 4   # Co(II)
        if d == 4:   return 3 if strong else 5   # Mn(III), Cr(II)
        if d == 8:   return 1 if strong else 3   # Ni(II) oct

    return None  # unknown or ambiguous — caller must require explicit spin


@mcp.tool()
async def build_coordination_complex(
    metal: str,
    ligands: list,
    geometry: str = "oct",
    oxidation_state: str = "II",
    spin: Optional[int] = None,
    multiplicity: Optional[int] = None,
    charge: int = 0,
    force_field: str = "uff",
    job_label: Optional[str] = None,
) -> str:
    """Build a 3D coordination complex geometry using molSimplify.

    Args:
        metal:           Element symbol lowercase: "fe", "co", "ni", "cu", "pt", etc.
        ligands:         List of molSimplify ligand names, one entry per coordination site.
                         Common: "cl", "water", "nh3", "co", "cn", "en", "bipy",
                         "acac", "acetate", "ox", "ncs".
                         Bidentate ligands (en, bipy, acac…) count as 2 sites each.
                         Coordination number is derived automatically.
        geometry:        "oct" (octahedral) | "sqp" (square planar) |
                         "tbp" (trigonal bipyramidal) | "thd" (tetrahedral)
        oxidation_state: Roman numeral string: "II", "III", "IV", etc.
        spin:            Spin multiplicity 2S+1. Auto-determined if omitted:
                           Cu(I)/Zn(II)/d10  → 1
                           Cu(II)/d9         → 2
                           Ni/Pd/Pt d8 sqp   → 1  (diamagnetic square planar)
                           Ni/Pd/Pt d8 thd   → 3  (triplet tetrahedral)
                           Fe/Co/Cr/Mn       → ANN prediction
                           Others            → error (must be specified)
        charge:          Total complex charge (integer).
        force_field:     "uff" (default) | "mmff94" | "n" (skip FF).
        job_label:       Optional label for this structure.

    Returns:
        JSON with status, geometry_xyz (no-header XYZ), n_atoms, charge, multiplicity.
    """
    try:
        from molSimplify.Scripts.generator import startgen_pythonic
    except ImportError as e:
        return json.dumps({"status": "error", "error": f"molSimplify not available: {e}"})

    if not metal:
        return json.dumps({"status": "error", "error": "metal must be specified"})
    if not ligands:
        return json.dumps({"status": "error", "error": "ligands list must not be empty"})

    # Resolve spin: explicit > ligand-field heuristic > error
    explicit_spin = multiplicity if multiplicity is not None else spin
    if explicit_spin is not None:
        effective_spin = int(explicit_spin)
        spin_source = "explicit"
    else:
        auto = _auto_spin(metal, oxidation_state, geometry, ligands)
        if auto is not None:
            effective_spin = auto
            spin_source = "auto"
        else:
            return json.dumps({
                "status": "error",
                "error": (f"spin not specified and no auto-rule for "
                          f"{metal}/{oxidation_state}/{geometry}. "
                          f"Please provide spin explicitly.")
            })

    lig_str = ",".join(str(l) for l in ligands)
    ligocc_str = ",".join("1" for _ in ligands)

    # Derive effective_coord by summing denticities from ligands.dict.
    # Format: "name:struct_file,abbrev,conn_atoms,groups,ff,charge"
    # Split on first ":" only, then split the value by "," — conn_atoms is index 2.
    def _get_denticity(lig_name: str) -> int:
        try:
            import os as _os
            import molSimplify as _ms
            db = _os.path.join(_os.path.dirname(_ms.__file__), "Ligands", "ligands.dict")
            with open(db) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(":", 1)
                    if len(parts) < 2:
                        continue
                    if parts[0].strip().lower() == lig_name.lower():
                        subfields = parts[1].split(",")
                        if len(subfields) < 3:
                            continue
                        conn = subfields[2].strip()   # e.g. "0 1" for bidentate
                        return sum(1 for t in conn.split() if t.isdigit())
        except Exception:
            pass
        return 1  # default monodentate

    effective_coord = sum(_get_denticity(l) for l in ligands)

    input_dict = {
        "-core":     metal.lower(),
        "-lig":      lig_str,
        "-ligocc":   ligocc_str,
        "-coord":    str(effective_coord),
        "-geometry": geometry.lower(),
        "-oxstate":  str(oxidation_state),
        "-spin":     str(effective_spin),
        "-charge":   str(charge),
        "-ff":       force_field,
        "-ffoption": "ba" if force_field != "n" else "n",
        "-skipANN":  "True",
    }
    if job_label:
        input_dict["-name"] = sanitize_label(job_label)

    try:
        import io, contextlib
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            strfiles, emsg, diag = startgen_pythonic(input_dict=input_dict, write=False)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"molSimplify exception: {e}"})

    if emsg:
        return json.dumps({"status": "error", "error": f"molSimplify error: {emsg}"})

    mol = diag.mol
    if mol is None or mol.natoms == 0:
        return json.dumps({"status": "error", "error": "molSimplify returned empty geometry"})

    # coords() returns "N\n\nxyz_block" — strip the natoms/comment header
    full_xyz = mol.coords()
    lines = full_xyz.strip().splitlines()
    body_lines = [l for l in lines if l.strip() and not l.strip().lstrip('-').isdigit()]
    geometry_xyz = "\n".join(body_lines)

    return json.dumps({
        "status":       "ok",
        "geometry_xyz": geometry_xyz,
        "n_atoms":      mol.natoms,
        "charge":       charge,
        "multiplicity": effective_spin,
        "metal":        metal,
        "geometry":     geometry,
        "spin_source":  spin_source,
        "text": (f"Status: OK\n"
                 f"Complex: {metal.capitalize()} {geometry} CN={effective_coord}\n"
                 f"Ligands: {lig_str}\n"
                 f"Atoms: {mol.natoms}  Charge: {charge}  Mult: {effective_spin} ({spin_source})"),
    })


@mcp.tool()
async def run_solvator_cluster(
    geometry_xyz: str,
    charge: int = 0,
    multiplicity: int = 1,
    nsolv: int = 3,
    wall_timeout_seconds: int = 600,
    job_label: Optional[str] = None,
) -> str:
    """
    Build a small explicit-solvent cluster with SOLVATOR (water) around the given solute.

    - Uses XTB + ALPB(WATER)
    - Uses SOLVATOR with nsolv water molecules, docking mode, fixed solute
    - Returns: cluster geometry (XYZ, no header) + status text
    """

    if job_label is None:
        job_label = f"solv_{os.getpid()}_{int(asyncio.get_event_loop().time())}"
    job_label = sanitize_label(job_label)
    # Run the sync helper in a thread; optionally you could add your own timeout logic here
    result = await asyncio.to_thread(
        _run_solvator_sync,
        geometry_xyz,
        charge,
        multiplicity,
        nsolv,
        job_label,
    )

    return json.dumps(result)



@mcp.tool()
async def inspect_job(
    job_label: str,
    tail_lines: int = 200,
) -> str:
    """Return a compact debug bundle for an existing job directory."""
    jobs_dir = Path(os.environ.get("ORCA_JOBS_DIR", "jobs"))
    workdir = jobs_dir / job_label
    if not workdir.exists():
        return json.dumps({"status": "error", "label": job_label, "error": "workdir not found"})
    dbg = _collect_debug(workdir, job_label, tail_lines=tail_lines)
    return json.dumps({"status": "ok", "label": job_label, "debug": dbg})

@mcp.tool()
async def structure_add_remove_proton(
    geometry_xyz: str,
    mode: Literal["add", "remove"],
    charge: int = 0,
    multiplicity: int = 1,
    site_selector: Optional[str] = None,
    variant: int = 0,
    h_index: Optional[int] = None,
    target_atom_index: Optional[int] = None,
    geometry_name: Optional[str] = None,
    strategy: Literal["auto", "distance"] = "auto",
) -> str:
    res = structure_proton_edit(
        xyz=geometry_xyz,
        mode=mode,
        charge=charge,
        multiplicity=multiplicity,
        site_selector=site_selector,
        variant=variant,
        h_index=h_index,
        target_atom_index=target_atom_index,
        geometry_name=geometry_name,
        strategy=strategy,
    )
    return json.dumps(res)


sys.stdout = _real_stdout  # restore stdout for MCP stdio transport
mcp.run(transport="stdio")