"""
parse_sp_property_labels.py

Extends rccs_collect_results.py's parsing: pulls additional per-job property LABELS out of the
same real ORCA .out files already retrieved from RCCS for the accuracy-benchmark SP jobs, beyond
just the final energy -- HOMO/LUMO orbital energies (and gap), dipole moment magnitude, and
open-shell spin contamination (<S**2>). Motivated directly by the paper outline's Chapter 7.7
"property-sensitivity-linked, not just choice-linked" data argument: the corpus currently only
records WHICH method was chosen, never any downstream property computed with it. These files
already sit locally with this data in them -- no new RCCS submission needed, just parsing.

For closed-shell (RHF/RKS) jobs: one ORBITAL ENERGIES block, HOMO = last row with OCC>0, LUMO =
first row with OCC==0 (ORCA always prints at least the first virtual orbital).
For open-shell (UHF/UKS) jobs: two blocks (SPIN UP / SPIN DOWN ORBITALS) -- HOMO/LUMO are taken
across both spin channels (max occupied energy / min virtual energy overall), which spin channel
noted for transparency.

Usage:
    python parse_sp_property_labels.py --dir rccs_batch_export_v2/organic_general
    python parse_sp_property_labels.py --dir rccs_batch_export_metal_full/metal_general
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ORBITAL_BLOCK_RE = re.compile(
    r"ORBITAL ENERGIES\n-+\n(.*?)(?=\n\n\S|\Z)", re.DOTALL
)
_SPIN_SECTION_RE = re.compile(
    r"SPIN (UP|DOWN) ORBITALS\n\s*NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s*\n((?:\s*\d+\s+[\d.]+\s+-?[\d.]+\s+-?[\d.]+\s*\n)+)"
)
_PLAIN_ORBITAL_RE = re.compile(
    r"NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s*\n((?:\s*\d+\s+[\d.]+\s+-?[\d.]+\s+-?[\d.]+\s*\n)+)"
)
_ORBITAL_ROW_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$", re.MULTILINE)

_DIPOLE_RE = re.compile(r"Magnitude \(Debye\)\s*:\s*([\d.]+)")
_S2_RE = re.compile(r"Expectation value of <S\*\*2>\s*:\s*([\d.]+)")


def _homo_lumo_from_rows(rows: list[tuple[int, float, float, float]]) -> tuple[float | None, float | None]:
    """rows: list of (no, occ, e_eh, e_ev). Returns (homo_eh, lumo_eh)."""
    occupied = [r for r in rows if r[1] > 0]
    virtual = [r for r in rows if r[1] == 0]
    homo = max(occupied, key=lambda r: r[2])[2] if occupied else None
    lumo = min(virtual, key=lambda r: r[2])[2] if virtual else None
    return homo, lumo


def parse_orbitals(text: str) -> dict:
    """Returns {'homo_eh', 'lumo_eh', 'gap_eh', 'gap_ev', 'open_shell'} or {} if not found."""
    block_match = _ORBITAL_BLOCK_RE.search(text)
    if not block_match:
        return {}
    block = block_match.group(1)

    spin_sections = _SPIN_SECTION_RE.findall(block)
    if spin_sections:
        all_rows: list[tuple[int, float, float, float]] = []
        for _spin, rows_text in spin_sections:
            for m in _ORBITAL_ROW_RE.finditer(rows_text):
                all_rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
        homo_eh, lumo_eh = _homo_lumo_from_rows(all_rows)
        open_shell = True
    else:
        plain_match = _PLAIN_ORBITAL_RE.search(block)
        if not plain_match:
            return {}
        rows = [
            (int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
            for m in _ORBITAL_ROW_RE.finditer(plain_match.group(1))
        ]
        homo_eh, lumo_eh = _homo_lumo_from_rows(rows)
        open_shell = False

    if homo_eh is None or lumo_eh is None:
        return {"open_shell": open_shell}
    gap_eh = lumo_eh - homo_eh
    return {
        "homo_eh": homo_eh, "lumo_eh": lumo_eh,
        "gap_eh": gap_eh, "gap_ev": gap_eh * 27.211386245988,
        "open_shell": open_shell,
    }


def parse_dipole(text: str) -> float | None:
    m = _DIPOLE_RE.search(text)
    return float(m.group(1)) if m else None


def parse_s2(text: str) -> float | None:
    m = _S2_RE.search(text)
    return float(m.group(1)) if m else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, action="append", required=True,
                   help="One per-cell batch dir (containing _meta.json + .out files, as "
                        "produced by run_accuracy_benchmark.py --export-rccs-batch + retrieval). "
                        "Repeat for multiple cells. Writes property_labels.json inside each dir.")
    return p.parse_args()


def process_dir(batch_dir: Path) -> dict[str, dict]:
    meta_path = batch_dir / "_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found -- not a --export-rccs-batch dir")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    out: dict[str, dict] = {}
    n_ok = n_homo = n_dipole = n_s2 = 0
    for node_id, m in meta.items():
        out_path = batch_dir / f"{node_id}.out"
        if not out_path.exists():
            continue
        text = out_path.read_text(encoding="utf-8", errors="replace")
        if "TERMINATED NORMALLY" not in text:
            continue
        n_ok += 1
        orbitals = parse_orbitals(text)
        dipole = parse_dipole(text)
        s2 = parse_s2(text)
        if "homo_eh" in orbitals:
            n_homo += 1
        if dipole is not None:
            n_dipole += 1
        if s2 is not None:
            n_s2 += 1
        out[node_id] = {
            **m,
            "dipole_debye": dipole,
            "s2_expectation": s2,
            **orbitals,
        }
    print(f"  {batch_dir}: {len(meta)} jobs in _meta.json, {n_ok} TERMINATED NORMALLY, "
          f"{n_homo} with HOMO/LUMO parsed, {n_dipole} with dipole, {n_s2} with <S**2> "
          f"(open-shell only)")
    return out


def main() -> None:
    args = parse_args()
    for batch_dir in args.dir:
        labels = process_dir(batch_dir)
        out_path = batch_dir / "property_labels.json"
        out_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
