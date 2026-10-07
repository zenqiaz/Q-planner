"""
build_custom_fe_complexes_local.py

Local, no-remote-server workaround for the custom Fe(II) ligand-field-series boundary
experiment (decision_boundary_experiment_design.md Sec. 8) after `build_coordination_complex`
(molSimplify on the zhang-ollama pod) was found broken 2026-09-14 -- see memory
`project_build_coordination_complex_broken_2026-09.md`. Builds idealized-Oh-symmetry starting
geometries directly with numpy vector math, no molSimplify/RDKit-embedding/remote-server
dependency. These are DELIBERATELY approximate starting guesses, not literature-verified
structures -- every geometry is meant to be passed through `run_opt_job` (a real DFT
optimization) before any energy is extracted; nothing here is a final answer.

Bond-length choice: Shannon ionic radii for Fe2+ are spin-state-dependent (6-coordinate:
HS 0.780 A, LS 0.610 A -- Shannon, Acta Cryst. A32, 1976), so the LS structure for a given
ligand is built with a shorter Fe-L distance than the HS structure for the SAME ligand,
reflecting the real physical spin-crossover bond-length contraction. Ligand-side radii/bond
geometry are rounded, standard textbook values (ionic radius for anionic donors, covalent
geometry for neutral H2O/NH3), not re-derived or literature-pinned per ligand -- precision here
does not matter since DFT optimization corrects the geometry regardless; only "reasonable,
clash-free starting point" matters.

Usage:
    cd D:\\brick\\D\\20260217\\working\\something_like_MOSIAC\\lora_train\\experiments
    python build_custom_fe_complexes_local.py
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent / "custom_complexes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fe2+ Shannon ionic radius (6-coordinate), spin-state dependent
FE_RADIUS = {1: 0.61, 5: 0.78}  # multiplicity -> radius (Angstrom); 1=LS, 5=HS

# donor-atom ionic/covalent radius (rough, textbook values) -- combined with FE_RADIUS to
# get an approximate Fe-L distance per spin state
LIGAND_DONOR_RADIUS = {
    "i":     2.06,   # I- ionic radius
    "cl":    1.67,   # Cl- ionic radius
    "water": 1.34,   # O, neutral donor, rough
    "nh3":   1.40,   # N, neutral donor, rough
    "cn":    1.30,   # C (cyanide is C-bound), rough
}
LIGAND_CHARGE = {"i": -1, "cl": -1, "water": 0, "nh3": 0, "cn": -1}
LIGAND_FIELD = {"i": "weak", "cl": "weak", "water": "weak", "nh3": "strong", "cn": "strong"}
METAL_CHARGE = 2

# Octahedral vertex directions
AXES = [
    np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
    np.array([0, 1.0, 0]), np.array([0, -1.0, 0]),
    np.array([0, 0, 1.0]), np.array([0, 0, -1.0]),
]


def _water_atoms(donor_pos: np.ndarray, radial_dir: np.ndarray, twist_deg: float) -> list[tuple[str, np.ndarray]]:
    """O at donor_pos; 2 H at standard bent geometry (104.5 deg, O-H 0.96 A), H-O-H bisector
    pointing away from the metal (along radial_dir), twisted azimuthally to reduce clashes."""
    oh = 0.96
    half_angle = np.radians(104.5 / 2)
    # build an orthonormal frame around radial_dir
    ref = np.array([0.0, 0, 1]) if abs(radial_dir[2]) < 0.9 else np.array([1.0, 0, 0])
    perp1 = np.cross(radial_dir, ref); perp1 /= np.linalg.norm(perp1)
    perp1 = perp1 * np.cos(np.radians(twist_deg)) + np.cross(radial_dir, perp1) * np.sin(np.radians(twist_deg))
    h1_dir = radial_dir * np.cos(half_angle) + perp1 * np.sin(half_angle)
    h2_dir = radial_dir * np.cos(half_angle) - perp1 * np.sin(half_angle)
    return [
        ("H", donor_pos + h1_dir * oh),
        ("H", donor_pos + h2_dir * oh),
    ]


def _nh3_atoms(donor_pos: np.ndarray, radial_dir: np.ndarray, twist_deg: float) -> list[tuple[str, np.ndarray]]:
    """N at donor_pos; 3 H in a pyramidal tripod (107 deg from the N-Fe axis, N-H 1.01 A),
    pointing away from the metal, staggered azimuthally at 120-degree intervals."""
    nh = 1.01
    cone_angle = np.radians(180 - 107)  # angle from radial_dir to each N-H bond
    ref = np.array([0.0, 0, 1]) if abs(radial_dir[2]) < 0.9 else np.array([1.0, 0, 0])
    perp1 = np.cross(radial_dir, ref); perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(radial_dir, perp1)
    atoms = []
    for k in range(3):
        ang = np.radians(twist_deg + 120 * k)
        tang = perp1 * np.cos(ang) + perp2 * np.sin(ang)
        h_dir = radial_dir * np.cos(cone_angle) + tang * np.sin(cone_angle)
        atoms.append(("H", donor_pos + h_dir * nh))
    return atoms


def build_complex(ligand: str, mult: int) -> dict:
    fe_r = FE_RADIUS[mult]
    bond = fe_r + LIGAND_DONOR_RADIUS[ligand]
    atoms: list[tuple[str, np.ndarray]] = [("Fe", np.zeros(3))]

    for i, axis in enumerate(AXES):
        donor_pos = axis * bond
        twist = 30.0 * i  # vary twist per site so H's don't perfectly eclipse between ligands
        if ligand == "i":
            atoms.append(("I", donor_pos))
        elif ligand == "cl":
            atoms.append(("Cl", donor_pos))
        elif ligand == "cn":
            atoms.append(("C", donor_pos))
            atoms.append(("N", donor_pos + axis * 1.16))  # C#N bond length, pointing further out
        elif ligand == "water":
            atoms.append(("O", donor_pos))
            atoms.extend(_water_atoms(donor_pos, axis, twist))
        elif ligand == "nh3":
            atoms.append(("N", donor_pos))
            atoms.extend(_nh3_atoms(donor_pos, axis, twist))
        else:
            raise ValueError(f"unknown ligand {ligand}")

    total_charge = METAL_CHARGE + 6 * LIGAND_CHARGE[ligand]
    xyz_lines = [f"{el} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for el, p in atoms]
    return {
        "n_atoms": len(atoms),
        "charge": total_charge,
        "multiplicity": mult,
        "geometry_xyz": "\n".join(xyz_lines),
        "field": LIGAND_FIELD[ligand],
        "fe_radius": fe_r,
        "bond_length_estimate": round(bond, 3),
    }


def main() -> None:
    manifest = []
    for ligand in LIGAND_DONOR_RADIUS:
        for mult, spin_label in ((1, "LS singlet"), (5, "HS quintet")):
            result = build_complex(ligand, mult)
            label = f"fe_{ligand}6_mult{mult}_local"
            xyz_path = OUT_DIR / f"{label}.xyz"
            header = (f"{result['n_atoms']}\n{label} charge={result['charge']} mult={mult} "
                      f"({result['field']} field, {spin_label}, Fe-L~{result['bond_length_estimate']}A "
                      f"APPROXIMATE STARTING GUESS -- needs run_opt_job before use)\n")
            xyz_path.write_text(header + result["geometry_xyz"], encoding="utf-8")
            entry = {
                "label": label, "ligand": ligand, "field": result["field"],
                "multiplicity": mult, "spin_label": spin_label,
                "charge": result["charge"], "n_atoms": result["n_atoms"],
                "bond_length_estimate_A": result["bond_length_estimate"],
                "xyz_path": str(xyz_path),
                "provenance": "local idealized-Oh template (numpy), NOT molSimplify -- "
                              "approximate starting guess only, needs DFT re-optimization",
            }
            manifest.append(entry)
            print(f"  [ok] {label:24s} n_atoms={result['n_atoms']}  charge={result['charge']:+d}  "
                  f"Fe-L~{result['bond_length_estimate']}A")

    manifest_path = OUT_DIR / "manifest_local.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)}/10 complexes built locally (no remote server, no compute spent). "
          f"Manifest: {manifest_path}")
    print("REMINDER: these are approximate idealized starting geometries, not final structures. "
          "Every one needs a real DFT geometry optimization (run_opt_job) before any energy "
          "comparison is meaningful.")


if __name__ == "__main__":
    main()
