"""
build_custom_fe_complex_inps.py

Builds 20 ORCA .inp files (10 local Fe complex geometries x 2 functionals: B3LYP/def2-TZVP,
TPSSH/def2-TZVP, OPT job type) for RCCS batch submission -- pivoted here after qcl's
kubectl/kubelet path hit a live cluster-networking outage 2026-09-15
("connect: connection refused" to the kubelet API, confirmed via `kubectl get pod`/`ps aux`
diagnostics -- the pod itself is healthy, the outage is at the cluster-networking layer).
RCCS is a separate, plain PBS cluster with no Kubernetes dependency at all, so this bypasses
the outage entirely. Same .inp format as write_orca_inp() in run_accuracy_benchmark.py
(RCCS_MAXCORE_MB=1700), OPT instead of SP.
"""
import json
from pathlib import Path

COMPLEX_DIR = Path(__file__).parent / "custom_complexes"
BATCH_DIR = Path(__file__).parent / "rccs_batch_custom_fe"
BATCH_DIR.mkdir(exist_ok=True)

FUNCTIONALS = [("B3LYP", "def2-TZVP"), ("TPSSH", "def2-TZVP")]
MAXCORE_MB = 1700

# Implicit solvation is REQUIRED here, not optional polish (added 2026-09-16 after a first
# gas-phase batch, PBS 1525057, was cancelled mid-run). Three of the five ligands give a -4
# complex ([FeI6]4-, [FeCl6]4-, [Fe(CN)6]4-), and a tetraanion of this size is not
# electronically bound in the gas phase -- the excess charge autodetaches in reality, and a
# finite basis set instead produces an artifact. Confirmed directly in that cancelled batch's
# own output: [FeCl6]4- HS/B3LYP had SEVEN OCCUPIED orbitals at POSITIVE energy (+0.220 to
# +0.273 Eh, i.e. +5.99 to +7.42 eV), while the +2 complex [Fe(H2O)6]2+ was textbook-clean
# (HOMO -0.597 Eh, LUMO -0.308 Eh). CPCM(water) stabilises the excess charge and is also the
# physically appropriate regime -- these species exist in solution, not isolated. Applied
# uniformly to ALL five ligands (not just the anions) so every point in the ligand-field
# series is treated identically and remains comparable.
SOLVENT_KEYWORD = "CPCM(water)"


def main() -> None:
    manifest = json.loads((COMPLEX_DIR / "manifest_local.json").read_text(encoding="utf-8"))
    written = []
    for entry in manifest:
        xyz_text = Path(entry["xyz_path"]).read_text(encoding="utf-8")
        lines = xyz_text.strip().splitlines()
        n = int(lines[0])
        geometry_xyz = "\n".join(lines[2:2 + n])
        for functional, basis in FUNCTIONALS:
            job_name = f"{entry['label']}_{functional}"
            inp = (
                f"! {functional} {basis} OPT RIJCOSX {SOLVENT_KEYWORD}\n"
                f"%maxcore {MAXCORE_MB}\n\n"
                f"* xyz {entry['charge']} {entry['multiplicity']}\n"
                f"{geometry_xyz}\n"
                f"*\n"
            )
            inp_path = BATCH_DIR / f"{job_name}.inp"
            inp_path.write_text(inp, encoding="utf-8")
            written.append({**entry, "functional": functional, "basis": basis,
                             "job_name": job_name, "inp_path": str(inp_path)})
            print(f"  wrote {inp_path.name}")

    (BATCH_DIR / "job_manifest.json").write_text(json.dumps(written, indent=2), encoding="utf-8")
    print(f"\n{len(written)} .inp files written to {BATCH_DIR}")


if __name__ == "__main__":
    main()
