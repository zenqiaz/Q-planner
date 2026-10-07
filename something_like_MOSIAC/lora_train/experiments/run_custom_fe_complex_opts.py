"""
run_custom_fe_complex_opts.py

Calculation step for the custom Fe(II) ligand-field-series boundary experiment
(decision_boundary_experiment_design.md Sec. 8). Takes the 10 locally-built starting
geometries (build_custom_fe_complexes_local.py's output, custom_complexes/*_local.xyz) and
runs a real DFT geometry optimization for each, under BOTH functionals proposed for the
metal_general specialist pair (B3LYP/def2-TZVP, TPSSH/def2-TZVP) -- 10 x 2 = 20 optimization
jobs total, matching the design doc's own cost estimate (20 opts for a first pass, without
frequency jobs yet).

Reuses the exact same MCP-over-SSH-to-qcl pattern already used throughout this project's
accuracy-benchmark scripts (session.call_tool("run_opt_job", ...) via a single shared
ClientSession) -- NOT the broken build_coordination_complex path.

Usage:
    cd D:\\brick\\D\\20260217\\working
    python something_like_MOSIAC\\lora_train\\experiments\\run_custom_fe_complex_opts.py
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(os.environ.get("ENV_FILE", os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

COMPLEX_DIR = Path(__file__).parent / "custom_complexes"
LOG_DIR = Path(__file__).parent / "custom_complex_logs"
LOG_DIR.mkdir(exist_ok=True)

FUNCTIONALS = [("B3LYP", "def2-TZVP"), ("TPSSH", "def2-TZVP")]
CONCURRENCY = 4  # conservative given this project's own documented qcl session-hang history
WALL_TIMEOUT = 1800


def _build_server_params() -> StdioServerParameters:
    _ssh_bin = os.getenv("MCP_SSH_BIN", "ssh")
    _ssh_key = os.getenv("MCP_SSH_KEY")
    _ssh_host = os.getenv("MCP_SSH_HOST")
    _ssh_cmd = os.getenv("MCP_SERVER_CMD")
    if not (_ssh_key and _ssh_host and _ssh_cmd):
        raise RuntimeError("MCP_SSH_KEY / MCP_SSH_HOST / MCP_SERVER_CMD must be set in .env")
    return StdioServerParameters(
        command=_ssh_bin,
        args=["-i", _ssh_key, "-o", "StrictHostKeyChecking=no",
              "-o", "BatchMode=yes", _ssh_host, _ssh_cmd],
        env=dict(os.environ),
    )


def _load_geometries() -> list[dict]:
    manifest = json.loads((COMPLEX_DIR / "manifest_local.json").read_text(encoding="utf-8"))
    out = []
    for entry in manifest:
        xyz_text = Path(entry["xyz_path"]).read_text(encoding="utf-8")
        lines = xyz_text.strip().splitlines()
        n = int(lines[0])
        geometry_xyz = "\n".join(lines[2:2 + n])  # strip the 2-line XYZ header
        out.append({**entry, "geometry_xyz": geometry_xyz})
    return out


async def run_one(session: ClientSession, sem: asyncio.Semaphore, entry: dict,
                   functional: str, basis: str) -> dict:
    job_label = f"{entry['label']}_{functional}"
    async with sem:
        t0 = time.monotonic()
        try:
            res = await session.call_tool("run_opt_job", {
                "geometry_xyz": entry["geometry_xyz"],
                "charge": entry["charge"],
                "multiplicity": entry["multiplicity"],
                "method": functional,
                "basis": basis,
                "job_label": job_label,
                "wall_timeout_seconds": WALL_TIMEOUT,
            })
            text = res.content[0].text if res.content else "{}"
            payload = json.loads(text)
        except Exception as e:  # noqa: BLE001 -- record and continue, don't abort the whole batch
            payload = {"status": "error", "error": f"client_exception: {e!r}"}
    elapsed = round(time.monotonic() - t0, 1)
    result = {
        "job_label": job_label, "ligand": entry["ligand"], "field": entry["field"],
        "multiplicity": entry["multiplicity"], "spin_label": entry["spin_label"],
        "functional": functional, "basis": basis, "elapsed_s": elapsed,
        "status": payload.get("status"),
    }
    if payload.get("status") in ("ok", "not_converged"):
        result["opt_converged"] = payload.get("opt_converged")
        result["energy_eh"] = payload.get("energy_eh")
        result["optimized_geometry_xyz"] = payload.get("geometry_xyz")
        tag = "ok" if payload.get("status") == "ok" else "NOT_CONV"
        print(f"  [{tag:8s}] {job_label:30s} E={result.get('energy_eh')}  ({elapsed}s)")
    else:
        result["error"] = payload.get("error") or payload.get("error_summary") or "unknown error"
        print(f"  [ERROR] {job_label:30s} {str(result['error'])[:100]}  ({elapsed}s)")
    return result


async def main() -> None:
    t0 = time.monotonic()
    geometries = _load_geometries()
    print(f"Loaded {len(geometries)} starting geometries. Running {len(geometries)} x "
          f"{len(FUNCTIONALS)} = {len(geometries) * len(FUNCTIONALS)} optimization jobs "
          f"(concurrency={CONCURRENCY}) ...\n")

    server_params = _build_server_params()
    results = []
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            sem = asyncio.Semaphore(CONCURRENCY)
            tasks = [
                run_one(session, sem, entry, functional, basis)
                for entry in geometries
                for functional, basis in FUNCTIONALS
            ]
            results = await asyncio.gather(*tasks)

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = LOG_DIR / f"custom_fe_complex_opts_{ts}.json"
    log_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    wall_s = round(time.monotonic() - t0, 1)
    print(f"\n{n_ok}/{len(results)} jobs OK in {wall_s}s. Log: {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
