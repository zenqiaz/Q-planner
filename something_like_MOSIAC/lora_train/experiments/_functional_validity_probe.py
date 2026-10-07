"""Throwaway probe: test which of the pool's distinct functional keywords ORCA actually
recognizes. Submits a trivial H-atom SP job per functional (def2-SVP, fixed) and records
status. Not part of the benchmark pipeline -- one-off data collection for a design question."""
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(r"D:\brick\D\20260217\working")
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(str(REPO_ROOT / ".env"))

_PAGEANT_PIPE = (
    "\\\\.\\pipe\\pageant.zrqrc."
    "14894499a8910a6f5b16b5130c181a0118bd7013f313045b54aa55b182f05ad4"
)
if os.name == "nt" and "SSH_AUTH_SOCK" not in os.environ:
    os.environ["SSH_AUTH_SOCK"] = _PAGEANT_PIPE

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


def _build_mcp_server_params() -> StdioServerParameters:
    ssh_bin = os.getenv("MCP_SSH_BIN", "ssh")
    ssh_key = os.getenv("MCP_SSH_KEY", "")
    ssh_host = os.getenv("MCP_SSH_HOST", "")
    ssh_cmd = os.getenv("MCP_SERVER_CMD", "")
    return StdioServerParameters(
        command=ssh_bin,
        args=["-i", ssh_key, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", ssh_host, ssh_cmd],
        env=dict(os.environ),
    )


FUNCTIONALS = [
    "TPSSH", "PBE0", "PM6", "B3LYP", "QCISD", "WB97X-D3", "BP86", "PBE", "BLYP", "MP2",
    "DLPNO-CCSD(T)", "BEPBE", "TPSS", "CCSD", "M06-L", "HF", "B3PW91", "MPW1PW91", "RHF",
    "CAM-B3LYP", "M06", "B1B95", "B2PLYP", "CCSD(T)", "LC-WPBE", "RI-MP2", "M06-2X", "M062X",
    "MP4", "BE1PBE", "CISD", "PBEH-3C", "AM1", "HSE06", "ROHF", "WB97X-D", "MP3", "B97-D",
]

H_XYZ = "H 0.0 0.0 0.0"


async def probe_one(session, sem, functional):
    async with sem:
        node_id = f"probe_{functional}".replace("(", "_").replace(")", "_").replace("*", "_")
        try:
            res = await session.call_tool("run_sp_energy", {
                "geometry_xyz": H_XYZ, "charge": 0, "multiplicity": 2,
                "method": functional, "basis": "def2-SVP",
                "job_label": node_id, "wall_timeout_seconds": 60,
            })
            text = res.content[0].text if res.content else "{}"
            payload = json.loads(text)
        except Exception as e:  # noqa: BLE001
            payload = {"status": "client_error", "error": repr(e)}
        return functional, payload


async def main():
    server_params = _build_mcp_server_params()
    sem = asyncio.Semaphore(8)
    results = {}
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            out = await asyncio.gather(*[probe_one(session, sem, f) for f in FUNCTIONALS])
            for f, payload in out:
                results[f] = payload

    valid, invalid = [], []
    for f in FUNCTIONALS:
        p = results[f]
        status = p.get("status")
        if status == "ok":
            valid.append(f)
        else:
            reason = p.get("error") or p.get("code") or (p.get("tail", "")[-200:] if p.get("tail") else status)
            invalid.append((f, status, reason))

    print(f"\nVALID ({len(valid)}/{len(FUNCTIONALS)}): {valid}")
    print(f"\nINVALID/ERROR ({len(invalid)}/{len(FUNCTIONALS)}):")
    for f, status, reason in invalid:
        print(f"  {f:15s} status={status:12s} {str(reason)[:150]}")

    out_path = Path(__file__).parent / "accuracy_benchmark_logs" / "functional_validity_probe.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
