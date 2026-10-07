"""Throwaway probe: does def2-SVP (candidate TM-safe replacement for 6-31G in FIXED_DEFAULT)
actually cover transition metals on this ORCA build? Tests a few real TM species/functionals."""
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
    return StdioServerParameters(
        command=os.getenv("MCP_SSH_BIN", "ssh"),
        args=["-i", os.getenv("MCP_SSH_KEY", ""), "-o", "StrictHostKeyChecking=no",
              "-o", "BatchMode=yes", os.getenv("MCP_SSH_HOST", ""), os.getenv("MCP_SERVER_CMD", "")],
        env=dict(os.environ),
    )


ED03_XYZ = (
    "Ni   -0.7629039   -0.2803608    0.4889495\n"
    "C     0.9417181   -0.5614688   -0.0291272\n"
    "C    -1.6112009    1.2380907    0.0119109\n"
    "C    -1.6202975   -1.5176032    1.4825547\n"
    "O     2.0270968   -0.7402781   -0.3595063\n"
    "O    -2.1515398    2.2049067   -0.2921948\n"
    "O    -2.1666934   -2.3052194    2.1152351"
)

# candidate blind (functional, basis) settings for metal_general
CANDIDATES = [
    ("B3LYP", "def2-SVP"),
    ("TPSSH", "def2-SVP"),
    ("PBE0", "def2-SVP"),
]


async def probe_one(session, sem, functional, basis):
    async with sem:
        node_id = f"probe_ED03_{functional}_{basis}".replace("(", "").replace(")", "")
        try:
            res = await session.call_tool("run_sp_energy", {
                "geometry_xyz": ED03_XYZ, "charge": 0, "multiplicity": 1,
                "method": functional, "basis": basis,
                "job_label": node_id, "wall_timeout_seconds": 120,
            })
            text = res.content[0].text if res.content else "{}"
            payload = json.loads(text)
        except Exception as e:  # noqa: BLE001
            payload = {"status": "client_error", "error": repr(e)}
        return (functional, basis), payload


async def main():
    server_params = _build_mcp_server_params()
    sem = asyncio.Semaphore(3)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            out = await asyncio.gather(*[probe_one(session, sem, f, b) for f, b in CANDIDATES])

    for (f, b), p in out:
        status = p.get("status")
        if status == "ok":
            print(f"{f}/{b}: OK, energy_eh={p.get('energy_eh')}")
        else:
            tail = p.get("tail", "") or p.get("text", "") or p.get("error", "")
            print(f"{f}/{b}: {status} -- {str(tail)[-200:]}")


if __name__ == "__main__":
    asyncio.run(main())
