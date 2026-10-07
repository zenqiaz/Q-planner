"""One-job smoke test before committing to the full 20-job custom-complex opt batch."""
import asyncio, json, os, sys, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def _params():
    return StdioServerParameters(
        command=os.getenv("MCP_SSH_BIN", "ssh"),
        args=["-i", os.getenv("MCP_SSH_KEY"), "-o", "StrictHostKeyChecking=no",
              "-o", "BatchMode=yes", os.getenv("MCP_SSH_HOST"), os.getenv("MCP_SERVER_CMD")],
        env=dict(os.environ),
    )

async def main():
    xyz_path = Path(__file__).parent / "custom_complexes" / "fe_cl6_mult1_local.xyz"
    lines = xyz_path.read_text(encoding="utf-8").strip().splitlines()
    n = int(lines[0])
    geometry_xyz = "\n".join(lines[2:2+n])
    t0 = time.monotonic()
    async with stdio_client(_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("run_opt_job", {
                "geometry_xyz": geometry_xyz, "charge": -4, "multiplicity": 1,
                "method": "B3LYP", "basis": "def2-TZVP",
                "job_label": "smoketest_fe_cl6_mult1_B3LYP",
                "wall_timeout_seconds": 900,
            })
            text = res.content[0].text if res.content else "{}"
            payload = json.loads(text)
    print(f"elapsed={round(time.monotonic()-t0,1)}s")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("tail",)}, indent=2)[:3000])

asyncio.run(main())
