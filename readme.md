# Q-planner

[![GitHub](https://img.shields.io/badge/GitHub-Q--planner-blue)](https://github.com/zenqiaz/Q-planner)

Q-planner turns natural-language requests (e.g., “optimize ethanol and compute a single-point energy”) into a **deterministic, reproducible workflow** that runs quantum‑chemistry jobs via **ORCA** (through an MCP tool server), and returns structured results plus run logs and reports.

> ### Also on this branch: the DFT method-selection study
>
> Alongside the agent documented below, the `rag-retrieval` branch carries the code and
> artifacts behind *"Route, Then Learn: A General Recipe for Automating DFT Method Selection
> from Literature Precedent"* (J. Chem. Inf. Model., submitted). That study builds the
> method/basis **selection** step that the planner described here assumes has already been
> decided, so the two fit together rather than overlapping.
>
> **Start at [`README_method_selection.md`](README_method_selection.md).** It maps each table in
> the paper to the script and result file that produced it, gives the reproduction order with its
> GPU and ORCA requirements, and states what is deliberately not redistributed.
>
> To cite that work's code, use the archived release DOI for the tagged commit, never this branch
> URL -- a branch is mutable. See its "Release and archiving" section. Licence: MIT
> ([`LICENSE`](LICENSE)), which also records the third-party terms covering ORCA, benchmark
> reference data and NOMAD inputs.

---

## What the agent does

- **Plans** a workflow (a JSON “plan”) consisting of nodes like geometry loading, optimization, frequency, NBO, single-point energy, and simple post-processing calculations.
- **Executes** the plan in a **deterministic executor** (LangGraph), so tool calls happen in a known order and outputs are captured consistently.
- **Tracks geometries** in a lightweight in‑state **GeometryRegistry** (multiple geometries, aliases, charge/multiplicity, current selection).
- **Persists reports**:
  - a **runtime report** for each run
  - a **bug report** if the orchestrator hits an unexpected exception

---

## Core concepts

### 1) Plan JSON

A plan is a JSON document with metadata and a `nodes[]` list. Each node is typically one of:

- `kind: "tool"` — call one tool (e.g., `run_opt_job`, `run_sp_energy`)
- `kind: "llm_task"` (treated as `llm`) — an LLM step (optional; can be used for summarization/report writing)
- `kind: "calc"` / `"expr"` — evaluate a safe arithmetic expression from values in state/artifacts

Nodes can list dependencies via `needs: [...]` and can optionally route with `next`.

### 2) State, artifacts, and node results

During execution the orchestrator maintains a shared state:

- `state["artifacts"]`: long‑lived outputs you care about (energies, G, report text, etc.)
- `state["node_results"]`: per-node raw payloads (useful for debugging)
- `state["run_log"]`: ordered event log (node timing, statuses, etc.)
- `state["result"]` / `state["final_report"]`: the final structured report returned from the graph invocation

The state builder pre-seeds `artifacts` keys from the plan so downstream code can treat them as “pending” instead of missing.  

### 3) GeometryRegistry (multi-geometry support)

Geometries are stored in state as flat dicts:

- `state["geometries"][geom_id] -> "atom-lines-only XYZ"`
- `state["geom_meta"][geom_id] -> {"charge":..., "multiplicity":..., "name":...}`
- `state["name_to_geom"][alias] -> geom_id`
- `state["current_geom"] -> geom_id`

Tool nodes should pass geometry references via:

- `input_id` / `output_id` (preferred)
- or `input_geom_id` / `output_geom_id` (accepted)

The executor injects the correct `geometry_xyz`, `charge`, and `multiplicity` into tools that require a single geometry.

---

## Quickstart

### Requirements

- **ORCA** installed and available to the tool server.
- A running **MCP** server exposing ORCA-related tools (see Tool Server below).
- Python environment with:
  - `langgraph`
  - ORCA “OPI” integration used by the tool server
  - RDKit (optional, for name→3D geometry resolution client-side)

### Typical usage pattern

1. **Load / define a molecule**
   - by name (PubChem/OPSIN), or
   - by XYZ file, or
   - by generating/modifying a geometry (e.g., add/remove proton)

2. **Create a plan**
   - optimizer → frequency → single-point energy, etc.

3. **Run the plan**
   - `out_state = await graph.ainvoke(init_state)`
   - read outputs from `out_state["result"]`, `out_state["artifacts"]`

4. **Inspect outputs and reports**
   - runtime report and bug report paths are stored in artifacts if generated.

---

## Tool server (MCP) and available tools

The MCP tool server exposes ORCA jobs as tools. Common tools include:

- `run_opt_job`: geometry optimization
- `run_sp_energy`: single-point energy
- `run_nbo_job`: NBO/NPA section extraction
- `run_solvator_cluster_thermo`: explicit-solvent cluster + thermo (E/H/G)
- `structure_add_remove_proton`: modify protonation state

Some utilities may run **client-side** (not on MCP), such as:

- `name_to_geometry_xyz`: resolve a name to a 3D geometry (via PubChem)

Tool outputs are normalized into JSON, with consistent keys like:
- `status`
- `label` (job label)
- `geometry_xyz` or `final_geometry_xyz` (when a new/updated structure exists)
- `product` / `energy` etc. for primary scalar outputs

---

## Plan example: DFT opt → SP energy

Below is an example plan that:
1) loads a geometry by name  
2) optimizes it  
3) runs a single-point calculation

```json
{
  "name": "opt_then_sp",
  "version": "1.0",
  "artifacts_to_save": ["sp_energy_hartree"],
  "nodes": [
    {
      "id": "load_geom",
      "kind": "tool",
      "tool": "name_to_geometry_xyz",
      "output_id": "mol",
      "args": { "name": "water" }
    },
    {
      "id": "opt",
      "kind": "tool",
      "tool": "run_opt_job",
      "input_id": "mol",
      "output_id": "mol_opt",
      "args": {
        "method": "B3LYP",
        "basis": "def2-SVP",
        "ncores": 4
      }
    },
    {
      "id": "sp",
      "kind": "tool",
      "tool": "run_sp_energy",
      "input_id": "mol_opt",
      "output_id": "mol_opt",
      "args": {
        "method": "B3LYP",
        "basis": "def2-TZVP",
        "ncores": 4
      },
      "product": { "sp_energy_hartree": "energy" }
    }
  ],
  "final_report": {
    "format": "json",
    "fields": ["sp_energy_hartree"],
    "collect_from": ["sp"]
  }
}
```

Notes:

- `input_id` / `output_id` are **geometry identifiers** managed by the GeometryRegistry.
- The `product` mapping tells the orchestrator which payload key to store into which artifact key.

---

## Outputs you should expect

### Final result
The graph invocation returns the **final state**. The final structured report is usually at:

- `out_state["result"]` (alias of `out_state["final_report"]`)

### Runtime and bug reports
At the end of a run, the orchestrator attempts to write:

- a **runtime report** (`runtime_reports/*.json` and `runtime_reports/*.md`)
- if a node crashes unexpectedly, a **bug report** (`bug_reports/*.json` and `bug_reports/*.md`)

Paths are recorded in `out_state["artifacts"]`:
- `runtime_report_json_path`, `runtime_report_md_path`
- `bug_report_json_path`, `bug_report_md_path`

---

## Troubleshooting

### “No geometry available”
Most tool nodes require a geometry. Ensure you have:
- loaded a geometry, and
- set a `current_geom`, or passed `input_id` explicitly.

### Optimization did not converge
`run_opt_job` may return `status: "not_converged"`. You can:
- increase `opt_max_iter`
- adjust method/basis or add tighter SCF options
- rerun from the last geometry

### Timeouts
Tools return `status: "timeout"` when wall limits are exceeded.
- increase `wall_timeout_seconds` (or the tool-specific timeout)
- reduce method cost (smaller basis, cheaper method, fewer cores mismatch, etc.)

### Debug bundles
Some tools (e.g., optimization) support `debug=True` to return:
- ORCA executable path
- input/output tail
- directory listing

---

## Design goals (what to expect)

- **Reproducibility:** deterministic execution order and structured outputs.
- **Robustness:** state keeps raw payloads and logs; errors generate persistent reports.
- **Extensibility:** add new tools and new plan node types without changing user‑facing workflows.

