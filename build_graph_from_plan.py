from __future__ import annotations

import asyncio
import os
from typing import TypedDict, Annotated, Any, Dict, Callable, List, Awaitable, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
import ast
import math
import json
import re
import time
from datetime import datetime, timezone
from orchestrator_reporter import report_bug_and_persist, report_runtime_and_persist
from prompts import CALCULATOR_SYSTEM_PROMPT
from typing import Tuple

class State(TypedDict, total=False):
    session: Any
    # Optional registry of purely local tools (not exposed by the MCP server).
    # If present, the deterministic tool runner can resolve e.g. name_to_geometry_xyz
    # without going through the session.
    client_side_tools: Dict[str, Any]
    run_id: str
    run_started_utc: str
    run_finished_utc: str
    last_status: str
    last_tool_result: Dict[str, Any]

    # Annotated with merge reducer so parallel nodes don't overwrite each other's entries.
    run_log:      Annotated[List[dict],      lambda a, b: (a or []) + (b or [])]
    artifacts:    Annotated[Dict[str, Any],  lambda a, b: {**(a or {}), **(b or {})}]
    node_results: Annotated[Dict[str, Any],  lambda a, b: {**(a or {}), **(b or {})}]

    # GeometryRegistry-backed storage (dict-state)
    # Annotated with merge reducer so parallel nodes don't overwrite each other's geometry entries.
    geometries:   Annotated[Dict[str, str],  lambda a, b: {**(a or {}), **(b or {})}]
    geom_meta:    Annotated[Dict[str, Any],  lambda a, b: {**(a or {}), **(b or {})}]
    name_to_geom: Annotated[Dict[str, str],  lambda a, b: {**(a or {}), **(b or {})}]
    current_geom: Optional[str]
    final_report: Dict[str, Any]
    result: Any
    token_usage: Dict[str, int]



def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


ToolCaller = Callable[[str, Dict[str, Any]], Dict[str, Any]]
LLMCaller = Callable[[str, State, Dict[str, Any]], Dict[str, Any]]

# New-style node runners (can wrap an LLM+tool-loop executor)
ToolNodeRunner = Callable[[State, Dict[str, Any]], "ToolNodeResult"]
AsyncToolNodeRunner = Callable[[State, Dict[str, Any]], Awaitable["ToolNodeResult"]]

ToolNodeResult = Dict[str, Any]


ALLOWED_FUNCS = {
    "log": math.log,
    "exp": math.exp,
    "sqrt": math.sqrt,
    "abs": abs,
}
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
    ast.Name, ast.Constant,
    ast.Call, ast.Load,
)


# ─── Template expansion ────────────────────────────────────────────────────────

def _deep_substitute(obj: Any, compound: Dict[str, Any]) -> Any:
    """Recursively replace {C} and {C.<field>} placeholders in strings.

    Supported placeholders:
      {C}         → compound["id"]
      {C.name}    → compound.get("name", id)
      {C.role}    → compound.get("role", "target")
      {C.<field>} → compound.get("<field>", "") for any other field in the compound dict
    """
    if isinstance(obj, str):
        import re as _re
        # Whole-value shortcut: if the entire string is exactly one placeholder,
        # return the raw Python value so lists/ints pass through without str().
        if obj == "{C}":
            return compound["id"]
        _whole_match = _re.fullmatch(r"\{C\.([^}]+)\}", obj)
        if _whole_match:
            return compound.get(_whole_match.group(1), "")

        def _replace(m: "re.Match") -> str:
            field = m.group(1)  # everything after "C."
            if field == "":
                return compound["id"]
            return str(compound.get(field, ""))
        # Replace {C} first, then {C.<field>}
        result = obj.replace("{C}", compound["id"])
        result = _re.sub(r"\{C\.([^}]+)\}", _replace, result)
        return result
    if isinstance(obj, dict):
        return {_deep_substitute(k, compound): _deep_substitute(v, compound)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_substitute(item, compound) for item in obj]
    return obj  # int, float, bool, None — pass through unchanged


def expand_template(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Expand template + compounds into a flat nodes list.

    If plan has no 'template' key, returns plan unchanged (backward compatible).
    Raises ValueError on invalid template structure or unresolved placeholders.
    """
    template = plan.get("template")
    if not template or not isinstance(template, dict):
        return plan

    compounds = plan.get("compounds")
    if not compounds or not isinstance(compounds, list):
        raise ValueError("plan has 'template' but no 'compounds' list")

    per_compound = template.get("per_compound") or []
    post_template = template.get("post_template") or []

    expanded_nodes: List[Dict[str, Any]] = []

    # Phase 1: expand per_compound nodes for every compound.
    # Nodes whose id contains no {C} placeholder are treated as "once" nodes —
    # they belong at the end (after all per-compound nodes) and appear exactly once.
    once_nodes: List[Dict[str, Any]] = []
    _seen_once_ids: set = set()
    for compound in compounds:
        for node_tmpl in per_compound:
            node_id_tmpl = (node_tmpl.get("id") or "") if isinstance(node_tmpl, dict) else ""
            if "{C" not in node_id_tmpl:
                # Shared node: collect once, add after per-compound loop
                resolved = _deep_substitute(node_tmpl, compound)
                nid = resolved.get("id") if isinstance(resolved, dict) else None
                if nid and nid not in _seen_once_ids:
                    _seen_once_ids.add(nid)
                    once_nodes.append(resolved)
            else:
                expanded_nodes.append(_deep_substitute(node_tmpl, compound))
    expanded_nodes.extend(once_nodes)

    # Phase 2: expand post_template nodes with applies_to filtering
    for node_tmpl in post_template:
        applies_to = node_tmpl.get("applies_to", "all")
        if applies_to == "once":
            node = dict(node_tmpl)
            node.pop("applies_to", None)
            expanded_nodes.append(node)
        else:
            if applies_to == "targets_only":
                subset = [c for c in compounds if c.get("role", "target") != "reference"]
            elif applies_to == "references_only":
                subset = [c for c in compounds if c.get("role", "target") == "reference"]
            else:  # "all"
                subset = compounds
            for compound in subset:
                node = _deep_substitute(node_tmpl, compound)
                if isinstance(node, dict):
                    node.pop("applies_to", None)
                expanded_nodes.append(node)

    # Phase 3: expand geom_ids and artifacts_to_save
    # Patterns containing any {C...} placeholder are expanded once per compound.
    def _expand_list(patterns: List[str]) -> List[str]:
        result: List[str] = []
        for pat in (patterns or []):
            if isinstance(pat, str) and "{C" in pat:
                subset = compounds
                for c in subset:
                    val = _deep_substitute(pat, c)
                    if val not in result:
                        result.append(val)
            else:
                if pat not in result:
                    result.append(pat)
        return result

    new_plan = dict(plan)
    new_plan["nodes"] = expanded_nodes
    new_plan["geom_ids"] = _expand_list(plan.get("geom_ids") or [])
    new_plan["artifacts_to_save"] = _expand_list(plan.get("artifacts_to_save") or [])

    # Also expand {C} placeholders in final_report.fields and summarizer.report_fields
    fr = plan.get("final_report")
    if isinstance(fr, dict) and fr.get("fields"):
        new_plan["final_report"] = {**fr, "fields": _expand_list(fr["fields"])}
    summ = plan.get("summarizer")
    if isinstance(summ, dict) and summ.get("report_fields"):
        new_plan["summarizer"] = {**summ, "report_fields": _expand_list(summ["report_fields"])}

    # Validation: no unresolved placeholders (catches both {C} and unknown variants like {C.foo})
    import re as _re
    raw = json.dumps(expanded_nodes)
    leftover = _re.findall(r"\{C[^}]*\}", raw)
    if leftover:
        raise ValueError(
            f"Unresolved placeholder(s) after template expansion: {set(leftover)}. "
            f"Only {{C}}, {{C.name}}, {{C.role}} are supported."
        )

    # Validation: unique node IDs
    ids = [n["id"] for n in expanded_nodes if isinstance(n, dict) and "id" in n]
    dupes = {x for x in ids if ids.count(x) > 1}
    if dupes:
        raise ValueError(f"Duplicate node IDs after template expansion: {dupes}")

    return new_plan


def build_state(
    plan: Dict[str, Any],
    session: Any,
    *,
    seed: Optional[Dict[str, Any]] = None,
    client_side_tools: Optional[Dict[str, Any]] = None,
) -> State:
    """Build an initial LangGraph state from a plan JSON.

    Key behavior:
    - Pre-seed state['artifacts'] with all artifact keys implied by the plan.
      This lets executors/UI treat absent values as "pending" rather than
      missing keys.
    - Does NOT overwrite any non-None artifact values present in `seed`.
    """
    plan = expand_template(plan)
    st: Dict[str, Any] = dict(seed or {})
    st["session"] = session
    st["plan"] = plan

    # If callers don't pass a seed containing client-side tools, allow passing them
    # explicitly via build_state(..., client_side_tools=...).
    if "client_side_tools" not in st and client_side_tools is not None:
        st["client_side_tools"] = client_side_tools

    st.setdefault("run_log", [])
    st.setdefault("node_results", {})
    st.setdefault("last_status", "init")
    st.setdefault("last_tool_result", {})
    st.setdefault("final_report", {})
    st.setdefault("token_usage", {"planner": 0, "calculator": 0, "reporter": 0, "total": 0})

    # Collect artifact keys from multiple plan sections
    artifact_keys: set[str] = set()
    artifact_keys.update(plan.get("artifacts_to_save") or [])

    fr = plan.get("final_report") or {}
    artifact_keys.update(fr.get("fields") or [])

    summ = plan.get("summarizer") or {}
    artifact_keys.update(summ.get("report_fields") or [])

    by_node: Dict[str, Any] = {}
    for n in (plan.get("nodes") or []):
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if not nid:
            continue

        product = n.get("product") or {}
        if isinstance(product, dict):
            artifact_keys.update(product.keys())

        expect = n.get("expect") or {}
        if isinstance(expect, dict):
            artifact_keys.update(expect.get("artifacts_required") or [])

        if n.get("kind") == "tool":
            by_node[nid] = {
                "tool": n.get("tool"),
                "artifacts_required": (expect.get("artifacts_required") or []) if isinstance(expect, dict) else [],
                "properties_required": (expect.get("properties_required") or []) if isinstance(expect, dict) else [],
                "product_keys": list(product.keys()) if isinstance(product, dict) else [],
            }

    artifacts = dict(st.get("artifacts") or {})
    for k in sorted(artifact_keys):
        artifacts.setdefault(k, None)
    st["artifacts"] = artifacts
    st.setdefault("artifacts_spec", {"artifact_keys": sorted(artifact_keys), "by_node": by_node})

    return st  # type: ignore[return-value]


def compile_expr(expr: str) -> Callable[[Dict[str, float]], float]:
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCS:
                raise ValueError("Disallowed function call")
        if isinstance(node, ast.Name):
            # Names are validated at runtime via the env dict;
            # this just blocks dunder / attribute tricks.
            if node.id.startswith("__"):
                raise ValueError("Disallowed name")

    code = compile(tree, "<expr>", "eval")

    def f(env: Dict[str, float]) -> float:
        safe_globals = {"__builtins__": {}}
        safe_locals = dict(ALLOWED_FUNCS)
        safe_locals.update(env)
        return float(eval(code, safe_globals, safe_locals))

    return f

# ─── patch_and_retry helpers ───────────────────────────────────────────────────

# Tool parameter names safe to patch (all ORCA tools expose these)
_PATCHABLE_TOOL_ARGS: set = {
    "method", "basis", "use_ri", "scf_max_iter", "opt_max_iter",
    "ncores", "wall_timeout_seconds", "nroots", "calc_hess", "xtb_preopt",
    "maxiter", "avas_variant",
}


def _infer_error_code(tool_payload: Dict[str, Any]) -> Optional[str]:
    """Return a structured error code from a failed tool result.

    Checks tool_payload["code"] first (server-supplied), then falls back to
    pattern-matching text fields so retries work even when the server predates
    the structured code field.
    """
    code = (tool_payload or {}).get("code")
    if code:
        return str(code)
    if (tool_payload or {}).get("status") == "timeout":
        return "RESOURCE_LIMIT"
    text = " ".join([
        str(tool_payload.get("error_summary") or ""),
        str(tool_payload.get("error") or ""),
        str(tool_payload.get("tail") or ""),
    ]).upper()
    # Same precedence as the server-side classifier: input that never parsed outranks any
    # convergence reading of the same tail.
    if "INPUT ERROR" in text:
        return "INPUT_ERROR"
    if "SCF NOT CONVERGED" in text:
        return "SCF_NOT_CONVERGED"
    if "IMAGINARY" in text and ("MODE" in text or "FREQ" in text):
        return "IMAG_FREQ"
    if "CASSCF" in text and "NOT CONVERGED" in text:
        return "CASSCF_NOT_CONVERGED"
    if "GEOM" in text and ("INVALID" in text or "FAILED" in text or "BAD" in text):
        return "GEOM_INVALID"
    if "RESOURCE" in text or "TIME LIMIT" in text:
        return "RESOURCE_LIMIT"
    # Client-side fallback: check ir_spectrum list for imaginary modes
    ir_spectrum = (tool_payload or {}).get("ir_spectrum")
    if isinstance(ir_spectrum, list):
        if any(isinstance(e.get("freq_cm1"), (int, float)) and e["freq_cm1"] < -10
               for e in ir_spectrum):
            return "IMAG_FREQ"
    return None


def _normalize_on_error_rule(rule: dict) -> Dict[str, Any]:
    """Normalize one on_error entry to canonical form.

    Handles three planner schema variants:
      A: {"if": {cond}, "action": ..., "patch": ..., "max_attempts": N}
      B: {"when": {cond}, "action": ..., "patch": ..., "max_attempts": N}
      C: {"if": {cond}, "then": {"action": ..., "patch": ..., "max_attempts": N}}
    """
    condition = rule.get("if") or rule.get("when") or {}
    inner     = rule.get("then") if "then" in rule else rule
    return {
        "codes":        list(condition.get("code_in") or []),
        "action":       str(inner.get("action") or ""),
        "patch":        dict(inner.get("patch") or {}),
        "max_attempts": int(inner.get("max_attempts") or 1),
    }


# Solvation is expressed inside the ORCA method string (e.g. "PBE0 CPCM(Water)")
# because the tools take no separate solvent parameter. A patch that replaces
# `method` wholesale would therefore silently drop solvation and report the
# result as solvated anyway. These keywords are carried across such a patch.
_SOLVATION_KEYWORDS = ("CPCM", "SMD", "COSMO", "PCM", "ALPB", "GBSA")


def _preserved_solvation(method_value: Any) -> str:
    """Solvation tokens present in a method string, e.g. 'CPCM(Water)'."""
    if not isinstance(method_value, str):
        return ""
    for tok in method_value.split():
        if any(tok.upper().startswith(k) for k in _SOLVATION_KEYWORDS):
            return tok
    return ""


def _apply_patch_to_args(current_args: dict, patch: dict) -> dict:
    """Merge a patch dict into tool args, mapping nested structures to param names.

    Safe: only known tool parameter names are applied; unknown keys are silently
    dropped to avoid TypeError in strict MCP tool signatures. A `method` patch
    additionally preserves any solvation keyword from the method being replaced,
    so an error-recovery retry cannot quietly turn a solvated calculation into a
    gas-phase one.
    """
    new_args = dict(current_args)
    for key, val in patch.items():
        if key == "scf" and isinstance(val, dict):
            # Only scf_max_iter is a real tool param today
            if "maxiter" in val:
                new_args["scf_max_iter"] = int(val["maxiter"])
        elif key == "resources" and isinstance(val, dict):
            if "ncores" in val:
                new_args["ncores"] = int(val["ncores"])
        elif key == "method":
            solv = _preserved_solvation(current_args.get("method"))
            if solv and isinstance(val, str) and not _preserved_solvation(val):
                new_args["method"] = f"{val} {solv}"
                print(f"  [patch] preserving solvation keyword {solv!r} "
                      f"across method patch -> {new_args['method']!r}")
            else:
                new_args["method"] = val
        elif key in _PATCHABLE_TOOL_ARGS:
            new_args[key] = val
        # else: silently drop (not a known tool parameter)
    return new_args

# ───────────────────────────────────────────────────────────────────────────────


def build_graph_from_plan(
    plan: Dict[str, Any],
    call_tool: Optional[ToolCaller] = None,
    get_tool_args: Optional[Callable[[Any, str, Dict[str, Any]], Dict[str, Any]]] = None,
    run_tool_node: Optional[ToolNodeRunner | AsyncToolNodeRunner] = None,
    openai_client: Optional[Any] = None,
):
    plan = expand_template(plan)
    plan_nodes: List[Dict[str, Any]] = plan.get("nodes") or []
    if not plan_nodes:
        raise ValueError("plan['nodes'] is empty")

    nodes_by_id = {n["id"]: n for n in plan_nodes if n.get("id")}
    if not nodes_by_id:
        raise ValueError("plan has no nodes with 'id' fields")

    ordered_ids = [n["id"] for n in plan_nodes if n.get("id") in nodes_by_id]
    entry = plan.get("entry") or ordered_ids[0]

    FINAL_NODE_ID = "__final_report__"

    # sequential fallback routing (when spec lacks 'next')
    next_in_order: Dict[str, str] = {}
    for i, nid in enumerate(ordered_ids):
        next_in_order[nid] = ordered_ids[i + 1] if (i + 1) < len(ordered_ids) else FINAL_NODE_ID

    artifacts_to_save = set(plan.get("artifacts_to_save") or [])

    async def _maybe_await(v: Any) -> Any:
        if asyncio.iscoroutine(v):
            return await v
        return v

    def _ensure_state_defaults(state: State) -> Dict[str, Any]:
        upd: Dict[str, Any] = {}
        if "run_id" not in state or not state.get("run_id"):
            upd["run_id"] = _utc_run_id()
        if "run_started_utc" not in state or not state.get("run_started_utc"):
            upd["run_started_utc"] = datetime.now(timezone.utc).isoformat()
        if "run_log" not in state:
            upd["run_log"] = []
        if "artifacts" not in state or not isinstance(state.get("artifacts"), dict):
            upd["artifacts"] = {}
        if "node_results" not in state or not isinstance(state.get("node_results"), dict):
            upd["node_results"] = {}
        return upd

    def _get_by_path(obj: Any, path: Any) -> Any:
        if obj is None or path is None:
            return None
        cur = obj
        if isinstance(path, str):
            parts = [p for p in path.split(".") if p]
        elif isinstance(path, list):
            parts = path
        else:
            parts = [path]

        for p in parts:
            if cur is None:
                return None
            if isinstance(p, int):
                if isinstance(cur, list) and 0 <= p < len(cur):
                    cur = cur[p]
                else:
                    return None
            else:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
        return cur

    def _summarize_value_for_log(v: Any) -> Any:
        """Keep run_log compact + JSON-safe."""
        if v is None or isinstance(v, (int, float, bool)):
            return v
        if isinstance(v, str):
            if len(v) <= 200:
                return v
            return v[:200] + f"...<len={len(v)}>"
        if isinstance(v, dict):
            ks = list(v.keys())
            return {"_type": "dict", "n_keys": len(ks), "keys_head": ks[:20]}
        if isinstance(v, list):
            return {"_type": "list", "len": len(v)}
        return {"_type": type(v).__name__}

    def _stash_artifacts(node_id: str, spec: Dict[str, Any], result: Dict[str, Any], state: State) -> Dict[str, Any]:
        upd: Dict[str, Any] = {}
        upd.update(_ensure_state_defaults(state))

        node_results = dict(state.get("node_results", {}))
        node_results[node_id] = result
        result = result.get("last_tool_result")
        upd["node_results"] = node_results

        if result is None:
            return upd

        artifacts = dict(state.get("artifacts", {}))

        # Convention: tools may return a primary output via a "product" field.
        # Example: {"status":"ok","product":"energy","energy":-76.32,...}
        product_spec = spec.get("product") or {}  # artifact key → result field path
        if isinstance(product_spec, dict) and product_spec:
            # Explicit product mapping: planner-specified artifact names.
            for out_key, src in product_spec.items():
                if isinstance(src, dict):
                    # src is {artifact_subkey: result_field_path, ...} → collect as dict
                    sub = {k: _get_by_path(result, p) for k, p in src.items()}
                    sub = {k: v for k, v in sub.items() if v is not None}
                    if sub:
                        artifacts[out_key] = sub
                else:
                    v = _get_by_path(result, src)
                    if v is not None:
                        artifacts[out_key] = v
        else:
            # Auto-artifact mode: no product dict → save every tool output field
            # as "{node_id}.{field}", skipping orchestration-internal keys.
            _INTERNAL = {"status", "product", "error"}
            for field, val in result.items():
                if field not in _INTERNAL and val is not None:
                    artifacts[f"{node_id}.{field}"] = val

        upd["artifacts"] = artifacts
        return upd
    
    def _set_by_path_root_update(state: State, path: str, value: Any) -> Dict[str, Any]:
        if not path or not isinstance(path, str):
            return {}
        parts = [p for p in path.split(".") if p]
        if not parts:
            return {}
        if len(parts) == 1:
            return {parts[0]: value}

        root = parts[0]
        tail = parts[1:]

        root_obj = state.get(root)
        if not isinstance(root_obj, dict):
            root_obj = {}
        new_root = dict(root_obj)
        cur = new_root
        for p in tail[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
            nxt = dict(nxt)
            cur[p] = nxt
            cur = nxt
        cur[tail[-1]] = value
        return {root: new_root}
    
    def _compile_final_report(state: State) -> Dict[str, Any]:
        fr = plan.get("final_report") or {}
        collect_from = fr.get("collect_from") or ordered_ids
        fields = fr.get("fields") or []

        node_results = state.get("node_results", {}) or {}
        artifacts = state.get("artifacts", {}) or {}

        report: Dict[str, Any] = {}
        for field in fields:
            val = artifacts.get(field)
            if val is None:
                for nid in collect_from:
                    res = node_results.get(nid)
                    if not isinstance(res, dict):
                        continue
                    if field in res:
                        val = res[field]
                        break
                    props = res.get("properties")
                    if isinstance(props, dict) and field in props:
                        val = props[field]
                        break
                    art = res.get("artifacts")
                    if isinstance(art, dict) and field in art:
                        val = art[field]
                        break
            report[field] = val

        report.setdefault("status", state.get("last_status", "ok"))
        report.setdefault("run_log", state.get("run_log", []))
        return report

    builder = StateGraph(State)

    async def _maybe_await(x: Any) -> Any:
        # LangGraph nodes may be sync or async; we support both runner styles.
        if asyncio.iscoroutine(x):
            return await x
        return x

    def make_node(node_id: str):
        spec = dict(nodes_by_id[node_id])

        async def _node(state: State) -> Command:
            updates: Dict[str, Any] = {}
            updates.update(_ensure_state_defaults(state))

            node_started_utc = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()

            try:
                kind = (spec.get("kind") or "tool")
                if kind == "llm_task":
                    kind = "llm"
                status = "ok"

                if kind == "tool":
                    tool_name = spec["tool"]
                    # Prefer plan schema "args"; keep "overrides" for backward compatibility.
                    node_args = spec.get("args")
                    if node_args is None:
                        node_args = spec.get("overrides")
                    node_args = node_args or {}

                    # --- patch_and_retry setup ---
                    on_error_rules = [
                        _normalize_on_error_rule(r)
                        for r in (spec.get("on_error") or [])
                    ]
                    on_error_rules = [r for r in on_error_rules if r["action"] == "patch_and_retry"]

                    attempt       = 0
                    attempt_args  = dict(node_args)
                    current_state = state

                    while True:
                        retry_spec = dict(spec)
                        retry_spec["id"]             = node_id
                        retry_spec["args"]           = attempt_args
                        retry_spec["_plan_settings"] = plan.get("settings") or {}

                        if run_tool_node is not None:
                            result = await _maybe_await(run_tool_node(current_state, retry_spec))
                            if not isinstance(result, dict):
                                result = {"status": "error", "error": "tool node runner returned non-dict"}
                        else:
                            args = get_tool_args(state["session"], tool_name, attempt_args)
                            result = call_tool(tool_name, args) or {}

                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except Exception:
                                result = {"status": "error", "error": "tool returned non-JSON string", "raw": result}

                        tool_payload = result.get("last_tool_result", result)
                        status = result.get(
                            "last_status",
                            tool_payload.get("status") if isinstance(tool_payload, dict) else None,
                        ) or "unknown"

                        if status == "ok" or not on_error_rules:
                            break

                        error_code   = _infer_error_code(tool_payload if isinstance(tool_payload, dict) else {})
                        matched_rule = next(
                            (r for r in on_error_rules
                             if error_code in r["codes"] and attempt < r["max_attempts"]),
                            None,
                        )
                        if matched_rule is None:
                            break  # no rule matches or max_attempts exhausted

                        attempt      += 1
                        attempt_args  = _apply_patch_to_args(attempt_args, matched_rule["patch"])
                        current_state = {**state, **{
                            k: result[k]
                            for k in ("geometries", "geom_meta", "name_to_geom", "current_geom")
                            if k in result
                        }}
                        print(f"  [retry] node={node_id!r} attempt={attempt} "
                              f"code={error_code!r} patch_keys={list(matched_rule['patch'].keys())}")
                    # --- end retry loop ---

                    ended_utc  = datetime.now(timezone.utc).isoformat()
                    duration_ms = int((time.perf_counter() - t0) * 1000)

                    # 1) merge the returned state updates first (this is what makes geometry persistent)
                    updates.update(result)

                    # 2) ensure per-node node_results stores the tool payload (not the whole state update dict)
                    node_results = dict(state.get("node_results") or {})
                    node_results[node_id] = tool_payload
                    updates["node_results"] = node_results

                    # 3) standard bookkeeping
                    print(f"  [tool_status] node={node_id!r} status={status!r}"
                          + (f" (after {attempt} retr{'y' if attempt==1 else 'ies'})" if attempt else ""))
                    log_entry: Dict[str, Any] = {
                        "node":         node_id,
                        "kind":         kind,
                        "tool":         tool_name,
                        "status":       status,
                        "started_utc":  node_started_utc,
                        "ended_utc":    ended_utc,
                        "duration_ms":  duration_ms,
                    }
                    if attempt:
                        log_entry["retry_attempts"] = attempt
                    updates.update({
                        "last_status":      status,
                        "last_tool_result": tool_payload,
                        "run_log":          [log_entry],
                    })
                    if isinstance(result, dict):
                        updates.update(_stash_artifacts(node_id, spec, result, state))
                elif kind == "llm":
                    task = spec.get("task") or spec.get("pattern") or "llm_task"
                    prompt = spec.get("prompt") or task
                    needs_artifacts = spec.get("needs_artifacts") or []

                    if openai_client is None:
                        out = {"status": "error", "error": "no openai_client provided for llm node"}
                    else:
                        # Gather requested artifacts from state
                        artifacts = state.get("artifacts") or {}
                        gathered = {}
                        missing = []
                        for key in needs_artifacts:
                            val = artifacts.get(key)
                            if val is None:
                                missing.append(key)
                            else:
                                gathered[key] = val

                        if missing:
                            print(f"  [llm_missing] node={node_id!r} missing={missing!r} available={list(artifacts.keys())}")
                            out = {"status": "error", "error": f"missing artifacts: {missing}"}
                        else:
                            # Build context for the calculator LLM
                            plan_settings = plan.get("settings") or {}
                            user_content = (
                                f"Task: {prompt}\n\n"
                                f"Artifacts:\n{json.dumps(gathered, indent=2)}\n\n"
                                f"Plan settings:\n{json.dumps(plan_settings, indent=2)}"
                            )
                            messages = [
                                {"role": "system", "content": CALCULATOR_SYSTEM_PROMPT},
                                {"role": "user", "content": user_content},
                            ]

                            # A calculator node's whole output is one json.loads call
                            # on a raw completion, so a fenced response
                            # ("```json\n{...}\n```", which some completions add even
                            # when told not to) or one truncated mid-object fails the
                            # node outright with no recovery. One retry, explicitly
                            # telling the model what was wrong with its own last
                            # answer, converts an intermittent parse failure into a
                            # deterministic pass without weakening what is checked --
                            # the JSON still has to be a valid dict either way.
                            call_messages = list(messages)
                            raw = ""
                            out = None
                            for attempt in range(2):
                                try:
                                    resp = openai_client.chat.completions.create(
                                        model=os.getenv("LLM_MODEL", "gpt-5.2"),
                                        messages=call_messages,
                                        response_format={"type": "json_object"},
                                    )
                                    _usage = getattr(resp, "usage", None)
                                    if _usage:
                                        _tu = state.get("token_usage") or {}
                                        _tu["calculator"] = _tu.get("calculator", 0) + (_usage.total_tokens or 0)
                                        _tu["total"]      = _tu.get("total", 0)      + (_usage.total_tokens or 0)
                                        state = {**state, "token_usage": _tu}
                                    raw = (resp.choices[0].message.content or "").strip()
                                    stripped = raw
                                    if stripped.startswith("```"):
                                        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
                                        stripped = re.sub(r"```\s*$", "", stripped).strip()
                                    parsed = json.loads(stripped)
                                    if not isinstance(parsed, dict):
                                        out = {"status": "error",
                                               "error": "LLM returned non-dict JSON",
                                               "raw": raw}
                                    else:
                                        out = parsed
                                        break
                                except json.JSONDecodeError as e:
                                    out = {"status": "error",
                                           "error": f"LLM returned non-JSON: {e}",
                                           "raw": raw}
                                except Exception as e:
                                    out = {"status": "error", "error": str(e), "raw": raw}
                                    break   # not a parse failure -- retrying will not help
                                if attempt == 0:
                                    call_messages = messages + [
                                        {"role": "assistant", "content": raw},
                                        {"role": "user", "content":
                                            "That response was not valid JSON "
                                            f"({out.get('error')}). Reply with the JSON "
                                            "object only -- no prose, no markdown "
                                            "code fence."},
                                    ]

                    status = out.get("status", "ok")

                    ended_utc = datetime.now(timezone.utc).isoformat()
                    duration_ms = int((time.perf_counter() - t0) * 1000)

                    # Store LLM output as last_tool_result so _stash_artifacts can read product
                    out_with_meta = dict(out)
                    out_with_meta.setdefault("task", task)

                    node_results = dict(state.get("node_results") or {})
                    node_results[node_id] = out_with_meta
                    updates["node_results"] = node_results

                    updates["last_status"] = status
                    updates["last_tool_result"] = out_with_meta
                    log_entry = {
                        "node": node_id,
                        "kind": kind,
                        "llm_task": task,
                        "status": status,
                        "started_utc": node_started_utc,
                        "ended_utc": ended_utc,
                        "duration_ms": duration_ms,
                    }
                    if status != "ok":
                        # Preserved specifically so a parse failure is diagnosable from
                        # the runtime report alone -- previously this information was
                        # produced (out["error"]/out["raw"]) and then discarded before
                        # it reached the log, so a failed llm node left no trace of why.
                        if out.get("error"):
                            log_entry["error"] = out["error"]
                        if out.get("raw"):
                            log_entry["raw"] = out["raw"][:2000]
                    updates["run_log"] = [log_entry]

                    # Stash artifacts from LLM output via product spec
                    # LLM may return values at top level or nested under "values"
                    product_spec = spec.get("product") or {}
                    if isinstance(product_spec, dict) and isinstance(out, dict):
                        cur_artifacts = dict(state.get("artifacts") or {})
                        cur_artifacts.update(updates.get("artifacts") or {})
                        values_dict = out.get("values") if isinstance(out.get("values"), dict) else {}
                        # Build a flat lookup merging top-level + values_dict
                        # Also try common key aliases (site_ranking_json → site_ranking, etc.)
                        _ALIAS_MAP = {
                            "site_ranking_json": "site_ranking",
                            "ranking_json":      "site_ranking",
                            "comparison_table_json": "comparison_table",
                            # pKa: LLM may return pKa_X (variable from formula) instead of lowercase pka
                            "pka": "pKa_X",
                        }
                        flat_out = {**values_dict, **{k: v for k, v in out.items() if k != "values"}}
                        # Build case-insensitive fallback map for LLM key lookup
                        _flat_out_lower = {k.lower(): v for k, v in flat_out.items()}
                        for art_key, src_key in product_spec.items():
                            val = flat_out.get(src_key)
                            if val is None:
                                val = flat_out.get(_ALIAS_MAP.get(src_key, src_key))
                            if val is None:
                                # Case-insensitive fallback (e.g. "pKa_X" vs "pka")
                                val = _flat_out_lower.get(src_key.lower())
                            if val is not None:
                                cur_artifacts[art_key] = val
                        updates["artifacts"] = cur_artifacts
                elif kind in ("calc_expr", "expr", "calc"):
                    expr = spec.get("expr")
                    inputs = spec.get("inputs") or {}
                    constants = spec.get("constants") or {}
                    needs_artifacts_calc = spec.get("needs_artifacts") or []
                    output_key = spec.get("output_key")  # optional, e.g. "results.pKa"

                    result: Dict[str, Any] = {"status": "error"}
                    try:
                        if not isinstance(expr, str) or not expr.strip():
                            raise ValueError("missing expr")

                        env: Dict[str, float] = {}

                        # Inject artifacts from needs_artifacts: key "a.b" → variable "a_b"
                        artifacts_state = state.get("artifacts") or {}
                        missing_arts = []
                        for art_key in needs_artifacts_calc:
                            val = artifacts_state.get(art_key)
                            if val is None:
                                missing_arts.append(art_key)
                            else:
                                sym = art_key.replace(".", "_")
                                try:
                                    env[sym] = float(val)
                                except (TypeError, ValueError):
                                    missing_arts.append(art_key)
                        if missing_arts:
                            raise KeyError(f"needs_artifacts not found or non-numeric: {missing_arts}")

                        if not isinstance(inputs, dict):
                            raise ValueError("inputs must be a dict")
                        for sym, path in inputs.items():
                            v = _get_by_path(state, path)
                            if v is None:
                                raise KeyError(f"input {sym!r} not found at path {path!r}")
                            env[str(sym)] = float(v)

                        if not isinstance(constants, dict):
                            raise ValueError("constants must be a dict")
                        for k, v in constants.items():
                            env[str(k)] = float(v)

                        fn = compile_expr(expr)
                        value = fn(env)

                        result = {"status": "ok", "expr": expr, "value": value, "env": env}

                        if isinstance(output_key, str) and output_key.strip():
                            updates.update(_set_by_path_root_update(state, output_key, value))

                    except Exception as e:
                        result = {"status": "error", "error": str(e), "expr": expr, "inputs": inputs}

                    status = (result.get("status") or "error")

                    ended_utc = datetime.now(timezone.utc).isoformat()
                    duration_ms = int((time.perf_counter() - t0) * 1000)

                    updates.update({
                        "last_status": status,
                        "last_tool_result": result,
                        "run_log": [{
                            "node": node_id,
                            "kind": kind,
                            "status": status,
                            "started_utc": node_started_utc,
                            "ended_utc": ended_utc,
                            "duration_ms": duration_ms,
                        }],
                    })
                    # _stash_artifacts expects a state-like dict with last_tool_result;
                    # merge updates into state so product mapping sees the result.
                    updates.update(_stash_artifacts(node_id, spec, {**state, **updates}, {**state, **updates}))

                else:
                    status = "error"
                    print(f"  [unknown_kind] node={node_id!r} kind={kind!r}")

                    ended_utc = datetime.now(timezone.utc).isoformat()
                    duration_ms = int((time.perf_counter() - t0) * 1000)

                    updates["last_status"] = status
                    updates["run_log"] = [{
                        "node": node_id,
                        "kind": kind,
                        "status": status,
                        "error": f"unknown kind: {kind}",
                        "started_utc": node_started_utc,
                        "ended_utc": ended_utc,
                        "duration_ms": duration_ms,
                    }]

                # routing
                if "next" in spec:
                    nxt = spec.get("next")
                    if isinstance(nxt, dict):
                        goto = nxt.get(status) or nxt.get("default") or next_in_order.get(node_id, FINAL_NODE_ID)
                    else:
                        goto = nxt
                else:
                    goto = next_in_order.get(node_id, FINAL_NODE_ID)

                # always run FINAL_REPORT if present/configured
                if goto in ("END", END, None):
                    goto = FINAL_NODE_ID

                if isinstance(goto, str) and goto not in nodes_by_id and goto != FINAL_NODE_ID:
                    print(f"  [unknown_goto] node={node_id!r} goto={goto!r} known={list(nodes_by_id.keys())}")
                    updates["last_status"] = "error"
                    updates["run_log"] = [{"node": node_id, "status": "error", "error": f"unknown next node: {goto!r}"}]
                    goto = FINAL_NODE_ID

                return Command(update=updates, goto=goto)

            except Exception as e:
                # Unexpected runtime exception: persist a bug report and force-finalize.
                print(f"  [node_exception] node={node_id!r} kind={spec.get('kind')!r} error={e!r}")
                import traceback; traceback.print_exc()
                node_spec = dict(spec)
                node_spec["id"] = node_id

                report: Dict[str, Any]
                json_path: Optional[str]
                md_path: Optional[str]

                try:
                    report, json_path, md_path = report_bug_and_persist(
                        node_id=node_id,
                        node_spec=node_spec,
                        state=state,
                        error=e,
                        stage="node_execution",
                    )
                except Exception as e2:
                    report = {
                        "type": "orchestrator_bug_reporter_failed",
                        "node_id": node_id,
                        "original_error": str(e),
                        "reporter_error": str(e2),
                    }
                    json_path, md_path = None, None

                # Make sure artifacts exist and stash paths.
                artifacts = state.get("artifacts")
                if not isinstance(artifacts, dict):
                    artifacts = {}
                else:
                    artifacts = dict(artifacts)

                artifacts["bug_report"] = report
                if json_path:
                    artifacts["bug_report_json_path"] = json_path
                if md_path:
                    artifacts["bug_report_md_path"] = md_path

                updates.update(_ensure_state_defaults(state))
                updates["artifacts"] = artifacts
                updates["last_status"] = "error"
                updates["last_tool_result"] = {"status": "error", "error": str(e)}

                ended_utc = datetime.now(timezone.utc).isoformat()
                duration_ms = int((time.perf_counter() - t0) * 1000)
                updates["run_log"] = [{
                    "node": node_id,
                    "kind": spec.get("kind", "tool"),
                    "status": "error",
                    "error": str(e),
                    "bug_report_json_path": json_path,
                    "bug_report_md_path": md_path,
                    "started_utc": node_started_utc,
                    "ended_utc": ended_utc,
                    "duration_ms": duration_ms,
                }]

                return Command(update=updates, goto=FINAL_NODE_ID)

        return _node

    for node_id in ordered_ids:
        builder.add_node(node_id, make_node(node_id))

    def _final_report_node(state: State) -> Command:
        updates: Dict[str, Any] = {}
        updates.update(_ensure_state_defaults(state))

        # Mark end-of-run timestamp (used by runtime report writer).
        if not state.get("run_finished_utc"):
            updates["run_finished_utc"] = datetime.now(timezone.utc).isoformat()

        report = _compile_final_report(state)
        updates["final_report"] = report
        updates["result"] = report
        updates["last_status"] = report.get("status", state.get("last_status", "ok"))

        # Persist a full runtime report (JSON + Markdown) for the entire graph.
        merged_state = dict(state)
        merged_state.update(updates)
        merged_state["final_report"] = report

        rt_report: Dict[str, Any]
        rt_json_path: Optional[str]
        rt_md_path: Optional[str]
        try:
            rt_report, rt_json_path, rt_md_path = report_runtime_and_persist(
                plan=plan,
                state=merged_state,
                final_report=report,
                stage="graph_complete",
            )
        except Exception as e:
            rt_report = {"type": "orchestrator_runtime_report_failed", "error": str(e)}
            rt_json_path, rt_md_path = None, None

        artifacts = merged_state.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        else:
            artifacts = dict(artifacts)

        artifacts["runtime_report"] = rt_report
        if rt_json_path:
            artifacts["runtime_report_json_path"] = rt_json_path
        if rt_md_path:
            artifacts["runtime_report_md_path"] = rt_md_path

        updates["artifacts"] = artifacts
        updates["run_log"] = [{
            "node": FINAL_NODE_ID,
            "kind": "final",
            "status": "ok",
            "runtime_report_json_path": rt_json_path,
            "runtime_report_md_path": rt_md_path,
        }]
        return Command(update=updates, goto=END)

    builder.add_node(FINAL_NODE_ID, _final_report_node)
    builder.add_edge(START, entry)

    return builder.compile()
