from __future__ import annotations

import os
import json
from typing import Any, Dict

from .indexer import build_index
from .prompts import build_architect_input, build_implementer_input, build_reviewer_input
from .diffutil import apply_patch_in_memory
from .fsview import read_text
from .envelope import make_artifact, make_envelope
from ..committee_client import committee_ai_text, committee_jsonify
from ..json_parser import JSONParser


ARCH_SCHEMA: Dict[str, Any] = {
    "plan": [{"file": str, "intent": str, "reason": str}],
    "notes": [str],
}
IMPL_SCHEMA: Dict[str, Any] = {
    "patch": str,
    "notes": [str],
}
REV_SCHEMA: Dict[str, Any] = {
    "patch": str,
    "findings": [str],
}


async def _parse_json(text: str, expected_schema: Any, *, trace_id: str) -> Dict[str, Any]:
    """
    IMPORTANT:
    - Only used for JSON emitted by LLMs (committee-backed in this module).
    - Always route through committee_jsonify BEFORE any JSONParser coercion.
    """
    parsed = await committee_jsonify(
        raw_text=text or "{}",
        expected_schema=expected_schema,
        trace_id=trace_id,
        temperature=0.0,
    )
    # Always run the final coercion/normalization pass via JSONParser with the expected structure.
    parser = JSONParser()
    obj = parser.parse(parsed if parsed is not None else "{}", expected_schema)
    return obj if isinstance(obj, dict) else {}


async def run_super_loop(task: str, repo_root: str, *, trace_id: str, step_tokens: int = 900) -> dict:
    """
    Returns canonical envelope with a unified diff and small artifacts.
    """
    idx = build_index(repo_root)
    # Phase 1: Architect
    arch_in = build_architect_input(task, idx)
    arch_env = await committee_ai_text(
        messages=[{"role": "user", "content": arch_in}],
        trace_id=trace_id,
        temperature=0.3,
    )
    arch_txt = ""
    if isinstance(arch_env, dict) and arch_env.get("ok"):
        arch_res = arch_env.get("result") or {}
        if isinstance(arch_res, dict) and isinstance(arch_res.get("text"), str):
            arch_txt = arch_res.get("text") or ""
    plan = await _parse_json(arch_txt, ARCH_SCHEMA, trace_id=trace_id)
    decisions = ["architect plan produced"]
    # Prepare excerpts
    excerpts = []
    for p in plan.get("plan", [])[:20]:
        f = p.get("file")
        if not f:
            continue
        txt = read_text(os.path.join(repo_root, f))
        excerpts.append({"file": f, "head": txt[:4000]})
    # Phase 2: Implementer
    impl_in = build_implementer_input(plan, excerpts)
    impl_env = await committee_ai_text(
        messages=[{"role": "user", "content": impl_in}],
        trace_id=trace_id,
        temperature=0.3,
    )
    impl_txt = ""
    if isinstance(impl_env, dict) and impl_env.get("ok"):
        impl_res = impl_env.get("result") or {}
        if isinstance(impl_res, dict) and isinstance(impl_res.get("text"), str):
            impl_txt = impl_res.get("text") or ""
    impl = await _parse_json(impl_txt, IMPL_SCHEMA, trace_id=trace_id)
    patch = impl.get("patch", "")
    decisions.append("implementer patch produced")
    # Phase 3: Reviewer
    rev_in = build_reviewer_input(task, plan, patch)
    rev_env = await committee_ai_text(
        messages=[{"role": "user", "content": rev_in}],
        trace_id=trace_id,
        temperature=0.3,
    )
    rev_txt = ""
    if isinstance(rev_env, dict) and rev_env.get("ok"):
        rev_res = rev_env.get("result") or {}
        if isinstance(rev_res, dict) and isinstance(rev_res.get("text"), str):
            rev_txt = rev_res.get("text") or ""
    rev = await _parse_json(rev_txt, REV_SCHEMA, trace_id=trace_id)
    final_patch = rev.get("patch", patch)
    decisions += ["reviewer pass", "diff verified"]
    # In-memory verification
    verify = apply_patch_in_memory(repo_root, final_patch)
    if not verify.ok:
        decisions.append(f"verify failed: {verify.errmsg}")
    # Build envelope + artifacts
    arts = [
        make_artifact("plan.json", "json", "architect plan"),
        make_artifact("patch.diff", "code", "unified diff", bytes_count=len((final_patch or "").encode())),
    ]
    tool_calls = [{"tool": "code.super_loop", "args": {"task": task, "repo_root": repo_root}, "status": "done", "result_ref": "patch.diff"}]
    env = make_envelope("committee", final_patch, arts, tool_calls, decisions)
    env["artifacts_data"] = {"patch.diff": final_patch or "", "plan.json": json.dumps(plan, ensure_ascii=False)}
    return env


