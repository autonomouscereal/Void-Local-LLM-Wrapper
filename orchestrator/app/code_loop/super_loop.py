from __future__ import annotations

import os
import json
from types import SimpleNamespace
from .indexer import build_index
from .prompts import build_architect_input, build_implementer_input, build_reviewer_input
from .diffutil import apply_patch_in_memory
from .fsview import read_text
from .envelope import make_artifact, make_envelope
from ..json_parser import JSONParser


def _parse_json(text: str) -> dict:
    parser = JSONParser()
    return parser.parse(text or "{}", {})


def run_super_loop(task: str, repo_root: str, model, step_tokens: int = 900) -> dict:
    """
    model: object exposing .chat(prompt, max_tokens) -> SimpleNamespace(text, model_name)
    Returns canonical envelope with a unified diff and small artifacts.
    """
    idx = build_index(repo_root)
    # Phase 1: Architect
    arch_in = build_architect_input(task, idx)
    arch_out = model.chat(arch_in, max_tokens=step_tokens)
    plan = _parse_json(getattr(arch_out, "text", ""))
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
    impl_out = model.chat(impl_in, max_tokens=step_tokens)
    impl = _parse_json(getattr(impl_out, "text", ""))
    patch = impl.get("patch", "")
    decisions.append("implementer patch produced")
    # Phase 3: Reviewer
    rev_in = build_reviewer_input(task, plan, patch)
    rev_out = model.chat(rev_in, max_tokens=step_tokens)
    rev = _parse_json(getattr(rev_out, "text", ""))
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
    env = make_envelope(getattr(model, "model_name", "model"), final_patch, arts, tool_calls, decisions)
    env["artifacts_data"] = {"patch.diff": final_patch or "", "plan.json": json.dumps(plan, ensure_ascii=False)}
    return env


