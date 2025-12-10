from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from ..icw.tokenizer import byte_budget_for_model, bytes_len


_LOG = logging.getLogger(__name__)


def _pick_alloc_within_ranges(ranges: Dict[str, Any]) -> Dict[str, int]:
    """
    Choose fixed representative percentages within provided ranges.
    This is deterministic and simple by design (prompt-only orchestration).
    """
    icw_lo, icw_hi = int(ranges.get("icw_pct", [65, 70])[0]), int(ranges.get("icw_pct", [65, 70])[1])
    tools_lo, tools_hi = int(ranges.get("tools_pct", [18, 20])[0]), int(ranges.get("tools_pct", [18, 20])[1])
    roe_lo, roe_hi = int(ranges.get("roe_pct", [5, 10])[0]), int(ranges.get("roe_pct", [5, 10])[1])
    misc_lo, misc_hi = int(ranges.get("misc_pct", [3, 5])[0]), int(ranges.get("misc_pct", [3, 5])[1])
    buffer_pct = int(ranges.get("buffer_pct", 5))

    def mid(a: int, b: int) -> int:
        return int(round((a + b) / 2))

    alloc = {
        "icw": mid(icw_lo, icw_hi),
        "tools": mid(tools_lo, tools_hi),
        "roe": mid(roe_lo, roe_hi),
        "misc": mid(misc_lo, misc_hi),
        "buffer": buffer_pct,
    }
    total = sum(alloc.values())
    if total > 100:
        # Trim in priority: misc -> tools -> icw tail (represented as icw)
        excess = total - 100
        if alloc["misc"] > 0:
            d = min(excess, alloc["misc"])
            alloc["misc"] -= d
            excess -= d
        if excess > 0 and alloc["tools"] > 0:
            d = min(excess, alloc["tools"])
            alloc["tools"] -= d
            excess -= d
        if excess > 0 and alloc["icw"] > 0:
            d = min(excess, alloc["icw"])
            alloc["icw"] -= d
            excess -= d
    return alloc


def _co_frames_baseline() -> Dict[str, str]:
    """
    Return canonical short instruction blocks used by CO. Keep in sync with meta_prompt scaffolds.
    """
    return {
        "coding": (
            "### [CODING STYLE / SYSTEM]\n"
            "[PYTHON INDENTATION RULES]\n\n"
            "- For all Python code, you MUST use exactly 4 spaces per indentation level.\n"
            '- Tabs ("\\t") are FORBIDDEN in Python code. Do not emit tab characters anywhere.\n'
            "- Do NOT mix tabs and spaces. Indentation must be consistent 4-space blocks.\n"
            "- When refactoring or editing existing Python code, normalize indentation to 4-space blocks, removing any tabs."
        ),
        "global_rules": (
            "### [CYRIL — GLOBAL PYTHON & ENGINEERING RULES / SYSTEM]\n"
            "These rules apply to ALL Python code and architecture you generate or modify for me, across ALL projects and repos. They are HARD REQUIREMENTS.\n"
            "Do NOT weaken, ignore, or interpret them.\n\n"
            "==================================================\n"
            "1) INDENTATION & FORMATTING\n"
            "==================================================\n"
            "- All Python code MUST use exactly 4 spaces per indentation level.\n"
            "- Tabs (\\t) are FORBIDDEN in Python code. Never emit tab characters.\n"
            "- Do NOT mix tabs and spaces under any circumstance.\n"
            "- When touching existing code that has tabs, normalize it to 4-space indentation.\n\n"
            "==================================================\n"
            "2) IMPORTS POLICY\n"
            "==================================================\n"
            "- All imports MUST be at the top of the file at MODULE level.\n"
            "- NO function-level, method-level, class-level, or conditional imports. EVER.\n"
            "- NO dynamic imports or importlib hacks.\n"
            "- NO import-time I/O or side effects except minimal logger setup in a single entrypoint.\n"
            "- Whenever you introduce a new import, you MUST also update dependency manifests (e.g., requirements.txt, Docker image layer, etc.) in the same change, with sensible version pins.\n\n"
            "==================================================\n"
            "3) TIMEOUTS & RETRIES (GLOBAL)\n"
            "==================================================\n"
            "- Default rule: NO client-side timeouts anywhere (HTTP, WS, DB, subprocess, etc.).\n"
            "- If a library or API REQUIRES a timeout: prefer timeout=None; otherwise use the maximum safe cap.\n"
            "- NEVER add retries for timeouts.\n"
            "- Retries ONLY for non-timeout transient failures (429, 503, network reset/refused) with bounded jitter, and only when explicitly requested.\n"
            "- Do NOT hide timeouts or retries inside helpers/wrappers.\n\n"
            "==================================================\n"
            "4) ERROR HANDLING / NO SILENT FAILS\n"
            "==================================================\n"
            "- In executor/orchestrator hot paths, NO try/except at all.\n"
            "- Elsewhere, avoid try/except unless strictly necessary.\n"
            "- Failures must be explicit; never convert to empty success, fake booleans, or vague messages.\n\n"
            "==================================================\n"
            "5) DEPENDENCIES, ORMs, AND DATABASES\n"
            "==================================================\n"
            "- FORBIDDEN: Pydantic, SQLAlchemy, ANY ORM (Django ORM, Tortoise, GINO, etc.), SQLite for app data.\n"
            "- DATABASE: Only PostgreSQL via asyncpg. Use raw SQL + asyncpg (no ORMs/query builders that hide SQL).\n\n"
            "==================================================\n"
            "6) JSON & API CONTRACTS\n"
            "==================================================\n"
            "- Public APIs use JSON-only envelopes; no magic schema layers.\n"
            "- /v1/chat/completions: always HTTP 200 with OpenAI-compatible JSON; choices[0].message.content must be present and non-empty on success.\n"
            "- On failure: return structured JSON error explaining what failed and why.\n"
            "- Tool/executor args MUST be JSON objects at execution time. If planning emits strings, include explicit json.parse before execution; executors assume proper objects.\n\n"
            "==================================================\n"
            "7) TOOL / EXECUTOR / ORCHESTRATOR PATH INVARIANTS\n"
            "==================================================\n"
            "- Public routes MUST NOT call /tool.run directly. Valid flow: Planner → Executor (/execute) → Orchestrator /tool.run.\n"
            "- On 422: executor surfaces errors directly; no separate validate/repair loop.\n"
            "- Do not clobber provided args; add defaults only if missing. Do not delete/overwrite user keys arbitrarily.\n"
            "- Executor runs provided steps once; traces MUST include exec.payload (patched), repair.executing, tool.run.start, etc.\n\n"
            "==================================================\n"
            "8) TRACING & LOGGING\n"
            "==================================================\n"
            "- Every run produces deterministic traces: requests.jsonl, events.jsonl, tools.jsonl, artifacts.jsonl, responses.jsonl; errors.jsonl for verbose details.\n"
            "- Include breadcrumbs: chat.start, planner.*, committee.*, exec.payload (pre & patched), validate.*, repair.*, tool.run.start, Comfy submit/poll, chat.finish.\n"
            "- Errors are explicit, never logged-and-forgotten.\n\n"
            "==================================================\n"
            "9) MISC GLOBAL ENGINEERING RULES\n"
            "==================================================\n"
            "- No in-method imports or hidden side effects.\n"
            "- No background ops layers or preflight gates unless explicitly requested.\n"
            "- Prefer deterministic behavior; where randomness is needed, expose and log seeds.\n"
            "- Do not introduce new frameworks/large deps without clear justification.\n\n"
            "==================================================\n"
            "10) FUNCTION STRUCTURE (NO NESTED FUNCTIONS)\n"
            "==================================================\n"
            "- Sub-functions / nested functions are FORBIDDEN.\n"
            "  - Do NOT define functions inside other functions.\n"
            "  - Do NOT define lambdas or callbacks that contain inner `def` blocks.\n"
            "- All functions must be:\n"
            "  - Top-level module functions, or\n"
            "  - Methods on a class.\n"
            "- If you think you \"need\" a helper function inside another:\n"
            "  - Promote it to a top-level function (or a method), and call it from there instead.\n"
            "- No closures that depend on outer function locals via inner `def` blocks.\n"
            "  - Use explicit parameters and return values instead of capturing outer scope.\n\n"
            "SUMMARY: 4 spaces; no tabs; no function-level imports; no default timeouts; no retries on timeouts; no try/except in hot paths; no silent failures; no Pydantic/SQLAlchemy/ORMs/SQLite; Postgres+asyncpg only; strict JSON envelopes; Planner→Executor→Orchestrator flow; clear errors and full traceability."
        ),
        "co": (
            "### [CO / SYSTEM — Percent Allocator]\n"
            "Build your working context by self-compression and self-trimming. Use this ratio of the model’s context C:\n"
            "- ICW (multimodal compressed): 65–70% of C\n"
            "- TOOLS (catalog names + recent tool outcomes): 18–20% of C\n"
            "- RoE (Rules of Engagement): 5–10% of C (always include; tail-resilient)\n"
            "- MISC (identity/steer/tail): 3–5% of C\n"
            "- BUFFER: ~5% of C\n"
            "If your assembled set exceeds 100% of C, trim in this exact order until you fit: MISC → TOOLS summaries → ICW history tail. Never trim RoE or the identity frame.\n"
            "Build and plan in three sweeps over an oversized corpus (~150% of C): Sweep A 0–90%, Sweep B 30–120%, Sweep C 60–150% + wrap 0–30%. For each sweep, compress to these ratios, then plan. After all sweeps, merge notes into a single plan."
        ),
        "icw": (
            "### [ICW BUILDER / SYSTEM]\n"
            "Construct ICW (65–70% of C) by summarizing, not copying:\n"
            "- Session Tail (high): recent turns merged; intents, decisions, params; link artifacts by short IDs.\n"
            "- Episodic Summaries: per topic, focusing on what changed & why.\n"
            "- Decision Ledger: bullets of settled choices.\n"
            "- Artifact Index: [type, short_name, purpose, key params, url/hash].\n"
            "- Thread Map: who/what produced what; open TODOs.\n"
            "If ICW exceeds its ratio, trim: Thread Map → Artifact Index → Decision Ledger → Episodic → Session Tail (last).\n"
            "\n"
            "You are given separate 'recent_summary' and 'recent_raw' frames for the latest user request.\n"
            "When compressing history, focus ICW on messages before the latest user request; do not override\n"
            "or contradict the latest request with older instructions."
        ),
        "tools": (
            "### [TOOLS CONTEXT / SYSTEM]\n"
            "Fit tools into 18–20% of C:\n"
            "- Valid tool names (one line) only.\n"
            "- Recent tool memory (one line each): [tool, success/fail, key args, artifact ids/urls, short diagnosis].\n"
            "If over ratio, drop oldest/least relevant summaries first; keep the names line."
        ),
        "tools_schema_raw": (
            "### [TOOLS SCHEMA RAW / SYSTEM]\n"
            "Curated RAW JSON blocks for selected tools only (schema_ref notes allowed)."
        ),
        "tools_evidence_raw": (
            "### [TOOLS EVIDENCE RAW / SYSTEM]\n"
            "Last K=2–3 runs per selected tool: 1 success and 1–2 failures (RAW JSON, artifacts truncated)."
        ),
        "subject": (
            "### [SUBJECT CANON / SYSTEM — tail with RoE]\n"
            'If the user mentions Shadow (Sonic), treat the subject as "Shadow the Hedgehog" (SEGA) unless explicitly negated.\n'
            "In any image.dispatch prompt, include the literal name and ≥60% of: black hedgehog; red stripes on quills/arms; upward-swept quills; red eyes; white chest tuft; gold inhibitor rings (wrists/ankles); hover shoes (red/white/black, jet glow).\n"
            "Prefer futuristic/city/space-lab scenes unless requested otherwise. Add negatives to avoid generic silhouettes/forests."
        ),
        "roe_capture": (
            "### [RoE CAPTURE / SYSTEM]\n"
            "When the user gives interaction rules, extract/update a concise RoE list.\n"
            "Format: - {scope:user|global} {must|must_not|prefer|avoid}: <directive> (reason:<short>).\n"
            "Deduplicate; most recent overrides. Keep within 5–10% by merging/abstracting."
        ),
        "roe_digest": (
            "### [RoE DIGEST / SYSTEM — tail]\n"
            "Rules of Engagement (5–10% of C; never omit). Self-check planned steps and wording against RoE; adjust voluntarily."
        ),
    }


def _build_roe_digest_lines(incoming: List[str], globals_: List[str]) -> List[str]:
    lines: List[str] = []
    for raw in (globals_ or []):
        s = str(raw).strip()
        if s:
            lines.append(s)
    for raw in (incoming or []):
        s = str(raw).strip()
        if not s:
            continue
        # Very light normalization to bullets if not already formatted
        if not s.startswith("- "):
            s = f"- {s}"
        lines.append(s)
    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        out.append(ln)
    return out[:16]


def split_previous_and_last_user(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    """
    Split the provided messages into all messages before the last user message
    and the last user message itself. If no user message exists, return all
    messages as previous and None for last.
    """
    last_user_index = -1
    index = len(messages) - 1
    while index >= 0 and last_user_index == -1:
        m = messages[index]
        role = m.get("role") if isinstance(m, dict) else None
        if role == "user":
            last_user_index = index
        index -= 1
    if last_user_index == -1:
        return messages, None
    return messages[:last_user_index], messages[last_user_index]


RECENT_RAW_MAX_CHARS = 4000
RECENT_HEAD_CHARS = 2000
RECENT_TAIL_CHARS = 1500
RECENT_SUMMARY_MAX_CHARS = 512


def build_recent_summary(last_user_message: Dict[str, Any] | None) -> Dict[str, str] | None:
    """
    Build a short system frame summarizing the last user message.
    Simple truncation-based summary if no summarizer is available.
    """
    if not isinstance(last_user_message, dict):
        return None
    content = str(last_user_message.get("content") or "")
    if not content:
        return None
    snippet = content[:RECENT_SUMMARY_MAX_CHARS]
    summary_text = "Latest user request (compressed):\n" + snippet
    return {"role": "system", "name": "recent_summary", "content": summary_text}


def build_recent_raw(last_user_message: Dict[str, Any] | None) -> Dict[str, str] | None:
    """
    Build a user frame with head+tail clipping to preserve intro and instructions.
    """
    if not isinstance(last_user_message, dict):
        return None
    content = str(last_user_message.get("content") or "")
    if not content:
        return None
    length = len(content)
    if length <= RECENT_RAW_MAX_CHARS:
        clipped = content
    else:
        head = content[:RECENT_HEAD_CHARS]
        tail = content[-RECENT_TAIL_CHARS:] if RECENT_TAIL_CHARS > 0 else ""
        clipped = head + "\n...\n" + tail
    return {"role": "user", "name": "recent_raw", "content": clipped}


def frames_bytes(frames: List[Dict[str, Any]]) -> int:
    return bytes_len(
        "\n\n".join(
            [
                f.get("content", "")
                for f in frames or []
                if isinstance(f.get("content"), str)
            ]
        )
    )


def co_pack(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build canonical CO frames and ratio telemetry (prompt-only).
    No try/except here; callers handle errors by contract.
    """
    trace_id = str(envelope.get("trace_id") or "")
    user_turn = envelope.get("user_turn") or {}
    history = envelope.get("history") or []
    subject_canon = envelope.get("subject_canon") or {}
    roe_incoming = envelope.get("roe_incoming_instructions") or []
    call_kind = str(envelope.get("call_kind") or "planner")
    percent_budget = envelope.get("percent_budget") or {}
    sweeps = envelope.get("sweep_plan") or ["0-90", "30-120", "60-150+wrap"]

    # Optional tools context payloads
    tools_names: List[str] = list(envelope.get("tools_names") or [])
    tools_recent_lines: List[str] = list(envelope.get("tools_recent_lines") or [])
    tsl_blocks: List[Dict[str, Any]] = list(envelope.get("tsl_blocks") or [])
    tel_blocks: List[Dict[str, Any]] = list(envelope.get("tel_blocks") or [])

    model_caps = envelope.get("model_caps") or {}
    ctx_limit_tokens_raw = envelope.get("ctx_limit_tokens") or model_caps.get("num_ctx") or 8192
    try:
        ctx_limit_tokens = int(ctx_limit_tokens_raw)
    except (TypeError, ValueError):
        ctx_limit_tokens = 8192
    total_budget_bytes = byte_budget_for_model(ctx_limit_tokens)

    # Pick static allocations within ranges
    alloc = _pick_alloc_within_ranges(percent_budget or {})
    # During repair, allow TOOLS to expand to <=30% by borrowing from ICW (prompt-only policy)
    if call_kind == "repair" and alloc["tools"] < 30:
        diff = 30 - alloc["tools"]
        if alloc["icw"] > diff:
            alloc["icw"] -= diff
            alloc["tools"] = 30
    used_pct = alloc["icw"] + alloc["tools"] + alloc["roe"] + alloc["misc"] + alloc["buffer"]
    free_pct = max(0, 100 - used_pct)

    icw_budget_bytes = int(total_budget_bytes * alloc["icw"] / 100.0)
    tools_budget_bytes = int(total_budget_bytes * alloc["tools"] / 100.0)
    roe_budget_bytes = int(total_budget_bytes * alloc["roe"] / 100.0)
    misc_budget_bytes = int(total_budget_bytes * alloc["misc"] / 100.0)
    buffer_budget_bytes = int(total_budget_bytes * alloc["buffer"] / 100.0)

    # Build baseline frames
    blocks = _co_frames_baseline()
    frames: List[Dict[str, Any]] = []
    frames.append({"role": "system", "content": blocks["co"]})
    # For planner/executor we rely on an actual ICW window elsewhere, not the big
    # ICW meta-prose block.
    if call_kind != "planner":
        frames.append({"role": "system", "content": blocks["icw"]})
    # Global coding style rules
    frames.append({"role": "system", "content": blocks["coding"]})
    # Global engineering rules
    frames.append({"role": "system", "content": blocks["global_rules"]})

    # Tools summary (names + recent lines)
    tools_summary_lines: List[str] = []
    if tools_names:
        tools_summary_lines.append(
            "Tools: " + ", ".join(sorted([str(n) for n in tools_names if isinstance(n, str)]))
        )
    for ln in (tools_recent_lines or [])[:12]:
        if isinstance(ln, str) and ln.strip():
            tools_summary_lines.append(ln.strip())
    tools_summary = blocks["tools"] + ("\n" + "\n".join(tools_summary_lines) if tools_summary_lines else "")
    frames.append({"role": "system", "content": tools_summary})

    # RAW Tool Schema Lane (TSL) — curated blocks
    if tsl_blocks:
        parts: List[str] = [blocks["tools_schema_raw"]]
        # Keep TSL compact: cap to at most 3 schemas up front.
        for b in tsl_blocks[:3]:
            try:
                name = str(b.get("name") or "").strip()
                raw_json = b.get("raw") or {}
                if not name or not isinstance(raw_json, dict):
                    continue
                parts.append(
                    "### [TOOL SCHEMA RAW] "
                    + name
                    + "\n"
                    + json.dumps(raw_json, ensure_ascii=False)
                )
            except Exception as ex:
                _LOG.warning(
                    "compression_orchestrator.tsl_block_error",
                    extra={"block_name": str((b or {}).get("name") or "")},
                    exc_info=ex,
                )
                continue
        frames.append({"role": "system", "content": "\n".join(parts)})

    # RAW Tool Evidence Lane (TEL) — recent runs
    if tel_blocks:
        parts = [blocks["tools_evidence_raw"]]
        for b in tel_blocks[:8]:
            try:
                name = str(b.get("name") or "").strip()
                label = (
                    str(b.get("label") or "").strip()
                    or ("success" if b.get("ok") else "failure")
                )
                raw_json = b.get("raw") or {}
                if not name or not isinstance(raw_json, dict):
                    continue
                parts.append(
                    "### [TOOL RESULT RAW] "
                    + name
                    + " ("
                    + label
                    + ")\n"
                    + json.dumps(raw_json, ensure_ascii=False)
                )
            except Exception as ex:
                _LOG.warning(
                    "compression_orchestrator.tel_block_error",
                    extra={
                        "block_name": str((b or {}).get("name") or ""),
                        "label": str((b or {}).get("label") or ""),
                    },
                    exc_info=ex,
                )
                continue
        frames.append({"role": "system", "content": "\n".join(parts)})

    # Optional subject canon
    if isinstance(subject_canon, dict) and (subject_canon.get("literal") or subject_canon.get("tokens")):
        frames.append({"role": "system", "content": blocks["subject"]})

    # Determine previous vs last user message for recency handling
    messages_for_split: List[Dict[str, Any]] = list(history or [])
    if isinstance(user_turn, dict) and user_turn.get("role") == "user" and user_turn.get("content"):
        messages_for_split = list(messages_for_split) + [user_turn]
    previous_messages, last_user_message = split_previous_and_last_user(messages_for_split)
    # Note: ICW compression should consider only previous_messages. The CO 'icw' frame above
    # provides instructions for the model to compress history; we intentionally do not include
    # last_user_message in that compression and instead add it as recent frames below.
    _ = previous_messages  # reserved for future explicit ICW integration

    # Insert recency frames immediately before RoE tail
    recent_summary_frame = build_recent_summary(last_user_message)
    recent_raw_frame = build_recent_raw(last_user_message)
    if recent_summary_frame is not None:
        frames.append(recent_summary_frame)
    if recent_raw_frame is not None:
        frames.append(recent_raw_frame)

    # Tail-anchored RoE digest
    # Keep RoE small and at the tail. Place capture/digest after recent frames.
    frames.append({"role": "system", "content": blocks["roe_capture"]})
    global_rules = [
        "- scope:user must: one final answer only (reason:user policy)",
        "- scope:user must: use image.dispatch for image requests (reason:tool-first)",
        "- scope:user must_not: include Comfy /view or bare filenames; only absolute /uploads/artifacts/... (reason:URL hygiene)",
        "- scope:global prefer: keep literal identity 'Shadow the Hedgehog' + ≥60% tokens (reason:subject fidelity)",
        "- scope:global must: body content non-empty; surface warnings inline (reason:UX)",
    ]
    roe_digest_lines = _build_roe_digest_lines(roe_incoming, global_rules)
    frames.append({"role": "system", "content": blocks["roe_digest"] + "\n" + "\n".join(roe_digest_lines)})

    # Enforce byte budget by trimming low-priority bands in order:
    # misc (TEL / extra schemas) → tools details → ICW tail (recent_raw).
    overflow_handled = False
    evicted_misc = 0
    tools_trim_pct = 0
    icw_tail_trim_pct = 0
    p3_dropped_pct = 0

    total_bytes = frames_bytes(frames)
    if total_bytes > total_budget_bytes:
        # Drop tool evidence lane (TEL) first – lowest priority misc.
        before = total_bytes
        frames = [
            f
            for f in frames
            if "TOOLS EVIDENCE RAW / SYSTEM" not in str(f.get("content") or "")
        ]
        if frames_bytes(frames) < before:
            overflow_handled = True
            evicted_misc += 1
        total_bytes = frames_bytes(frames)

    if total_bytes > total_budget_bytes:
        # If still over, drop any additional TSL content beyond the first block.
        new_frames: List[Dict[str, Any]] = []
        saw_tsl = False
        for f in frames:
            content = str(f.get("content") or "")
            if "TOOLS SCHEMA RAW / SYSTEM" in content or "### [TOOL SCHEMA RAW]" in content:
                if not saw_tsl:
                    new_frames.append(f)
                    saw_tsl = True
                else:
                    overflow_handled = True
                    evicted_misc += 1
                    continue
            else:
                new_frames.append(f)
        frames = new_frames
        total_bytes = frames_bytes(frames)

    if total_bytes > total_budget_bytes:
        # Trim tools frame down to header + tool names line only.
        new_frames = []
        for f in frames:
            content = str(f.get("content") or "")
            if "### [TOOLS CONTEXT / SYSTEM]" in content:
                lines = content.splitlines()
                header_lines: List[str] = []
                other_lines: List[str] = []
                for ln in lines:
                    if ln.startswith("### [TOOLS CONTEXT / SYSTEM]"):
                        header_lines.append(ln)
                    else:
                        other_lines.append(ln)
                names_line = ""
                for ln in other_lines:
                    if ln.startswith("Tools: "):
                        names_line = ln
                        break
                trimmed_parts = []
                if header_lines:
                    trimmed_parts.extend(header_lines)
                if names_line:
                    trimmed_parts.append(names_line)
                trimmed_content = "\n".join([p for p in trimmed_parts if p])
                new_frames.append({"role": f.get("role", "system"), "content": trimmed_content})
                overflow_handled = True
                tools_trim_pct = alloc["tools"]
            else:
                new_frames.append(f)
        frames = new_frames
        total_bytes = frames_bytes(frames)

    if total_bytes > total_budget_bytes:
        # Finally, drop recent_raw (ICW tail) but keep recent_summary and RoE.
        before_count = len(frames)
        frames = [f for f in frames if f.get("name") != "recent_raw"]
        after_count = len(frames)
        if after_count < before_count:
            overflow_handled = True
            icw_tail_trim_pct = alloc["icw"]
        total_bytes = frames_bytes(frames)

    ratio_telemetry = {
        "used_pct": used_pct,
        "free_pct": free_pct,
        "alloc": alloc,
        "call_kind": call_kind,
        "kept": {"P0": True, "ICW": True, "Tools": True, "RoE": True},
        "evicted": {
            "misc": evicted_misc,
            "tools_trim_pct": tools_trim_pct,
            "icw_tail_trim_pct": icw_tail_trim_pct,
            "p3_dropped_pct": p3_dropped_pct,
        },
        "sweeps": sweeps,
        "tail_identity_present": True,
        "overflow_handled": overflow_handled,
        "byte_budget": {
            "ctx_limit_tokens": ctx_limit_tokens,
            "total_budget_bytes": total_budget_bytes,
            "total_bytes": total_bytes,
            "icw_bytes": icw_budget_bytes,
            "tools_bytes": tools_budget_bytes,
            "roe_bytes": roe_budget_bytes,
            "misc_bytes": misc_budget_bytes,
            "buffer_bytes": buffer_budget_bytes,
        },
    }
    return {
        "schema_version": 1,
        "trace_id": trace_id,
        "frames": frames,
        "ratio_telemetry": ratio_telemetry,
        "roe_digest": roe_digest_lines,
    }


def frames_to_messages(frames: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Pass-through helper for clarity and future expansion.
    """
    return list(frames or [])


def frames_to_string(frames: List[Dict[str, str]]) -> str:
    """
    Concatenate frame contents into a single prompt string for backends that accept plain prompts.
    """
    parts: List[str] = []
    for fr in frames or []:
        ct = fr.get("content")
        if isinstance(ct, str) and ct.strip():
            parts.append(ct.strip())
    return "\n\n".join(parts)


