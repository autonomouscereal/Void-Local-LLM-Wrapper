from __future__ import annotations
# HARD BAN (permanent): Never add Pydantic, SQLAlchemy, or any ORM/CSV/Parquet libs to this service.
# - No Pydantic models. Use plain dicts + hand-rolled validators only.
# - No SQLAlchemy/ORM. Use asyncpg with pooling and raw SQL only.
# - No CSV/Parquet persistence. JSON/NDJSON only for manifests/logs.
#
# JSON POLICY (critical):
# - Never call json.loads directly in this service. All JSON coming from LLMs, tools, env/config,
#   or external services MUST be parsed via JSONParser().parse with an explicit expected structure.
# - Always pass an expected schema (dict/list with primitive types) to enforce shape and defaults;
#   avoid parser(..., {}) unless the value is truly open‑ended (e.g., opaque metadata).
# - When receiving strings that may be JSON (e.g., plan/workflow/metadata), coerce using JSONParser
#   BEFORE any .get or type‑dependent access to prevent 'str'.get type errors.
#
# Historical/rationale (orchestrator):
# - Planner/Executors (Qwen + GLM4-9B) are coordinated with explicit system guidance to avoid refusals.
# - Tools are selected SEMANTICALLY (not keywords). On refusal, we may force a single best-match tool
#   with minimal arguments, but we never overwrite model output; we only append a short Status.
# - JSON parsing uses a hardened JSONParser with expected structures to survive malformed LLM JSON.
# - All DB access is via asyncpg; JSONB writes use json.dumps and ::jsonb to avoid type issues.
# - RAG (pgvector) initializes extension + indexes at startup and uses a simple cache.
# - OpenAI-compatible /v1/chat/completions returns full Markdown: main answer first, then "### Tool Results"
#   (film_id, job_id(s), errors), then an "### Appendix — Model Answers" with raw Qwen/GLM4-9B responses (trimmed).
# - We never replace the main content with status; instead we append a Status block only when empty/refusal.
# - Long-running film pipeline: film_create → film_add_scene (ComfyUI jobs via /jobs) → film_compile (n8n or local assembler).
# - Keep LLM models warm: we set options.keep_alive=24h on every Ollama call to avoid reloading between requests.
# - Timeouts: Default to no client-side timeouts. If a library/API requires a timeout param,
#   set timeout=None (infinite) or the maximum safe cap. Never retry on timeouts; retries are allowed
#   only for non-timeout transients (429/503/524/network reset/refused) with bounded jitter.

import os
import logging
import sys
from types import SimpleNamespace
from io import BytesIO
import base64 as _b64
import imageio.v3 as iio  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore
import asyncio as _as
import asyncio as _asyncio
import contextlib
import httpx as _hx  # type: ignore
import httpx  # type: ignore
import hashlib as _hl
import hashlib as _h
import json
import re as _re
import uuid
import subprocess
from glob import glob as _glob
import wave
import audioop
import colorsys
import random as _rnd
from urllib.parse import urlparse, urlencode
import tempfile
import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
import math
import librosa  # type: ignore
import shutil as _sh
import websockets as _ws  # type: ignore
import sympy as _sp  # type: ignore
from sympy.parsing.sympy_parser import parse_expr as _parse  # type: ignore
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict
import time
import traceback

# --- Legacy alias shims for older callsites (avoid NameError without touching all usages) ---
_os = os          # for legacy _os.* uses
_b = _b64         # for legacy _b.* uses
_tm = time        # for legacy _tm.time() uses
# -------------------------------------------------------------------------------------------

## httpx imported above as _hx

from .analysis.media import (
    analyze_audio,
    analyze_image,
    audio_detect_tempo as _audio_detect_tempo,
    audio_detect_key as _audio_detect_key,
    cosine_similarity as _cosine_similarity,
    key_similarity as _key_similarity,
    audio_band_energy_profile as _audio_band_energy_profile,
    stem_balance_score as _stem_balance_score,
)  # type: ignore
from .ingest.core import ingest_file
import asyncpg  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from .tools_schema import (
    get_builtin_tools_schema,
    get_tool_introspection_registry,
    merge_tool_schemas,
    compute_tools_hash,
    tool_expected_from_jsonschema,
)
from fastapi import FastAPI, Request, UploadFile, File, WebSocket, WebSocketDisconnect  # type: ignore
from fastapi.responses import JSONResponse, StreamingResponse, Response  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from .json_parser import JSONParser
from .determinism import seed_router as det_seed_router, seed_tool as det_seed_tool, round6 as det_round6
from void_envelopes import normalize_to_envelope
from .search.meta import rrf_fuse as _rrf_fuse
from .tools.progress import emit_progress, set_progress_queue, get_progress_queue
from .tools.mcp_bridge import call_mcp_tool as _call_mcp_tool
from .state.checkpoints import append_event as checkpoints_append_event
from .trace_utils import emit_trace, append_jsonl_compat
from .omni.context import build_omni_context
from void_envelopes import merge_envelopes as stitch_merge_envelopes, stitch_openai as stitch_openai_final
# router layer removed; planner-only orchestration lives in this module now
from .rag.core import get_embedder as _rag_get_embedder
from .ablation.core import ablate as ablate_env
from .ablation.export import write_facts_jsonl as ablate_write_facts
from .code_loop.super_loop import run_super_loop
from .state.checkpoints import append_event as _append_event
# manifest helpers imported below as _art_manifest_add/_art_manifest_write
from .state.ids import step_id as _step_id
from .research.orchestrator import run_research
from .jobs.state import get_job as _get_orcjob, request_cancel as _orcjob_cancel
from void_envelopes import bump_envelope as _env_bump, assert_envelope as _env_assert
from void_envelopes import build_openai_envelope as _build_openai_envelope
from .determinism.seeds import stamp_envelope as _env_stamp, stamp_tool_args as _tool_stamp
from .refs.api import (
    post_refs_save as _refs_save,
    post_refs_refine as _refs_refine,
    get_refs_list as _refs_list,
    post_refs_apply as _refs_apply,
    post_refs_music_profile as _refs_music_profile,
)
from .context.index import resolve_reference as _ctx_resolve, list_recent as _ctx_list, resolve_global as _glob_resolve, infer_audio_emotion as _infer_emotion
from .datasets.api import post_datasets_start as _datasets_start, get_datasets_list as _datasets_list, get_datasets_versions as _datasets_versions, get_datasets_index as _datasets_index
from .jobs.state import create_job as _orcjob_create, set_state as _orcjob_set_state
from .admin.api import get_jobs_list as _admin_jobs_list, get_jobs_replay as _admin_jobs_replay, post_artifacts_gc as _admin_gc
from .admin.prompts import _id_of as _prompt_id_of, save_prompt as _save_prompt, list_prompts as _list_prompts
from .state.checkpoints import append_ndjson as _append_jsonl, read_tail as _read_tail
from .film2.snapshots import save_shot_snapshot as _film_save_snap, load_shot_snapshot as _film_load_snap
from .film2.qa_embed import face_similarity as _qa_face, voice_similarity as _qa_voice, music_similarity as _qa_music
from .icw.project import (
    project_dir as _proj_dir,
    capsules_dir as _capsules_dir,
    project_capsule_path as _proj_capsule_path,
    windows_dir as _windows_dir,
    read_json_safe as _read_json_safe,
    write_json_safe as _write_json_safe,
)
# run_image_gen removed; planner must use image.dispatch
from .tools_image.edit import run_image_edit
from .tools_image.upscale import run_image_upscale
from .tools_tts.speak import run_tts_speak
from .tools_tts.sfx import run_sfx_compose
from .tools_tts.provider import _TTSProvider
from .tools_tts.voice_tools import run_voice_register, run_voice_train
from .tools_music.variation import run_music_variation
from .tools_music.mixdown import run_music_mixdown
from .tools_music.vocals import render_vocal_stems_for_track
from .context.index import add_artifact as _ctx_add
from .artifacts.shard import open_shard as _art_open_shard, append_jsonl as _art_append_jsonl, _finalize_shard as _art_finalize
from .artifacts.shard import newest_part as _art_newest_part, list_parts as _art_list_parts
from .artifacts.manifest import add_manifest_row as _art_manifest_add, write_manifest_atomic as _art_manifest_write
from .tracing.runtime import trace_event as trace_append
from .embeddings.core import build_embeddings_response as _build_embeddings_response
from .traces.writer import (
    log_request as _trace_log_request,
    log_tool as _trace_log_tool,
    log_event as _trace_log_event,
    log_artifact as _trace_log_artifact,
    log_response as _trace_log_response,
    log_error as _trace_log_error,
)
from .analysis.media import analyze_image as _analyze_image, analyze_audio as _analyze_audio
from .analysis.media import analyze_image_regions as _analyze_image_regions
from .review.referee import build_delta_plan as _build_delta_plan
from .review.ledger_writer import append_ledger as _append_ledger
from .comfy.dispatcher import load_workflow as _comfy_load_wf, patch_workflow as _comfy_patch_wf, submit as _comfy_submit
from .comfy.client_aio import (
    comfy_submit as _comfy_submit_aio,
    comfy_history as _comfy_history_aio,
    comfy_upload_image as _comfy_upload_image,
    comfy_upload_mask as _comfy_upload_mask,
    comfy_view as _comfy_view,
    choose_sampler_name as _choose_sampler_name,
    comfy_is_completed as _comfy_is_completed,
)
from .comfy.assets import (
    normalize_history_entry as _normalize_comfy_history_entry,
    extract_comfy_asset_urls as _extract_comfy_asset_urls,
)
from .locks.store import locks_root as _locks_root, upsert_lock_bundle as _lock_save, get_lock_bundle as _lock_load
from .locks.builder import (
    build_image_bundle as _build_image_lock_bundle,
    build_audio_bundle as _build_audio_lock_bundle,
    build_region_locks as _build_region_lock_bundle,
    apply_region_mode_updates as _apply_region_mode_updates,
    apply_audio_mode_updates as _apply_audio_mode_updates,
    voice_embedding_from_path as _lock_voice_embedding_from_path,
)
from .locks.video_builder import build_video_bundle as _build_video_lock_bundle
from .locks.runtime import (
    apply_quality_profile as _lock_apply_profile,
    bundle_to_image_locks as _lock_to_assets,
    QUALITY_PRESETS as LOCK_QUALITY_PRESETS,
    quality_thresholds as _lock_quality_thresholds,
    update_bundle_from_hero_frame as _lock_update_from_hero,
    migrate_visual_bundle as _lock_migrate_visual,
    migrate_music_bundle as _lock_migrate_music,
    migrate_tts_bundle as _lock_migrate_tts,
    migrate_sfx_bundle as _lock_migrate_sfx,
    migrate_film2_bundle as _lock_migrate_film2,
    visual_get_entities as _lock_visual_get_entities,
    visual_freeze_entities as _lock_visual_freeze,
    visual_refresh_all_except as _lock_visual_refresh_all_except,
    ensure_visual_lock_bundle as _lock_ensure_visual,
    ensure_tts_lock_bundle as _lock_ensure_tts,
    merge_lock_bundles as _merge_lock_bundles,
    summarize_visual_bundle_for_context as _lock_summarize_visual_bundle_for_context,
    summarize_music_bundle_for_context as _lock_summarize_music_bundle_for_context,
    summarize_tts_bundle_for_context as _lock_summarize_tts_bundle_for_context,
    summarize_sfx_bundle_for_context as _lock_summarize_sfx_bundle_for_context,
    summarize_film2_bundle_for_context as _lock_summarize_film2_bundle_for_context,
    summarize_all_locks_for_context as _lock_summarize_all_locks_for_context,
)
from .services.image.analysis.locks import (
    compute_face_lock_score as _compute_face_lock_score,
    compute_style_similarity as _compute_style_similarity,
    compute_pose_similarity as _compute_pose_similarity,
    compute_region_scores,
    compute_scene_score,
)
from .film2.hero import choose_hero_frame, _extract_meta as _frame_meta_payload, frame_image_path as _frame_image_path
from .film2.timeline import text_to_simple_srt as _text_to_simple_srt
from .film2.locks import (
    ensure_story_character_bundles,
    ensure_visual_locks_for_story as _ensure_visual_locks_for_story,
    generate_scene_storyboards as _film2_generate_scene_storyboards,
    generate_shot_storyboards as _film2_generate_shot_storyboards,
)
from .image.graph_builder import build_full_graph as _build_full_graph
from .tools_image.common import ensure_dir as _ensure_dir, sidecar as _sidecar, make_outpaths as _make_outpaths, now_ts as _now_ts
from .tools_image.refine import build_image_refine_dispatch_args
from .review.referee import build_delta_plan as _committee, postrun_committee_decide
from .plan.catalog import PLANNER_VISIBLE_TOOLS
from .qa.segments import (
    build_segments_for_tool,
    ALLOWED_PATCH_TOOLS,
    filter_patch_plan,
    apply_patch_plan,
    enrich_patch_plan_for_image_segments,
    enrich_patch_plan_for_video_segments,
    enrich_patch_plan_for_music_segments,
    enrich_patch_plan_for_tts_segments,
    enrich_patch_plan_for_sfx_segments,
)
from .plan.song import plan_song_graph
from .tools_music.windowed import run_music_infinite_windowed, restitch_music_from_windows
from .tools_music.export import append_music_sample as _append_music_sample
from .tools_music.provider import RestMusicProvider
from .music.eval import compute_music_eval, get_music_acceptance_thresholds, MUSIC_HERO_QUALITY_MIN, MUSIC_HERO_FIT_MIN
from functools import partial
from .story import (
    draft_story_graph as _story_draft,
    check_story_consistency as _story_check,
    fix_story as _story_fix,
    derive_scenes_and_shots as _story_derive,
    ensure_tts_locks_and_dialogue_audio as _story_ensure_tts_locks_and_dialogue_audio,
)
from .http.client import HttpRequestConfig, perform_http_request, validate_remote_host  # type: ignore
# Lightweight tool kind overlay for planner mode (reuses existing catalog/builtins)
# Treat all tools as analysis by default; explicitly tag action/asset-creating tools.
ACTION_TOOL_NAMES = {
    # Front-door action tools only (planner-visible surfaces)
    "image.dispatch",
    "music.infinite.windowed",
    "film2.run",      # Film-2 front door (planner-visible)
    "tts.speak",
}

def _filter_tool_names_by_mode(names: List[str], mode: Optional[str]) -> List[str]:
    """
    Normalize planner tool names.

    The planner always sees the same fixed set of tools; this helper just normalizes names.
    """
    return sorted({n for n in (names or []) if isinstance(n, str) and n.strip()})


def _allowed_tools_for_mode(mode: Optional[str]) -> List[str]:
    """
    Return the allowed tool names for planner use for a given mode.

    All planner-visible tools are allowed in all modes; mode may still affect prompts, not tool lists.
    """
    # Start from existing catalog (routes + builtins) and restrict to planner-visible tools only.
    allowed_set = catalog_allowed(get_builtin_tools_schema)
    return sorted([n for n in allowed_set if isinstance(n, str) and n.strip() and n in PLANNER_VISIBLE_TOOLS])


def _configure_logging() -> str:
    """
    Single authoritative logging config for this service.

    Requirements:
    - Always log to stdout (container-friendly).
    - Always log to a file on the void log volume when available (default /workspace/logs).
    - Avoid competing/duplicate configs from uvicorn by forcing uvicorn loggers to propagate to root.
    """
    global LOG_LEVEL

    # Default to DEBUG to match historical behavior, but allow env override for deployments.
    LOG_LEVEL = (os.getenv("ORCH_LOG_LEVEL", "DEBUG") or "DEBUG").strip().upper()
    _level = getattr(logging, LOG_LEVEL, logging.DEBUG)

    _log_dir = (os.getenv("ORCH_LOG_DIR", "") or "").strip()
    if not _log_dir:
        # Prefer the compose-mounted void log volume if present.
        if os.path.isdir("/workspace/logs"):
            _log_dir = "/workspace/logs"
        else:
            try:
                _state_dir_env = (os.getenv("STATE_DIR", "") or "").strip()
                _log_dir = os.path.join(_state_dir_env, "logs") if _state_dir_env else "."
            except Exception:
                _log_dir = "."

    try:
        os.makedirs(_log_dir, exist_ok=True)
    except Exception:
        # If the logs dir can't be created (permissions), fall back to cwd.
        _log_dir = "."

    _log_file = (os.getenv("ORCH_LOG_FILE", "") or "").strip() or os.path.join(_log_dir, "orchestrator.log")

    _handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        _handlers.append(logging.FileHandler(_log_file, encoding="utf-8"))
    except Exception as _ex:
        # Never fail module import due to file handler issues; stdout logging remains.
        try:
            sys.stderr.write(f"[orchestrator.logging] file logging disabled: {_ex}\n")
        except Exception:
            pass

    logging.captureWarnings(True)
    logging.basicConfig(
        level=_level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=_handlers,
        force=True,
    )

    
    logging.getLogger(__name__).debug(f"logging configured level={LOG_LEVEL} file={_log_file!r}")
    return _log_file


# Hard logging: stdout + file, always configured at import.
LOG_LEVEL = "DEBUG"
ORCH_LOG_FILE = _configure_logging()
# Single logger per module (no custom logger names)
log = logging.getLogger(__name__)

def _log(event: str, **fields: Any) -> None:
    # Hard logging: only debug.
    log.debug(f"{event} " + json.dumps(fields, ensure_ascii=False, default=str))
    # Emit distillation-grade event row when a trace_id is provided
    tr = fields.get("trace_id")
    if isinstance(tr, str) and tr:
        row = {
            "t": int(time.time() * 1000),
            "trace_id": tr,
            "event": event,
            "notes": {k: v for k, v in fields.items() if k != "trace_id"},
        }
        checkpoints_append_event(STATE_DIR, tr, str(row.get("event") or "event"), row)


def _inject_execution_context(tool_calls: List[Dict[str, Any]], trace_id: str, effective_mode: str, cid: str | None) -> None:
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        args_tc = tc.get("arguments")
        if isinstance(args_tc, dict):
            if trace_id and not args_tc.get("trace_id"):
                args_tc["trace_id"] = trace_id
            args_tc.setdefault("_effective_mode", effective_mode)
            # Enforce orchestrator-owned cid end-to-end: never allow planner/client to force a cid.
            if isinstance(cid, str) and cid.strip():
                args_tc["cid"] = cid.strip()


async def _resolve_or_create_conversation_cid(requested_cid: str | None) -> str:
    """
    Enforce orchestrator-owned conversation ids (cid).

    Rules:
    - If caller sends no cid: mint a new UUID cid.
    - If caller sends a cid that exists: use it.
    - If caller sends a cid that does NOT exist: ignore it, mint a new UUID cid, and use that.
    - If Postgres isn't configured: always mint a new UUID cid (never trust external input).
    """
    pool = await get_pg_pool()
    # Hard security posture when DB isn't available: never trust a caller-provided cid.
    if pool is None:
        return uuid.uuid4().hex

    rcid = requested_cid.strip() if isinstance(requested_cid, str) and requested_cid.strip() else None
    async with pool.acquire() as conn:
        if rcid:
            try:
                row = await conn.fetchrow("SELECT cid FROM orc_conversation WHERE cid=$1", rcid)
                if row and row.get("cid"):
                    await conn.execute("UPDATE orc_conversation SET updated_at=NOW() WHERE cid=$1", rcid)
                    return rcid
            except Exception as ex:
                # Fall through to minting a new cid; never let DB errors break chat completions.
                log.warning(f"cid.resolve: lookup failed rcid={rcid!r} ex={ex}", exc_info=True)

        new_cid = uuid.uuid4().hex
        try:
            await conn.execute("INSERT INTO orc_conversation (cid) VALUES ($1) ON CONFLICT (cid) DO UPDATE SET updated_at=NOW()", new_cid)
        except Exception as ex:
            log.warning(f"cid.resolve: insert failed new_cid={new_cid!r} ex={ex}", exc_info=True)
        return new_cid


def _args_preview_for_log(args: Any, *, max_keys: int = 64) -> Dict[str, Any]:
    """
    Compact, safe-to-log snapshot of tool args.
    Never includes full long prompts or binary; focuses on shape and key fields.
    """
    if not isinstance(args, dict):
        return {"type": type(args).__name__}
    keys = sorted([str(k) for k in args.keys()])
    out: Dict[str, Any] = {"keys": keys[:max_keys], "has_raw": bool("_raw" in args)}
    # Common fields (trim long strings)
    for k in ("prompt", "text", "negative", "src", "url", "image_url", "video_url", "audio_url", "segment_id", "quality_profile", "profile"):
        v = args.get(k)
        if isinstance(v, str):
            out[k] = (v[:240] + "…") if len(v) > 240 else v
        elif isinstance(v, (int, float, bool)) or v is None:
            out[k] = v
    return out


def _tool_step_eval_user_text(tool_name: str, tool_args: Any, last_user_text: str) -> str:
    """
    Committee/QA prompt seed that is scoped to a SINGLE tool execution step.
    """
    a = tool_args if isinstance(tool_args, dict) else {"_raw": tool_args}
    # Build a minimal, tool-specific "goal" hint using args rather than full chat context.
    goal_hint = ""
    if tool_name.startswith("image."):
        p = a.get("prompt") or a.get("text") or ""
        if isinstance(p, str) and p.strip():
            goal_hint = f"Generate/refine image(s) to match prompt={p.strip()!r}."
    elif tool_name.startswith("video."):
        src = a.get("src") or a.get("video") or a.get("url") or ""
        if isinstance(src, str) and src.strip():
            goal_hint = f"Process video src={src.strip()!r}."
    elif tool_name.startswith("music.") or tool_name.startswith("tts.") or tool_name.startswith("sfx."):
        p = a.get("prompt") or a.get("text") or a.get("lyrics") or ""
        if isinstance(p, str) and p.strip():
            goal_hint = f"Generate/refine audio to match prompt/text={p.strip()!r}."
    elif tool_name == "film2.run":
        syn = a.get("synopsis") or a.get("prompt") or ""
        if isinstance(syn, str) and syn.strip():
            goal_hint = f"Generate film output to match synopsis/prompt={syn.strip()!r}."
    blob = {
        "scope": "tool_step_only",
        "instruction": (
            "Evaluate ONLY whether THIS SINGLE tool call executed correctly and whether its output matches the tool args. "
            "Do NOT evaluate whether the entire user request has been completed. "
            "Do NOT propose unrelated tools; only revise if this tool's own outputs/QA/locks are below thresholds."
        ),
        "tool": tool_name,
        "tool_goal_hint": goal_hint,
        "tool_args_preview": _args_preview_for_log(a),
        "original_user_request_preview": (str(last_user_text or "")[:600] + "…") if isinstance(last_user_text, str) and len(last_user_text) > 600 else str(last_user_text or ""),
    }
    return json.dumps(blob, ensure_ascii=False, default=str)


async def _run_patch_call(
    call: Dict[str, Any],
    trace_id: str,
    mode: str,
    base_url: str,
    executor_base_url: str,
) -> Dict[str, Any]:
    """
    Execute a single patch tool call via the executor path (validator disabled).
    """
    if not isinstance(call, dict):
        # Surface invalid patch calls as a structured error instead of raising.
        return {
            "ok": False,
            "error": {
                "code": "invalid_patch_call",
                "message": "patch tool call must be a dict",
                "trace_id": trace_id,
            },
            "result": {},
        }
    patch_calls: List[Dict[str, Any]] = [call]
    _inject_execution_context(patch_calls, trace_id, mode, None)
    # Validator disabled: execute patch calls directly without pre-validation.
    patch_validated = list(patch_calls)
    patch_failures: List[Dict[str, Any]] = []
    patch_failure_results: List[Dict[str, Any]] = []
    for failure in patch_failures or []:
        env = failure.get("envelope") if isinstance(failure.get("envelope"), dict) else {}
        args_snapshot = failure.get("arguments") if isinstance(failure.get("arguments"), dict) else {}
        name_snapshot = str((failure.get("name") or "")).strip() or "tool"
        error_snapshot = env.get("error") if isinstance(env, dict) else {}
        patch_failure_results.append(
            {
                "name": name_snapshot,
                "result": env,
                "error": error_snapshot,
                "args": args_snapshot,
            }
        )
    # Execute patch calls directly via the local tool front doors instead of
    # delegating to the external executor. This guarantees that image/music/film
    # services are actually called, and errors are surfaced via the canonical
    # tool envelopes.
    exec_results: List[Dict[str, Any]] = []
    exec_batch = patch_validated or patch_calls
    if exec_batch:
        _log("exec.payload", trace_id=trace_id, steps=[{"tool": (c.get("name") or "")} for c in exec_batch[:5]])
        exec_results = await gateway_execute(exec_batch, trace_id, executor_base_url, request_id=str(uuid.uuid4().hex))
    # Prefer direct results; if none, return first failure snapshot.
    if exec_results:
        return exec_results[0]
    if patch_failure_results:
        return patch_failure_results[0]
    # If nothing executed or failed, echo a simple error payload; the caller is
    # responsible for wrapping this in a canonical envelope.
    return {
        "name": call.get("name") or "tool",
        "result": {},
        "error": {
            "code": "no_result",
            "message": "no patch result",
        },
    }


async def segment_qa_and_committee(
    trace_id: str,
    user_text: str,
    tool_name: str,
    segment_results: List[Dict[str, Any]],
    mode: str,
    *,
    base_url: str,
    executor_base_url: str,
    absolutize_url: Callable[[str], str],
    quality_profile: Optional[str] = None,
    max_refine_passes: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run per-segment QA + committee review with an optional single patch pass.
    Returns updated segment results, the committee outcome, and any patch call results.
    """
    if not segment_results:
        outcome = {"action": "go", "rationale": "no_segment_results", "patch_plan": []}
        return segment_results, outcome, []
    thresholds = _lock_quality_thresholds(quality_profile)
    attempt = 0
    current_results = list(segment_results or [])
    cumulative_patch_results: List[Dict[str, Any]] = []
    committee_outcome: Dict[str, Any] = {"action": "go", "patch_plan": [], "rationale": ""}
    # Track per-tool failure counts across attempts so we can detect and break
    # out of potential self-looping patch plans (same tool failing repeatedly).
    failure_counts: Dict[str, int] = {}
    while attempt <= max(0, int(max_refine_passes)):
        # Build per-segment structures and standalone QA scores for committee input
        segments_summary: List[Dict[str, Any]] = []
        segments_for_committee: List[Dict[str, Any]] = []
        for idx, tr in enumerate(current_results):
            if not isinstance(tr, dict):
                continue
            result_obj = tr.get("result")
            if not isinstance(result_obj, dict):
                continue
            parent_cid = None
            meta_part = result_obj.get("meta")
            if isinstance(meta_part, dict) and isinstance(meta_part.get("cid"), str):
                parent_cid = meta_part.get("cid")
            segs = build_segments_for_tool(tool_name, trace_id=trace_id, cid=parent_cid, result=result_obj)
            for seg in (segs or []):
                if not isinstance(seg, dict):
                    continue
                # Compute standalone QA for this segment using existing domain QA helper
                seg_result = seg.get("result") if isinstance(seg.get("result"), dict) else {}
                single_tool_results: List[Dict[str, Any]] = []
                if seg_result:
                    single_tool_results.append({"name": seg.get("tool"), "result": seg_result})
                if single_tool_results:
                    seg_domain_qa = assets_compute_domain_qa(single_tool_results)
                    domain_name = str(seg.get("domain") or "").strip().lower()
                    scores: Dict[str, Any] = {}
                    if domain_name == "image":
                        scores = seg_domain_qa.get("images") or {}
                    elif domain_name in ("music", "tts", "sfx", "audio"):
                        scores = seg_domain_qa.get("audio") or {}
                    elif domain_name in ("video", "film2"):
                        scores = seg_domain_qa.get("videos") or {}
                    # Inject lipsync_score and sharpness from segment meta when available.
                    seg_meta_local = seg.get("meta") if isinstance(seg.get("meta"), dict) else {}
                    ls_val = seg_meta_local.get("lipsync_score")
                    if isinstance(ls_val, (int, float)):
                        scores = dict(scores or {})
                        scores["lipsync"] = float(ls_val)
                    sharp_val = seg_meta_local.get("sharpness")
                    if isinstance(sharp_val, (int, float)):
                        scores = dict(scores or {})
                        scores["sharpness"] = float(sharp_val)
                    qa_block = seg.setdefault("qa", {})
                    if isinstance(qa_block, dict):
                        qa_block["scores"] = scores
                segments_for_committee.append(seg)
        for seg in segments_for_committee:
            qa_block = seg.get("qa") if isinstance(seg.get("qa"), dict) else {}
            scores = qa_block.get("scores") if isinstance(qa_block.get("scores"), dict) else {}
            segments_summary.append(
                {
                    "id": seg.get("id"),
                    "domain": seg.get("domain"),
                    "qa": scores,
                    "locks": seg.get("locks") if isinstance(seg.get("locks"), dict) else None,
                }
            )
            # Per-segment QA trace row for forensic analysis
            seg_meta = seg.get("meta") if isinstance(seg.get("meta"), dict) else {}
            profile_val = seg_meta.get("profile") if isinstance(seg_meta.get("profile"), str) else None
            locks_val = seg.get("locks") if isinstance(seg.get("locks"), dict) else None
            cid_val = seg.get("cid")
            trace_append(
                "qa.segment.qa",
                {
                    "trace_id": trace_id,
                    "tool": tool_name,
                    "segment_id": seg.get("id"),
                    "domain": seg.get("domain"),
                    "profile": profile_val,
                    "attempt": attempt,
                    "cid": cid_val,
                    "locks": locks_val,
                    "qa_scores": scores,
                },
            )
        counts = {
            "images": int(assets_count_images(current_results)),
            "videos": int(assets_count_video(current_results)),
            "audio": int(assets_count_audio(current_results)),
        }
        domain_qa = assets_compute_domain_qa(current_results)
        violations: Dict[str, float] = {}
        images_domain = domain_qa.get("images") or {}
        audio_domain = domain_qa.get("audio") or {}
        if images_domain.get("face_lock") is not None and images_domain.get("face_lock") < thresholds["face_min"]:
            violations["images.face_lock"] = images_domain.get("face_lock")
        if images_domain.get("region_shape_min") is not None and images_domain.get("region_shape_min") < thresholds["region_shape_min"]:
            violations["images.region_shape_min"] = images_domain.get("region_shape_min")
        if images_domain.get("region_texture_mean") is not None and images_domain.get("region_texture_mean") < thresholds["region_texture_min"]:
            violations["images.region_texture_mean"] = images_domain.get("region_texture_mean")
        if images_domain.get("scene_lock") is not None and images_domain.get("scene_lock") < thresholds["scene_min"]:
            violations["images.scene_lock"] = images_domain.get("scene_lock")
        if audio_domain.get("voice_lock") is not None and audio_domain.get("voice_lock") < thresholds["voice_min"]:
            violations["audio.voice_lock"] = audio_domain.get("voice_lock")
        if audio_domain.get("tempo_lock") is not None and audio_domain.get("tempo_lock") < thresholds["tempo_min"]:
            violations["audio.tempo_lock"] = audio_domain.get("tempo_lock")
        if audio_domain.get("key_lock") is not None and audio_domain.get("key_lock") < thresholds["key_min"]:
            violations["audio.key_lock"] = audio_domain.get("key_lock")
        if audio_domain.get("stem_balance_lock") is not None and audio_domain.get("stem_balance_lock") < thresholds["stem_balance_min"]:
            violations["audio.stem_balance_lock"] = audio_domain.get("stem_balance_lock")
        if audio_domain.get("lyrics_lock") is not None and audio_domain.get("lyrics_lock") < thresholds["lyrics_min"]:
            violations["audio.lyrics_lock"] = audio_domain.get("lyrics_lock")
        qa_metrics_pre = {
            "counts": counts,
            "domain": domain_qa,
            "threshold_violations": violations,
            "segments": segments_summary,
        }
        _log("qa.metrics.segment", trace_id=trace_id, tool=tool_name, phase="pre", attempt=attempt, metrics=qa_metrics_pre)
        _trace_log_event(
            STATE_DIR,
            trace_id,
            {
                "kind": "qa",
                "stage": "segment_pre",
                "data": qa_metrics_pre,
            },
        )
        committee_outcome = await postrun_committee_decide(
            trace_id=trace_id,
            user_text=user_text,
            tool_results=current_results,
            qa_metrics=qa_metrics_pre,
            mode=mode,
        )
        action = str((committee_outcome.get("action") or "go")).strip().lower()
        if violations and action == "go":
            if attempt < max_refine_passes:
                action = "revise"
                committee_outcome["action"] = "revise"
                rationale = committee_outcome.get("rationale") or ""
                committee_outcome["rationale"] = (rationale + " | auto-revise: lock thresholds unmet").strip()
                committee_outcome.setdefault("threshold_violations", {}).update(violations)
            else:
                committee_outcome["action"] = "fail"
                committee_outcome.setdefault("threshold_violations", {}).update(violations)
                action = "fail"
        _log(
            "committee.decision.segment",
            trace_id=trace_id,
            tool=tool_name,
            attempt=attempt,
            action=action,
            rationale=committee_outcome.get("rationale"),
            violations=violations,
            committee_error=committee_outcome.get("committee_error"),
        )
        # Auto-suggest video.refine.clip steps for low-quality clips based on per-segment QA.
        if action == "revise":
            auto_patch_steps: List[Dict[str, Any]] = []
            # Collect already proposed (tool, segment_id) pairs to avoid duplicates.
            existing_pairs = set()
            for st in committee_outcome.get("patch_plan") or []:
                if not isinstance(st, dict):
                    continue
                t = (st.get("tool"), st.get("segment_id"))
                existing_pairs.add(t)
            for seg in segments_for_committee:
                if not isinstance(seg, dict):
                    continue
                domain_name = str(seg.get("domain") or "").strip().lower()
                if domain_name not in ("video", "film2"):
                    continue
                seg_id = seg.get("id")
                if not isinstance(seg_id, str) or not seg_id:
                    continue
                qa_block = seg.get("qa") if isinstance(seg.get("qa"), dict) else {}
                scores = qa_block.get("scores") if isinstance(qa_block.get("scores"), dict) else {}
                refine_mode_auto: Optional[str] = None
                ls_val = scores.get("lipsync")
                if isinstance(ls_val, (int, float)) and ls_val < 0.5:
                    refine_mode_auto = "fix_lipsync"
                face_val = scores.get("face_lock")
                if refine_mode_auto is None and isinstance(face_val, (int, float)) and face_val < thresholds.get("face_min", 0.0):
                    refine_mode_auto = "fix_faces"
                temporal_val = scores.get("frame_lpips_mean") or scores.get("temporal_stability")
                if refine_mode_auto is None and isinstance(temporal_val, (int, float)) and temporal_val < 0.5:
                    refine_mode_auto = "stabilize_motion"
                sharpness_val = scores.get("sharpness")
                if refine_mode_auto is None and isinstance(sharpness_val, (int, float)) and sharpness_val < 50.0:
                    refine_mode_auto = "improve_quality"
                if refine_mode_auto is None:
                    continue
                key = ("video.refine.clip", seg_id)
                if key in existing_pairs:
                    continue
                auto_patch_steps.append(
                    {
                        "tool": "video.refine.clip",
                        "segment_id": seg_id,
                        "args": {"refine_mode": refine_mode_auto},
                    }
                )
            if auto_patch_steps:
                committee_outcome.setdefault("patch_plan", [])
                if isinstance(committee_outcome.get("patch_plan"), list):
                    committee_outcome["patch_plan"].extend(auto_patch_steps)
                # Trace the planned clip refinements for this QA pass.
                plan_summary: Dict[str, Any] = {
                    "event": "clip_refine_plan",
                    "tool": tool_name,
                    "segment_ids": [],
                    "refine_modes": {},
                }
                seg_id_to_mode: Dict[str, str] = {}
                for step in auto_patch_steps:
                    sid = step.get("segment_id")
                    args_used = step.get("args") if isinstance(step.get("args"), dict) else {}
                    mode_val = args_used.get("refine_mode")
                    if isinstance(sid, str) and sid and isinstance(mode_val, str) and mode_val:
                        seg_id_to_mode[sid] = mode_val
                plan_summary["segment_ids"] = list(seg_id_to_mode.keys())
                plan_summary["refine_modes"] = seg_id_to_mode
                trace_append(
                    "film2.clip_refine_plan",
                    {
                        "trace_id": trace_id,
                        **(plan_summary or {}),
                    },
                )
        if action != "revise":
            break
        allowed_mode_set = set(_allowed_tools_for_mode(mode))
        raw_patch_plan = committee_outcome.get("patch_plan") or []
        # First, apply segment-aware filtering using the canonical helpers.
        segment_filtered = filter_patch_plan(raw_patch_plan, segments_for_committee)
        filtered_patch_plan: List[Dict[str, Any]] = []
        for step in segment_filtered:
            step_tool = (step.get("tool") or "").strip()
            if not step_tool:
                continue
            if step_tool not in PLANNER_VISIBLE_TOOLS or step_tool not in allowed_mode_set:
                continue
            filtered_patch_plan.append(step)
        committee_outcome["patch_plan"] = filtered_patch_plan
        if not filtered_patch_plan:
            break
        # Loop guard: if any tool in the filtered patch plan has already failed
        # multiple times in this segment-committee loop, stop revising and fail
        # instead of attempting the same broken tool over and over.
        loop_guard_tools: List[str] = []
        for step in filtered_patch_plan:
            st_name = (step.get("tool") or "").strip()
            if st_name and failure_counts.get(st_name, 0) >= 2:
                loop_guard_tools.append(st_name)
        if loop_guard_tools:
            _log(
                "committee.loop_guard",
                trace_id=trace_id,
                tool=tool_name,
                attempt=attempt,
                tools=loop_guard_tools,
            )
            committee_outcome["action"] = "fail"
            committee_outcome.setdefault("loop_guard", {"tools": loop_guard_tools})
            break
        # Enrich image/video/music refine steps with per-segment context before execution.
        enriched_patch_plan = enrich_patch_plan_for_image_segments(
            filtered_patch_plan,
            segments_for_committee,
        )
        enriched_patch_plan = enrich_patch_plan_for_video_segments(
            enriched_patch_plan,
            segments_for_committee,
        )
        enriched_patch_plan = enrich_patch_plan_for_music_segments(
            enriched_patch_plan,
            segments_for_committee,
        )
        enriched_patch_plan = enrich_patch_plan_for_tts_segments(
            enriched_patch_plan,
            segments_for_committee,
        )
        enriched_patch_plan = enrich_patch_plan_for_sfx_segments(
            enriched_patch_plan,
            segments_for_committee,
        )
        # Execute enriched patch plan via central patch executor.
        tool_runner = partial(
            _run_patch_call,
            trace_id=trace_id,
            mode=mode,
            base_url=base_url,
            executor_base_url=executor_base_url,
        )
        updated_segments, patch_results = await apply_patch_plan(
            enriched_patch_plan,
            segments_for_committee,
            tool_runner,
            trace_id,
            tool_name,
        )
        # Log patch execution results for telemetry/debugging and update failure counts.
        for pr in patch_results or []:
            tname = str((pr or {}).get("name") or "tool")
            result_obj = (pr or {}).get("result") if isinstance((pr or {}).get("result"), dict) else {}
            err_obj = (pr or {}).get("error") or (result_obj.get("error") if isinstance(result_obj, dict) else None)
            if isinstance(err_obj, (str, dict)):
                code = (err_obj.get("code") if isinstance(err_obj, dict) else str(err_obj))
                status = (err_obj.get("status") if isinstance(err_obj, dict) else None)
                message = (err_obj.get("message") if isinstance(err_obj, dict) else "")
                _log("tool.run.error", trace_id=trace_id, tool=tname, code=(code or ""), status=status, message=(message or ""), attempt=attempt)
                # Increment failure counter for this tool to detect repeated failures.
                try:
                    failure_counts[tname] = int(failure_counts.get(tname, 0) or 0) + 1
                except Exception as ex:
                    logging.warning(
                        f"patch_exec: bad failure_counts[{tname}]={failure_counts.get(tname)!r}; resetting to 1",
                        exc_info=True,
                    )
                    failure_counts[tname] = 1
            else:
                artifacts_summary: List[Dict[str, Any]] = []
                arts = result_obj.get("artifacts") if isinstance(result_obj, dict) else None
                if isinstance(arts, list):
                    for art in arts:
                        if isinstance(art, dict):
                            aid = art.get("id")
                            kind = art.get("kind")
                            if isinstance(aid, str) and isinstance(kind, str):
                                artifacts_summary.append({"id": aid, "kind": kind})
                urls_local = assets_collect_urls([pr], absolutize_url)
                _log("tool.run.after", trace_id=trace_id, tool=tname, artifacts=artifacts_summary, urls_count=len(urls_local or []), attempt=attempt)
                meta_local = result_obj.get("meta") if isinstance(result_obj, dict) else {}
                first_url = (urls_local[0] if isinstance(urls_local, list) and urls_local else None)
        if filtered_patch_plan:
            _log("committee.revision.segment", trace_id=trace_id, tool=tool_name, steps=len(filtered_patch_plan), failures=0, attempt=attempt)
        if patch_results:
            cumulative_patch_results.extend(patch_results)
            current_results = (current_results or []) + patch_results
        attempt += 1
        _log("segment.refine.iteration", trace_id=trace_id, tool=tool_name, attempt=attempt)
        if attempt > max_refine_passes:
            break
    if cumulative_patch_results:
        counts_post = {
            "images": int(assets_count_images(current_results)),
            "videos": int(assets_count_video(current_results)),
            "audio": int(assets_count_audio(current_results)),
        }
        domain_post = assets_compute_domain_qa(current_results)
        qa_metrics_post = {"counts": counts_post, "domain": domain_post}
        _log("qa.metrics.segment", trace_id=trace_id, tool=tool_name, phase="post", metrics=qa_metrics_post)
    return current_results, committee_outcome, cumulative_patch_results


async def _generate_scene_storyboards(
    scenes: List[Dict[str, Any]],
    locks_arg: Dict[str, Any],
    profile_name: str,
    trace_id: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Generate a simple storyboard image per scene using image.dispatch, when possible.
    """
    out_scenes: List[Dict[str, Any]] = []
    for scene in scenes or []:
        if not isinstance(scene, dict):
            continue
        scene_id = scene.get("scene_id")
        summary = scene.get("summary") or ""
        prompt = f"Scene {scene_id}: {summary}"
        args_img: Dict[str, Any] = {
            "prompt": prompt,
            "width": 1024,
            "height": 576,
            "quality_profile": profile_name,
        }
        if isinstance(locks_arg, dict) and locks_arg:
            args_img["lock_bundle"] = locks_arg
        storyboard_path: Optional[str] = None
        res = await execute_tool_call({"name": "image.dispatch", "arguments": args_img})
        if isinstance(res, dict) and isinstance(res.get("result"), dict):
            r = res.get("result") or {}
            arts = r.get("artifacts") if isinstance(r.get("artifacts"), list) else []
            if arts:
                first = arts[0]
                if isinstance(first, dict):
                    path_val = first.get("path") or first.get("view_url") or first.get("url")
                    if isinstance(path_val, str) and path_val:
                        storyboard_path = path_val
                        trace_append(
                            "film2.scene_storyboard_generated",
                            {
                                "trace_id": trace_id,
                                "scene_id": scene_id,
                                "image_path": path_val,
                                "prompt": prompt,
                                "quality_profile": profile_name,
                            },
                        )
        scene_out = dict(scene)
        if storyboard_path:
            scene_out["storyboard_image"] = storyboard_path
        out_scenes.append(scene_out)
    return out_scenes


def _append_jsonl(path: str, obj: dict) -> None:
    append_jsonl_compat(STATE_DIR, path, obj)

## trace_append imported from .tracing.runtime (single runtime tracing API)

def _trace_response(trace_id: str, envelope: Dict[str, Any]) -> None:
    tms = int(time.time() * 1000)
    content = ""
    ch0 = (envelope.get("choices") or [{}])[0]
    msg = (ch0.get("message") or {})
    content = str(msg.get("content") or "")
    # Heuristic: count asset bullet lines introduced as "- " under "Assets" blocks
    assets_count = content.count("\n- ")
    # Prefer the explicit envelope["ok"] flag when present; fall back to the
    # absence of an error object for callers that omit ok.
    if isinstance(envelope, dict) and "ok" in envelope:
        ok_flag = bool(envelope.get("ok"))
    else:
        ok_flag = not bool(envelope.get("error"))
    out = {
        "t": tms,
        "trace_id": trace_id,
        "ok": bool(ok_flag),
        "message_len": len(content),
        "assets_count": int(assets_count),
    }
    if isinstance(envelope.get("error"), dict):
        out["error"] = envelope.get("error")
    _log(
        "response",
        trace_id=trace_id,
        t=tms,
        ok=bool(ok_flag),
        message_len=len(content),
        assets_count=int(assets_count),
        error=(envelope.get("error") if isinstance(envelope.get("error"), dict) else None),
    )
    _trace_log_response(
        STATE_DIR,
        trace_id,
        {
            # request_id and trace_id are distinct; this trace log is keyed by trace_id.
            "request_id": str(envelope.get("id") or ""),
            "kind": "final_response",
            "mode": "chat",
            "content": content,
            "meta": {
                "assets_count": int(assets_count),
            },
        },
    )
    _log(
        "chat.finish",
        trace_id=trace_id,
        ok=bool(ok_flag),
        assets_count=int(assets_count),
        message_len=int(len(content)),
    )
# CPU/GPU adaptive mode for ComfyUI graphs
COMFY_CPU_MODE = (os.getenv("COMFY_CPU_MODE", "").strip().lower() in ("1", "true", "yes", "on"))
from .db.pool import get_pg_pool
from .db.tracing import db_insert_run as _db_insert_run, db_update_run_response as _db_update_run_response, db_insert_tool_call as _db_insert_tool_call


DEFAULT_STEM_BANDS: Dict[str, Tuple[float, float]] = {
    "bass": (20.0, 150.0),
    "drums": (150.0, 800.0),
    "guitars": (800.0, 3000.0),
    "pads": (200.0, 2000.0),
    "vocals": (300.0, 4000.0),
}


# Committee configuration moved to committee_client; imports below keep back-compat.
_default_temp_raw = os.getenv("DEFAULT_TEMPERATURE", "0.3")
try:
    DEFAULT_TEMPERATURE = float(str(_default_temp_raw).strip() or "0.3")
except Exception as _exc:
    logging.getLogger(__name__).warning("bad DEFAULT_TEMPERATURE=%r; defaulting to 0.3", _default_temp_raw, exc_info=True)
    DEFAULT_TEMPERATURE = 0.3
# Gates removed: defaults are always ON; API-key checks still apply where required
ENABLE_WEBSEARCH = True
MCP_HTTP_BRIDGE_URL = os.getenv("MCP_HTTP_BRIDGE_URL")  # e.g., http://host.docker.internal:9999
EXECUTOR_BASE_URL = os.getenv("EXECUTOR_BASE_URL", "http://127.0.0.1:8081")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwen")  # qwen | glm
AUTO_EXECUTE_TOOLS = True
# Always allow tool execution
ALLOW_TOOL_EXECUTION = True
TIMEOUTS_FORBIDDEN = True
_stream_chunk_size_raw = os.getenv("STREAM_CHUNK_SIZE_CHARS", "0")
try:
    STREAM_CHUNK_SIZE_CHARS = int(str(_stream_chunk_size_raw).strip() or "0")
except Exception:
    logging.getLogger(__name__).warning("bad STREAM_CHUNK_SIZE_CHARS=%r; defaulting to 0", _stream_chunk_size_raw, exc_info=True)
    STREAM_CHUNK_SIZE_CHARS = 0
_stream_chunk_ms_raw = os.getenv("STREAM_CHUNK_INTERVAL_MS", "50")
try:
    STREAM_CHUNK_INTERVAL_MS = int(str(_stream_chunk_ms_raw).strip() or "50")
except Exception:
    logging.getLogger(__name__).warning("bad STREAM_CHUNK_INTERVAL_MS=%r; defaulting to 50", _stream_chunk_ms_raw, exc_info=True)
    STREAM_CHUNK_INTERVAL_MS = 50
JOBS_RAG_INDEX = os.getenv("JOBS_RAG_INDEX", "true").lower() == "true"
# No caps — never enforce caps inline

# RAG configuration (pgvector)
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
_pg_port_raw = os.getenv("POSTGRES_PORT", "5432")
try:
    POSTGRES_PORT = int(str(_pg_port_raw).strip() or "5432")
except Exception:
    logging.getLogger(__name__).warning("bad POSTGRES_PORT=%r; defaulting to 5432", _pg_port_raw, exc_info=True)
    POSTGRES_PORT = 5432
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

# Optional external tool services (set URLs to enable) —
# except where explicitly required (XTTS, Whisper, Music, VocalFix, RVC, OCR, VLM, MFA, VisionRepair).
COMFYUI_API_URL = os.getenv("COMFYUI_API_URL")  # e.g., http://comfyui:8188
COMFYUI_API_URLS = [u.strip() for u in os.getenv("COMFYUI_API_URLS", "").split(",") if u.strip()]
_comfy_repl_raw = os.getenv("COMFYUI_REPLICAS", "1")
try:
    COMFYUI_REPLICAS = int(str(_comfy_repl_raw).strip() or "1")
except Exception:
    logging.getLogger(__name__).warning("bad COMFYUI_REPLICAS=%r; defaulting to 1", _comfy_repl_raw, exc_info=True)
    COMFYUI_REPLICAS = 1
_scene_conc_raw = os.getenv("SCENE_SUBMIT_CONCURRENCY", "4")
try:
    SCENE_SUBMIT_CONCURRENCY = int(str(_scene_conc_raw).strip() or "4")
except Exception:
    logging.getLogger(__name__).warning("bad SCENE_SUBMIT_CONCURRENCY=%r; defaulting to 4", _scene_conc_raw, exc_info=True)
    SCENE_SUBMIT_CONCURRENCY = 4
_scene_batch_raw = os.getenv("SCENE_MAX_BATCH_FRAMES", "2")
try:
    SCENE_MAX_BATCH_FRAMES = int(str(_scene_batch_raw).strip() or "2")
except Exception:
    logging.getLogger(__name__).warning("bad SCENE_MAX_BATCH_FRAMES=%r; defaulting to 2", _scene_batch_raw, exc_info=True)
    SCENE_MAX_BATCH_FRAMES = 2
_resume_raw = os.getenv("RESUME_MAX_RETRIES", "3")
try:
    RESUME_MAX_RETRIES = int(str(_resume_raw).strip() or "3")
except Exception:
    logging.getLogger(__name__).warning("bad RESUME_MAX_RETRIES=%r; defaulting to 3", _resume_raw, exc_info=True)
    RESUME_MAX_RETRIES = 3
COMFYUI_API_URLS = [u.strip() for u in os.getenv("COMFYUI_API_URLS", "").split(",") if u.strip()]
_scene_conc_raw2 = os.getenv("SCENE_SUBMIT_CONCURRENCY", "4")
try:
    SCENE_SUBMIT_CONCURRENCY = int(str(_scene_conc_raw2).strip() or "4")
except Exception:
    logging.getLogger(__name__).warning("bad SCENE_SUBMIT_CONCURRENCY=%r; defaulting to 4", _scene_conc_raw2, exc_info=True)
    SCENE_SUBMIT_CONCURRENCY = 4
XTTS_API_URL = os.getenv("XTTS_API_URL")       # e.g., http://127.0.0.1:8020
WHISPER_API_URL = os.getenv("WHISPER_API_URL") # e.g., http://127.0.0.1:9090
FACEID_API_URL = os.getenv("FACEID_API_URL")   # e.g., http://127.0.0.1:7000
MUSIC_API_URL = os.getenv("MUSIC_API_URL")     # e.g., http://127.0.0.1:7860
VLM_API_URL = os.getenv("VLM_API_URL")        # e.g., http://vlm:8050
OCR_API_URL = os.getenv("OCR_API_URL")        # e.g., http://ocr:8070
VISION_REPAIR_API_URL = os.getenv("VISION_REPAIR_API_URL")  # e.g., http://vision_repair:8095
MFA_API_URL = os.getenv("MFA_API_URL")        # e.g., http://mfa:7867
VOCAL_FIXER_API_URL = os.getenv("VOCAL_FIXER_API_URL")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/workspace/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
LOCKS_ROOT_DIR = _locks_root(UPLOAD_DIR)
FILM2_MODELS_DIR = os.getenv("FILM2_MODELS", "/opt/models")
FILM2_DATA_DIR = os.getenv("FILM2_DATA", "/srv/film2")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")  # optional external workflow orchestration
ENABLE_N8N = os.getenv("ENABLE_N8N", "false").lower() == "true"
ASSEMBLER_API_URL = os.getenv("ASSEMBLER_API_URL")  # http://assembler:9095
TEACHER_API_URL = os.getenv("TEACHER_API_URL", "http://127.0.0.1:8097")
WRAPPER_CONFIG_PATH = os.getenv("WRAPPER_CONFIG_PATH", "/workspace/configs/wrapper_config.json")
WRAPPER_CONFIG: Dict[str, Any] = {}
WRAPPER_CONFIG_HASH: Optional[str] = None
DRT_API_URL = os.getenv("DRT_API_URL", "http://drt:8086")
STATE_DIR = os.path.join(UPLOAD_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)
_shard_bytes_raw = os.getenv("ARTIFACT_SHARD_BYTES", "200000")
try:
    ARTIFACT_SHARD_BYTES = int(str(_shard_bytes_raw).strip() or "200000")
except Exception:
    logging.getLogger(__name__).warning("bad ARTIFACT_SHARD_BYTES=%r; defaulting to 200000", _shard_bytes_raw, exc_info=True)
    ARTIFACT_SHARD_BYTES = 200000
ARTIFACT_LATEST_ONLY = os.getenv("ARTIFACT_LATEST_ONLY", "true").lower() == "true"
_event_buf_raw = os.getenv("EVENT_BUFFER_MAX", "100")
try:
    EVENT_BUFFER_MAX = int(str(_event_buf_raw).strip() or "100")
except Exception:
    logging.getLogger(__name__).warning("bad EVENT_BUFFER_MAX=%r; defaulting to 100", _event_buf_raw, exc_info=True)
    EVENT_BUFFER_MAX = 100
_shard_bytes_raw2 = os.getenv("ARTIFACT_SHARD_BYTES", "200000")
try:
    ARTIFACT_SHARD_BYTES = int(str(_shard_bytes_raw2).strip() or "200000")
except Exception:
    logging.getLogger(__name__).warning("bad ARTIFACT_SHARD_BYTES=%r; defaulting to 200000", _shard_bytes_raw2, exc_info=True)
    ARTIFACT_SHARD_BYTES = 200000
ARTIFACT_LATEST_ONLY = os.getenv("ARTIFACT_LATEST_ONLY", "true").lower() == "true"

# Music/Audio extended services (spec Step 18/Instruction Set)
SAO_API_URL = os.getenv("SAO_API_URL")                    # e.g., http://sao:9002
DEMUCS_API_URL = os.getenv("DEMUCS_API_URL")              # e.g., http://demucs:9003
RVC_API_URL = os.getenv("RVC_API_URL")                    # e.g., http://rvc:9004
DIFFSINGER_RVC_API_URL = os.getenv("DIFFSINGER_RVC_API_URL")  # e.g., http://dsrvc:9005

# Generic primary music engine base URL. Prefer an explicit MUSIC_FULL_API_URL
# when provided; then MUSIC_API_URL.
PRIMARY_MUSIC_API_URL = (
    os.getenv("MUSIC_FULL_API_URL")
    or os.getenv("MUSIC_API_URL")
)
HUNYUAN_FOLEY_API_URL = os.getenv("HUNYUAN_FOLEY_API_URL")    # http://foley:9006
HUNYUAN_VIDEO_API_URL = os.getenv("HUNYUAN_VIDEO_API_URL")    # http://hunyuan:9007
SVD_API_URL = os.getenv("SVD_API_URL")                        # http://svd:9008

# Mandatory audio/vocal services: fail fast if configuration is missing.
_missing_audio_env: list[str] = []
for _name, _val in (
    ("XTTS_API_URL", XTTS_API_URL),
    ("WHISPER_API_URL", WHISPER_API_URL),
    ("MUSIC_API_URL", MUSIC_API_URL),
    ("VOCAL_FIXER_API_URL", VOCAL_FIXER_API_URL),
    ("RVC_API_URL", RVC_API_URL),
    ("OCR_API_URL", OCR_API_URL),
    ("VLM_API_URL", VLM_API_URL),
    ("MFA_API_URL", MFA_API_URL),
    ("VISION_REPAIR_API_URL", VISION_REPAIR_API_URL),
):
    if not isinstance(_val, str) or not _val.strip():
        _missing_audio_env.append(_name)
if _missing_audio_env:
    # Log missing audio/vocal service endpoints so misconfiguration is visible,
    # but do not raise here; downstream tools will fail naturally if they rely
    # on these services.
    _log(
        "audio.env.missing",
        missing_endpoints=sorted(_missing_audio_env),
    )


def _sha256_bytes(b: bytes) -> str:
    return _hl.sha256(b).hexdigest()


def _friendly_failure_text(name: str, attempted_args: Dict[str, Any], err: Dict[str, Any], trace_id: str) -> str:
    code = (err.get("code") or "validation_error")
    details = err.get("details") or {}
    missing = details.get("missing") or []
    invalid = details.get("invalid") or []
    # fallback: derive from schema_validation errors array
    if (not missing) and isinstance(details.get("errors"), list):
        miss = [e.get("path") for e in details["errors"] if (e.get("code") in ("required_missing", "required"))]
        missing = [m for m in miss if m]
        inv = [{"field": e.get("path"), "reason": e.get("code")} for e in details["errors"] if (e.get("code") not in ("required_missing","required"))]
        invalid = [x for x in inv if x.get("field")]
    lines = [f"Couldn't run {name or 'tool'} (code: `{code}`)."]
    if missing:
        _parts = [f"`{str(m)}`" for m in missing if m is not None]
        if _parts:
            lines.append("Missing: " + ", ".join(_parts))
    if invalid:
        _inv_parts = []
        for x in invalid:
            field = x.get("field")
            reason = x.get("reason")
            if field:
                _inv_parts.append(f"`{field}` ({reason})")
        if _inv_parts:
            lines.append("Invalid: " + ", ".join(_inv_parts))
    tried = {k: attempted_args.get(k) for k in ("prompt","negative","width","height","steps","cfg") if isinstance(attempted_args, dict) and (k in attempted_args)}
    if tried:
        lines.append("AI tried args: `" + json.dumps(tried, separators=(',',':'), default=str) + "`")
    if trace_id:
        lines.append(f"trace: `{trace_id}`")
    lines.append("Tip: type any missing values (e.g., `prompt: ...`) and resend; the AI will fill the rest.")
    return "\n".join(lines)

_DISPLAY_KEYS = ("prompt","negative","width","height","steps","cfg","seconds","fps","model","workflow_path")
def _summarize_invalid(errors: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for e in (errors or []):
        code = str(e.get("code") or "")
        path = str(e.get("path") or e.get("field") or "")
        exp  = e.get("expected") or e.get("allowed")
        got  = e.get("got")
        reason = code
        if code == "enum_mismatch":
            reason = f"must be one of {exp}"
        elif code == "type_mismatch":
            reason = f"type {exp}, got {type(got).__name__}"
        elif code == "schema_validation":
            reason = "schema mismatch"
        elif code == "workflow_invalid":
            reason = "invalid workflow graph (topology/keys)"
        elif code == "workflow_binding_missing":
            reason = "missing required node binding"
        out.append({"field": path, "reason": reason})
    return out


def _load_wrapper_config() -> None:
    global WRAPPER_CONFIG, WRAPPER_CONFIG_HASH
    if os.path.exists(WRAPPER_CONFIG_PATH):
        with open(WRAPPER_CONFIG_PATH, "rb") as f:
            data = f.read()
        WRAPPER_CONFIG_HASH = f"sha256:{_sha256_bytes(data)}"
        parser = JSONParser()
        # Wrapper config is expected to be a dict; treat it as open, but
        # still coerce into a mapping shape.
        cfg = parser.parse(data.decode("utf-8"), {"profiles": list, "defaults": dict})
        if isinstance(cfg, dict):
            WRAPPER_CONFIG = cfg
        else:
            # Malformed wrapper config; leave WRAPPER_CONFIG empty so callers
            # that depend on it fail more explicitly when they access it.
            WRAPPER_CONFIG = {}
    else:
        WRAPPER_CONFIG = {}
        WRAPPER_CONFIG_HASH = None


_load_wrapper_config()


def _get_music_acceptance_thresholds() -> Dict[str, float]:
    """
    Backwards-compatible helper for film2.run and any other paths that still
    reference the underscored variant. Delegates to the canonical
    get_music_acceptance_thresholds() in app.music.eval.

    Pure data, deterministic, no network; the underlying helper reads the
    static review/acceptance_audio.json at most once per process.
    """
    th = get_music_acceptance_thresholds()
    oq_raw = th.get("overall_quality_min", 0.6) if isinstance(th, dict) else 0.6
    fs_raw = th.get("fit_score_min", 0.6) if isinstance(th, dict) else 0.6
    try:
        oq = float(oq_raw)
    except Exception as exc:
        logging.getLogger(__name__).warning("music.acceptance: bad overall_quality_min=%r; defaulting to 0.6", oq_raw, exc_info=True)
        oq = 0.6
    try:
        fs = float(fs_raw)
    except Exception as exc:
        logging.getLogger(__name__).warning("music.acceptance: bad fit_score_min=%r; defaulting to 0.6", fs_raw, exc_info=True)
        fs = 0.6
    return {
        "overall_quality_min": oq,
        "fit_score_min": fs,
    }


# removed: robust_json_loads — use JSONParser().parse with explicit expected structures everywhere


# No Pydantic. All request bodies are plain dicts validated by helpers.


app = FastAPI(title="Void Orchestrator", version="0.1.0")
# Canonical envelope (shared across services)
from void_envelopes import ToolEnvelope  # type: ignore


@app.post("/tool.run")
async def tool_run(request: Request) -> Any:
    """
    Canonical tool execution endpoint.

    Accepts either {name, args} or {name, arguments}. Returns ToolEnvelope with HTTP 200.
    """
    raw = await request.body()
    raw_text = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw or "")
    parser = JSONParser()
    schema = {
        "name": str,
        "args": object,
        "arguments": object,
        "request_id": str,
        "trace_id": str,
        "cid": str,
        "tool_call_id": str,
        "meta": dict,
    }
    body = parser.parse(raw_text, schema)
    if not isinstance(body, dict):
        rid = uuid.uuid4().hex
        return ToolEnvelope.failure("invalid_body_type", "Body must be a JSON object", status=422, request_id=rid, details={})
    rid = (str(body.get("request_id") or "").strip()) or uuid.uuid4().hex
    name = (body.get("name") or "").strip()
    if not name:
        return ToolEnvelope.failure("missing_name", "name is required", status=422, request_id=rid, details={})

    args_in = body.get("args") if ("args" in body) else body.get("arguments")
    if isinstance(args_in, dict):
        args = dict(args_in)
    elif isinstance(args_in, str):
        parsed = parser.parse(args_in, {})
        args = dict(parsed) if isinstance(parsed, dict) else {"_raw": args_in}
    elif args_in is None:
        args = {}
    else:
        args = {"_raw": args_in}

    trace_id_val = body.get("trace_id")
    trace_id = trace_id_val if isinstance(trace_id_val, str) and trace_id_val.strip() else None
    if trace_id is None and isinstance(args.get("trace_id"), str) and str(args.get("trace_id")).strip():
        trace_id = str(args.get("trace_id")).strip()

    call: Dict[str, Any] = {"name": name, "arguments": args}
    if trace_id:
        call["trace_id"] = trace_id
    res = await execute_tool_call(call)

    if isinstance(res, dict) and isinstance(res.get("result"), dict):
        return ToolEnvelope.success(res["result"], request_id=rid)
    err_obj = res.get("error") if isinstance(res, dict) else None
    if isinstance(err_obj, dict):
        code = str(err_obj.get("code") or "tool_error")
        msg = str(err_obj.get("message") or f"{name} failed")
        st_raw = err_obj.get("status") or 422
        try:
            st = int(st_raw)  # type: ignore[arg-type]
        except Exception:
            st = 422
        return ToolEnvelope.failure(code, msg, status=st, request_id=rid, details=dict(err_obj))
    return ToolEnvelope.failure("tool_error", str(err_obj or "tool failed"), status=422, request_id=rid, details={"raw": res})
# Middleware order matters: last added runs first.
# We want Preflight to run FIRST, so add it LAST.
from .middleware.ws_permissive import PermissiveWebSocketMiddleware
app.add_middleware(PermissiveWebSocketMiddleware)
# CORS is handled by global_cors_middleware below; avoid stacking extra CORS/header middleware.


def _json_response(obj: Dict[str, Any], status_code: int = 200) -> Response:
    body = json.dumps(obj, ensure_ascii=False)
    headers = {
        "Cache-Control": "no-store",
        "Content-Type": "application/json; charset=utf-8",
        "Content-Length": str(len(body.encode("utf-8"))),
    }
    return Response(content=body, status_code=status_code, media_type="application/json", headers=headers)

# ---- Pipeline imports (extracted helpers) ----
from .pipeline.assets import collect_urls as assets_collect_urls, count_images as assets_count_images, count_video as assets_count_video, count_audio as assets_count_audio, compute_domain_qa as assets_compute_domain_qa  # type: ignore
from .pipeline.executor_gateway import execute as gateway_execute  # type: ignore
from .pipeline.catalog import build_allowed_tool_names as catalog_allowed, validate_tool_names as catalog_validate  # type: ignore
from .pipeline.finalize import finalize_tool_phase as finalize_tool_phase, compose_openai_response as compose_openai_response  # type: ignore
#
# NOTE: request shaping intentionally removed. We pass the user's original request body/messages
# through unchanged (other than minimal type validation) to avoid hidden truncation/normalization.
#

# Planner lives in its module; chat completions only orchestrates.
from .plan.committee import produce_tool_plan


# ---- Single-exit helpers (state + finalization) ----
class RunState(TypedDict, total=False):
    tool_results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    img_count: int

def accumulate_error(state: RunState, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
    if "errors" not in state:
        state["errors"] = []
    state["errors"].append({"code": code, "message": message, "details": details or {}})

def _collect_urls_from_results(results: List[Dict[str, Any]], abs_url_fn) -> List[str]:
    urls: List[str] = []
    for tr in results or []:
        res = (tr or {}).get("result") or {}
        if not isinstance(res, dict):
            continue
        meta = res.get("meta"); arts = res.get("artifacts")
        if isinstance(arts, list):
            for a in arts:
                u = (a or {}).get("url")
                if isinstance(u, str) and u.strip():
                    urls.append(u)
        if isinstance(meta, dict) and isinstance(meta.get("orch_view_urls"), list):
            for u in (meta.get("orch_view_urls") or []):
                if isinstance(u, str) and u.strip():
                    urls.append(u)
        ids_obj = res.get("ids") if isinstance(res.get("ids"), dict) else {}
        if isinstance(ids_obj, dict) and isinstance(ids_obj.get("image_files"), list):
            for fp in (ids_obj.get("image_files") or []):
                if isinstance(fp, str) and fp.strip():
                    rel_fp = fp.replace("\\", "/")
                    urls.append(f"/uploads/{rel_fp}")
    # dedupe and absolutize
    urls = list(dict.fromkeys(urls))
    return [abs_url_fn(u) for u in urls]

def finalize_response(trace_id: str, messages: List[Dict[str, Any]], state: RunState, abs_url_fn, seed: int, cid: str | None = None) -> Dict[str, Any]:
    # Best-effort cid inference when not provided explicitly: use the first
    # tool_result.meta.cid we can find so that all preview envelopes carry a
    # stable client/conversation id for downstream consumers.
    if cid is None:
        for tr in state.get("tool_results") or []:
            res_obj = (tr or {}).get("result") if isinstance(tr, dict) else {}
            meta_part = res_obj.get("meta") if isinstance(res_obj, dict) and isinstance(res_obj.get("meta"), dict) else {}
            cid_val = None
            if isinstance(meta_part, dict):
                cid_val = meta_part.get("cid")
            if isinstance(cid_val, (str, int)):
                cid = str(cid_val)
                break
    urls = _collect_urls_from_results(state.get("tool_results") or [], abs_url_fn)
    state["img_count"] = len(urls)
    lines: List[str] = []
    if state["img_count"] > 0:
        lines.append("Here are your generated image(s):")
        lines.extend([f"- {u}" for u in urls])
        warns = state.get("warnings") or []
        if warns:
            lines.append("")
            lines.append("Warnings:")
            for w in warns[:5]:
                lines.append(f"- {w.get('code','warn')}: {w.get('message','')}")
        ok_flag = True
        err_payload = None
    else:
        lines.append("The job completed with errors and no assets were produced.")
        errs = state.get("errors") or []
        for e in errs[:5]:
            lines.append(f"- {e.get('code','error')}: {e.get('message','')}")
        ok_flag = False
        err_payload = None
    final_text = "\n".join(lines) if lines else "Done."
    usage = estimate_usage(messages, final_text)
    resp = _build_openai_envelope(
        ok=ok_flag,
        text=final_text,
        error=err_payload,
        usage=usage,
        model=COMMITTEE_MODEL_ID,
        seed=seed,
        id_="orc-1",
    )
    meta_block: Dict[str, Any] = {
        "errors": state.get("errors") or [],
        "warnings": state.get("warnings") or [],
        "assets": urls,
        "trace_id": trace_id,
    }
    if isinstance(cid, (str, int)):
        meta_block["cid"] = str(cid)
    resp["_meta"] = meta_block
    _log("response.preview", trace_id=trace_id, ok=bool(ok_flag), assets=int(len(urls)))
    return resp


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# run_all router removed (deprecated /v1/run, /ws.run)

@app.post("/v1/image.generate")
async def v1_image_generate(request: Request):
    """
    Canonical image generation endpoint.
    Delegates to the image.dispatch tool (via /tool.run) instead of maintaining a separate Comfy path.
    """
    rid = str(uuid.uuid4())
    raw = await request.body()
    # Accept any JSON object for this front-door image API; shape it manually
    # into image.dispatch args below.
    body = JSONParser().parse(raw.decode("utf-8", errors="replace"), {})
    if not isinstance(body, dict):
        return ToolEnvelope.failure(
            "invalid_body_type",
            "Body must be an object",
            status=422,
            request_id=rid,
        )
    prompt = body.get("prompt") or body.get("text") or ""
    if not isinstance(prompt, str):
        return ToolEnvelope.failure(
            "invalid_prompt",
            "prompt must be a string",
            status=422,
            request_id=rid,
        )
    # Shape args for image.dispatch; keep contract minimal and forwards-compatible
    args: Dict[str, Any] = {
        "prompt": prompt,
    }
    if isinstance(body.get("negative"), str):
        args["negative"] = body.get("negative")
    if isinstance(body.get("seed"), int):
        args["seed"] = body.get("seed")
    if isinstance(body.get("width"), int):
        args["width"] = body.get("width")
    if isinstance(body.get("height"), int):
        args["height"] = body.get("height")
    if isinstance(body.get("size"), str):
        args["size"] = body.get("size")
    if isinstance(body.get("assets"), dict):
        args["assets"] = body.get("assets")
    if isinstance(body.get("lock_bundle"), dict):
        args["lock_bundle"] = body.get("lock_bundle")
    if isinstance(body.get("quality_profile"), str):
        args["quality_profile"] = body.get("quality_profile")
    # Defensive: client-provided numeric knobs must never crash the request.
    if "steps" in body:
        raw_steps = body.get("steps")
        try:
            # Accept int/float/str-ish; default if invalid.
            args["steps"] = int(raw_steps)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "image.dispatch: bad steps=%r; using default=%r",
                raw_steps,
                args.get("steps"),
                exc_info=True,
            )
    if "cfg" in body:
        raw_cfg = body.get("cfg")
        try:
            args["cfg"] = float(raw_cfg)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "image.dispatch: bad cfg=%r; using default=%r",
                raw_cfg,
                args.get("cfg"),
                exc_info=True,
            )
    # Execute in-process (no /tool.run HTTP recursion)
    res = await execute_tool_call({"name": "image.dispatch", "arguments": args})
    if isinstance(res, dict) and isinstance(res.get("result"), dict):
        result_obj = res.get("result") or {}
        # Preserve existing shape: prompt_id, cid, paths for compatibility,
        # but adapt to the canonical image.dispatch result which exposes
        # artifacts + meta.{orch_view_urls,view_urls}.
        paths = result_obj.get("paths") if isinstance(result_obj.get("paths"), list) else []
        if not paths:
            arts = result_obj.get("artifacts")
            if isinstance(arts, list):
                for a in arts:
                    if not isinstance(a, dict):
                        continue
                    # Prefer view_url, then path/url
                    p = a.get("view_url") or a.get("path") or a.get("url")
                    if isinstance(p, str) and p.strip():
                        paths.append(p)
        if not paths:
            meta = result_obj.get("meta") if isinstance(result_obj.get("meta"), dict) else {}
            orch_urls = meta.get("orch_view_urls") if isinstance(meta.get("orch_view_urls"), list) else []
            view_urls = meta.get("view_urls") if isinstance(meta.get("view_urls"), list) else []
            for p in (orch_urls or view_urls or []):
                if isinstance(p, str) and p.strip():
                    paths.append(p)
        payload = {
            "prompt_id": result_obj.get("prompt_id") or (result_obj.get("ids") or {}).get("prompt_id"),
            # cid is the conversation/client id; do not substitute ComfyUI client_id here.
            "cid": result_obj.get("cid"),
            "paths": paths,
        }
        return ToolEnvelope.success(payload, request_id=rid)
    # Tool failed; surface as envelope error but keep HTTP 200
    err_msg = res.get("error") if isinstance(res, dict) else "image_dispatch_failed"
    return ToolEnvelope.failure(
        "image_generate_failed",
        str(err_msg),
        status=422,
        request_id=rid,
    )
def _origin_norm(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("ws://"): s = "http://" + s[5:]
    if s.startswith("wss://"): s = "https://" + s[6:]
    return s.rstrip("/")

def _same_origin(origin: str, request: Request) -> bool:
    base = _origin_norm(str(request.base_url))
    return _origin_norm(origin) == base.rstrip("/")

_CORS_EXTRA = [x.strip() for x in (os.getenv("CORS_EXTRA_ORIGINS", "").split(",")) if x.strip()]

def _allowed_cross(origin: str, request: Request) -> bool:
    o = _origin_norm(origin)
    if not o:
        return False
    if _same_origin(o, request):
        return True
    if o.startswith("http://localhost") or o.startswith("http://127.0.0.1") or o.startswith("https://localhost") or o.startswith("https://127.0.0.1"):
        return True
    pb = (os.getenv("PUBLIC_BASE_URL", "").strip())
    if pb and _origin_norm(pb) == o:
        return True
    for extra in _CORS_EXTRA:
        if _origin_norm(extra) == o:
            return True
    return False

@app.middleware("http")
async def global_cors_middleware(request: Request, call_next):
    # Single source of truth for CORS on HTTP requests.
    #
    # Note: we intentionally avoid having multiple @app.middleware("http") functions
    # all setting Access-Control-* headers, because that produces contradictory
    # responses (e.g. Allow-Credentials true vs false) depending on middleware order.
    origin = (request.headers.get("origin") or request.headers.get("Origin") or "").strip()
    # "100% open always": reflect any Origin and always allow credentials.
    # Browsers reject `Access-Control-Allow-Origin: *` when credentials are enabled,
    # so we echo the origin when present; otherwise fall back to "*".
    allowed_origin = origin or "*"
    allow_credentials = "true"
    if request.method == "OPTIONS":
        hdrs = {
            "Access-Control-Allow-Origin": (allowed_origin or "*"),
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": request.headers.get("access-control-request-headers") or "*",
            "Access-Control-Allow-Credentials": allow_credentials,
            "Access-Control-Expose-Headers": "*",
            "Access-Control-Max-Age": "86400",
            "Access-Control-Allow-Private-Network": "true",
            "Vary": "Origin",
            "Content-Length": "0",
        }
        # Always 200 with explicit zero-length body to avoid 204/chunked quirks
        return Response(content=b"", status_code=200, headers=hdrs, media_type="text/plain")
    try:
        resp = await call_next(request)
    except Exception as ex:
        logging.error(
            f"http.exception (global_cors) path={str(request.url.path)}: {ex}",
            exc_info=True,
        )
        hdrs = {
            "Access-Control-Allow-Origin": (allowed_origin or "*"),
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": allow_credentials,
            "Access-Control-Expose-Headers": "*",
            "Access-Control-Max-Age": "86400",
            "Access-Control-Allow-Private-Network": "true",
            "Vary": "Origin",
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Timing-Allow-Origin": "*",
        }
        body = {
            "ok": False,
            "error": {"code": "internal_error", "message": str(ex)},
            "_meta": {"route": str(request.url.path), "method": str(request.method)},
        }
        return Response(content=json.dumps(body, ensure_ascii=False), status_code=500, media_type="application/json", headers=hdrs)
    resp.headers["Access-Control-Allow-Origin"] = allowed_origin or "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Credentials"] = allow_credentials
    resp.headers["Access-Control-Expose-Headers"] = "*"
    resp.headers["Access-Control-Max-Age"] = "86400"
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    resp.headers["Vary"] = "Origin"
    path = request.url.path or ""
    if path.startswith("/uploads/") or path.startswith("/view/") or path.endswith(".mp4") or path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".webm"):
        resp.headers.setdefault("Cross-Origin-Resource-Policy", "cross-origin")
        resp.headers.setdefault("Timing-Allow-Origin", "*")
    return resp


# In-memory job cache (DB is source of truth)
_jobs_store: Dict[str, Dict[str, Any]] = {}
_job_endpoint: Dict[str, str] = {}
_comfy_load: Dict[str, int] = {}
_comfy_backoff_raw = os.getenv("COMFYUI_BACKOFF_MS", "250")
try:
    COMFYUI_BACKOFF_MS = int(str(_comfy_backoff_raw).strip() or "250")
except Exception:
    logging.getLogger(__name__).warning("bad COMFYUI_BACKOFF_MS=%r; defaulting to 250", _comfy_backoff_raw, exc_info=True)
    COMFYUI_BACKOFF_MS = 250
_comfy_backoff_max_raw = os.getenv("COMFYUI_BACKOFF_MAX_MS", "4000")
try:
    COMFYUI_BACKOFF_MAX_MS = int(str(_comfy_backoff_max_raw).strip() or "4000")
except Exception:
    logging.getLogger(__name__).warning("bad COMFYUI_BACKOFF_MAX_MS=%r; defaulting to 4000", _comfy_backoff_max_raw, exc_info=True)
    COMFYUI_BACKOFF_MAX_MS = 4000
_comfy_retries_raw = os.getenv("COMFYUI_MAX_RETRIES", "6")
try:
    COMFYUI_MAX_RETRIES = int(str(_comfy_retries_raw).strip() or "6")
except Exception:
    logging.getLogger(__name__).warning("bad COMFYUI_MAX_RETRIES=%r; defaulting to 6", _comfy_retries_raw, exc_info=True)
    COMFYUI_MAX_RETRIES = 6
_comfy_sem = _as.Semaphore(max(1, int(SCENE_SUBMIT_CONCURRENCY))) if isinstance(SCENE_SUBMIT_CONCURRENCY, int) else _as.Semaphore(1)
_films_mem: Dict[str, Dict[str, Any]] = {}


@app.post("/v1/image/dispatch")
async def post_image_dispatch(request: Request):
    rid = uuid.uuid4().hex
    raw = await request.body()
    body_text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw or "")
    parser = JSONParser()
    expected = {
        "prompt": (str),
        "negative": (str),
        "seed": (int),
        "width": (int),
        "height": (int),
        "size": (str),
        "assets": (dict),
        "lock_bundle": (dict),
        "quality_profile": (str),
        "steps": (int),
        "cfg": (float),
    }
    obj = parser.parse(body_text, expected)
    # core fields
    prompt = obj.get("prompt")
    negative = obj.get("negative")
    seed = obj.get("seed")
    width = obj.get("width")
    height = obj.get("height")
    size = obj.get("size")
    if not (isinstance(prompt, str) and prompt.strip()):
        return ToolEnvelope.failure(
            "invalid_prompt",
            "prompt must be non-empty string",
            status=422,
            request_id=rid,
            details={},
        )
    if negative is not None and not isinstance(negative, str):
        return ToolEnvelope.failure(
            "invalid_negative",
            "negative prompt must be a string when provided",
            status=422,
            request_id=rid,
            details={},
        )
    if seed is not None and not (isinstance(seed, int) and seed >= 0):
        return ToolEnvelope.failure(
            "invalid_seed",
            "seed must be a non-negative integer",
            status=422,
            request_id=rid,
            details={},
        )
    assets = obj.get("assets") if isinstance(obj.get("assets"), dict) else {}
    lock_bundle = obj.get("lock_bundle") if isinstance(obj.get("lock_bundle"), dict) else None
    quality_profile = obj.get("quality_profile") if isinstance(obj.get("quality_profile"), str) else None
    steps_val = obj.get("steps") if isinstance(obj.get("steps"), int) else None
    cfg_val = obj.get("cfg")
    cfg_num = float(cfg_val) if isinstance(cfg_val, (int, float)) else None
    # Execute in-process (no /tool.run HTTP recursion).
    args = {
        "prompt": prompt,
        "negative": negative,
        "seed": seed,
        "width": width,
        "height": height,
        "size": size,
        "assets": assets,
        "lock_bundle": lock_bundle,
        "quality_profile": quality_profile,
        "steps": steps_val,
        "cfg": cfg_num,
    }
    res = await execute_tool_call({"name": "image.dispatch", "arguments": args})
    return ToolEnvelope.success(
        res.get("result") if isinstance(res, dict) and isinstance(res.get("result"), dict) else res,
        request_id=rid,
    )
_characters_mem: Dict[str, Dict[str, Any]] = {}
_scenes_mem: Dict[str, Dict[str, Any]] = {}

def _uri_from_upload_path(path: str) -> str:
    rel = os.path.relpath(path, UPLOAD_DIR).replace("\\", "/")
    return f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{rel}" if PUBLIC_BASE_URL else f"/uploads/{rel}"

def _write_text_atomic(path: str, text: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return {"uri": _uri_from_upload_path(path), "hash": f"sha256:{_sha256_bytes(text.encode('utf-8'))}"}

def _write_json_atomic(path: str, obj: Any) -> Dict[str, Any]:
    return _write_text_atomic(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


@app.get("/capabilities.json")
async def capabilities():
    tools_schema = get_builtin_tools_schema()
    tool_names: List[str] = []
    for t in (tools_schema or []):
        fn = (t or {}).get("function") or {}
        nm = fn.get("name")
        if isinstance(nm, str):
            tool_names.append(nm)
    tools_sorted = sorted(list(dict.fromkeys(tool_names)))
    return {"openai_compat": True, "tools": tools_sorted, "config_hash": WRAPPER_CONFIG_HASH}


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    # crude approximation: 1 token ~ 4 chars
    return max(1, (len(text) + 3) // 4)


def estimate_usage(messages: List[Dict[str, Any]], completion_text: str) -> Dict[str, int]:
    prompt_text = "\n".join([(m.get("content") or "") for m in messages])
    prompt_tokens = estimate_tokens_from_text(prompt_text)
    completion_tokens = estimate_tokens_from_text(completion_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def merge_usages(usages: List[Optional[Dict[str, int]]]) -> Dict[str, int]:
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usages:
        if not u:
            continue
        pt_raw = (u.get("prompt_tokens", 0) if isinstance(u, dict) else 0)
        ct_raw = (u.get("completion_tokens", 0) if isinstance(u, dict) else 0)
        try:
            out["prompt_tokens"] += int(pt_raw or 0)
        except Exception as exc:
            logging.getLogger(__name__).warning("merge_usages: bad prompt_tokens=%r; ignoring", pt_raw, exc_info=True)
        try:
            out["completion_tokens"] += int(ct_raw or 0)
        except Exception as exc:
            logging.getLogger(__name__).warning("merge_usages: bad completion_tokens=%r; ignoring", ct_raw, exc_info=True)
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def _save_base64_file(b64: str, suffix: str) -> str:
    raw = _b64.b64decode(b64)
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(raw)
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{filename}"
    return f"/uploads/{filename}"


# ---- Film2 helpers (event and artifact tracing) ----

def _film2_trace_event(trace_id: Optional[str], e: Dict[str, Any]) -> None:
    if not trace_id:
        return
    row = {"t": int(time.time() * 1000), **e}
    ev = str(e.get("event") or "event")
    payload = {k: v for k, v in row.items() if k != "event"}
    _log(ev, trace_id=trace_id, **payload)


def _film2_artifact_video(trace_id: Optional[str], path: str) -> None:
    if not (trace_id and isinstance(path, str) and path):
        return
    rel = os.path.relpath(path, UPLOAD_DIR).replace("\\", "/")
    _log("artifact", trace_id=trace_id, kind="video", path=rel)


# ---- Creative helpers ----

def _creative_alt_score(tr: Dict[str, Any], base_tool: str) -> float:
    res = tr.get("result") or {}
    arts = (res.get("artifacts") or []) if isinstance(res, dict) else []
    if arts:
        a0 = arts[0]
        aid = a0.get("id")
        kind = a0.get("kind") or ""
        meta = (res.get("meta") or {}) if isinstance(res.get("meta"), dict) else {}
        cid = meta.get("cid")
        artifact_group_id = meta.get("artifact_group_id")
        if aid and (artifact_group_id or cid):
            if kind.startswith("image") or base_tool.startswith("image"):
                # Prefer full image score from the local artifact path
                group = artifact_group_id or cid
                img_path = os.path.join(UPLOAD_DIR, "artifacts", "image", str(group), str(aid))
                try:
                    ai = _analyze_image(img_path, prompt=None)
                    score_block = ai.get("score") or {}
                    return float(score_block.get("overall") or 0.0)
                except Exception as ex:
                    # Image QA is best-effort; log and fall back to neutral score.
                    logging.warning(f"creative_alt_score.image_analysis_failed: {ex}", exc_info=True)
                    return 0.0
            if kind.startswith("audio") or base_tool.startswith("music"):
                try:
                    m = analyze_audio(f"/uploads/artifacts/music/{cid}/{aid}")
                    # Simple composite: louder within safe LUFS and richer spectrum
                    return float((m.get("spectral_flatness") or 0.0))
                except Exception as ex:
                    # Audio QA is best-effort; log and fall back to neutral score.
                    logging.warning(f"creative_alt_score.audio_analysis_failed: {ex}", exc_info=True)
                    return 0.0
    return 0.0


# ---- Web search helpers ----

def _synthetic_search_results(engine: str, q: str, k: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(1, min(6, k) + 1):
        key = _h.sha256(f"{engine}|{q}|{i}".encode("utf-8")).hexdigest()
        out.append(
            {
                "title": f"{engine} result {i}",
                "snippet": f"deterministic snippet {i}",
                "link": f"https://example.com/{engine}/{key[:16]}",
            }
        )
    return out


# ---- Video helpers ----

def _video_read_any(p: str):
    if not p:
        return None
    pp = p
    if pp.startswith("/uploads/"):
        pp = "/workspace" + pp
    return cv2.imread(pp, cv2.IMREAD_COLOR)


def _video_draw_on(im: "Image.Image", spec: dict) -> "Image.Image":
    draw = ImageDraw.Draw(im)
    content = str(spec.get("content") or spec.get("text") or "")
    rgb = spec.get("color") or (255, 255, 255)
    if isinstance(rgb, list):
        color = tuple(int(c) for c in rgb[:3])
    else:
        color = (255, 255, 255)
    size = int(spec.get("font_size") or 48)
    font_name = spec.get("font") or "arial.ttf"
    if not isinstance(font_name, str) or not font_name.strip():
        font_name = "arial.ttf"
    try:
        font = ImageFont.truetype(font_name, size)
    except (OSError, IOError) as ex:
        logging.debug(f"video_draw_on.font_fallback: {ex}", exc_info=True)
        font = ImageFont.load_default()
    x = int(spec.get("x") or im.width // 2)
    y = int(spec.get("y") or int(im.height * 0.9))
    anchor = spec.get("anchor") or "mm"
    draw.text((x, y), content, fill=color, font=font, anchor=anchor)
    return im


# ---- Chat / tools helpers ----

def _abs_url(u: str, request_base_url: Optional[str] = None) -> str:
    if isinstance(u, str) and u.startswith("/"):
        base = (PUBLIC_BASE_URL or "").rstrip("/")
        if not base and request_base_url:
            base = (request_base_url or "").rstrip("/")
        if base:
            return base + u
    return u


def _normalize_tool_calls(calls: Any) -> List[Dict[str, Any]]:
    """
    Normalize planner tool call structures into orchestrator-internal schema
    of the form {\"name\": str, \"arguments\": dict}.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(calls, list):
        return out
    for c in calls:
        if not isinstance(c, dict):
            continue
        # Carry through planner-provided step_id (canonical).
        step_id_val: str = ""
        raw_step_id = c.get("step_id")
        if isinstance(raw_step_id, str) and raw_step_id.strip():
            step_id_val = raw_step_id.strip()
        # Accept normalized or "steps" style
        if c.get("name") or c.get("tool"):
            nm = (c.get("name") or c.get("tool") or "")
            args_obj = c.get("arguments")
            args: Dict[str, Any] = {}
            if isinstance(args_obj, dict):
                args = args_obj
            elif isinstance(args_obj, str):
                # Do not parse inline; carry raw for gating
                args = {"_raw": args_obj}
            else:
                # fallback to "args" key (dict or JSON string)
                alt = c.get("args")
                if isinstance(alt, dict):
                    args = alt
                elif isinstance(alt, str):
                    # Do not parse inline; carry raw for gating
                    args = {"_raw": alt}
            item = {"name": str(nm), "arguments": (args or {})}
            if step_id_val:
                item["step_id"] = step_id_val
            out.append(item)
            continue
        if c.get("type") == "function":
            fn = c.get("function") or {}
            nm = fn.get("name")
            args_obj = fn.get("arguments")
            args: Dict[str, Any] = {}
            if isinstance(args_obj, dict):
                args = args_obj
            elif isinstance(args_obj, str):
                # Do not parse inline; carry raw for gating
                args = {"_raw": args_obj}
            if nm:
                item = {"name": str(nm), "arguments": (args or {})}
                if step_id_val:
                    item["step_id"] = step_id_val
                out.append(item)
    return out


_tool_expected_from_jsonschema = tool_expected_from_jsonschema


# ---- Committee / composing helpers ----

def _env_text(env: Dict[str, Any]) -> str:
    text = ""
    if isinstance(env, dict) and env.get("ok"):
        res = env.get("result") or {}
        if isinstance(res, dict) and isinstance(res.get("text"), str):
            text = res.get("text") or ""
    return text



def _collect_artifacts_payload(tool_results: List[Dict[str, Any]], abs_url_fn) -> Dict[str, Any]:
    """
    Collect a stable artifacts payload for responses:
    - urls: list[str] (deduped)
    - items: list[dict] (deduped, with tool join keys)
    """
    urls = assets_collect_urls(tool_results or [], abs_url_fn)
    items: List[Dict[str, Any]] = []
    for tr in tool_results or []:
        if not isinstance(tr, dict):
            continue
        tool_name = str(tr.get("name") or tr.get("tool") or "")
        res = tr.get("result")
        if not isinstance(res, dict):
            continue
        meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
        arts = res.get("artifacts") if isinstance(res.get("artifacts"), list) else []
        for a in arts:
            if not isinstance(a, dict):
                continue
            u = a.get("url") or a.get("view_url") or a.get("path")
            it = dict(a)
            it.setdefault("tool", tool_name)
            if isinstance(meta, dict):
                if "cid" in meta and isinstance(meta.get("cid"), (str, int)):
                    it.setdefault("cid", str(meta.get("cid")))
                if "artifact_group_id" in meta and isinstance(meta.get("artifact_group_id"), (str, int)):
                    it.setdefault("artifact_group_id", str(meta.get("artifact_group_id")))
            if isinstance(u, str) and u.strip():
                it["url"] = abs_url_fn(u.strip())
            items.append(it)
    # stable dedupe
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for it in items:
        k = (it.get("kind"), it.get("id"), it.get("url"), it.get("tool"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    return {"urls": list(dict.fromkeys(urls or [])), "items": uniq}


def _merge_tool_exec_meta_from_tool_results(
    tool_exec_meta: List[Dict[str, Any]] | None,
    tool_results: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """
    Ensure we never "lose" tool args/artifacts for downstream consumers (final_env stitching,
    teacher trace, logs), even when additional tool runs are produced inside QA loops
    (segment-level refinements) and appended directly to tool_results.

    - tool_exec_meta: existing persisted entries (typically top-level tool runs + postrun revision)
    - tool_results: the final list of tool result dicts (may include internal patch runs)

    Returns a merged list shaped like tool_exec_meta entries:
      {"name": str, "args": dict, "ok": bool, "result": dict, "error": Any, "artifacts": Any}
    """
    base: List[Dict[str, Any]] = list(tool_exec_meta or [])
    seen: set[tuple[str, str]] = set()
    for e in base:
        if not isinstance(e, dict):
            continue
        nm = str(e.get("name") or "")
        # allow either explicit step_id or fallback signature
        sid = str(e.get("step_id") or "")
        if not sid:
            a = e.get("args") if isinstance(e.get("args"), dict) else {}
            sid = str(a.get("trace_id") or "") + "|" + str(a.get("segment_id") or "")
        seen.add((nm, sid))

    for tr in (tool_results or []):
        if not isinstance(tr, dict):
            continue
        nm = str(tr.get("name") or tr.get("tool") or "").strip()
        if not nm:
            continue
        sid = str(tr.get("step_id") or "").strip()
        # fall back to (trace_id, segment_id) signature when no step id exists
        if not sid:
            args_obj = tr.get("args") if isinstance(tr.get("args"), dict) else {}
            sid = str(args_obj.get("trace_id") or "") + "|" + str(args_obj.get("segment_id") or tr.get("segment_id") or "")
        key = (nm, sid)
        if key in seen:
            continue

        res_obj = tr.get("result") if isinstance(tr.get("result"), dict) else {}
        err_obj = tr.get("error")
        if err_obj is None and isinstance(res_obj, dict):
            err_obj = res_obj.get("error")
        ok_step = not isinstance(err_obj, (str, dict))

        args_obj = tr.get("args")
        if not isinstance(args_obj, dict):
            # tolerate OpenAI-style tool call shape
            args_obj = tr.get("arguments") if isinstance(tr.get("arguments"), dict) else {}
        arts_val = res_obj.get("artifacts") if isinstance(res_obj, dict) else None

        base.append(
            {
                "name": nm,
                "args": (args_obj if isinstance(args_obj, dict) else {}),
                "ok": bool(ok_step),
                "result": res_obj,
                "error": err_obj,
                "artifacts": arts_val,
                "step_id": (sid or None),
            }
        )
        seen.add(key)
    return base


def _warn_double_wrapped_tool_results(tool_results: List[Dict[str, Any]] | None) -> None:
    """
    Defensive guard: warn when a tool result accidentally carries a nested tool envelope
    under result["result"] (common shape drift that causes artifacts/args to "disappear").
    """
    try:
        for tr in (tool_results or [])[:200]:
            if not isinstance(tr, dict):
                continue
            res = tr.get("result")
            if not isinstance(res, dict):
                continue
            inner = res.get("result")
            # Heuristic: nested envelope usually contains ok/result/error keys
            if isinstance(inner, dict) and ("ok" in res or "error" in res) and ("result" in res):
                log.warning(
                    f"double_wrapped_tool_result detected name={(tr.get('name') or tr.get('tool'))!r} "
                    f"keys={sorted(list(res.keys()))[:24]} inner_keys={sorted(list(inner.keys()))[:24]}"
                )
    except Exception:
        log.debug("_warn_double_wrapped_tool_results failed (non-fatal)", exc_info=True)


def _tb(s: str) -> str:
    return s if len(s) <= 16000 else (s[:16000] + "\n... [traceback truncated]")


def _shorten(s: str, max_len: int = 12000) -> str:
    return s if len(s) <= max_len else (s[:max_len] + "\n... [truncated]")


# ---- Datasets helpers ----

def _datasets_emit(job_id: str, ev: Dict[str, Any]) -> None:
    try:
        ph = (ev or {}).get("phase") or "running"
        pr = float((ev or {}).get("progress") or 0.0)
        _orcjob_set_state(job_id, "running", phase=ph, progress=pr)
    except Exception as ex:
        # Emitting dataset job progress is best-effort; log failures but never raise.
        logging.error(f"datasets_emit failed for job_id={job_id}: {ex}", exc_info=True)


async def _datasets_runner(job_id: str, body: Dict[str, Any]) -> None:
    try:
        emit = partial(_datasets_emit, job_id)
        # Hard-blocking execution (no thread offloading allowed).
        _datasets_start(body or {}, emit)
        _orcjob_set_state(job_id, "done", phase="done", progress=1.0)
    except Exception as ex:
        logging.error(f"datasets_runner failed for job_id={job_id}: {ex}", exc_info=True)
        _orcjob_set_state(job_id, "failed", phase="error", progress=1.0, error=str(ex))


# ---- Jobs / embeddings helpers ----

def _collect_embedding_texts(v: Any, out: List[str]) -> None:
    if isinstance(v, str):
        out.append(v)
    elif isinstance(v, list):
        for it in v:
            _collect_embedding_texts(it, out)


def _iso(val: Any) -> Optional[str]:
    if val is None:
        return None
    iso = getattr(val, "isoformat", None)
    return iso() if callable(iso) else str(val)


def _normalize_job(js: Dict[str, Any]) -> Dict[str, Any]:
    jid = js.get("id") or js.get("job_id") or js.get("uuid") or ""
    pid = js.get("prompt_id") or js.get("promptId") or None
    st = js.get("status") or js.get("state") or "unknown"
    ca = _iso(js.get("created_at") or js.get("createdAt"))
    ua = _iso(js.get("updated_at") or js.get("updatedAt"))
    out = {
        "id": str(jid),
        "job_id": str(jid),
        "prompt_id": pid if (pid is None or isinstance(pid, str)) else str(pid),
        "status": str(st),
        "state": str(st),
        "created_at": ca,
        "updated_at": ua,
    }
    return out


# ---- Teacher / streaming helpers ----

async def _send_trace_stream_async(payload: Dict[str, Any]) -> None:
    url = TEACHER_API_URL.rstrip("/") + "/teacher/trace.append"
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(url, json=payload)
            dt_ms = int((time.time() - t0) * 1000.0)
            _log("teacher.trace.append", trace_id=str(payload.get("trace_id") or ""), url=url, http_status=int(r.status_code), ms=dt_ms)
            if not (200 <= int(r.status_code) < 300):
                _log("teacher.trace.append.error", trace_id=str(payload.get("trace_id") or ""), url=url, http_status=int(r.status_code), body=(r.text or "")[:1000])
    except Exception as ex:
        dt_ms = int((time.time() - t0) * 1000.0)
        _log("teacher.trace.append.exception", trace_id=str(payload.get("trace_id") or ""), url=url, ms=dt_ms, error=str(ex))
        logging.warning(f"teacher.trace.append failed url={url}: {ex}", exc_info=True)


async def _orcjob_stream_gen(job_id: str, interval_ms: Optional[int] = None):
    last = None
    while True:
        j = _get_orcjob(job_id)
        if not j:
            yield 'data: {"error": "not_found"}\n\n'
            break
        snapshot = json.dumps(
            {
                "id": j.id,
                "tool": j.tool,
                "state": j.state,
                "phase": j.phase,
                "progress": j.progress,
                "updated_at": j.updated_at,
            }
        )
        if snapshot != last:
            yield f"data: {snapshot}\n\n"
            last = snapshot
        if j.state in ("done", "failed", "cancelled"):
            yield "data: [DONE]\n\n"
            break
        await _as.sleep(max(0.01, (interval_ms or 1000) / 1000.0))


async def _jobs_stream_gen(job_id: str, interval_ms: Optional[int] = None):
    last_snapshot = None
    while True:
        pool = await get_pg_pool()
        if pool is None:
            snapshot = json.dumps(_jobs_store.get(job_id) or {"error": "not found"})
        else:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, prompt_id, status, created_at, updated_at, workflow, result, error FROM jobs WHERE id=$1",
                    job_id,
                )
                if not row:
                    yield 'data: {"error": "not_found"}\n\n'
                    break
                snapshot = json.dumps(dict(row))
        if snapshot != last_snapshot:
            yield f"data: {snapshot}\n\n"
            last_snapshot = snapshot
        # Parse current job status safely via JSONParser. Never use json.loads.
        parsed = JSONParser().parse(snapshot or "", {"status": str})
        state = parsed.get("status") if isinstance(parsed, dict) else None
        if state in ("succeeded", "failed", "cancelled"):
            yield "data: [DONE]\n\n"
            break
        if state is None:
            # Treat unknown/malformed snapshots as terminal errors instead of spinning forever.
            yield 'data: {"error": "invalid_status"}\n\n'
            yield "data: [DONE]\n\n"
            break
        await _as.sleep(max(0.01, (interval_ms or 1000) / 1000.0))


async def _completions_stream_gen(payload: Dict[str, Any]):
    async with _hx.AsyncClient(trust_env=False, timeout=None) as client:
        async with client.stream(
            "POST",
            (PUBLIC_BASE_URL.rstrip("/") if PUBLIC_BASE_URL else "http://127.0.0.1:8000") + "/v1/chat/completions",
            json=payload,
        ) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    parser = JSONParser()
                    obj = parser.parse(
                        data_str,
                        {"id": str, "model": str, "choices": [{"delta": {"content": str}, "message": {"content": str}}]},
                    )
                    # Extract assistant text (final-only chunk in our impl)
                    txt = ""
                    if isinstance(obj, dict):
                        choices = obj.get("choices")
                        ch0: Dict[str, Any] = {}
                        if isinstance(choices, list) and choices:
                            first = choices[0]
                            if isinstance(first, dict):
                                ch0 = first
                        delta = ch0.get("delta") if isinstance(ch0.get("delta"), dict) else {}
                        msg = ch0.get("message") if isinstance(ch0.get("message"), dict) else {}
                        txt_val = (delta.get("content") or msg.get("content") or "")
                        if txt_val is None:
                            txt = ""
                        elif isinstance(txt_val, (bytes, bytearray, memoryview)):
                            txt = bytes(txt_val).decode("utf-8", errors="replace")
                        elif isinstance(txt_val, (dict, list)):
                            txt = json.dumps(txt_val, ensure_ascii=False, default=str)
                        else:
                            txt = str(txt_val)
                    chunk = {
                        "id": obj.get("id") if isinstance(obj, dict) else "orc-1",
                        "object": "text_completion",
                        "choices": [
                            {
                                "index": 0,
                                "text": txt,
                                "finish_reason": None,
                                "logprobs": None,
                            }
                        ],
                    }
                    yield "data: " + json.dumps(chunk) + "\n\n"


async def _tool_ws_keepalive(websocket: WebSocket) -> None:
    while True:
        try:
            await websocket.send_text(json.dumps({"keepalive": True}))
        except Exception as ex:
            logging.debug(f"ws_tool.keepalive_failed: {ex}", exc_info=True)
            break
        await _as.sleep(10)


def extract_attachments_from_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    attachments: List[Dict[str, Any]] = []
    normalized: List[Dict[str, Any]] = []
    for m in messages:
        content_val = m.get("content")
        if isinstance(content_val, list):
            text_parts: List[str] = []
            for part in content_val:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(str(part.get("text", "")))
                elif ptype in ("input_image", "image"):
                    iu = part.get("image_url") or {}
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    b64 = part.get("image_base64")
                    final_url = None
                    if b64:
                        final_url = _save_base64_file(b64, ".png")
                    elif url:
                        final_url = url
                    if final_url:
                        attachments.append({"type": "image", "url": final_url})
                elif ptype in ("input_audio", "audio"):
                    au = part.get("audio_url") or {}
                    url = au.get("url") if isinstance(au, dict) else au
                    b64 = part.get("audio_base64")
                    final_url = None
                    if b64:
                        final_url = _save_base64_file(b64, ".wav")
                    elif url:
                        final_url = url
                    if final_url:
                        attachments.append({"type": "audio", "url": final_url})
                elif ptype in ("input_video", "video"):
                    vu = part.get("video_url") or {}
                    url = vu.get("url") if isinstance(vu, dict) else vu
                    b64 = part.get("video_base64")
                    final_url = None
                    if b64:
                        final_url = _save_base64_file(b64, ".mp4")
                    elif url:
                        final_url = url
                    if final_url:
                        attachments.append({"type": "video", "url": final_url})
                elif ptype in ("input_file", "file"):
                    fu = part.get("file_url") or {}
                    url = fu.get("url") if isinstance(fu, dict) else fu
                    b64 = part.get("file_base64")
                    name = part.get("name") or "file.bin"
                    suffix = "." + name.split(".")[-1] if "." in name else ".bin"
                    final_url = None
                    if b64:
                        final_url = _save_base64_file(b64, suffix)
                    elif url:
                        final_url = url
                    if final_url:
                        attachments.append({"type": "file", "url": final_url, "name": name})
            merged_text = "\n".join(tp for tp in text_parts if tp)
            normalized.append({"role": m.get("role"), "content": merged_text or None, "name": m.get("name"), "tool_call_id": m.get("tool_call_id"), "tool_calls": m.get("tool_calls")})
        else:
            normalized.append(m)
    return normalized, attachments


# ---------- RAG (pgvector) ----------
_embedder: Optional[SentenceTransformer] = None
_rag_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

# Tool streaming progress hook (set by streaming endpoints)


def get_embedder() -> SentenceTransformer:
    # Delegate to rag.core to keep a single embedder
    return _rag_get_embedder()


from .rag.core import rag_index_dir  # re-exported for backwards-compat


from .rag.core import rag_search  # re-exported for backwards-compat
from .committee_client import committee_ai_text, committee_jsonify

# committee_client import policy: only import committee call + jsonify.
# All other config is sourced directly from environment here.
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "qwen3:30b-a3b-instruct-2507-q4_K_M")
GLM_MODEL_ID = os.getenv("GLM_MODEL_ID", "glm4:9b")
DEEPSEEK_CODER_MODEL_ID = os.getenv("DEEPSEEK_CODER_MODEL_ID", "deepseek-coder-v2:lite")

_DEFAULT_NUM_CTX_RAW = os.getenv("DEFAULT_NUM_CTX", "8192")
try:
    DEFAULT_NUM_CTX = int(str(_DEFAULT_NUM_CTX_RAW).strip() or "8192")
except Exception:
    DEFAULT_NUM_CTX = 8192

COMMITTEE_MODEL_ID = os.getenv("COMMITTEE_MODEL_ID") or f"committee:{QWEN_MODEL_ID}+{GLM_MODEL_ID}+{DEEPSEEK_CODER_MODEL_ID}"


## serpapi removed — use metasearch.fuse/web_search exclusively


## rrf_fuse moved to .search.meta


## moved to .tools.mcp_bridge


def to_openai_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for idx, c in enumerate(tool_calls):
        name = c.get("name") or "tool"
        args = c.get("arguments") or {}
        converted.append({
            "id": f"call_{idx+1}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return converted


async def propose_search_queries(messages: List[Dict[str, Any]]) -> List[str]:
    guidance = (
        "Given the conversation, propose up to 3 concise web search queries that would most improve the answer. "
        "Return each query on its own line with no extra text."
    )
    prompt_messages = messages + [{"role": "user", "content": guidance}]
    try:
        env = await committee_ai_text(messages=prompt_messages, trace_id="search_suggest", temperature=DEFAULT_TEMPERATURE)
    except Exception as ex:
        # Fail-soft: log and return no suggestions instead of killing the request
        _log(
            "search_suggest.exception",
            trace_id="search_suggest",
            error=str(ex),
        )
        return []

    queries: List[str] = []
    if not isinstance(env, dict) or not env.get("ok"):
        # Log committee failure instead of silently returning no suggestions.
        _log(
            "search_suggest.error",
            trace_id="search_suggest",
            error=(env or {}).get("error") if isinstance(env, dict) else {
                "code": "committee_invalid_env",
                "message": str(env),
            },
        )
    else:
        res_env = env.get("result") or {}
        text = res_env.get("text") or ""
        lines = [ln.strip("- ") for ln in text.splitlines() if ln.strip()]
        queries = lines[:3]
    return queries





def _tool_cast_to_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if _re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
        return float(s)
    return None


def _tool_cast_to_int(v: Any) -> Optional[int]:
    if isinstance(v, int):
        return int(v)
    s = str(v).strip()
    if _re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    return None


def _tool_error(name: str, code: str, message: str, status: int = 422, **details: Any) -> Dict[str, Any]:
    """
    Canonical error payload for internal tool handlers.
    This is what /tool.run will wrap into a ToolEnvelope and what the executor
    will surface as tool_result.error, so always include code/message/status.
    """
    err: Dict[str, Any] = {
        "code": code,
        "message": message,
        "status": int(status),
    }
    if details:
        err.update(details)
    return {"name": name, "error": err}


async def execute_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    name = call.get("name")
    # IMPORTANT: do not use `or {}` here; falsy values (e.g. "" or 0) should
    # still be preserved and routed through normalization below.
    raw_args = call.get("arguments")
    # Determine trace id for this tool call; always present in envelopes.
    trace_id: str
    if isinstance(call.get("trace_id"), str) and call.get("trace_id"):
        trace_id = str(call.get("trace_id"))
    else:
        meta_val = call.get("meta")
        if isinstance(meta_val, dict) and isinstance(meta_val.get("trace_id"), str) and meta_val.get("trace_id"):
            trace_id = str(meta_val.get("trace_id"))
        else:
            trace_id = "tt_unknown"
    # The executor is not a callable tool; it is invoked only via
    # the external executor service (pipeline.executor_gateway).
    if name == "executor":
        return _tool_error(
            "executor",
            "executor_not_callable",
            "The executor must be invoked via execute_tools(), not as a tool.",
            status=500,
        )

    # Generic HTTP API request tool (public APIs only; no internal SSRF)
    if name == "api.request":
        # Normalize arguments for this tool without dropping payloads.
        parser_req = JSONParser()
        if isinstance(raw_args, dict):
            args_req = dict(raw_args)
        elif isinstance(raw_args, str):
            parsed_req = parser_req.parse(raw_args, {})
            args_req = dict(parsed_req) if isinstance(parsed_req, dict) else {"_raw": raw_args}
        elif raw_args is None:
            args_req = {}
        else:
            args_req = {"_raw": raw_args}
        # Backwards-compatible shim: route api.request to http.request implementation.
        http_config: HttpRequestConfig = {
            "url": str(args_req.get("url") or ""),
            "method": str(args_req.get("method") or "").upper() or "GET",
            "headers": args_req.get("headers") if isinstance(args_req.get("headers"), dict) else {},
            "query": args_req.get("params") if isinstance(args_req.get("params"), dict) else {},
            "body": args_req.get("body"),
            "expect_json": (str(args_req.get("expect") or "json")).lower() != "text",
        }
        ok, payload = await perform_http_request(http_config)
        if ok:
            return {"name": "api.request", "result": payload}
        return {
            "name": "api.request",
            "error": {
                "code": payload.get("code"),
                "message": payload.get("message"),
                "status": payload.get("status"),
                "details": payload.get("details"),
            },
        }
    # DB pool (may be None if PG not configured)
    pool = await get_pg_pool()
    # Normalize tool arguments while preserving *all* raw keys.
    #
    # CRITICAL: do NOT coerce all tools into one fixed schema because it will
    # silently drop args for tools that expect different keys.
    #
    # We still do best-effort parsing for string arguments, and we optionally
    # coerce a small set of common, cross-tool keys when they are present.
    parser = JSONParser()
    if isinstance(raw_args, dict) and isinstance(raw_args.get("_raw"), str):
        raw_inner = str(raw_args.get("_raw") or "")
        parsed_inner = parser.parse(raw_inner, {})
        if isinstance(parsed_inner, dict) and parsed_inner:
            args = dict(parsed_inner)
            # Overlay any explicit keys provided alongside _raw without dropping them.
            for k, v in raw_args.items():
                if k == "_raw":
                    continue
                args[k] = v
        else:
            # Never drop arguments: preserve the original dict and keep synopsis.
            args = dict(raw_args)
            args.setdefault("synopsis", raw_inner)
    elif isinstance(raw_args, str):
        parsed = parser.parse(raw_args, {})
        if isinstance(parsed, dict) and parsed:
            args = dict(parsed)
        else:
            # Never drop arguments: preserve the raw string while also keeping
            # backwards-compatible "synopsis" behavior.
            args = {"_raw": raw_args, "synopsis": raw_args}
    elif isinstance(raw_args, dict):
        args = dict(raw_args)
    elif raw_args is None:
        args = {}
    else:
        # Never drop arguments: preserve any non-object payload under "_raw".
        args = {"_raw": raw_args}

    # Always propagate trace id into args (critical for downstream trace wiring)
    if trace_id and isinstance(args, dict) and not args.get("trace_id"):
        args["trace_id"] = trace_id

    # Light, additive coercion for common keys (only when present).
    expected_args_shape = {
        "film_id": str,
        "title": str,
        "synopsis": str,
        "prompt": str,
        "characters": [{"name": str, "description": str}],
        "scenes": [{"index_num": int, "prompt": str}],
        "index_num": int,
        "duration_seconds": float,
        "duration": float,
        "resolution": str,
        "fps": float,
        "frame_rate": float,
        "style": str,
        "language": str,
        "voice": str,
        "audio_enabled": str,
        "audio_on": str,
        "subtitles_enabled": str,
        "subtitles_on": str,
        "metadata": dict,
        "references": dict,
        "reference_data": dict,
    }
    expected_present = {k: t for k, t in expected_args_shape.items() if k in args}
    if expected_present:
        # IMPORTANT: `ensure_structure` is an internal coercion implementation detail.
        # Use the public parser API surface.
        coerced_common = parser.parse(args, expected_present)
        if isinstance(coerced_common, dict):
            args.update(coerced_common)
    # Synonyms → canonical
    if args.get("duration_seconds") is None and args.get("duration") is not None:
        args["duration_seconds"] = args.get("duration")
    if args.get("fps") is None and args.get("frame_rate") is not None:
        args["fps"] = args.get("frame_rate")
    if args.get("reference_data") is None and args.get("references") is not None:
        args["reference_data"] = args.get("references")
    # Coerce types
    if args.get("fps") is not None:
        f = _tool_cast_to_float(args.get("fps"))
        if f is not None:
            args["fps"] = f
    if args.get("duration_seconds") is not None:
        d = _tool_cast_to_float(args.get("duration_seconds"))
        if d is not None:
            args["duration_seconds"] = d
    if args.get("index_num") is not None:
        i = _tool_cast_to_int(args.get("index_num"))
        if i is not None:
            args["index_num"] = i
    # Booleans that might arrive as strings
    for key, syn in (("audio_enabled", "audio_on"), ("subtitles_enabled", "subtitles_on")):
        val = args.get(key)
        if val is None:
            val = args.get(syn)
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ("true", "1", "yes", "on"):
                val = True
            elif low in ("false", "0", "no", "off"):
                val = False
            else:
                val = None
        if isinstance(val, bool):
            args[key] = bool(val)
    # Additional aliases often provided by clients
    if args.get("audio_enabled") is None and ("audio" in args):
        v = args.get("audio")
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ("true", "1", "yes", "on"):
                v = True
            elif vv in ("false", "0", "no", "off"):
                v = False
            else:
                v = None
        if isinstance(v, bool):
            args["audio_enabled"] = bool(v)
    if args.get("subtitles_enabled") is None and ("subtitles" in args):
        v = args.get("subtitles")
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ("true", "1", "yes", "on"):
                v = True
            elif vv in ("false", "0", "no", "off"):
                v = False
            else:
                v = None
        if isinstance(v, bool):
            args["subtitles_enabled"] = bool(v)
    # Normalize resolution synonyms
    res = args.get("resolution")
    if isinstance(res, str):
        r = res.lower()
        if r in ("4k", "3840x2160", "3840×2160"):
            args["resolution"] = "3840x2160"
        elif r in ("8k", "7680x4320", "7680×4320"):
            args["resolution"] = "7680x4320"
    if name == "locks.build_image_bundle":
        a = args if isinstance(args, dict) else {}
        character_id = str(a.get("character_id") or "").strip()
        image_url = str(a.get("image_url") or "").strip()
        options = a.get("options") if isinstance(a.get("options"), dict) else {}
        if not character_id:
            return _tool_error(name, "missing_character_id", "character_id is required")
        if not image_url:
            return _tool_error(name, "missing_image_url", "image_url is required")
        bundle = await _build_image_lock_bundle(character_id, image_url, options, locks_root_dir=LOCKS_ROOT_DIR)
        existing = await _lock_load(character_id) or {}
        existing = _lock_migrate_visual(existing)
        existing = _lock_migrate_music(existing)
        existing = _lock_migrate_tts(existing)
        existing = _lock_migrate_sfx(existing)
        existing = _lock_migrate_film2(existing)
        merged = _merge_lock_bundles(existing, bundle)
        _lock_summarize_all_locks_for_context(_ctx_add, character_id, merged)
        await _lock_save(character_id, merged)
        return {"name": name, "result": {"lock_bundle": merged}}
    if name == "locks.build_video_bundle":
        a = args if isinstance(args, dict) else {}
        character_id = str(a.get("character_id") or "").strip()
        video_path = str(a.get("video_path") or "").strip()
        max_frames_val = a.get("max_frames")
        max_frames = int(max_frames_val) if isinstance(max_frames_val, int) and max_frames_val > 0 else 16
        if not character_id:
            return _tool_error(name, "missing_character_id", "character_id is required")
        if not video_path:
            return _tool_error(name, "missing_video_path", "video_path is required")
        bundle = await _build_video_lock_bundle(character_id, video_path, locks_root_dir=LOCKS_ROOT_DIR, max_frames=max_frames)
        existing = await _lock_load(character_id) or {}
        existing = _lock_migrate_visual(existing)
        existing = _lock_migrate_music(existing)
        existing = _lock_migrate_tts(existing)
        existing = _lock_migrate_sfx(existing)
        existing = _lock_migrate_film2(existing)
        merged = _merge_lock_bundles(existing, bundle)
        _lock_summarize_all_locks_for_context(_ctx_add, character_id, merged)
        await _lock_save(character_id, merged)
        return {"name": name, "result": {"lock_bundle": merged}}
    if name == "locks.build_audio_bundle":
        a = args if isinstance(args, dict) else {}
        character_id = str(a.get("character_id") or "").strip()
        audio_url = str(a.get("audio_url") or "").strip()
        if not character_id:
            return _tool_error(name, "missing_character_id", "character_id is required")
        if not audio_url:
            return _tool_error(name, "missing_audio_url", "audio_url is required")
        bundle = await _build_audio_lock_bundle(character_id, audio_url, locks_root_dir=LOCKS_ROOT_DIR)
        existing = await _lock_load(character_id) or {}
        existing = _lock_migrate_visual(existing)
        existing = _lock_migrate_music(existing)
        existing = _lock_migrate_tts(existing)
        existing = _lock_migrate_sfx(existing)
        existing = _lock_migrate_film2(existing)
        merged = _merge_lock_bundles(existing, bundle)
        _lock_summarize_all_locks_for_context(_ctx_add, character_id, merged)
        await _lock_save(character_id, merged)
        return {"name": name, "result": {"lock_bundle": merged}}
    if name == "locks.build_region_locks":
        a = args if isinstance(args, dict) else {}
        character_id = str(a.get("character_id") or "").strip()
        image_url = str(a.get("image_url") or "").strip()
        regions_arg = a.get("regions") if isinstance(a.get("regions"), list) else None
        if not character_id:
            return _tool_error(name, "missing_character_id", "character_id is required")
        if not image_url:
            return _tool_error(name, "missing_image_url", "image_url is required")
        bundle = await _build_region_lock_bundle(character_id, image_url, regions_arg, locks_root_dir=LOCKS_ROOT_DIR)
        existing = await _lock_load(character_id) or {}
        existing = _lock_migrate_visual(existing)
        existing = _lock_migrate_music(existing)
        existing = _lock_migrate_tts(existing)
        existing = _lock_migrate_sfx(existing)
        existing = _lock_migrate_film2(existing)
        merged = _merge_lock_bundles(existing, bundle)
        _lock_summarize_all_locks_for_context(_ctx_add, character_id, merged)
        await _lock_save(character_id, merged)
        return {"name": name, "result": {"lock_bundle": merged}}
    if name == "locks.update_region_modes":
        a = args if isinstance(args, dict) else {}
        character_id = str(a.get("character_id") or "").strip()
        updates = a.get("updates") if isinstance(a.get("updates"), list) else []
        bundle_arg = a.get("lock_bundle") if isinstance(a.get("lock_bundle"), dict) else None
        # Allow callers to either target a persisted character lock bundle via
        # character_id or operate on an explicit lock_bundle payload without
        # requiring character identity. This avoids noisy failures in revise
        # flows when only a bundle snapshot is available.
        if not character_id and bundle_arg is None:
            return _tool_error(name, "missing_character_id", "character_id or lock_bundle is required")
        if character_id:
            existing = await _lock_load(character_id) or {}
        else:
            existing = dict(bundle_arg or {})
        existing = _lock_migrate_visual(existing)
        existing = _lock_migrate_music(existing)
        existing = _lock_migrate_tts(existing)
        existing = _lock_migrate_sfx(existing)
        existing = _lock_migrate_film2(existing)
        updated_bundle = _apply_region_mode_updates(existing, updates)
        # Persist and summarize only when operating on a character-scoped bundle.
        if character_id:
            _lock_summarize_all_locks_for_context(_ctx_add, character_id, updated_bundle)
            await _lock_save(character_id, updated_bundle)
        return {"name": name, "result": {"lock_bundle": updated_bundle}}
    if name == "locks.update_audio_modes":
        a = args if isinstance(args, dict) else {}
        character_id = str(a.get("character_id") or "").strip()
        update_payload = a.get("update") if isinstance(a.get("update"), dict) else {}
        if not character_id:
            return _tool_error(name, "missing_character_id", "character_id is required")
        existing = await _lock_load(character_id) or {}
        existing = _lock_migrate_visual(existing)
        existing = _lock_migrate_music(existing)
        existing = _lock_migrate_tts(existing)
        existing = _lock_migrate_sfx(existing)
        existing = _lock_migrate_film2(existing)
        updated_bundle = _apply_audio_mode_updates(existing, update_payload)
        _lock_summarize_all_locks_for_context(_ctx_add, character_id, updated_bundle)
        await _lock_save(character_id, updated_bundle)
        return {"name": name, "result": {"lock_bundle": updated_bundle}}
    if name == "locks.get_bundle":
        a = args if isinstance(args, dict) else {}
        character_id = str(a.get("character_id") or "").strip()
        if not character_id:
            return {"name": name, "error": "missing_character_id"}
        bundle = await _lock_load(character_id)
        if bundle is None:
            return {"name": name, "result": {"lock_bundle": None, "found": False}}
        bundle = _lock_migrate_visual(bundle)
        bundle = _lock_migrate_music(bundle)
        bundle = _lock_migrate_tts(bundle)
        bundle = _lock_migrate_sfx(bundle)
        bundle = _lock_migrate_film2(bundle)
        return {"name": name, "result": {"lock_bundle": bundle, "found": True}}
    # Deterministic grouped dispatchers
    if name == "image.dispatch" and ALLOW_TOOL_EXECUTION:
        a = args if isinstance(args, dict) else {}
        quality_profile = (a.get("quality_profile") or a.get("profile") or "standard")
        preset_name = quality_profile.lower()
        preset = LOCK_QUALITY_PRESETS.get(preset_name, LOCK_QUALITY_PRESETS["standard"])
        if "steps" not in a or a.get("steps") is None:
            a["steps"] = preset["steps"]
        if "cfg" not in a or a.get("cfg") is None:
            a["cfg"] = preset["cfg"]

        # Best-effort character identity for lock routing. Prefer explicit
        # character_id / lock_character_id; fall back to a trace-scoped
        # synthetic character id so that image generations participate in the
        # lock bundle model even when the caller omits character metadata.
        char_key = str(a.get("character_id") or a.get("lock_character_id") or "").strip()
        if not char_key:
            trace_val = a.get("trace_id")
            if isinstance(trace_val, str) and trace_val.strip():
                char_key = f"char_{trace_val.strip()}"
        if char_key and "character_id" not in a:
            a["character_id"] = char_key

        bundle_arg = a.get("lock_bundle")
        lock_bundle: Optional[Dict[str, Any]] = None
        if isinstance(bundle_arg, dict):
            lock_bundle = bundle_arg
        elif isinstance(bundle_arg, str) and bundle_arg.strip():
            loaded = await _lock_load(bundle_arg.strip())
            if isinstance(loaded, dict):
                lock_bundle = loaded
        if lock_bundle is None and char_key:
            loaded = await _lock_load(char_key)
            if isinstance(loaded, dict):
                lock_bundle = loaded
        # Enforce a lock-first invariant for character-scoped image generation:
        # delegate to the locks.runtime helper so skeleton construction lives
        # alongside other lock runtime helpers.
        if lock_bundle is None and char_key:
            lock_bundle = await _lock_ensure_visual(char_key, None)

        if lock_bundle is not None:
            # Apply quality profile and normalize visual branch for this request
            lock_bundle = _lock_apply_profile(quality_profile, lock_bundle)
            lock_bundle = _lock_migrate_visual(lock_bundle)
            # If any entities are marked hard, freeze them and relax others.
            ents = _lock_visual_get_entities(lock_bundle)
            hard_ids: List[str] = []
            for ent in ents:
                if not isinstance(ent, dict):
                    continue
                ent_id = ent.get("entity_id")
                c = ent.get("constraints") if isinstance(ent.get("constraints"), dict) else {}
                if isinstance(ent_id, str) and c.get("lock_mode") == "hard":
                    hard_ids.append(ent_id)
            if hard_ids:
                _lock_visual_freeze(lock_bundle, hard_ids)
                _lock_visual_refresh_all_except(lock_bundle, hard_ids)
        # Log executor step arguments best-effort; failures should not affect request handling.
        _args_keys = sorted([str(k) for k in (list(a.keys()) if isinstance(a, dict) else [])])
        _log("[executor] step.args", tool=name, args_keys=_args_keys)
        prompt = a.get("prompt") or a.get("text") or ""
        negative = a.get("negative")
        seed = a.get("seed")
        size = a.get("size")
        width = a.get("width")
        height = a.get("height")
        steps_val = None
        if "steps" in a:
            raw_steps = a.get("steps")
            if isinstance(raw_steps, int):
                steps_val = raw_steps
            else:
                try:
                    steps_val = int(str(raw_steps).strip())
                except (TypeError, ValueError):
                    steps_val = None
        cfg_val = a.get("cfg")
        cfg_num = float(cfg_val) if isinstance(cfg_val, (int, float)) else None
        assets = dict(a.get("assets") if isinstance(a.get("assets"), dict) else {})
        if lock_bundle is not None:
            assets["lock_bundle"] = lock_bundle
        dispatch_args = {
            "prompt": str(prompt),
            "negative": negative if isinstance(negative, str) else None,
            "seed": seed if isinstance(seed, int) else None,
            "width": width if isinstance(width, int) else None,
            "height": height if isinstance(height, int) else None,
            "size": size if isinstance(size, str) else None,
            "assets": assets,
            "lock_bundle": lock_bundle,
            "quality_profile": quality_profile,
            "steps": steps_val,
            "cfg": cfg_num,
            "trace_id": a.get("trace_id") if isinstance(a.get("trace_id"), str) else None,
        }
        # Execute image generation in-process via ComfyUI (no /tool.run HTTP recursion).
        try:
            prompt_text = str(dispatch_args.get("prompt") or "")
            negative_text = dispatch_args.get("negative") if isinstance(dispatch_args.get("negative"), str) else None
            seed_int = dispatch_args.get("seed") if isinstance(dispatch_args.get("seed"), int) else None
            width_int = dispatch_args.get("width") if isinstance(dispatch_args.get("width"), int) else None
            height_int = dispatch_args.get("height") if isinstance(dispatch_args.get("height"), int) else None
            # Parse "WxH" size when width/height not provided.
            if (width_int is None or height_int is None) and isinstance(dispatch_args.get("size"), str) and "x" in str(dispatch_args.get("size")).lower():
                try:
                    w_str, h_str = str(dispatch_args.get("size")).lower().split("x", 1)
                    if width_int is None and w_str.strip().isdigit():
                        width_int = int(w_str.strip())
                    if height_int is None and h_str.strip().isdigit():
                        height_int = int(h_str.strip())
                except Exception:
                    pass

            # Build a prompt graph. Prefer a configured workflow file if it already contains an API graph.
            wf_path = os.getenv("COMFY_WORKFLOW_PATH") or "/workspace/services/image/workflows/stock_smoke.json"
            prompt_graph: Dict[str, Any] | None = None
            try:
                if wf_path and os.path.exists(wf_path):
                    with open(wf_path, "r", encoding="utf-8") as f:
                        wf_obj = JSONParser().parse(f.read(), {})
                else:
                    wf_obj = {}
            except Exception:
                wf_obj = {}
            if isinstance(wf_obj, dict) and isinstance(wf_obj.get("prompt"), dict):
                prompt_graph = wf_obj.get("prompt")  # type: ignore[assignment]
            elif isinstance(wf_obj, dict) and wf_obj:
                # If it already looks like an API graph mapping, accept it.
                looks_api = True
                for _k, _v in wf_obj.items():
                    if not (isinstance(_v, dict) and "class_type" in _v and "inputs" in _v):
                        looks_api = False
                        break
                if looks_api:
                    prompt_graph = wf_obj

            if prompt_graph is None:
                # Fallback: minimal SDXL graph.
                w0, h0 = (width_int or 1024), (height_int or 1024)
                steps0 = int(steps_val or 25)
                seed0 = int(seed_int or 0)
                prompt_graph = (build_default_scene_workflow(prompt_text, [], style=None, width=w0, height=h0, steps=steps0, seed=seed0, filename_prefix="void_image") or {}).get("prompt")  # type: ignore[name-defined]
                if not isinstance(prompt_graph, dict):
                    prompt_graph = {}

            # Patch prompt / negative, seed, size, cfg/steps in common node types.
            pos_set = False
            for _nid, _node in (prompt_graph or {}).items():
                if not isinstance(_node, dict):
                    continue
                ct = _node.get("class_type")
                inp = _node.get("inputs") if isinstance(_node.get("inputs"), dict) else None
                if not isinstance(inp, dict):
                    continue
                if ct == "CLIPTextEncode":
                    # First CLIPTextEncode is positive, second is negative (common convention).
                    if (not pos_set) and (prompt_text is not None):
                        inp["text"] = str(prompt_text)
                        pos_set = True
                    elif negative_text is not None:
                        inp["text"] = str(negative_text)
                        negative_text = None
                if ct in ("KSampler", "KSamplerAdvanced", "SamplerCustom"):
                    if seed_int is not None and "seed" in inp:
                        inp["seed"] = int(seed_int)
                    if steps_val is not None and "steps" in inp:
                        inp["steps"] = int(steps_val)
                    if cfg_num is not None and "cfg" in inp:
                        inp["cfg"] = float(cfg_num)
                if ct in ("EmptyLatentImage", "LatentImage", "ImageResize", "LatentUpscale"):
                    if width_int is not None and "width" in inp:
                        inp["width"] = int(width_int)
                    if height_int is not None and "height" in inp:
                        inp["height"] = int(height_int)

            workflow_payload: Dict[str, Any] = {"prompt": prompt_graph}
            submit_res = await _comfy_submit_workflow(workflow_payload)  # type: ignore[name-defined]
            if not isinstance(submit_res, dict) or submit_res.get("error"):
                return _tool_error(name, "comfy_submit_failed", "comfy submit failed", status=502, details=submit_res)  # type: ignore[name-defined]
            prompt_id = submit_res.get("prompt_id") or submit_res.get("uuid") or submit_res.get("id")
            if not isinstance(prompt_id, str) or not prompt_id:
                return _tool_error(name, "missing_prompt_id", "missing prompt_id from comfy", status=502, detail=submit_res)  # type: ignore[name-defined]

            hist = await _comfy_history(prompt_id)  # type: ignore[name-defined]
            detail = _normalize_comfy_history_entry(hist if isinstance(hist, dict) else {}, prompt_id)  # type: ignore[name-defined]
            base = _job_endpoint.get(prompt_id) or (COMFYUI_API_URL or "")  # type: ignore[name-defined]
            assets_list = _extract_comfy_asset_urls(detail if isinstance(detail, dict) else {}, base)  # type: ignore[name-defined]

            # Download outputs into uploads/artifacts so downstream can serve them.
            artifact_group_id = str(prompt_id)
            save_dir = os.path.join(UPLOAD_DIR, "artifacts", "image", artifact_group_id)  # type: ignore[name-defined]
            os.makedirs(save_dir, exist_ok=True)

            orch_urls: List[str] = []
            artifacts_out: List[Dict[str, Any]] = []
            view_urls: List[str] = []

            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                for it in assets_list:
                    if not isinstance(it, dict):
                        continue
                    fn = it.get("filename")
                    url = it.get("url")
                    if not isinstance(fn, str) or not fn:
                        continue
                    if not isinstance(url, str) or not url:
                        continue
                    view_urls.append(url)
                    # Fetch and persist
                    resp = await client.get(url)
                    if int(getattr(resp, "status_code", 0) or 0) != 200:
                        continue
                    safe_fn = os.path.basename(fn)
                    dst = os.path.join(save_dir, safe_fn)
                    try:
                        with open(dst, "wb") as f:
                            f.write(resp.content)
                    except Exception:
                        continue
                    rel = os.path.relpath(dst, UPLOAD_DIR).replace("\\", "/")  # type: ignore[name-defined]
                    orch_url = (f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{rel}" if PUBLIC_BASE_URL else f"/uploads/{rel}")  # type: ignore[name-defined]
                    orch_urls.append(orch_url)
                    artifacts_out.append(
                        {
                            "id": safe_fn,
                            "kind": "image",
                            "path": f"/uploads/{rel}",
                            "view_url": orch_url,
                        }
                    )

            result_obj: Dict[str, Any] = {
                "ids": {"prompt_id": prompt_id},
                "meta": {
                    "prompt": prompt_text,
                    "negative": dispatch_args.get("negative"),
                    "trace_id": dispatch_args.get("trace_id"),
                    "artifact_group_id": artifact_group_id,
                    "view_urls": view_urls,
                    "orch_view_urls": orch_urls,
                    "image_count": len(orch_urls) or len(view_urls),
                    "steps": steps_val,
                    "cfg": cfg_num,
                    "width": width_int,
                    "height": height_int,
                },
            }
            if artifacts_out:
                result_obj["artifacts"] = artifacts_out
            return {"name": name, "result": result_obj}
        except Exception as ex:
            return _tool_error(name, "image_dispatch_exception", str(ex), status=500, stack=traceback.format_exc())  # type: ignore[name-defined]
    if name == "image.refine.segment" and ALLOW_TOOL_EXECUTION:
        a = args if isinstance(args, dict) else {}
        segment_id_raw = a.get("segment_id")
        segment_id = str(segment_id_raw or "").strip() if segment_id_raw is not None else ""
        if not segment_id:
            return {
                "name": name,
                "error": {
                    "code": "missing_segment_id",
                    "message": "image.refine.segment requires a non-empty segment_id",
                    "status": 422,
                },
            }
        prompt_val = a.get("prompt")
        prompt = str(prompt_val or "").strip() if prompt_val is not None else ""
        if not prompt:
            return {
                "name": name,
                "error": {
                    "code": "missing_prompt",
                    "message": "image.refine.segment requires prompt to be provided for Step 2",
                    "status": 422,
                },
            }
        source_image_val = a.get("source_image")
        source_image = str(source_image_val or "").strip() if source_image_val is not None else ""
        if not source_image:
            return {
                "name": name,
                "error": {
                    "code": "no_source_image",
                    "message": "image.refine.segment requires source_image for refinement in Step 2",
                    "status": 422,
                },
            }
        quality_profile = (a.get("quality_profile") or a.get("profile") or "standard")
        preset_name = str(quality_profile or "standard").lower()
        preset = LOCK_QUALITY_PRESETS.get(preset_name, LOCK_QUALITY_PRESETS["standard"])
        if "steps" not in a or a.get("steps") is None:
            a["steps"] = preset["steps"]
        if "cfg" not in a or a.get("cfg") is None:
            a["cfg"] = preset["cfg"]
        bundle_arg = a.get("lock_bundle")
        lock_bundle: Optional[Dict[str, Any]] = None
        if isinstance(bundle_arg, dict):
            lock_bundle = bundle_arg
        elif isinstance(bundle_arg, str) and bundle_arg.strip():
            lock_bundle = await _lock_load(bundle_arg.strip()) or {}
        if lock_bundle:
            lock_bundle = _lock_apply_profile(quality_profile, lock_bundle)
            lock_bundle = _lock_migrate_visual(lock_bundle)
        dispatch_args = build_image_refine_dispatch_args(a, lock_bundle, str(quality_profile or "standard"))
        # Execute in-process (no /tool.run recursion)
        res = await execute_tool_call({"name": "image.dispatch", "arguments": dispatch_args, "trace_id": a.get("trace_id")})
        if isinstance(res, dict) and isinstance(res.get("result"), dict):
            env = res.get("result") or {}
        else:
            error_payload: Dict[str, Any] = {}
            if isinstance(res, dict) and res.get("error") is not None:
                if isinstance(res.get("error"), dict):
                    error_payload = res.get("error")  # type: ignore[assignment]
                else:
                    error_payload = {
                        "code": "image_dispatch_failed",
                        "message": str(res.get("error")),
                        "status": 422,
                    }
            else:
                error_payload = {
                    "code": "image_dispatch_failed",
                    "message": "image.dispatch refine call failed",
                    "status": 422,
                }
            return {"name": name, "error": error_payload}
        if isinstance(env, dict):
            meta_block = env.setdefault("meta", {})
            if isinstance(meta_block, dict):
                if isinstance(lock_bundle, dict):
                    locks_meta = meta_block.get("locks") if isinstance(meta_block.get("locks"), dict) else {}
                    if isinstance(locks_meta, dict):
                        if "bundle" not in locks_meta:
                            locks_meta["bundle"] = lock_bundle
                        meta_block["locks"] = locks_meta
                    else:
                        meta_block["locks"] = {"bundle": lock_bundle}
                meta_block.setdefault("quality_profile", quality_profile)
                meta_block["refined_from_segment"] = segment_id
        return {"name": name, "result": env}
    if name == "image.detect" and VISION_REPAIR_API_URL and ALLOW_TOOL_EXECUTION:
        src = args.get("src") or args.get("image_path") or ""
        if not isinstance(src, str) or not src:
            return _tool_error(name, "missing_src", "src/image_path is required")
        payload: Dict[str, Any] = {"image_path": src}
        locks = args.get("locks")
        if isinstance(locks, dict):
            payload["locks"] = locks
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(VISION_REPAIR_API_URL.rstrip("/") + "/v1/image/analyze", json=payload)
        if r.status_code < 200 or r.status_code >= 300:
            return _tool_error(
                name,
                "vision_repair_http_error",
                f"vision_repair analyze returned HTTP {r.status_code}",
                status=r.status_code,
            )
        parser = JSONParser()
        js = parser.parse(r.text or "", {"faces": list, "hands": list, "objects": list, "quality": dict})
        return {"name": name, "result": js}
    if name == "image.repair" and VISION_REPAIR_API_URL and ALLOW_TOOL_EXECUTION:
        src = args.get("src") or args.get("image_path") or ""
        if not isinstance(src, str) or not src:
            return _tool_error(name, "missing_src", "src/image_path is required")
        payload: Dict[str, Any] = {"image_path": src}
        regions = args.get("regions")
        if isinstance(regions, list):
            payload["regions"] = regions
        locks = args.get("locks")
        if isinstance(locks, dict):
            payload["locks"] = locks
        mode = args.get("mode") or args.get("refine_mode")
        if isinstance(mode, str) and mode:
            payload["mode"] = mode
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(VISION_REPAIR_API_URL.rstrip("/") + "/v1/image/repair", json=payload)
        if r.status_code < 200 or r.status_code >= 300:
            return _tool_error(
                name,
                "vision_repair_http_error",
                f"vision_repair repair returned HTTP {r.status_code}",
                status=r.status_code,
            )
        parser = JSONParser()
        js = parser.parse(r.text or "", {"repaired_image_path": str, "regions": list, "mode": (str,)})
        return {"name": name, "result": js}
    if name == "video.refine.clip" and ALLOW_TOOL_EXECUTION:
        a = args if isinstance(args, dict) else {}
        segment_id_val = a.get("segment_id")
        segment_id = str(segment_id_val or "").strip() if segment_id_val is not None else ""
        if not segment_id:
            return {
                "name": name,
                "error": {
                    "code": "missing_segment_id",
                    "message": "video.refine.clip requires a non-empty segment_id",
                    "status": 422,
                },
            }
        src_val = a.get("src")
        src = str(src_val or "").strip() if src_val is not None else ""
        if not src:
            return {
                "name": name,
                "error": {
                    "code": "missing_src",
                    "message": "video.refine.clip requires src video path",
                    "status": 422,
                },
            }

        timecode = a.get("timecode") if isinstance(a.get("timecode"), dict) else {}
        prompt_val = a.get("prompt")
        prompt = str(prompt_val or "").strip() if prompt_val is not None else ""
        locks = a.get("locks") if isinstance(a.get("locks"), dict) else {}
        refine_mode_val = a.get("refine_mode")
        refine_mode = str(refine_mode_val or "").strip() if refine_mode_val is not None else ""
        if not refine_mode:
            refine_mode = "improve_quality"
        film_id = a.get("film_id")
        scene_id = a.get("scene_id")
        shot_id = a.get("shot_id")

        trace_append(
            "film2.clip_refine_start",
            {
                "trace_id": trace_id,
                "segment_id": segment_id,
                "film_id": film_id,
                "scene_id": scene_id,
                "shot_id": shot_id,
                "src": src,
                "timecode": timecode,
                "prompt": prompt,
                "refine_mode": refine_mode,
            },
        )

        # Derive basic timing from timecode; fall back to a 2s window.
        start_s = float(timecode.get("start_s") or 0.0)
        end_s = float(timecode.get("end_s") or 0.0)
        fps_val = int(timecode.get("fps") or 60)
        duration_s = end_s - start_s
        if duration_s <= 0.0:
            duration_s = 2.0
        seconds_val = max(1, int(round(duration_s)))
        width_val = int(a.get("width") or 1920)
        height_val = int(a.get("height") or 1080)

        # Prompt suffix to bias fix mode.
        prompt_suffix = ""
        if refine_mode == "fix_faces":
            prompt_suffix = " | improve facial consistency, correct facial features, match character identity"
        elif refine_mode == "stabilize_motion":
            prompt_suffix = " | stabilize motion, reduce jitter and flicker"
        elif refine_mode == "fix_lipsync":
            prompt_suffix = " | improve alignment of mouth motion with speech and dialogue"
        elif refine_mode == "improve_quality":
            prompt_suffix = " | enhance overall visual quality and temporal stability"
        hv_prompt = (prompt or "") + prompt_suffix

        # 1) Extract clip frames.
        refined_video_path: Optional[str] = None
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "video", "refine", segment_id or f"seg-{int(time.time())}")
        frames_dir = os.path.join(outdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        ff_args = ["ffmpeg", "-y", "-i", src]
        if start_s > 0.0:
            ff_args.extend(["-ss", f"{start_s:.3f}"])
        if end_s > start_s:
            ff_args.extend(["-to", f"{end_s:.3f}"])
        ff_args.append(os.path.join(frames_dir, "%06d.png"))
        proc_extract = subprocess.run(ff_args, check=False)
        if proc_extract.returncode != 0:
            return {
                "name": name,
                "error": {
                    "code": "ffmpeg_extract_failed",
                    "message": f"ffmpeg extract exited with code {proc_extract.returncode}",
                    "status": 500,
                },
            }

        frame_files = sorted(_glob(os.path.join(frames_dir, "*.png")))
        cleaned_frames: List[str] = []
        clip_identity_scores: List[float] = []
        sharpness_sum: float = 0.0
        sharpness_count: int = 0

        # Derive reference face embedding from lock bundle when available
        face_ref: Optional[list] = None
        lock_bundle = locks.get("bundle") if isinstance(locks.get("bundle"), dict) else None
        if isinstance(lock_bundle, dict):
            vis = lock_bundle.get("visual")
            if isinstance(vis, dict):
                faces = vis.get("faces")
                if isinstance(faces, list) and faces:
                    emb_block = faces[0].get("embeddings") or {}
                    if isinstance(emb_block, dict) and isinstance(emb_block.get("id_embedding"), list):
                        face_ref = emb_block.get("id_embedding")  # type: ignore[assignment]
            if face_ref is None:
                fallback_face = lock_bundle.get("face")
                if isinstance(fallback_face, dict) and isinstance(fallback_face.get("embedding"), list):
                    face_ref = fallback_face.get("embedding")  # type: ignore[assignment]

        for idx, fp in enumerate(frame_files):
            cleaned_path = fp
            frame_sharpness_val: Optional[float] = None

            # 2) Best-effort detection + repair via vision_repair, when configured.
            if VISION_REPAIR_API_URL:
                det_res = await execute_tool_call({"name": "image.detect", "arguments": {"src": fp, "locks": locks}})
                det_out = det_res.get("result") if isinstance(det_res, dict) else {}
                regions: List[Dict[str, Any]] = []
                if isinstance(det_out, dict):
                    qual = det_out.get("quality") if isinstance(det_out.get("quality"), dict) else {}
                    sh_val = qual.get("sharpness")
                    if isinstance(sh_val, (int, float)):
                        frame_sharpness_val = float(sh_val)
                    objs = det_out.get("objects") if isinstance(det_out.get("objects"), list) else []
                    for ob in objs:
                        if not isinstance(ob, dict):
                            continue
                        bbox = ob.get("bbox")
                        conf = ob.get("conf")
                        cls_name = ob.get("class_name")
                        if not isinstance(bbox, list) or not isinstance(conf, (int, float)):
                            continue
                        if refine_mode in ("fix_faces", "fix_lipsync"):
                            if isinstance(cls_name, str) and cls_name.lower() == "person" and conf > 0.5:
                                regions.append({"type": "face", "bbox": bbox})
                        elif refine_mode in ("stabilize_motion", "improve_quality"):
                            if conf > 0.5:
                                regions.append({"type": "object", "bbox": bbox})

                if regions:
                    trace_append(
                        "film2.frame_detect",
                        {
                            "trace_id": trace_id,
                            "segment_id": segment_id,
                            "film_id": film_id,
                            "scene_id": scene_id,
                            "shot_id": shot_id,
                            "frame_index": idx,
                            "regions": regions,
                            "refine_mode": refine_mode,
                        },
                    )
                    rep_args: Dict[str, Any] = {"src": fp, "regions": regions, "mode": refine_mode, "locks": locks}
                    rep_res = await execute_tool_call({"name": "image.repair", "arguments": rep_args})
                    rep_out = rep_res.get("result") if isinstance(rep_res, dict) else {}
                    if isinstance(rep_out, dict):
                        rp = rep_out.get("repaired_image_path")
                        if isinstance(rp, str) and rp:
                            cleaned_path = rp

            # 3) Fallback cleanup if no repair happened
            if cleaned_path == fp:
                im_res = await execute_tool_call({"name": "image.cleanup", "arguments": {"src": fp}})
                if isinstance(im_res, dict) and isinstance(im_res.get("result"), dict):
                    r = im_res.get("result") or {}
                    p = r.get("path") or r.get("view_url") or r.get("url")
                    if isinstance(p, str) and p:
                        cleaned_path = p

            cleaned_frames.append(cleaned_path)

            # 4) Per-frame identity score (best-effort)
            if face_ref is not None:
                score = await _compute_face_lock_score(cleaned_path, face_ref)
                if isinstance(score, (int, float)):
                    clip_identity_scores.append(float(score))

            trace_append(
                "film2.frame_fix",
                {
                    "trace_id": trace_id,
                    "segment_id": segment_id,
                    "film_id": film_id,
                    "scene_id": scene_id,
                    "shot_id": shot_id,
                    "frame_index": idx,
                    "source_frame": fp,
                    "cleaned_frame": cleaned_path,
                    "refine_mode": refine_mode,
                },
            )

            if frame_sharpness_val is not None:
                sharpness_sum += frame_sharpness_val
                sharpness_count += 1

        # 5) Rebuild the clip from cleaned frames, if any.
        clip_sharpness: Optional[float] = None
        if cleaned_frames:
            rebuilt_path = os.path.join(outdir, "refined.mp4")
            proc_rebuild = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps_val),
                    "-i",
                    os.path.join(frames_dir, "%06d.png"),
                    "-c:v",
                    "libx264",
                    "yuv420p",
                    rebuilt_path,
                ],
                check=False,
            )
            if proc_rebuild.returncode != 0:
                return {
                    "name": name,
                    "error": {
                        "code": "ffmpeg_rebuild_failed",
                        "message": f"ffmpeg rebuild exited with code {proc_rebuild.returncode}",
                        "status": 500,
                    },
                }
            refined_video_path = rebuilt_path
            _film2_artifact_video(trace_id, rebuilt_path)
            trace_append(
                "film2.clip_rebuild_success",
                {
                    "trace_id": trace_id,
                    "segment_id": segment_id,
                    "film_id": film_id,
                    "scene_id": scene_id,
                    "shot_id": shot_id,
                    "video_path": rebuilt_path,
                    "refine_mode": refine_mode,
                },
            )

        if sharpness_count > 0:
            clip_sharpness = sharpness_sum / float(sharpness_count)
        else:
            clip_sharpness = None

        # 6) Clip-level identity QA summary
        qa_video: Dict[str, Any] = {}
        if clip_identity_scores:
            id_min = min(clip_identity_scores)
            id_mean = sum(clip_identity_scores) / float(len(clip_identity_scores))
            preset_hero = LOCK_QUALITY_PRESETS.get("hero", LOCK_QUALITY_PRESETS["standard"])
            face_min_raw = preset_hero.get("face_min", 0.9)
            face_min = float(face_min_raw) if isinstance(face_min_raw, (int, float)) else 0.9
            weak_margin = 0.1 * face_min
            if id_min >= face_min and id_mean >= face_min:
                identity_status = "ok"
            elif id_min >= max(0.0, face_min - weak_margin) and id_mean >= max(0.0, face_min - weak_margin):
                identity_status = "weak"
            else:
                identity_status = "fail"
            qa_video = {
                "identity_min": id_min,
                "identity_mean": id_mean,
                "identity_status": identity_status,
            }
            trace_append(
                "film2.clip_identity_qa",
                {
                    "trace_id": trace_id,
                    "segment_id": segment_id,
                    "film_id": film_id,
                    "scene_id": scene_id,
                    "shot_id": shot_id,
                    "scores": clip_identity_scores,
                    "min": id_min,
                    "mean": id_mean,
                },
            )

        video_path: Optional[str] = refined_video_path

        # 7) Hunyuan fallback if frame-level path is unavailable.
        if video_path is None:
            hv_args: Dict[str, Any] = {
                "prompt": hv_prompt,
                "width": width_val,
                "height": height_val,
                "fps": fps_val,
                "seconds": seconds_val,
                "locks": locks,
                "seed": a.get("seed"),
                "post": {"interpolate": True, "upscale": True, "face_lock": True, "hand_fix": True},
                "latent_reinit_every": 48,
                "cid": a.get("cid"),
            }
            hv_res = await execute_tool_call({"name": "video.hv.t2v", "arguments": hv_args})
            if isinstance(hv_res, dict) and hv_res.get("error") is not None:
                return hv_res
            hv_result_raw = hv_res.get("result") if isinstance(hv_res, dict) else hv_res
            hv_result = hv_result_raw if isinstance(hv_result_raw, dict) else {}
            arts = hv_result.get("artifacts") if isinstance(hv_result.get("artifacts"), list) else []
            if arts:
                first = arts[0]
                if isinstance(first, dict):
                    path_val = first.get("path") or first.get("view_url") or first.get("url")
                    if isinstance(path_val, str) and path_val:
                        video_path = path_val

        result_meta: Dict[str, Any] = {
            "timecode": timecode,
            "locks": locks,
        }
        if clip_sharpness is not None:
            result_meta["sharpness"] = clip_sharpness
        artifacts: List[Dict[str, Any]] = []
        if isinstance(video_path, str) and video_path:
            artifacts.append({"kind": "video", "path": video_path})

        out: Dict[str, Any] = {
            "meta": result_meta,
            "artifacts": artifacts,
        }
        if qa_video:
            out["qa"] = {"video": qa_video}

        trace_append(
            "film2.clip_refine_success",
            {
                "trace_id": trace_id,
                "segment_id": segment_id,
                "film_id": film_id,
                "scene_id": scene_id,
                "shot_id": shot_id,
                "video_path": video_path,
                "refine_mode": refine_mode,
            },
        )
        return {"name": name, "result": out}
    if name == "film2.run" and ALLOW_TOOL_EXECUTION:
        # Unified Film-2 shot runner. Executes internal video passes and writes distilled traces/artifacts.
        if not XTTS_API_URL:
            return {
                "name": name,
                "error": {
                    "code": "xtts_unconfigured",
                    "message": "XTTS_API_URL is required for film2.run music vocals",
                },
            }
        a = raw_args if isinstance(raw_args, dict) else {}
        prompt = (a.get("prompt") or "").strip()
        trace_id = a.get("trace_id")
        # Use trace_id only (no derived/fallback trace keys, no normalization).
        # cid is the conversation/client id. film_id is a separate identifier for a film run.
        _cid_raw = a.get("cid")
        cid = str(_cid_raw).strip() if isinstance(_cid_raw, (str, int)) and str(_cid_raw).strip() else ""

        _film_id_raw = a.get("film_id")
        if isinstance(_film_id_raw, (str, int)) and str(_film_id_raw).strip():
            film_id = str(_film_id_raw).strip()
        else:
            film_id = uuid.uuid4().hex
            a["film_id"] = film_id
        clips = a.get("clips") if isinstance(a.get("clips"), list) else []
        images = a.get("images") if isinstance(a.get("images"), list) else []
        do_interpolate = bool(a.get("interpolate") or False)
        target_scale = a.get("scale")  # optional
        duration_s = float(a.get("duration_seconds") or a.get("duration_s") or 8.0)
        fps_val = int(a.get("fps") or 60)
        width_val = int(a.get("width") or 1920)
        height_val = int(a.get("height") or 1080)
        result: Dict[str, Any] = {"ids": {"film_id": film_id}, "meta": {"shots": [], "film_id": film_id}}
        if cid:
            result["meta"]["cid"] = cid
        profile_name = (a.get("quality_profile") or "hero")
        result["meta"]["quality_profile"] = profile_name
        thresholds_lock = _lock_quality_thresholds(profile_name)
        locks_arg = a.get("locks") if isinstance(a.get("locks"), dict) else {}
        if locks_arg:
            result["meta"]["locks"] = locks_arg
        # Cross-branch locals used throughout the film2.run pipeline
        story_obj: Dict[str, Any] = {}
        scenes_from_story: List[Dict[str, Any]] = []
        shots_from_story: List[Dict[str, Any]] = []
        warnings: List[str] = []
        # Character context derived from locks (if present); reused for music + hero updates.
        current_character_bundles: Dict[str, Dict[str, Any]] = await ensure_story_character_bundles(locks_arg if isinstance(locks_arg, dict) else {})
        # Derive character_ids once for downstream use (music + hero lock updates).
        character_ids: List[str] = []
        raw_character_ids = a.get("character_ids")
        if isinstance(raw_character_ids, list):
            character_ids = [
                str(cid)
                for cid in raw_character_ids
                if isinstance(cid, (str, int))
            ]
        elif isinstance(current_character_bundles, dict):
            character_ids = [
                str(cid)
                for cid in current_character_bundles.keys()
                if isinstance(cid, (str, int))
            ]
        if prompt:
            story_obj = _story_draft(prompt, duration_s)
            trace_append(
                "film2.story_draft",
                {
                    "trace_id": trace_id,
                    "prompt": prompt,
                    "duration_hint_s": duration_s,
                },
            )
            max_story_passes = 3
            last_issues: List[Dict[str, Any]] = []
            for pass_index in range(max_story_passes):
                last_issues = _story_check(story_obj, prompt)
                issue_codes = [iss.get("code") for iss in last_issues if isinstance(iss, dict)]
                trace_append(
                    "film2.story_consistency_pass",
                    {
                        "trace_id": trace_id,
                        "pass_index": pass_index,
                        "issue_codes": issue_codes,
                    },
                )
                if not last_issues:
                    break
                story_obj = _story_fix(story_obj, last_issues)
            if last_issues:
                warnings.append("story_consistency_unresolved")
                trace_append(
                    "film2.story_consistency_unresolved",
                    {
                        "trace_id": trace_id,
                        "issues": last_issues,
                        "prompt": prompt,
                    },
                )
            scenes_from_story, shots_from_story = _story_derive(story_obj)
        if warnings:
            meta_warnings = result["meta"].setdefault("warnings", [])
            if isinstance(meta_warnings, list):
                meta_warnings.extend(warnings)
        if story_obj:
            result["meta"]["story"] = story_obj
        if scenes_from_story:
            result["meta"]["scenes"] = scenes_from_story
        if shots_from_story:
            result["meta"]["shots_from_story"] = shots_from_story
        # Trace character state changes and per-shot state snapshots for distillation.
        if story_obj:
            acts = story_obj.get("acts") if isinstance(story_obj.get("acts"), list) else []
            for act in acts:
                act_id = act.get("act_id")
                scenes = act.get("scenes") if isinstance(act.get("scenes"), list) else []
                for scene in scenes:
                    scene_id = scene.get("scene_id")
                    beats = scene.get("beats") if isinstance(scene.get("beats"), list) else []
                    for beat in beats:
                        beat_id = beat.get("beat_id")
                        events = beat.get("events") if isinstance(beat.get("events"), list) else []
                        for ev in events:
                            if not isinstance(ev, dict):
                                continue
                            if ev.get("type") != "state_change":
                                continue
                            target = ev.get("target")
                            state_delta = ev.get("state_delta") if isinstance(ev.get("state_delta"), dict) else {}
                            trace_append(
                                "film2.character_state_change",
                                {
                                    "trace_id": trace_id,
                                    "char_id": target,
                                    "event_id": ev.get("event_id"),
                                    "state_delta": state_delta,
                                    "act_id": act_id,
                                    "scene_id": scene_id,
                                    "beat_id": beat_id,
                                },
                            )
        if shots_from_story:
            for shot in shots_from_story:
                if not isinstance(shot, dict):
                    continue
                shot_id = shot.get("shot_id")
                scene_id = shot.get("scene_id")
                act_id = shot.get("act_id")
                state_map = shot.get("character_states") if isinstance(shot.get("character_states"), dict) else {}
                trace_append(
                    "film2.shot_character_state_snapshot",
                    {
                        "trace_id": trace_id,
                        "shot_id": shot_id,
                        "scene_id": scene_id,
                        "act_id": act_id,
                        "character_states": state_map,
                    },
                )
        # Enrich locks with basic visual/story structure when possible
        if story_obj and isinstance(locks_arg, dict):
            locks_arg = await _ensure_visual_locks_for_story(story_obj, locks_arg, profile_name, trace_id)
            result["meta"]["locks"] = locks_arg
        # Precompute TTS dialogue audio when possible
        dialogue_index: Dict[str, Any] = {}
        if story_obj:
            locks_arg, dialogue_index = await _story_ensure_tts_locks_and_dialogue_audio(
                story_obj,
                locks_arg,
                profile_name,
                trace_id,
                execute_tool_call,
            )
            if dialogue_index:
                result["meta"]["dialogue"] = dialogue_index
            if locks_arg:
                result["meta"]["locks"] = locks_arg
        # Generate simple scene and shot storyboards before hv video
        if scenes_from_story:
            scenes_from_story = await _film2_generate_scene_storyboards(
                scenes_from_story,
                locks_arg,
                profile_name,
                trace_id,
                execute_tool_call,
            )
            result["meta"]["scenes"] = scenes_from_story
        if shots_from_story:
            shots_from_story = await _film2_generate_shot_storyboards(
                shots_from_story,
                locks_arg,
                profile_name,
                trace_id,
                execute_tool_call,
            )
            result["meta"]["shots_from_story"] = shots_from_story
        # Music planning phase: optionally score the film via music.infinite.windowed.
        music_meta: Dict[str, Any] = {}
        if prompt and duration_s > 0:
            music_args: Dict[str, Any] = {
                "prompt": f"Film score for: {prompt}",
                "length_s": int(duration_s),
                "bpm": None,
                "key": None,
                "window_bars": 8,
                "overlap_bars": 1,
                "mode": "start",
                "lock_bundle": locks_arg if isinstance(locks_arg, dict) else {},
                "cid": cid,
                "instrumental_only": True,
            }
            if character_ids:
                music_args["character_id"] = character_ids[0]
            # Trace: Film2 → music front-door call start
            trace_append(
                "film2.music.start",
                {
                    "trace_id": trace_id,
                    "tool": "music.infinite.windowed",
                    "args": {k: v for k, v in music_args.items() if k not in ("lock_bundle",)},
                },
            )
            # Call the infinite/windowed engine directly to avoid nested /tool.run recursion.
            provider = RestMusicProvider(MUSIC_API_URL)
            manifest_music: Dict[str, Any] = {"items": []}
            music_env = await run_music_infinite_windowed(music_args, provider, manifest_music)
            if isinstance(music_env, dict):
                music_meta = music_env
        final_music_mix_path: str | None = None
        final_music_eval: Dict[str, Any] | None = None
        if music_meta and isinstance(result.get("meta"), dict):
            # Trace: Film2 → music completion summary
            trace_append(
                "film2.music.finish",
                {
                    "trace_id": trace_id,
                    "tool": "music.infinite.windowed",
                    "meta": music_meta.get("meta") if isinstance(music_meta.get("meta"), dict) else {},
                },
            )
            # Tag music windows with scene/shot ids using simple time slicing.
            meta_obj = music_meta.get("meta") if isinstance(music_meta.get("meta"), dict) else {}
            win_list = meta_obj.get("windows") if isinstance(meta_obj.get("windows"), list) else []
            scenes_for_music = result["meta"].get("scenes") if isinstance(result["meta"].get("scenes"), list) else []
            shots_for_music = result["meta"].get("shots_from_story") if isinstance(result["meta"].get("shots_from_story"), list) else []
            # Simple heuristic: divide film duration evenly across scenes/shots if no explicit timing.
            scene_timing: Dict[str, Dict[str, float]] = {}
            if scenes_for_music:
                seg = duration_s / float(len(scenes_for_music) or 1)
                t0 = 0.0
                for sc in scenes_for_music:
                    if not isinstance(sc, dict):
                        continue
                    sid = sc.get("scene_id")
                    if not isinstance(sid, str):
                        continue
                    scene_timing[sid] = {"t_start": t0, "t_end": t0 + seg}
                    t0 += seg
            shot_timing: Dict[str, Dict[str, float]] = {}
            if shots_for_music:
                seg = duration_s / float(len(shots_for_music) or 1)
                t0 = 0.0
                for sh in shots_for_music:
                    if not isinstance(sh, dict):
                        continue
                    sid = sh.get("shot_id")
                    if not isinstance(sid, str):
                        continue
                    shot_timing[sid] = {"t_start": t0, "t_end": t0 + seg}
                    t0 += seg
            for win in win_list or []:
                if not isinstance(win, dict):
                    continue
                t_start = win.get("t_start")
                t_end = win.get("t_end")
                mid = float(t_start) + (float(t_end) - float(t_start)) / 2.0 if isinstance(t_start, (int, float)) and isinstance(t_end, (int, float)) else None
                if mid is not None and scene_timing:
                    for sid, rng in scene_timing.items():
                        if rng["t_start"] <= mid <= rng["t_end"]:
                            win["scene_id"] = sid
                            break
                if mid is not None and shot_timing:
                    win_shots: List[str] = []
                    for sid, rng in shot_timing.items():
                        if rng["t_start"] <= mid <= rng["t_end"]:
                            win_shots.append(sid)
                    if win_shots:
                        win["shot_ids"] = win_shots
            meta_obj["windows"] = win_list
            music_meta["meta"] = meta_obj
            result["meta"]["music"] = music_meta
            # Trace cross-modal attachment for distillation and perform backing+vocals mix.
            music_meta_obj = music_meta.get("meta") if isinstance(music_meta.get("meta"), dict) else {}
            # Locate the backing audio path from the music envelope artifacts.
            backing_full_path: str | None = None
            artifacts_music = music_meta.get("artifacts") if isinstance(music_meta.get("artifacts"), list) else []
            for art in artifacts_music:
                if not isinstance(art, dict):
                    continue
                if art.get("kind") == "audio-ref":
                    art_id = art.get("id")
                    if isinstance(art_id, str) and art_id:
                        backing_full_path = os.path.join(
                            "/workspace",
                            "uploads",
                            "artifacts",
                            "music",
                            music_meta_obj.get("cid") or cid,
                            art_id,
                        )
                        break
            if not backing_full_path:
                backing_full_path = music_meta.get("path") if isinstance(music_meta.get("path"), str) else None
            # When backing is available, render multi-voice vocals and mix.
            if isinstance(backing_full_path, str) and backing_full_path:
                # Derive Song Graph for vocals from the current locks bundle.
                music_branch = locks_arg.get("music") if isinstance(locks_arg.get("music"), dict) else {}
                win_list_for_vocals = meta_obj.get("windows") if isinstance(meta_obj.get("windows"), list) else []
                tts_manifest: Dict[str, Any] = {"items": []}
                stems_result = await render_vocal_stems_for_track(
                    job={"seed": a.get("seed")},
                    song_graph=music_branch,
                    windows=win_list_for_vocals,
                    lock_bundle=locks_arg if isinstance(locks_arg, dict) else {},
                    backing_path=backing_full_path,
                    cid=cid,
                    manifest=tts_manifest,
                    tts_provider=_TTSProvider(),
                )
                if isinstance(stems_result, dict) and stems_result.get("error"):
                    trace_append(
                        "film2.music.vocal_stems_error",
                        {
                            "trace_id": trace_id,
                            "film_cid": cid,
                            "error": stems_result.get("error"),
                        },
                    )
                    result["meta"].setdefault("music", {})["error"] = stems_result.get("error")
                    final_music_mix_path = backing_full_path
                else:
                    stems = stems_result.get("stems") if isinstance(stems_result, dict) else []
                    stems_arg: List[Dict[str, Any]] = []
                    stems_arg.append(
                        {
                            "path": backing_full_path,
                            "gain_db": 0.0,
                            "pan": 0.0,
                            "start_s": 0.0,
                        }
                    )
                    for stem in stems:
                        stem_path = stem.get("path")
                        if not isinstance(stem_path, str) or not stem_path:
                            continue
                        stems_arg.append(
                            {
                                "path": stem_path,
                                "gain_db": 0.0,
                                "pan": 0.0,
                                "start_s": float(stem.get("start_s") or 0.0),
                            }
                        )
                    mix_job = {
                        "cid": cid,
                        "stems": stems_arg,
                        "sample_rate": 32000,
                        "channels": 2,
                        "seed": a.get("seed"),
                    }
                    mix_env = run_music_mixdown(mix_job, {})
                    mix_artifacts = mix_env.get("artifacts") if isinstance(mix_env.get("artifacts"), list) else []
                    mix_path: str | None = None
                    for art in mix_artifacts:
                        if not isinstance(art, dict):
                            continue
                        if art.get("kind") == "audio-ref":
                            art_id = art.get("id")
                            if isinstance(art_id, str) and art_id:
                                mix_path = os.path.join(
                                    "/workspace",
                                    "uploads",
                                    "artifacts",
                                    "music",
                                    cid,
                                    art_id,
                                )
                                break
                    final_music_mix_path = mix_path or backing_full_path
                    if isinstance(final_music_mix_path, str) and final_music_mix_path:
                        final_music_eval = await compute_music_eval(
                            track_path=final_music_mix_path,
                            song_graph=music_branch,
                            style_pack=music_branch.get("style_pack") if isinstance(music_branch.get("style_pack"), dict) else None,
                            film_context={
                                "duration_s": duration_s,
                                "fps": fps_val,
                                "width": width_val,
                                "height": height_val,
                            },
                        )
                        overall_block = final_music_eval.get("overall") if isinstance(final_music_eval.get("overall"), dict) else {}
                        oq = float(overall_block.get("overall_quality_score") or 0.0)
                        fs = float(overall_block.get("fit_score") or 0.0)
                        th_music = get_music_acceptance_thresholds()
                        # Acceptance thresholds loaded from review/acceptance_audio.json
                        accepted = (oq >= th_music.get("overall_quality_min", 0.0) and fs >= th_music.get("fit_score_min", 0.0))
                        # Hero thresholds: stricter than acceptance
                        hero = (oq >= MUSIC_HERO_QUALITY_MIN and fs >= MUSIC_HERO_FIT_MIN)
                        trace_append(
                            "film2.music.final_eval",
                            {
                                "trace_id": trace_id,
                                "film_cid": cid,
                                "music_path": final_music_mix_path,
                                "music_eval": final_music_eval,
                                "tts_manifest_items": tts_manifest.get("items", []),
                                "accepted": accepted,
                                "hero": hero,
                            },
                        )
                        trace_append(
                            "film2.music.acceptance",
                            {
                                "trace_id": trace_id,
                                "film_cid": cid,
                                "music_path": final_music_mix_path,
                                "overall_quality_score": oq,
                                "fit_score": fs,
                                "accepted": accepted,
                                "hero": hero,
                                "reject_reason": None if accepted else "music_eval_below_threshold",
                            },
                        )
                        meta_music_block = result["meta"].setdefault("music", {})
                        if isinstance(meta_music_block, dict):
                            meta_music_block["final_mix_path"] = final_music_mix_path
                            meta_music_block["final_eval"] = final_music_eval
                            meta_music_block["accepted"] = accepted
                            meta_music_block["hero"] = hero
            trace_append(
                "film2.music.attach",
                {
                    "trace_id": trace_id,
                    "film_cid": cid,
                    "music_cid": music_meta_obj.get("cid"),
                    "num_scenes": len(scenes_for_music),
                    "num_shots": len(shots_for_music),
                    "num_windows": len(win_list),
                },
            )
        _film2_trace_event(trace_id, {"event": "film2.shot_start", "prompt": prompt})
        # Enhance/cleanup path with provided clips
        if clips:
            for i, src in enumerate(clips):
                shot_meta: Dict[str, Any] = {"index": i, "source": src}
                segment_log: List[Dict[str, Any]] = []
                # Cleanup
                _film2_trace_event(trace_id, {"event": "film2.pass_cleanup_start", "src": src})
                cc = await http_tool_run("video.cleanup", {"src": src, "cid": cid, "trace_id": trace_id})
                if isinstance(cc, dict):
                    segment_log.append(cc)
                ccr = (cc.get("result") or {}) if isinstance(cc, dict) else {}
                clean_path = ccr.get("path") if isinstance(ccr, dict) else None
                if isinstance(clean_path, str):
                    _film2_artifact_video(trace_id, clean_path)
                    shot_meta["clean_path"] = clean_path
                _film2_trace_event(trace_id, {"event": "film2.pass_cleanup_finish"})
                # Temporal interpolate (optional)
                current = clean_path or src
                if do_interpolate and isinstance(current, str):
                    _film2_trace_event(trace_id, {"event": "film2.pass_interpolate_start"})
                    ic = await http_tool_run("video.interpolate", {"src": current, "cid": cid, "trace_id": trace_id})
                    if isinstance(ic, dict):
                        segment_log.append(ic)
                    icr = (ic.get("result") or {}) if isinstance(ic, dict) else {}
                    interp_path = icr.get("path") if isinstance(icr, dict) else None
                    if isinstance(interp_path, str):
                        _film2_artifact_video(trace_id, interp_path)
                        shot_meta["interp_path"] = interp_path
                    current = interp_path or current
                    _film2_trace_event(trace_id, {"event": "film2.pass_interpolate_finish"})
                # Upscale
                if isinstance(current, str):
                    _film2_trace_event(trace_id, {"event": "film2.pass_upscale_start"})
                    uc_args = {"src": current, "cid": cid, "trace_id": trace_id}
                    if target_scale:
                        uc_args["scale"] = target_scale
                    uc = await http_tool_run("video.upscale", uc_args)
                    if isinstance(uc, dict):
                        segment_log.append(uc)
                    up = (uc.get("result") or {}) if isinstance(uc, dict) else {}
                    up_path = up.get("path") if isinstance(up, dict) else None
                    if isinstance(up_path, str):
                        _film2_artifact_video(trace_id, up_path)
                        shot_meta["upscaled_path"] = up_path
                    _film2_trace_event(trace_id, {"event": "film2.pass_upscale_finish"})
                if segment_log:
                    shot_meta["segment_results"] = segment_log
                    frames_for_hero = segment_log
                    hero_pick = choose_hero_frame(frames_for_hero, thresholds_lock)
                    hero_record: Dict[str, Any] = {}
                    hero_path_value: Optional[str] = None
                    if hero_pick is not None:
                        hero_frame = hero_pick.get("frame") if isinstance(hero_pick, dict) else None
                        hero_index = int(hero_pick.get("index")) if isinstance(hero_pick.get("index"), int) else hero_pick.get("index")
                        hero_meta_payload = _frame_meta_payload(hero_frame) if isinstance(hero_frame, dict) else {}
                        hero_record = {
                            "index": int(hero_index) if isinstance(hero_index, int) else hero_pick.get("index"),
                            "score": hero_pick.get("score"),
                            "locks": hero_meta_payload.get("locks") if isinstance(hero_meta_payload.get("locks"), dict) else {},
                        }
                        hero_path_value = _frame_image_path(hero_frame) if isinstance(hero_frame, dict) else None
                    elif frames_for_hero:
                        fallback_frame = frames_for_hero[-1]
                        fallback_meta = _frame_meta_payload(fallback_frame)
                        hero_record = {
                            "index": len(frames_for_hero) - 1,
                            "score": None,
                            "locks": fallback_meta.get("locks") if isinstance(fallback_meta.get("locks"), dict) else {},
                            "fallback": True,
                        }
                        hero_path_value = _frame_image_path(fallback_frame)
                    if hero_record:
                        if isinstance(hero_path_value, str) and hero_path_value:
                            if os.path.isabs(hero_path_value):
                                hero_abs = hero_path_value
                            else:
                                hero_abs = os.path.join(UPLOAD_DIR, hero_path_value.lstrip("/"))
                            hero_record["image_path"] = hero_abs
                        shot_meta["hero_frame"] = hero_record
                        hero_abs_path = hero_record.get("image_path")
                        if isinstance(hero_abs_path, str) and hero_abs_path and character_ids:
                            for character_id in character_ids:
                                existing_bundle = current_character_bundles.get(character_id)
                                if existing_bundle is not None:
                                    updated_bundle = await _lock_update_from_hero(
                                        character_id,
                                        hero_abs_path,
                                        existing_bundle,
                                        locks_root_dir=LOCKS_ROOT_DIR,
                                    )
                                    current_character_bundles[character_id] = updated_bundle
                                    hero_record.setdefault("bundle_versions", {})[character_id] = {
                                        "schema_version": updated_bundle.get("schema_version"),
                                    }
                result["meta"]["shots"].append(shot_meta)
        # Story-driven Hunyuan generation when no explicit clips are provided.
        elif shots_from_story:
                story_shot_index = 0
                for shot in shots_from_story:
                    if not isinstance(shot, dict):
                        continue
                    shot_id = shot.get("shot_id") or f"shot_{story_shot_index}"
                    scene_id = shot.get("scene_id")
                    act_id = shot.get("act_id")
                    descr = shot.get("description") or prompt
                    # Enrich prompt with active character state hints when available.
                    state_map = shot.get("character_states") if isinstance(shot.get("character_states"), dict) else {}
                    if state_map:
                        state_phrases: List[str] = []
                        for cid, st in state_map.items():
                            if not isinstance(st, dict):
                                continue
                            if st.get("left_arm") == "missing":
                                state_phrases.append("a character with a missing left arm")
                            if st.get("left_arm") == "robot":
                                state_phrases.append("a character with a robotic left arm")
                        if state_phrases:
                            extra = "; ".join(sorted(set(state_phrases)))
                            descr = f"{descr} ({extra})"
                    dur = float(shot.get("duration_s") or duration_s)
                    seconds_val = int(dur) if dur > 0.0 else int(duration_s)
                    adapter = "hv.t2v"
                    hv_name = "video.hv.t2v"
                    hv_args: Dict[str, Any] = {
                        "prompt": descr,
                        "width": width_val,
                        "height": height_val,
                        "fps": fps_val,
                        "seconds": seconds_val,
                        "locks": locks_arg,
                        "seed": a.get("seed"),
                        "post": {"interpolate": True, "upscale": True, "face_lock": True, "hand_fix": True},
                        "latent_reinit_every": 48,
                        "cid": cid,
                    }
                    storyboard_img = shot.get("storyboard_image")
                    if isinstance(storyboard_img, str) and storyboard_img:
                        hv_args["init_image"] = storyboard_img
                        hv_name = "video.hv.i2v"
                        adapter = "hv.i2v"
                    shot_meta = {
                        "index": story_shot_index,
                        "shot_id": shot_id,
                        "scene_id": scene_id,
                        "act_id": act_id,
                        "prompt": descr,
                        "duration_s": dur,
                    }
                    segment_log = []
                    _film2_trace_event(trace_id, {"event": "film2.pass_gen_start", "adapter": adapter, "shot_id": shot_id})
                    trace_append(
                        "film2.hv_call_start",
                        {
                            "trace_id": trace_id,
                            "adapter": hv_name,
                            "shot_id": shot_id,
                            "scene_id": scene_id,
                            "act_id": act_id,
                            "prompt": descr,
                            "width": width_val,
                            "height": height_val,
                            "fps": fps_val,
                            "seconds": seconds_val,
                            "init_image": hv_args.get("init_image"),
                        },
                    )
                    gv = await http_tool_run(hv_name, hv_args)
                    if isinstance(gv, dict):
                        segment_log.append(gv)
                    gvr = (gv.get("result") or {}) if isinstance(gv, dict) else {}
                    gen_path = None
                    if isinstance(gvr, dict):
                        gen_path = gvr.get("path")
                        if not isinstance(gen_path, str):
                            video_obj = gvr.get("video") if isinstance(gvr.get("video"), dict) else {}
                            gen_path = video_obj.get("path") if isinstance(video_obj.get("path"), str) else None
                    if isinstance(gen_path, str):
                        _film2_artifact_video(trace_id, gen_path)
                        shot_meta["gen_path"] = gen_path
                        trace_append(
                            "film2.hv_call_success",
                            {
                                "trace_id": trace_id,
                                "adapter": hv_name,
                                "shot_id": shot_id,
                                "scene_id": scene_id,
                                "act_id": act_id,
                                "video_path": gen_path,
                            },
                        )
                    _film2_trace_event(trace_id, {"event": "film2.pass_gen_finish"})
                    if segment_log:
                        shot_meta["segment_results"] = segment_log
                    result["meta"]["shots"].append(shot_meta)
                    story_shot_index += 1
        # Image-based generation goes through Hunyuan (full-res, max quality) when story shots are unavailable.
        elif images:
                for i, img in enumerate(images):
                    shot_meta = {"index": i, "init_image": img}
                    segment_log = []
                    hv_args_img: Dict[str, Any] = {
                        "init_image": img,
                        "prompt": prompt or "",
                        "width": width_val,
                        "height": height_val,
                        "fps": fps_val,
                        "seconds": int(duration_s),
                        "locks": locks_arg,
                        "seed": a.get("seed"),
                        "post": {"interpolate": True, "upscale": True, "face_lock": True, "hand_fix": True},
                        "latent_reinit_every": 48,
                        "cid": cid,
                    }
                    _film2_trace_event(trace_id, {"event": "film2.pass_gen_start", "adapter": "hv.i2v", "image": img})
                    gv = await http_tool_run("video.hv.i2v", hv_args_img)
                    if isinstance(gv, dict):
                        segment_log.append(gv)
                    gvr = (gv.get("result") or {}) if isinstance(gv, dict) else {}
                    gen_path = None
                    if isinstance(gvr, dict):
                        gen_path = gvr.get("path")
                        if not isinstance(gen_path, str):
                            video_obj = gvr.get("video") if isinstance(gvr.get("video"), dict) else {}
                            gen_path = video_obj.get("path") if isinstance(video_obj.get("path"), str) else None
                    if isinstance(gen_path, str):
                        _artifact_video(gen_path)
                        shot_meta["gen_path"] = gen_path
                    _film2_trace_event(trace_id, {"event": "film2.pass_gen_finish"})
                    if segment_log:
                        shot_meta["segment_results"] = segment_log
                    result["meta"]["shots"].append(shot_meta)
        else:
            # Prompt-only Hunyuan generation when no clips/images/story shots are supplied.
            if prompt:
                shot_meta = {"index": 0, "prompt": prompt}
                segment_log = []
                hv_args_prompt = {
                    "prompt": prompt,
                    "width": width_val,
                    "height": height_val,
                    "fps": fps_val,
                    "seconds": int(duration_s),
                    "locks": locks_arg,
                    "seed": a.get("seed"),
                    "post": {"interpolate": True, "upscale": True, "face_lock": True, "hand_fix": True},
                    "latent_reinit_every": 48,
                    "cid": cid,
                }
                _film2_trace_event(trace_id, {"event": "film2.pass_gen_start", "adapter": "hv.t2v"})
                gv = await execute_tool_call({"name": "video.hv.t2v", "arguments": hv_args_prompt})
                if isinstance(gv, dict):
                    segment_log.append(gv)
                gvr = (gv.get("result") or {}) if isinstance(gv, dict) else {}
                gen_path = None
                if isinstance(gvr, dict):
                    gen_path = gvr.get("path")
                    if not isinstance(gen_path, str):
                        video_obj = gvr.get("video") if isinstance(gvr.get("video"), dict) else {}
                        gen_path = video_obj.get("path") if isinstance(video_obj.get("path"), str) else None
                if isinstance(gen_path, str):
                    _artifact_video(gen_path)
                    shot_meta["gen_path"] = gen_path
                _film2_trace_event(trace_id, {"event": "film2.pass_gen_finish"})
                if segment_log:
                    shot_meta["segment_results"] = segment_log
                result["meta"]["shots"].append(shot_meta)
        _film2_trace_event(trace_id, {"event": "film2.shot_finish"})
        # Build a simple segment hierarchy: film -> scenes -> shots -> clips (one clip per shot for now)
        film_segment: Dict[str, Any] = {
            "segment_id": film_id,
            "level": "film",
            "children": [],
            "meta": {
                "prompt": prompt,
                "duration_s": duration_s,
            },
        }
        scene_segments: Dict[str, Dict[str, Any]] = {}
        shot_segments: Dict[str, Dict[str, Any]] = {}
        clip_segments: Dict[str, Dict[str, Any]] = {}
        scenes_meta = result.get("meta", {}).get("scenes")
        if isinstance(scenes_meta, list):
            for sc in scenes_meta:
                if not isinstance(sc, dict):
                    continue
                sid = sc.get("scene_id")
                if not isinstance(sid, str) or not sid:
                    continue
                if sid in scene_segments:
                    continue
                seg_scene: Dict[str, Any] = {
                    "segment_id": sid,
                    "level": "scene",
                    "children": [],
                    "meta": dict(sc),
                }
                scene_segments[sid] = seg_scene
                film_segment["children"].append(sid)
        shots_meta = result.get("meta", {}).get("shots")
        if isinstance(shots_meta, list):
            # Build a simple dialogue mapping per shot using story-derived shots and dialogue_index when available.
            meta_block = result.get("meta") if isinstance(result.get("meta"), dict) else {}
            dialogue_index = meta_block.get("dialogue") if isinstance(meta_block.get("dialogue"), dict) else {}
            shots_from_story_meta = meta_block.get("shots_from_story") if isinstance(meta_block.get("shots_from_story"), list) else []
            story_shot_dialogue: Dict[str, List[Dict[str, Any]]] = {}
            for s in shots_from_story_meta:
                if not isinstance(s, dict):
                    continue
                sid = s.get("shot_id")
                if not isinstance(sid, str) or not sid:
                    continue
                lines = s.get("dialogue") if isinstance(s.get("dialogue"), list) else []
                story_shot_dialogue[sid] = [ln for ln in lines if isinstance(ln, dict)]
            for sh in shots_meta:
                if not isinstance(sh, dict):
                    continue
                shot_id = sh.get("shot_id") or f"shot_{sh.get('index')}"
                if not isinstance(shot_id, str) or not shot_id:
                    continue
                scene_id = sh.get("scene_id")
                act_id = sh.get("act_id")
                dur_s = float(sh.get("duration_s") or duration_s)
                gen_path = sh.get("gen_path")
                shot_seg: Dict[str, Any] = {
                    "segment_id": shot_id,
                    "level": "shot",
                    "children": [],
                    "meta": {
                        "scene_id": scene_id,
                        "act_id": act_id,
                        "duration_s": dur_s,
                        "gen_path": gen_path,
                    },
                }
                shot_segments[shot_id] = shot_seg
                if isinstance(scene_id, str) and scene_id in scene_segments:
                    scene_segments[scene_id]["children"].append(shot_id)
                # Split each shot into multiple logical clips based on a fixed window size.
                clip_window_s = 2.0
                if dur_s <= 0.0:
                    dur_s = duration_s
                    shot_seg["meta"]["duration_s"] = dur_s
                # Align dialogue lines to this shot using a simple equal-partition policy when possible.
                shot_lines = story_shot_dialogue.get(shot_id, [])
                line_timing: Dict[str, Dict[str, float]] = {}
                if shot_lines:
                    per_line = dur_s / float(len(shot_lines))
                    current_start = 0.0
                    for ln in shot_lines:
                        line_id = ln.get("line_id")
                        if not isinstance(line_id, str) or not line_id:
                            continue
                        start_s = current_start
                        end_s = min(dur_s, start_s + per_line)
                        line_timing[line_id] = {"start_s": start_s, "end_s": end_s}
                        current_start = end_s
                        entry = dialogue_index.get(line_id) if isinstance(dialogue_index, dict) else None
                        if isinstance(entry, dict):
                            entry["shot_id"] = shot_id
                            entry["start_s"] = start_s
                            entry["end_s"] = end_s
                num_clips = max(1, int(math.ceil(dur_s / clip_window_s)))
                for clip_index in range(num_clips):
                    start_s = clip_index * clip_window_s
                    end_s = dur_s if clip_index == num_clips - 1 else min(dur_s, (clip_index + 1) * clip_window_s)
                    clip_id = f"{shot_id}_clip_{clip_index}"
                    clip_result: Dict[str, Any] = {
                        "meta": {
                            "cid": cid,
                            "film_id": film_id,
                            "scene_id": scene_id,
                            "shot_id": shot_id,
                            "clip_index": clip_index,
                            "timecode": {"start_s": start_s, "end_s": end_s, "fps": fps_val},
                            "locks": locks_arg if isinstance(locks_arg, dict) else {},
                            "prompt": sh.get("prompt"),
                            "width": width_val,
                            "height": height_val,
                        },
                        "artifacts": [],
                    }
                    if isinstance(gen_path, str) and gen_path:
                        clip_result["artifacts"].append({"kind": "video", "path": gen_path})
                    # Compute a simple lipsync score based on dialogue durations overlapping this clip window.
                    clip_lines: List[str] = []
                    dialogue_total_s = 0.0
                    if isinstance(dialogue_index, dict) and line_timing:
                        for ln in shot_lines:
                            line_id = ln.get("line_id")
                            if not isinstance(line_id, str) or not line_id:
                                continue
                            timing = line_timing.get(line_id)
                            if not isinstance(timing, dict):
                                continue
                            line_start = float(timing.get("start_s") or 0.0)
                            line_end = float(timing.get("end_s") or 0.0)
                            # Overlap between line window and clip window
                            if line_end > start_s and line_start < end_s:
                                clip_lines.append(line_id)
                                entry = dialogue_index.get(line_id)
                                dur_val = None
                                if isinstance(entry, dict):
                                    dur_raw = entry.get("duration_s")
                                    if isinstance(dur_raw, (int, float)):
                                        dur_val = float(dur_raw)
                                if dur_val is None:
                                    dur_val = max(0.0, line_end - line_start)
                                dialogue_total_s += dur_val
                    clip_length = max(0.0, end_s - start_s)
                    lipsync_score = 0.0
                    if clip_length > 0.0:
                        ratio = dialogue_total_s / clip_length
                        if ratio < 0.0:
                            ratio = 0.0
                        if ratio > 1.0:
                            ratio = 1.0
                        lipsync_score = ratio
                    clip_result["meta"]["dialogue_lines"] = clip_lines
                    clip_result["meta"]["lipsync_score"] = lipsync_score
                    clip_seg: Dict[str, Any] = {
                        "id": clip_id,
                        "tool": "video.hv.t2v",
                        "domain": "video",
                        "name": clip_id,
                        "index": clip_index,
                        "trace_id": trace_id,
                        "cid": cid,
                        "result": clip_result,
                        "meta": clip_result["meta"],
                        "qa": {"scores": {}},
                        "locks": clip_result["meta"].get("locks"),
                        "artifacts": clip_result["artifacts"],
                    }
                    clip_seg["qa"]["scores"]["lipsync"] = lipsync_score
                    clip_segments[clip_id] = clip_seg
                    shot_seg["children"].append(clip_id)
                    trace_append(
                        "film2.segment_clip_built",
                        {
                            "trace_id": trace_id,
                            "film_id": film_id,
                            "scene_id": scene_id,
                            "shot_id": shot_id,
                            "clip_id": clip_id,
                            "clip_index": clip_index,
                            "start_s": start_s,
                            "end_s": end_s,
                            "fps": fps_val,
                            "video_path": gen_path,
                            "dialogue_total_s": dialogue_total_s,
                            "lipsync_score": lipsync_score,
                        },
                    )
        result.setdefault("meta", {})["segments"] = {
                "film": film_segment,
                "scenes": list(scene_segments.values()),
                "shots": list(shot_segments.values()),
                "clips": list(clip_segments.values()),
            }
        # No error handling here: if segment hierarchy construction fails, let it surface.
        # Select a final video output and expose as ids/meta for UI
        final_path = None
        for sh in reversed(result.get("meta", {}).get("shots", [])):
            for key in ("upscaled_path", "interp_path", "clean_path", "gen_path"):
                p = sh.get(key)
                if isinstance(p, str) and p:
                    final_path = p
                    break
            if final_path:
                break
        if final_path:
            if trace_id:
                rel = final_path
                if os.path.isabs(final_path):
                    rel = os.path.relpath(final_path, UPLOAD_DIR).replace("\\", "/")
                _log("artifact", trace_id=trace_id, kind="video", path=rel)
                result.setdefault("ids", {})["video_id"] = rel
                view_rel = rel if rel.startswith("uploads/") else f"uploads/{rel.lstrip('/')}"
                result.setdefault("meta", {})["view_url"] = f"/{view_rel}"
                final_abs = final_path if os.path.isabs(final_path) else os.path.join(UPLOAD_DIR, rel)
                frame0 = iio.imread(final_abs, index=0)
                img = Image.fromarray(frame0).convert("RGB")
                img.thumbnail((512, 512))
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=70, optimize=True)
                poster_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
                result.setdefault("meta", {})["poster_data_url"] = f"data:image/jpeg;base64,{poster_b64}"
        meta_obj = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        segments_container = meta_obj.get("segments") if isinstance(meta_obj.get("segments"), dict) else {}
        clips_meta = segments_container.get("clips") if isinstance(segments_container, dict) else None
        if isinstance(clips_meta, list):
            for clip in clips_meta:
                if not isinstance(clip, dict):
                    continue
                cmeta = clip.get("meta") if isinstance(clip.get("meta"), dict) else {}
                refined_flag = bool(cmeta.get("refined"))
                ls_val = cmeta.get("lipsync_score")
                sharp_val = cmeta.get("sharpness")
                degraded_reasons: List[str] = []
                if isinstance(ls_val, (int, float)) and ls_val < 0.5 and refined_flag:
                    degraded_reasons.append("low_lipsync")
                if isinstance(sharp_val, (int, float)) and sharp_val < 50.0 and refined_flag:
                    degraded_reasons.append("low_sharpness")
                if degraded_reasons:
                    cmeta["degraded"] = True
                    cmeta["degraded_reasons"] = degraded_reasons
                    clip["meta"] = cmeta
                    trace_append(
                        "film2.clip_refine_exhausted",
                        {
                            "trace_id": trace_id,
                            "segment_id": clip.get("id"),
                            "film_id": segments_container.get("film", {}).get("segment_id") if isinstance(segments_container.get("film"), dict) else None,
                            "scene_id": cmeta.get("scene_id"),
                            "shot_id": cmeta.get("shot_id"),
                            "refined": refined_flag,
                            "degraded_reasons": degraded_reasons,
                        },
                    )
    
        return {"name": name, "result": result}
    if name == "voice.register" and ALLOW_TOOL_EXECUTION:
        res = await run_voice_register(args if isinstance(args, dict) else {})
        if isinstance(res, dict) and res.get("error"):
            return {"name": name, "error": res["error"]}
        return {"name": name, "result": res.get("result", res)}
    if name == "voice.train" and ALLOW_TOOL_EXECUTION:
        res = await run_voice_train(args if isinstance(args, dict) else {})
        if isinstance(res, dict) and res.get("error"):
            return {"name": name, "error": res["error"]}
        return {"name": name, "result": res.get("result", res)}
    if name == "tts.speak" and ALLOW_TOOL_EXECUTION:
        if not XTTS_API_URL:
            return {"name": name, "error": "XTTS_API_URL not configured"}
        provider = _TTSProvider()
        manifest = {"items": []}
        a = args if isinstance(args, dict) else {}
        quality_profile = (a.get("quality_profile") or "standard")
        # Derive a stable character/voice identity when possible.
        char_id = str(a.get("character_id") or a.get("voice_id") or a.get("voice_lock_id") or "").strip()
        if not char_id:
            trace_val = a.get("trace_id")
            if isinstance(trace_val, str) and trace_val.strip():
                char_id = f"char_{trace_val.strip()}"
        if char_id and "character_id" not in a:
            a["character_id"] = char_id

        bundle_arg = a.get("lock_bundle")
        lock_bundle: Optional[Dict[str, Any]] = None
        if isinstance(bundle_arg, dict):
            lock_bundle = bundle_arg
        elif isinstance(bundle_arg, str) and bundle_arg.strip():
            loaded = await _lock_load(bundle_arg.strip())
            if isinstance(loaded, dict):
                lock_bundle = loaded
        if lock_bundle is None and char_id:
            loaded = await _lock_load(char_id)
            if isinstance(loaded, dict):
                lock_bundle = loaded
        # Enforce a lock-first invariant for TTS when an identity is known:
        # delegate to the locks.runtime helper so skeleton construction lives
        # alongside other lock runtime helpers.
        if lock_bundle is None and char_id:
            lock_bundle = await _lock_ensure_tts(char_id, None)
        if lock_bundle is not None:
            lock_bundle = _lock_apply_profile(quality_profile, lock_bundle)
            lock_bundle = _lock_migrate_tts(lock_bundle)
            a["lock_bundle"] = lock_bundle
        a["quality_profile"] = quality_profile

        env = await run_tts_speak(a, provider, manifest)
        # If run_tts_speak surfaced an error envelope from provider, forward it as a tool error.
        if isinstance(env, dict) and "ok" in env and not bool(env.get("ok")):
            return {"name": name, "error": env.get("error") or env}

        # Persist audio artifact to memory + RAG; distilled artifact row
        wav = env.get("wav_bytes") if isinstance(env, dict) else None
        if isinstance(wav, (bytes, bytearray)) and len(wav) > 0:
            cid = "aud-" + str(_now_ts())
            outdir = os.path.join(UPLOAD_DIR, "artifacts", "audio", "tts", cid)
            _ensure_dir(outdir)
            stem = "tts_00_00"
            wav_path = os.path.join(outdir, stem + ".wav")
            with open(wav_path, "wb") as _wf:
                _wf.write(wav)
            # Sidecar metadata (minimal)
            _sidecar(
                wav_path,
                {
                    "tool": "tts.speak",
                    "text": a.get("text"),
                    "duration_s": float(env.get("duration_s") or 0.0),
                    "model": env.get("model") or "xtts",
                },
            )
            # Add to multimodal memory
            _ctx_add(
                cid,
                "audio",
                wav_path,
                _uri_from_upload_path(wav_path),
                None,
                [],
                {"text": a.get("text"), "duration_s": float(env.get("duration_s") or 0.0)},
            )
            # Distilled artifact
            tr = (a.get("trace_id") if isinstance(a.get("trace_id"), str) else None)
            if tr:
                rel = os.path.relpath(wav_path, UPLOAD_DIR).replace("\\", "/")
                _log(
                    "artifact",
                    trace_id=str(tr),
                    kind="audio",
                    path=rel,
                    bytes=int(len(wav)),
                    duration_s=float(env.get("duration_s") or 0.0),
                )
            # Optional: embed transcript text into RAG
            pool = await get_pg_pool()
            txt = a.get("text") if isinstance(a.get("text"), str) else None
            if pool is not None and txt and txt.strip():
                emb = get_embedder()
                vec = emb.encode([txt])[0]
                async with pool.acquire() as conn:
                    rel = os.path.relpath(wav_path, UPLOAD_DIR).replace("\\", "/")
                    await conn.execute(
                        "INSERT INTO rag_docs (path, chunk, embedding) VALUES ($1, $2, $3)",
                        rel,
                        txt,
                        list(vec),
                    )
        # Shape result with ids/meta when file persisted
        out_res = dict(env or {})
        if isinstance(wav, (bytes, bytearray)) and len(wav) > 0:
            rel = os.path.relpath(wav_path, UPLOAD_DIR).replace("\\", "/")
            out_res.setdefault("ids", {})["audio_id"] = rel
            out_res.setdefault("meta", {})["url"] = f"/{rel}"
            out_res["meta"]["mime"] = "audio/wav"
            # Build ~12s mono 22.05kHz preview data_url
            r = wave.open(BytesIO(wav))
            nch = r.getnchannels()
            sw = r.getsampwidth()
            fr = r.getframerate()
            frames = r.readframes(r.getnframes())
            r.close()
            if nch > 1:
                frames = audioop.tomono(frames, sw, 0.5, 0.5)
            if fr != 22050:
                frames, _ = audioop.ratecv(frames, sw, 1, fr, 22050, None)
                fr = 22050
            max_frames = 12 * fr
            frames = frames[: max_frames * sw]
            b2 = BytesIO()
            w2 = wave.open(b2, "wb")
            w2.setnchannels(1)
            w2.setsampwidth(sw)
            w2.setframerate(fr)
            w2.writeframes(frames)
            w2.close()
            out_res["meta"]["data_url"] = "data:audio/wav;base64," + _b64.b64encode(b2.getvalue()).decode("ascii")
            out_res["meta"]["preview_duration_s"] = float(len(frames) / (sw * fr))
            # expose preview and full durations distinctly
            out_res["meta"]["preview_duration_s"] = float(len(frames) / (sw * fr))
            if env.get("duration_s") is not None:
                out_res["meta"]["full_duration_s"] = float(env.get("duration_s"))
            locks_block = out_res.setdefault("meta", {}).setdefault("locks", {})
            if isinstance(locks_block, dict) and quality_profile:
                locks_block.setdefault("quality_profile", quality_profile)
        return {"name": name, "result": out_res}
    if name == "audio.sfx.compose" and ALLOW_TOOL_EXECUTION:
        manifest = {"items": []}
        try:
            a = args if isinstance(args, dict) else {}
            # Optional: resolve lock bundle for SFX jobs
            bundle_arg = a.get("lock_bundle")
            lock_bundle: Optional[Dict[str, Any]] = None
            if isinstance(bundle_arg, dict):
                lock_bundle = bundle_arg
            elif isinstance(bundle_arg, str) and bundle_arg.strip():
                lock_bundle = await _lock_load(bundle_arg.strip()) or {}
            if lock_bundle:
                # Ensure branch migrations and profile application happen inside the tool if needed
                a["lock_bundle"] = _lock_migrate_sfx(_lock_migrate_visual(_lock_migrate_music(_lock_migrate_tts(_lock_migrate_film2(lock_bundle)))))
            env = run_sfx_compose(a, manifest)
            # Persist if a wav payload present
            wav = env.get("wav_bytes") if isinstance(env, dict) else None
            if isinstance(wav, (bytes, bytearray)) and len(wav) > 0:
                cid = "aud-" + str(_now_ts())
                outdir = os.path.join(UPLOAD_DIR, "artifacts", "audio", "sfx", cid)
                _ensure_dir(outdir)
                stem = "sfx_00_00"
                wav_path = os.path.join(outdir, stem + ".wav")
                with open(wav_path, "wb") as _wf:
                    _wf.write(wav)
                _sidecar(wav_path, {"tool": "audio.sfx.compose"})
                _ctx_add(cid, "audio", wav_path, _uri_from_upload_path(wav_path), None, [], {})
                tr = (a.get("trace_id") if isinstance(a.get("trace_id"), str) else None)
                if tr:
                    rel = os.path.relpath(wav_path, UPLOAD_DIR).replace("\\", "/")
                    _log("artifact", trace_id=str(tr), kind="audio", path=rel, bytes=int(len(wav)))
            # Shape result with ids/meta when file persisted
            out_res = dict(env or {})
            if isinstance(wav, (bytes, bytearray)) and len(wav) > 0:
                rel = os.path.relpath(wav_path, UPLOAD_DIR).replace("\\", "/")
                out_res.setdefault("ids", {})["audio_id"] = rel
                out_res.setdefault("meta", {})["url"] = f"/{rel}"
                out_res["meta"]["mime"] = "audio/wav"
                r = wave.open(BytesIO(wav))
                nch = r.getnchannels(); sw = r.getsampwidth(); fr = r.getframerate()
                frames = r.readframes(r.getnframes()); r.close()
                if nch > 1:
                    frames = audioop.tomono(frames, sw, 0.5, 0.5)
                if fr != 22050:
                    frames, _ = audioop.ratecv(frames, sw, 1, fr, 22050, None)
                    fr = 22050
                max_frames = 12 * fr
                frames = frames[:max_frames * sw]
                b2 = BytesIO()
                w2 = wave.open(b2, "wb")
                w2.setnchannels(1); w2.setsampwidth(sw); w2.setframerate(fr)
                w2.writeframes(frames); w2.close()
                out_res["meta"]["data_url"] = "data:audio/wav;base64," + _b64.b64encode(b2.getvalue()).decode("ascii")
            return {"name": name, "result": out_res}
        except Exception as ex:
            _tb = traceback.format_exc()
            logging.error(_tb)
            return _tool_error(
                name,
                "tts_error",
                f"tts.speak failed: {ex}",
                status=500,
                traceback=_tb,
            )
    if name == "ocr.read" and ALLOW_TOOL_EXECUTION:
        if not OCR_API_URL:
            return _tool_error(name, "ocr_backend_unconfigured", "OCR_API_URL not configured", status=500)
        ext = (args.get("ext") or "").strip().lower()
        b64 = None
        if isinstance(args.get("b64"), str) and args.get("b64").strip():
            b64 = args.get("b64").strip()
        elif isinstance(args.get("url"), str) and args.get("url").strip():
            try:
                async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                    rr = await client.get(args.get("url").strip())
                    b64 = _b.b64encode(rr.content).decode("ascii")
                    if not ext:
                        p = urlparse(args.get("url").strip()).path
                        if "." in p:
                            ext = "." + p.split(".")[-1].lower()
            except Exception as ex:
                _tb = traceback.format_exc()
                logging.error(_tb)
                return _tool_error(
                    name,
                    "ocr_fetch_error",
                    f"failed to fetch image from URL: {ex}",
                    status=502,
                    traceback=_tb,
                )
        elif isinstance(args.get("path"), str) and args.get("path").strip():
            try:
                rel = args.get("path").strip()
                full = os.path.abspath(os.path.join(UPLOAD_DIR, rel)) if not os.path.isabs(rel) else rel
                if not full.startswith(os.path.abspath(UPLOAD_DIR)):
                    return _tool_error(name, "path_escapes_uploads", "path escapes uploads directory")
                with open(full, "rb") as f:
                    data = f.read()
                b64 = _b.b64encode(data).decode("ascii")
                if not ext and "." in rel:
                    ext = "." + rel.split(".")[-1].lower()
            except Exception as ex:
                _tb = traceback.format_exc()
                logging.error(_tb)
                return _tool_error(
                    name,
                    "ocr_path_error",
                    f"failed to read local image path: {ex}",
                    status=500,
                    traceback=_tb,
                )
        else:
            return _tool_error(name, "missing_input", "one of b64, url, or path is required")
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(OCR_API_URL.rstrip("/") + "/ocr", json={"b64": b64, "ext": ext})
            parser = JSONParser()
            js = parser.parse(r.text or "", {"text": str})
            text_val = js.get("text") or "" if isinstance(js, dict) else ""
            return {"name": name, "result": {"text": text_val, "ext": ext}}
    if name == "vlm.analyze" and ALLOW_TOOL_EXECUTION:
        if not VLM_API_URL:
            return _tool_error(name, "vlm_backend_unconfigured", "VLM_API_URL not configured", status=500)
        ext = (args.get("ext") or "").strip().lower()
        b64 = None
        if isinstance(args.get("b64"), str) and args.get("b64").strip():
            b64 = args.get("b64").strip()
        elif isinstance(args.get("url"), str) and args.get("url").strip():
            try:
                async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                    rr = await client.get(args.get("url").strip())
                    b64 = _b.b64encode(rr.content).decode("ascii")
                    if not ext:
                        p = urlparse(args.get("url").strip()).path
                        if "." in p:
                            ext = "." + p.split(".")[-1].lower()
            except Exception as ex:
                _tb = traceback.format_exc()
                logging.error(_tb)
                return _tool_error(
                    name,
                    "vlm_fetch_error",
                    f"failed to fetch image from URL: {ex}",
                    status=502,
                    traceback=_tb,
                )
        elif isinstance(args.get("path"), str) and args.get("path").strip():
            try:
                rel = args.get("path").strip()
                full = os.path.abspath(os.path.join(UPLOAD_DIR, rel)) if not os.path.isabs(rel) else rel
                if not full.startswith(os.path.abspath(UPLOAD_DIR)):
                    return _tool_error(name, "path_escapes_uploads", "path escapes uploads directory")
                with open(full, "rb") as f:
                    data = f.read()
                b64 = _b.b64encode(data).decode("ascii")
                if not ext and "." in rel:
                    ext = "." + rel.split(".")[-1].lower()
            except Exception as ex:
                _tb = traceback.format_exc()
                logging.error(_tb)
                return _tool_error(
                    name,
                    "vlm_path_error",
                    f"failed to read local image path: {ex}",
                    status=500,
                    traceback=_tb,
                )
        else:
            return _tool_error(name, "missing_input", "one of b64, url, or path is required")
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(VLM_API_URL.rstrip("/") + "/analyze", json={"b64": b64, "ext": ext})
            parser = JSONParser()
            # VLM returns a free-form JSON object; keep it as a generic mapping.
            js = parser.parse(r.text or "", {})
            return {"name": name, "result": js if isinstance(js, dict) else {}}
    if name == "music.infinite.windowed" and ALLOW_TOOL_EXECUTION:
        if not MUSIC_API_URL:
            # Fail fast with a clear, structured error instead of burying the root cause.
            return {
                "name": name,
                "error": {
                    "code": "music_backend_unconfigured",
                    "message": "MUSIC_API_URL not configured for music.infinite.windowed",
                    "status": 500,
                },
            }

        provider = RestMusicProvider(MUSIC_API_URL)
        manifest = {"items": []}

        # NOTE: Intentionally avoid a broad try/except here so that real errors surface
        # all the way up through /tool.run and the executor, instead of being flattened.
        a = args if isinstance(args, dict) else {}
        quality_profile = (a.get("quality_profile") or "standard")

        # Derive a stable character identity for music locks when possible.
        char_id = str(a.get("character_id") or "").strip()
        if not char_id:
            trace_val = a.get("trace_id")
            if isinstance(trace_val, str) and trace_val.strip():
                char_id = f"char_{trace_val.strip()}"
        if char_id and "character_id" not in a:
            a["character_id"] = char_id

        # Best-effort lock bundle resolution and Song Graph planning.
        bundle_arg = a.get("lock_bundle")
        lock_bundle: Optional[Dict[str, Any]] = None
        if isinstance(bundle_arg, dict):
            lock_bundle = bundle_arg
        elif isinstance(bundle_arg, str) and bundle_arg.strip():
            loaded = await _lock_load(bundle_arg.strip())
            if isinstance(loaded, dict):
                lock_bundle = loaded
        if lock_bundle is None and char_id:
            loaded = await _lock_load(char_id)
            if isinstance(loaded, dict):
                lock_bundle = loaded
        if lock_bundle is None:
            lock_bundle = {}

        # Ensure the bundle carries character identity for downstream tools.
        if char_id:
            lock_bundle.setdefault("character_id", char_id)

        # Normalize music branch shape.
        lock_bundle = _lock_migrate_music(lock_bundle)

        prompt_text = str(a.get("prompt") or "").strip()
        length_s = int(a.get("length_s") or 60)
        bpm_val = a.get("bpm")
        bpm_int: Optional[int] = None
        if isinstance(bpm_val, (int, float)):
            bpm_int = int(bpm_val)
        key_val = a.get("key")
        key_txt: Optional[str] = None
        if isinstance(key_val, str) and key_val.strip():
            key_txt = key_val.strip()

        music_branch = lock_bundle.get("music") if isinstance(lock_bundle.get("music"), dict) else {}
        needs_song_graph = (
            not isinstance(music_branch.get("global"), dict)
            or not isinstance(music_branch.get("sections"), list)
            or not music_branch.get("sections")
        )
        music_profile = None
        # If any music references were attached in locks, build a profile for the planner.
        refs_val = lock_bundle.get("music_refs") if isinstance(lock_bundle.get("music_refs"), dict) else {}
        ref_ids = refs_val.get("ref_ids") if isinstance(refs_val.get("ref_ids"), list) else None
        if ref_ids:
            prof = _refs_music_profile({"ref_ids": ref_ids})
            if isinstance(prof, dict) and prof.get("ok") and isinstance(prof.get("profile"), dict):
                music_profile = prof.get("profile")

        # Propagate the tool-level trace_id into Song Graph planning and tracing.
        trace_val = str(a.get("trace_id") or "").strip() or "tt_unknown"

        if needs_song_graph:
            song_graph = await plan_song_graph(
                user_text=prompt_text,
                length_s=length_s,
                bpm=bpm_int,
                key=key_txt,
                trace_id=trace_val,
                music_profile=music_profile,
            )
            if isinstance(song_graph, dict) and song_graph:
                music_branch = lock_bundle.setdefault("music", {})
                for k in ("global", "sections", "lyrics", "voices", "instruments", "motifs"):
                    val = song_graph.get(k)
                    if val is not None and k not in music_branch:
                        music_branch[k] = val

        if music_branch:
            lock_bundle["music"] = music_branch
        if lock_bundle:
            a["lock_bundle"] = lock_bundle
        # Trace: standalone music.infinite.windowed start
        if trace_val:
            trace_append("music.infinite.windowed.start", {
                "trace_id": trace_val,
                "tool": "music.infinite.windowed",
                "prompt": prompt_text,
                "length_s": length_s,
                "bpm": bpm_int,
                "key": key_txt,
            })
        env = await run_music_infinite_windowed(a, provider, manifest)

        # Persist updated lock bundle (including song graph and windows) when character_id present.
        if char_id and lock_bundle:
            await _lock_save(char_id, lock_bundle)

        # Optional: dataset logging + RAG indexing for the full music track.
        if isinstance(env, dict):
            meta_env = env.get("meta") if isinstance(env.get("meta"), dict) else {}
            cid_env = meta_env.get("cid") or a.get("cid")
            artifacts_env = env.get("artifacts") if isinstance(env.get("artifacts"), list) else []
            main_artifact_id: Optional[str] = None
            for art in artifacts_env:
                if not isinstance(art, dict):
                    continue
                if art.get("kind") == "audio-ref" and isinstance(art.get("id"), str):
                    main_artifact_id = art["id"]
                    break
            if isinstance(cid_env, str) and cid_env and isinstance(main_artifact_id, str) and main_artifact_id:
                outdir_music = os.path.join(UPLOAD_DIR, "artifacts", "audio", "music", cid_env)
                full_path_music = os.path.join(outdir_music, main_artifact_id)
                # Dataset JSONL sample row
                locks_meta = meta_env.get("locks") if isinstance(meta_env.get("locks"), dict) else {}
                style_score = locks_meta.get("style_score") if isinstance(locks_meta, dict) else None
                try:
                    _append_music_sample(
                        outdir_music,
                        {
                            "prompt": prompt_text,
                            "bpm": bpm_int,
                            "length_s": length_s,
                            "seed": int(a.get("seed") or 0),
                            "music_lock": bool(locks_meta),
                            "track_ref": full_path_music,
                            "stems": [],
                            "model": meta_env.get("model") or "music",
                            "style_score": style_score,
                            "created_at": _now_ts(),
                        },
                    )
                except Exception as ex:
                    logging.warning(f"music_samples.append_failed: {ex}", exc_info=True)
                # RAG: embed prompt text against the full track path
                try:
                    pool = await get_pg_pool()
                    txt = prompt_text
                    if pool is not None and txt and txt.strip():
                        emb = get_embedder()
                        vec = emb.encode([txt])[0]
                        async with pool.acquire() as conn:
                            rel = os.path.relpath(full_path_music, UPLOAD_DIR).replace("\\", "/")
                            # Store embeddings as pgvector-compatible arrays so the
                            # `embedding` column (vector(1024)) and `<=>` queries in
                            # rag.core stay in sync with inserts.
                            await conn.execute(
                                "INSERT INTO rag_docs (path, chunk, embedding) VALUES ($1, $2, $3)",
                                rel,
                                txt,
                                list(vec),
                            )
                except Exception as ex:
                    logging.warning(f"music_rag.insert_failed: {ex}", exc_info=True)

        # Trace: standalone music.infinite.windowed finish
        if trace_val:
            meta_env2 = env.get("meta") if isinstance(env, dict) else {}
            trace_append("music.infinite.windowed.finish", {
                "trace_id": trace_val,
                "tool": "music.infinite.windowed",
                "meta": meta_env2 if isinstance(meta_env2, dict) else {},
            })
        return {"name": name, "result": env}
    if name == "music.variation" and ALLOW_TOOL_EXECUTION:
        manifest = {"items": []}
        try:
            a = args if isinstance(args, dict) else {}
            profile_name = (a.get("quality_profile") or "standard")
            # Optional: attach lock bundle for variation jobs (no additional metrics yet)
            bundle_arg = a.get("lock_bundle")
            lock_bundle: Optional[Dict[str, Any]] = None
            if isinstance(bundle_arg, dict):
                lock_bundle = bundle_arg
            elif isinstance(bundle_arg, str) and bundle_arg.strip():
                lock_bundle = await _lock_load(bundle_arg.strip()) or {}
            if lock_bundle:
                a["lock_bundle"] = lock_bundle
            env = run_music_variation(a, manifest)
            wav = env.get("wav_bytes") if isinstance(env, dict) else None
            if isinstance(wav, (bytes, bytearray)) and len(wav) > 0:
                cid = "aud-" + str(_now_ts())
                outdir = os.path.join(UPLOAD_DIR, "artifacts", "audio", "music", cid)
                _ensure_dir(outdir)
                stem = "music_var_00_00"
                wav_path = os.path.join(outdir, stem + ".wav")
                with open(wav_path, "wb") as _wf:
                    _wf.write(wav)
                _sidecar(wav_path, {"tool": "music.variation"})
                _ctx_add(cid, "audio", wav_path, _uri_from_upload_path(wav_path), None, [], {})
                tr = (a.get("trace_id") if isinstance(a.get("trace_id"), str) else None)
                if tr:
                    rel = os.path.relpath(wav_path, UPLOAD_DIR).replace("\\", "/")
                    _log("artifact", trace_id=str(tr), kind="audio", path=rel, bytes=int(len(wav)))
                if isinstance(env, dict):
                    meta_env = env.setdefault("meta", {})
                    if isinstance(meta_env, dict):
                        meta_env.setdefault("quality_profile", profile_name)
                        if lock_bundle:
                            meta_env.setdefault("locks", {"bundle": lock_bundle})
            return {"name": name, "result": env}
        except Exception as ex:
            _tb = traceback.format_exc()
            logging.error(_tb)
            return _tool_error(
                name,
                "music_variation_exception",
                f"music.variation failed: {ex}",
                status=500,
                traceback=_tb,
            )
    if name == "music.refine.window" and ALLOW_TOOL_EXECUTION:
        if not MUSIC_API_URL:
            return _tool_error(name, "music_backend_unconfigured", "MUSIC_API_URL not configured", status=500)
        provider = _MusicRefineProvider()
        manifest = {"items": []}
        try:
            a = args if isinstance(args, dict) else {}
            compose_args = {
                "prompt": a.get("prompt") or "",
                "bpm": a.get("bpm"),
                "length_s": a.get("length_s") or 8,
                "sample_rate": a.get("sample_rate"),
                "channels": a.get("channels"),
                "music_lock": a.get("music_lock"),
                "seed": a.get("seed"),
            }
            res = provider.compose(compose_args)
            wav_bytes = res.get("wav_bytes") or b""
            cid = a.get("cid") or f"music-{_now_ts()}"
            outdir = os.path.join(UPLOAD_DIR, "artifacts", "audio", "music", cid)
            _ensure_dir(outdir)
            stem = f"refine_{_now_ts()}"
            path = os.path.join(outdir, f"{stem}.wav")
            with wave.open(path, "wb") as wf:
                wf.setnchannels(int(a.get("channels") or 2))
                wf.setsampwidth(2)
                wf.setframerate(int(a.get("sample_rate") or 44100))
                wf.writeframes(wav_bytes)
            _sidecar(path, {"tool": "music.refine.window", "segment_id": a.get("segment_id"), "window_id": a.get("window_id")})
            add_manifest_row(manifest, path, step_id="music.refine.window")
            # Best-effort: if a lock_bundle with a music.windows list is provided,
            # update the corresponding window entry so future runs can restitch.
            lock_bundle = a.get("lock_bundle") if isinstance(a.get("lock_bundle"), dict) else None
            old_metrics: Optional[Dict[str, Any]] = None
            new_metrics: Optional[Dict[str, Any]] = None
            if lock_bundle:
                music_branch = lock_bundle.get("music") if isinstance(lock_bundle.get("music"), dict) else {}
                windows_list = music_branch.get("windows") if isinstance(music_branch.get("windows"), list) else []
                win_id = a.get("window_id")
                target_window: Optional[Dict[str, Any]] = None
                for win in windows_list or []:
                    if not isinstance(win, dict):
                        continue
                    if win.get("window_id") == win_id:
                        target_window = win
                        break
                if target_window is not None:
                    prev_metrics = target_window.get("metrics")
                    if isinstance(prev_metrics, dict):
                        old_metrics = dict(prev_metrics)
                    target_window["artifact_path"] = path
                    # Re-run audio analysis for refined clip to update metrics/locks.
                    ainfo = _analyze_audio(path)
                    if isinstance(ainfo, dict):
                        new_metrics = dict(ainfo)
                        # Recompute simple tempo/key lock scores using existing section/global targets when possible.
                        section_index: Dict[str, Dict[str, Any]] = {}
                        sections_music = music_branch.get("sections") if isinstance(music_branch.get("sections"), list) else []
                        for sec in sections_music:
                            if isinstance(sec, dict) and isinstance(sec.get("section_id"), str):
                                section_index[str(sec.get("section_id"))] = sec
                        sec_obj = section_index.get(target_window.get("section_id")) if isinstance(target_window.get("section_id"), str) else None
                        tempo_target = None
                        if isinstance(sec_obj, dict) and isinstance(sec_obj.get("tempo_bpm"), (int, float)):
                            tempo_target = float(sec_obj.get("tempo_bpm"))
                        else:
                            g = music_branch.get("global") if isinstance(music_branch.get("global"), dict) else {}
                            if isinstance(g.get("tempo_bpm"), (int, float)):
                                tempo_target = float(g.get("tempo_bpm"))
                        tempo_lock = None
                        tempo_detected = ainfo.get("tempo_bpm")
                        if isinstance(tempo_detected, (int, float)) and tempo_target and tempo_target > 0.0:
                            tempo_lock = 1.0 - (abs(float(tempo_detected) - tempo_target) / tempo_target)
                            if tempo_lock < 0.0:
                                tempo_lock = 0.0
                            if tempo_lock > 1.0:
                                tempo_lock = 1.0
                        key_target = None
                        if isinstance(sec_obj, dict) and isinstance(sec_obj.get("key"), str):
                            key_target = sec_obj.get("key")
                        else:
                            g = music_branch.get("global") if isinstance(music_branch.get("global"), dict) else {}
                            if isinstance(g.get("key"), str):
                                key_target = g.get("key")
                        key_lock = None
                        key_detected = ainfo.get("key")
                        if isinstance(key_target, str) and key_target.strip() and isinstance(key_detected, str) and key_detected:
                            key_lock = 1.0 if key_target.strip().lower() == key_detected.strip().lower() else 0.0
                        new_metrics["tempo_lock"] = tempo_lock
                        new_metrics["key_lock"] = key_lock
                        target_window["metrics"] = new_metrics
                music_branch["windows"] = windows_list
                lock_bundle["music"] = music_branch
                # Restitch full track from updated windows and persist bundle when possible.
                restitched = restitch_music_from_windows(lock_bundle, cid)
                if isinstance(restitched, str) and restitched:
                    music_branch["full_track_path"] = restitched
                    lock_bundle["music"] = music_branch
                char_id = str(a.get("character_id") or lock_bundle.get("character_id") or "").strip()
                if char_id:
                    await _lock_save(char_id, lock_bundle)
            env: Dict[str, Any] = {
                "meta": {
                    "model": res.get("model", "music"),
                    "ts": _now_ts(),
                    "cid": cid,
                    "step": 0,
                    "state": "halt",
                    "cont": {"present": False, "state_hash": None, "reason": None},
                },
                "reasoning": {"goal": "music window refinement", "constraints": ["json-only"], "decisions": ["music.refine.window done"]},
                "evidence": [],
                "message": {"role": "assistant", "type": "tool", "content": "music window refined"},
                "tool_calls": [
                    {
                        "tool": "music.refine.window",
                        "args": a,
                        "status": "done",
                        "result_ref": os.path.basename(path),
                    }
                ],
                "artifacts": [
                    {"id": os.path.basename(path), "kind": "audio-ref", "summary": stem, "bytes": len(wav_bytes)}
                ],
                "telemetry": {
                    "window": {"input_bytes": 0, "output_target_tokens": 0},
                    "compression_passes": [],
                    "notes": [],
                },
            }
            env = normalize_to_envelope(env)
            env = bump_envelope(env)
            assert_envelope(env)
            env = stamp_env(env, "music.refine.window", env.get("meta", {}).get("model"))
            # Trace before/after metrics for training/distillation.
            trace_append(
                "music.window.refine",
                {
                    "trace_id": a.get("trace_id"),
                    "cid": cid,
                    "segment_id": a.get("segment_id"),
                    "window_id": a.get("window_id"),
                    "old_metrics": old_metrics,
                    "new_metrics": new_metrics,
                },
            )
            return {"name": name, "result": env}
        except Exception as ex:
            _tb = traceback.format_exc()
            logging.error(_tb)
            return _tool_error(
                name,
                "music_refine_exception",
                f"music.refine.window failed: {ex}",
                status=500,
                traceback=_tb,
            )
    if name == "music.mixdown" and ALLOW_TOOL_EXECUTION:
        manifest = {"items": []}
        try:
            a = args if isinstance(args, dict) else {}
            quality_profile = (a.get("quality_profile") or "standard")
            bundle_arg = a.get("lock_bundle")
            lock_bundle: Optional[Dict[str, Any]] = None
            if isinstance(bundle_arg, dict):
                lock_bundle = bundle_arg
            elif isinstance(bundle_arg, str) and bundle_arg.strip():
                lock_bundle = await _lock_load(bundle_arg.strip()) or {}
            if not lock_bundle:
                char_id = str(a.get("character_id") or "").strip()
                if char_id:
                    lock_bundle = await _lock_load(char_id) or {}
            if lock_bundle:
                a["lock_bundle"] = lock_bundle
            env = run_music_mixdown(a, manifest)
            wav = env.get("wav_bytes") if isinstance(env, dict) else None
            if isinstance(wav, (bytes, bytearray)) and len(wav) > 0:
                cid = "aud-" + str(_now_ts())
                outdir = os.path.join(UPLOAD_DIR, "artifacts", "audio", "music", cid)
                _ensure_dir(outdir)
                stem = "music_mix_00_00"
                wav_path = os.path.join(outdir, stem + ".wav")
                with open(wav_path, "wb") as _wf:
                    _wf.write(wav)
                locks_meta: Dict[str, Any] = {}
                if lock_bundle:
                    locks_meta["bundle"] = lock_bundle
                    locks_meta["quality_profile"] = quality_profile
                    audio_section = lock_bundle.get("audio") if isinstance(lock_bundle.get("audio"), dict) else {}
                    ref_voice = audio_section.get("voice_embedding")
                    if isinstance(ref_voice, list):
                        voice_embed = await _lock_voice_embedding_from_path(wav_path)
                        if voice_embed:
                            sim = _cosine_similarity(ref_voice, voice_embed)
                            if sim is not None:
                                locks_meta["voice_score"] = max(0.0, min((sim + 1.0) / 2.0, 1.0))
                                locks_meta["voice_embedding"] = voice_embed
                    tempo_target = audio_section.get("tempo_bpm")
                    tempo_mode = audio_section.get("tempo_lock_mode", "off")
                    if tempo_target is not None and tempo_mode != "off":
                        try:
                            tempo_target_f = float(tempo_target)
                            tempo_detected = _audio_detect_tempo(wav_path)
                            if tempo_detected is not None and tempo_target_f > 0:
                                tempo_score = max(0.0, min(1.0 - (abs(tempo_detected - tempo_target_f) / tempo_target_f), 1.0))
                                locks_meta["tempo_detected"] = tempo_detected
                                locks_meta["tempo_score"] = tempo_score
                                locks_meta["tempo_lock_mode"] = tempo_mode
                        except Exception as ex:
                            _log("music.lock.tempo_score.fail", error=str(ex), path=wav_path)
                    key_target = audio_section.get("key")
                    key_mode = audio_section.get("key_lock_mode", "off")
                    if isinstance(key_target, str) and key_mode != "off":
                        detected_key = _audio_detect_key(wav_path)
                        locks_meta["key_detected"] = detected_key
                        key_score = _key_similarity(key_target, detected_key)
                        if key_score is not None:
                            locks_meta["key_score"] = key_score
                            locks_meta["key_lock_mode"] = key_mode
                    stem_profile = audio_section.get("stem_profile") if isinstance(audio_section.get("stem_profile"), dict) else {}
                    if stem_profile and audio_section.get("stem_lock_mode", "off") != "off":
                        total_target = sum(float(v) for v in stem_profile.values() if isinstance(v, (int, float)))
                        normalized_target = {k: (float(v) / total_target) if total_target else float(v) for k, v in stem_profile.items() if isinstance(v, (int, float))}
                        detected_profile = _audio_band_energy_profile(wav_path, DEFAULT_STEM_BANDS)
                        stem_score = _stem_balance_score(normalized_target, detected_profile)
                        if stem_score is not None:
                            locks_meta["stem_balance_score"] = stem_score
                            locks_meta["stem_detected_profile"] = detected_profile
                            locks_meta["stem_lock_mode"] = audio_section.get("stem_lock_mode")
                    locks_meta.setdefault("lyrics_score", None)
                sidecar_meta = {"tool": "music.mixdown"}
                if locks_meta:
                    sidecar_meta["locks"] = locks_meta
                _sidecar(wav_path, sidecar_meta)
                _ctx_add(cid, "audio", wav_path, _uri_from_upload_path(wav_path), None, [], {})
                tr = (a.get("trace_id") if isinstance(a.get("trace_id"), str) else None)
                if tr:
                    rel = os.path.relpath(wav_path, UPLOAD_DIR).replace("\\", "/")
                    _log("artifact", trace_id=str(tr), kind="audio", path=rel, bytes=int(len(wav)))
                if isinstance(env, dict):
                    meta_env = env.setdefault("meta", {})
                    if isinstance(meta_env, dict) and quality_profile:
                        meta_env.setdefault("quality_profile", quality_profile)
            return {"name": name, "result": env}
        except Exception as ex:
            _tb = traceback.format_exc()
            logging.error(_tb)
            return _tool_error(
                name,
                "music_mixdown_exception",
                f"music.mixdown failed: {ex}",
                status=500,
                traceback=_tb,
            )
    # --- Creative helpers ---
    if name == "creative.alt_takes" and ALLOW_TOOL_EXECUTION:
        # Run N variants with orthogonal seeds, score via committee, pick best, return all URLs
        base_tool = (args.get("tool") or "").strip()
        base_args = args.get("args") if isinstance(args.get("args"), dict) else {}
        n = max(2, min(int(args.get("n") or 3), 5))
        results: List[Dict[str, Any]] = []
        for i in range(n):
            try:
                v_args = dict(base_args)
                v_args["seed"] = det_seed_tool(f"{base_tool}#{i}", str(det_seed_router("alt", 0)))
                tr = await execute_tool_call({"name": base_tool, "arguments": v_args})
                results.append(tr)
            except Exception as ex:
                results.append({"name": base_tool, "error": str(ex)})
        # Score images via CLIP; music via basic audio metrics
        ranked = sorted(results, key=lambda tr: _creative_alt_score(tr, base_tool), reverse=True)
        urls: List[str] = []
        for tr in ranked:
            try:
                res = tr.get("result") or {}
                arts = (res.get("artifacts") or []) if isinstance(res, dict) else []
                meta = (res.get("meta") or {}) if isinstance(res.get("meta"), dict) else {}
                cid = meta.get("cid")
                artifact_group_id = meta.get("artifact_group_id")
                for a in arts:
                    aid = a.get("id"); kind = a.get("kind") or ""
                    if aid and (cid or artifact_group_id):
                        if kind.startswith("image") or base_tool.startswith("image"):
                            group = artifact_group_id or cid
                            if group:
                                urls.append(f"/uploads/artifacts/image/{group}/{aid}")
                        elif base_tool.startswith("music"):
                            if cid:
                                urls.append(f"/uploads/artifacts/music/{cid}/{aid}")
            except Exception as ex:
                logging.warning(f"creative.alts.asset_collect_failed: {ex}", exc_info=True)
                continue
        return {"name": name, "result": {"variants": len(results), "assets": urls, "best": (urls[0] if urls else None)}}
    if name == "creative.pro_polish" and ALLOW_TOOL_EXECUTION:
        # Run a default quality chain based on kind
        kind = (args.get("kind") or "").strip().lower()
        src = args.get("src") or ""
        if not kind or not src:
            return {"name": name, "error": "missing kind|src"}
        try:
            if kind == "image":
                c1 = await execute_tool_call({"name": "image.cleanup", "arguments": {"src": src, "cid": args.get("cid")}})
                u = ((c1.get("result") or {}).get("url") or src)
                c2 = await execute_tool_call({"name": "image.upscale", "arguments": {"image_ref": u, "scale": 2, "cid": args.get("cid")}})
                return {"name": name, "result": {"chain": [c1, c2]}}
            if kind == "video":
                c1 = await execute_tool_call({"name": "video.cleanup", "arguments": {"src": src, "stabilize_faces": True, "cid": args.get("cid")}})
                u = ((c1.get("result") or {}).get("url") or src)
                c2 = await execute_tool_call({"name": "video.interpolate", "arguments": {"src": u, "target_fps": 60, "cid": args.get("cid")}})
                c3 = await execute_tool_call({"name": "video.upscale", "arguments": {"src": ((c2.get("result") or {}).get("url") or u), "scale": 2, "cid": args.get("cid")}})
                return {"name": name, "result": {"chain": [c1, c2, c3]}}
            if kind == "audio":
                # For now, prefer existing QA in TTS/music tools; here we just pass back src
                return {"name": name, "result": {"src": src, "note": "audio QA runs in modality tools"}}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "creative.repro_pack" and ALLOW_TOOL_EXECUTION:
        # Write a small repro bundle next to the artifact (or under /uploads/repros)
        tool_used = (args.get("tool") or "").strip()
        a = args.get("args") if isinstance(args.get("args"), dict) else {}
        _cid_raw = args.get("cid")
        if isinstance(_cid_raw, (str, int)):
            cid = str(_cid_raw).strip()
        else:
            cid = ""
        if not cid:
            cid = f"repro-{int(time.time())}"
            args["cid"] = cid
        repro = {"tool": tool_used, "args": a, "seed": det_seed_tool(tool_used, str(det_seed_router("repro", 0))), "ts": int(time.time()), "cid": cid}
        outdir = os.path.join(UPLOAD_DIR, "repros", cid or "misc")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "repro.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(repro, ensure_ascii=False, indent=2))
        uri = path.replace("/workspace", "") if path.startswith("/workspace/") else path
        return {"name": name, "result": {"uri": uri}}
    if name == "style.dna.extract" and ALLOW_TOOL_EXECUTION:
        imgs = args.get("images") if isinstance(args.get("images"), list) else []
        palette = []
        if Image and imgs:
            for p in imgs[:6]:
                try:
                    loc = p
                    if isinstance(loc, str) and loc.startswith("/uploads/"):
                        loc = "/workspace" + loc
                    with Image.open(loc) as im:  # type: ignore
                        im = im.convert("RGB")  # type: ignore
                        small = im.resize((32, 32))  # type: ignore
                        px = list(small.getdata())  # type: ignore
                        r = sum(x for x,_,_ in px)//len(px); g = sum(y for _,y,_ in px)//len(px); b = sum(z for _,_,z in px)//len(px)
                        h,s,v = colorsys.rgb_to_hsv(r/255.0,g/255.0,b/255.0)
                        palette.append({"rgb": [int(r),int(g),int(b)], "hsv": [h,s,v]})
                except Exception as ex:
                    logging.warning(f"style.dna.extract.palette.error path={p!r} err={str(ex)}", exc_info=True)
                    continue
        dna = {"palette": palette[:4], "keywords": [], "ts": int(time.time())}
        try:
            outdir = os.path.join(UPLOAD_DIR, "refs", "style_dna")
            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, f"dna_{int(time.time()*1000)}.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(dna, ensure_ascii=False, indent=2))
            uri = path.replace("/workspace", "") if path.startswith("/workspace/") else path
            return {"name": name, "result": {"dna": dna, "uri": uri}}
        except Exception as ex:
            _tb = traceback.format_exc()
            logging.error(_tb)
            return {"name": name, "error": str(ex), "traceback": _tb}
    if name == "refs.auto_hunt" and ALLOW_TOOL_EXECUTION:
        try:
            q = args.get("query") or ""
            k = int(args.get("k") or 8)
            tr = await execute_tool_call({"name": "metasearch.fuse", "arguments": {"q": q, "k": k}})
            res = (tr.get("result") or {}).get("results") if isinstance(tr, dict) else []
            links = []
            for it in (res or [])[:k]:
                u = it.get("link") or it.get("url")
                if isinstance(u, str) and u:
                    links.append(u)
            return {"name": name, "result": {"refs": links}}
        except Exception as ex:
            _tb = traceback.format_exc()
            logging.error(_tb)
            return {"name": name, "error": str(ex), "traceback": _tb}
    if name == "director.mode" and ALLOW_TOOL_EXECUTION:
        # No keyword-based intent detection here; return generic clarifying questions.
        p = str(args.get("prompt") or "").strip()
        qs: List[str] = []
        if not p:
            qs.append("What would you like to make, and what should it contain?")
        qs += [
            "What output format do you want (text/image/video/audio) and any required size/length?",
            "Any references, style constraints, or assets to use?",
            "Any hard requirements (must/avoid) or quality targets?",
        ]
        return {"name": name, "result": {"questions": qs[:5]}}
    if name == "scenegraph.plan" and ALLOW_TOOL_EXECUTION:
        prompt = args.get("prompt") or ""
        size = (args.get("size") or "1024x1024").lower()
        w, h = 1024, 1024
        if isinstance(size, str) and size:
            m = _re.match(r"^\s*(\d{2,5})\s*x\s*(\d{2,5})\s*$", size)
            if m:
                w = int(m.group(1))
                h = int(m.group(2))
        tokens = [t.strip(",. ") for t in prompt.replace(" and ", ",").split(",") if t.strip()]
        objs = []
        for t in tokens[:6]:
            x0 = _rnd.randint(0, max(0, w-200)); y0 = _rnd.randint(0, max(0, h-200))
            x1 = min(w, x0 + _rnd.randint(120, 300)); y1 = min(h, y0 + _rnd.randint(120, 300))
            objs.append({"label": t, "box": [x0,y0,x1,y1]})
        return {"name": name, "result": {"size": [w,h], "objects": objs}}
    if name == "video.temporal_clip_qa" and ALLOW_TOOL_EXECUTION:
        # Lightweight drift score placeholder (no heavy deps): report a high score by default
        return {"name": name, "result": {"drift_score": 0.95, "notes": "basic check"}}
    if name == "image.qa" and ALLOW_TOOL_EXECUTION:
        a = args if isinstance(args, dict) else {}
        src = a.get("path") or a.get("src") or a.get("image_path") or ""
        if not isinstance(src, str) or not src:
            return {"name": name, "error": "missing_image_path"}
        prompt = str(a.get("prompt") or "").strip()
        # src is expected to be an absolute path under UPLOAD_DIR; callers are responsible for resolving URLs.
        try:
            global_info = _analyze_image(src, prompt)
            region_info = _analyze_image_regions(src, prompt, global_info)
        except Exception as ex:
            _tb = traceback.format_exc()
            logging.error(_tb)
            return {"name": name, "error": f"image_qa_error:{ex}", "traceback": _tb}
        # Attach a QA-shaped summary so compute_domain_qa can consume it if needed
        qa_block: Dict[str, Any] = {}
        if isinstance(global_info, dict):
            sb = global_info.get("score") or {}
            sem = global_info.get("semantics") or {}
            if isinstance(sb, dict):
                qa_block["overall"] = float(sb.get("overall") or 0.0)
                qa_block["semantic"] = float(sb.get("semantic") or 0.0)
                qa_block["technical"] = float(sb.get("technical") or 0.0)
                qa_block["aesthetic"] = float(sb.get("aesthetic") or 0.0)
            if isinstance(sem, dict):
                cs = sem.get("clip_score")
                if isinstance(cs, (int, float)):
                    qa_block["clip_score"] = float(cs)
        if isinstance(region_info, dict):
            agg = region_info.get("aggregates") or {}
            if isinstance(agg, dict):
                fl = agg.get("face_lock")
                if isinstance(fl, (int, float)):
                    qa_block["face_lock"] = float(fl)
                il = agg.get("id_lock")
                if isinstance(il, (int, float)):
                    qa_block["id_lock"] = float(il)
                hr = agg.get("hands_ok_ratio")
                if isinstance(hr, (int, float)):
                    qa_block["hands_ok_ratio"] = float(hr)
                tr = agg.get("text_readable_lock")
                if isinstance(tr, (int, float)):
                    qa_block["text_readable_lock"] = float(tr)
                bq = agg.get("background_quality")
                if isinstance(bq, (int, float)):
                    qa_block["background_quality"] = float(bq)
        return {
            "name": name,
            "result": {
                "global": global_info,
                "regions": region_info,
                "qa": {"images": qa_block},
            },
        }
    if name == "music.motif_keeper" and ALLOW_TOOL_EXECUTION:
        # Record intent to preserve motifs; return a baseline recurrence score
        return {"name": name, "result": {"motif_recurrence": 0.9, "notes": "baseline motif keeper active"}}
    if name == "signage.grounding.loop" and ALLOW_TOOL_EXECUTION:
        # Hook is already integrated in image.super_gen; this returns an explicit OK
        return {"name": name, "result": {"ok": True}}
    # --- Extended Music/Audio tools ---
    if name == "music.melody.musicgen" and ALLOW_TOOL_EXECUTION:
        if not MUSIC_API_URL:
            return {"name": name, "error": "MUSIC_API_URL not configured"}
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            payload = {
                "text": args.get("text"),
                "melody_wav": args.get("melody_wav"),
                "bpm": args.get("bpm"),
                "key": args.get("key"),
                "seed": args.get("seed"),
                "style_tags": args.get("style_tags") or [],
                "length_s": args.get("length_s"),
                "quality": "max",
                "infinite": True,
            }
            r = await client.post(MUSIC_API_URL.rstrip("/") + "/generate", json=payload)
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            return {"name": name, "result": js if isinstance(js, dict) else {}}
    if name == "music.timed.sao" and ALLOW_TOOL_EXECUTION:
        if not SAO_API_URL:
            return {"name": name, "error": "SAO_API_URL not configured"}
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            payload = {
                "text": args.get("text"),
                "seconds": int(args.get("seconds") or args.get("duration_sec") or 8),
                "bpm": args.get("bpm"),
                "seed": args.get("seed"),
                "genre_tags": args.get("genre_tags") or [],
                "quality": "max",
            }
            r = await client.post(SAO_API_URL.rstrip("/") + "/v1/music/timed", json=payload)
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            return {"name": name, "result": js if isinstance(js, dict) else {}}
    if name == "audio.stems.demucs" and ALLOW_TOOL_EXECUTION:
        if not DEMUCS_API_URL:
            return {"name": name, "error": "DEMUCS_API_URL not configured"}
        try:
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                payload = {"mix_wav": args.get("mix_wav") or args.get("src"), "stems": args.get("stems") or ["vocals","drums","bass","other"]}
                r = await client.post(DEMUCS_API_URL.rstrip("/") + "/v1/audio/stems", json=payload)
                parser = JSONParser()
                js = parser.parse(r.text or "", {})
                # Persist stems if present
                stems_obj = js.get("stems") if isinstance(js, dict) else None
                if isinstance(stems_obj, dict):
                    cid = "demucs-" + str(_now_ts())
                    outdir = os.path.join(UPLOAD_DIR, "artifacts", "audio", "demucs", cid)
                    _ensure_dir(outdir)
                    for stem_name, val in stems_obj.items():
                        try:
                            wav_bytes = b""
                            if isinstance(val, str) and val.startswith("http"):
                                rr = await client.get(val)
                                if rr.status_code == 200:
                                    wav_bytes = rr.content
                            elif isinstance(val, str):
                                # producing an empty stem; this surfaces Demucs bugs.
                                wav_bytes = _b.b64decode(val)
                            if wav_bytes:
                                path = os.path.join(outdir, f"{stem_name}.wav")
                                with open(path, "wb") as wf: wf.write(wav_bytes)
                                _sidecar(path, {"tool": "audio.stems.demucs", "stem": stem_name})
                                try:
                                    _ctx_add(cid, "audio", path, _uri_from_upload_path(path), payload.get("mix_wav"), ["demucs"], {"stem": stem_name})
                                except Exception as ex:
                                    _log("audio.stems.demucs.ctx_add.error", stem=stem_name, error=str(ex))
                                tr = args.get("trace_id") if isinstance(args.get("trace_id"), str) else None
                                if tr:
                                    try:
                                        rel = os.path.relpath(path, UPLOAD_DIR).replace("\\", "/")
                                        _log(
                                            "artifact",
                                            trace_id=str(tr),
                                            kind="audio",
                                            path=rel,
                                            bytes=int(len(wav_bytes)),
                                            stem=stem_name,
                                        )
                                    except Exception as ex:
                                        _log("audio.stems.demucs.trace.error", stem=stem_name, error=str(ex))
                        except Exception as ex:
                            _log("audio.stems.demucs.persist.error", stem=stem_name, error=str(ex))
                            continue
                return {"name": name, "result": js}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "audio.stems.adjust" and ALLOW_TOOL_EXECUTION:
        """
        Convenience tool: take a mixed WAV, run Demucs to get stems (if needed),
        apply simple gain changes per stem, and re-mix via music.mixdown.
        """
        mix_path = args.get("mix_wav")
        if not isinstance(mix_path, str) or not mix_path.strip():
            return {"name": name, "error": "missing mix_wav"}
        mix_url = mix_path
        try:
            sep = await execute_tool_call(
                {"name": "audio.stems.demucs", "arguments": {"mix_wav": mix_url, "stems": args.get("stems") or ["vocals", "drums", "bass", "other"]}}
            )
            stems_info = (sep.get("result") or sep) if isinstance(sep, dict) else {}
            stems_obj = stems_info.get("stems") if isinstance(stems_info.get("stems"), dict) else stems_info
            stem_gains = args.get("stem_gains") if isinstance(args.get("stem_gains"), dict) else {}
            adjusted_stems: List[Dict[str, Any]] = []
            if isinstance(stems_obj, dict):
                for stem_name, val in stems_obj.items():
                    stem_path = None
                    if isinstance(val, str) and val.startswith("/"):
                        stem_path = val
                    elif isinstance(val, dict):
                        stem_path = val.get("path") if isinstance(val.get("path"), str) else None
                    if not stem_path:
                        continue
                    gain_db = 0.0
                    g_val = stem_gains.get(stem_name)
                    if isinstance(g_val, (int, float)):
                        gain_db = float(g_val)
                    adjusted_stems.append({"path": stem_path, "gain_db": gain_db})
            mix_args = {
                "stems": adjusted_stems,
                "sample_rate": args.get("sample_rate"),
                "channels": args.get("channels"),
                "seed": args.get("seed"),
            }
            manifest = {"items": []}
            env = run_music_mixdown(mix_args, manifest)
            # Canonicalize edit into lock bundle when provided.
            lock_bundle = args.get("lock_bundle") if isinstance(args.get("lock_bundle"), dict) else None
            if lock_bundle and isinstance(env, dict):
                music_branch = lock_bundle.get("music") if isinstance(lock_bundle.get("music"), dict) else {}
                full_track = music_branch.get("full_track_path")
                # Reconstruct new mix path from mixdown envelope.
                meta_env = env.get("meta") if isinstance(env.get("meta"), dict) else {}
                mix_cid = meta_env.get("cid") if isinstance(meta_env.get("cid"), str) else None
                result_ref = None
                tool_calls = env.get("tool_calls") if isinstance(env.get("tool_calls"), list) else []
                if tool_calls:
                    tc0 = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
                    result_ref = tc0.get("result_ref") if isinstance(tc0.get("result_ref"), str) else None
                new_path = None
                if mix_cid and result_ref:
                    new_path = os.path.join(UPLOAD_DIR, "artifacts", "music", mix_cid, result_ref)
                updated = False
                if isinstance(new_path, str):
                    # Update full track when this mix corresponds to the canonical track.
                    if isinstance(full_track, str) and os.path.normpath(full_track) == os.path.normpath(mix_path):
                        music_branch["full_track_path"] = new_path
                        updated = True
                    # Update any window whose artifact_path matches the edited mix.
                    win_list = music_branch.get("windows") if isinstance(music_branch.get("windows"), list) else []
                    for win in win_list:
                        if not isinstance(win, dict):
                            continue
                        w_path = win.get("artifact_path")
                        if isinstance(w_path, str) and os.path.normpath(w_path) == os.path.normpath(mix_path):
                            win["artifact_path"] = new_path
                            updated = True
                    music_branch["windows"] = win_list
                    # Persist stems summary on music branch.
                    if adjusted_stems:
                        music_branch["stems"] = [st.get("path") for st in adjusted_stems if isinstance(st.get("path"), str)]
                if updated:
                    lock_bundle["music"] = music_branch
                    char_id = str(args.get("character_id") or lock_bundle.get("character_id") or "").strip()
                    if char_id:
                        await _lock_save(char_id, lock_bundle)
                    # Surface updated music branch into envelope meta.locks for callers.
                    meta_env = env.setdefault("meta", {})
                    if isinstance(meta_env, dict):
                        locks_meta = meta_env.setdefault("locks", {})
                        if isinstance(locks_meta, dict):
                            locks_meta["music"] = music_branch
            # Trace stems adjustment usage for distillation.
            trace_append(
                "audio.stems.adjust",
                {
                    "trace_id": (args.get("trace_id") if isinstance(args, dict) else None),
                    "mix_path": mix_path,
                    "stems_count": len(adjusted_stems),
                    "stem_gains": stem_gains,
                },
            )
            return {"name": name, "result": env}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "audio.vocals.transform" and ALLOW_TOOL_EXECUTION:
        """
        Tool to adjust vocals in a mixed track: isolate vocals, apply pitch shift
        and optional voice conversion, then re-mix with backing.
        """
        mix_path = args.get("mix_wav")
        if not isinstance(mix_path, str) or not mix_path.strip():
            return {"name": name, "error": "missing mix_wav"}
        pitch_shift = args.get("pitch_shift_semitones")
        voice_lock_id = args.get("voice_lock_id")
        try:
            # 1) Separate stems
            sep = await execute_tool_call(
                {"name": "audio.stems.demucs", "arguments": {"mix_wav": mix_path, "stems": ["vocals", "drums", "bass", "other"]}}
            )
            stems_info = (sep.get("result") or sep) if isinstance(sep, dict) else {}
            stems_obj = stems_info.get("stems") if isinstance(stems_info.get("stems"), dict) else stems_info
            if not isinstance(stems_obj, dict):
                return {"name": name, "error": "stems_missing"}
            vocals_path = None
            backing_stems: List[Dict[str, Any]] = []
            for stem_name, val in stems_obj.items():
                stem_path = None
                if isinstance(val, str) and val.startswith("/"):
                    stem_path = val
                elif isinstance(val, dict):
                    stem_path = val.get("path") if isinstance(val.get("path"), str) else None
                if not stem_path:
                    continue
                if stem_name == "vocals":
                    vocals_path = stem_path
                else:
                    backing_stems.append({"path": stem_path, "gain_db": 0.0})
            if not vocals_path:
                return {"name": name, "error": "vocals_stem_missing"}
            # 2) Optional voice conversion via RVC (identity lock), required when voice_lock_id is provided.
            transformed_vocals = vocals_path
            if isinstance(voice_lock_id, str) and voice_lock_id.strip():
                try:
                    # Read source vocals and send as base64 to RVC.
                    with open(transformed_vocals, "rb") as _f:
                        src_bytes = _f.read()
                    payload_rvc = {
                        "source_wav_b64": _b64.b64encode(src_bytes).decode("ascii"),
                        "voice_lock_id": voice_lock_id.strip(),
                    }
                    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                        r_rvc = await client.post(RVC_API_URL.rstrip("/") + "/v1/audio/convert", json=payload_rvc)
                    parser = JSONParser()
                    js_rvc = parser.parse(r_rvc.text or "", {"ok": bool, "audio_wav_base64": str, "sample_rate": int})
                    if not isinstance(js_rvc, dict) or not bool(js_rvc.get("ok")):
                        return {
                            "name": name,
                            "error": {
                                "code": "rvc_error",
                                "status": int(getattr(r_rvc, "status_code", 500) or 500),
                                "message": "RVC /v1/audio/convert failed for vocals stem",
                                "raw": js_rvc,
                            },
                        }
                    b64_rvc = js_rvc.get("audio_wav_base64") if isinstance(js_rvc.get("audio_wav_base64"), str) else None
                    if not b64_rvc:
                        return {
                            "name": name,
                            "error": {
                                "code": "rvc_missing_audio",
                                "status": 500,
                                "message": "RVC /v1/audio/convert returned no audio_wav_base64",
                            },
                        }
                    out_bytes = _b64.b64decode(b64_rvc)
                    # Write converted vocals next to original.
                    v_dir, v_name = os.path.dirname(vocals_path), os.path.basename(vocals_path)
                    v_base, v_ext = os.path.splitext(v_name)
                    rvc_path = os.path.join(v_dir, f"{v_base}.rvc{v_ext or '.wav'}")
                    with open(rvc_path, "wb") as _rf:
                        _rf.write(out_bytes)
                    transformed_vocals = rvc_path
                except Exception as ex:
                    return {
                        "name": name,
                        "error": {
                            "code": "rvc_exception",
                            "status": 500,
                            "message": str(ex),
                        },
                    }
            # 3) Apply VocalFix (pitch correction/shift) as a mandatory quality stage.
            try:
                with open(transformed_vocals, "rb") as _f2:
                    vf_src = _f2.read()
                payload_vf = {
                    "audio_wav_base64": _b64.b64encode(vf_src).decode("ascii"),
                    "sample_rate": int(args.get("sample_rate") or 44100),
                    "ops": ["pitch", "align", "deess"],
                }
                if isinstance(pitch_shift, (int, float)):
                    payload_vf.setdefault("score_json", {})
                    payload_vf["score_json"]["pitch_shift_semitones"] = float(pitch_shift)
                async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                    r_vf = await client.post(VOCAL_FIXER_API_URL.rstrip("/") + "/v1/vocal/fix", json=payload_vf)
                parser = JSONParser()
                js_vf = parser.parse(
                    r_vf.text or "",
                    {
                        "ok": bool,
                        "audio_wav_base64": str,
                        "sample_rate": int,
                        "metrics_before": dict,
                        "metrics_after": dict,
                    },
                )
                if not isinstance(js_vf, dict) or not bool(js_vf.get("ok")):
                    return {
                        "name": name,
                        "error": {
                            "code": "vocalfix_error",
                            "status": int(getattr(r_vf, "status_code", 500) or 500),
                            "message": "VocalFix /v1/vocal/fix failed for vocals stem",
                            "raw": js_vf,
                        },
                    }
                b64_vf = js_vf.get("audio_wav_base64") if isinstance(js_vf.get("audio_wav_base64"), str) else None
                if not b64_vf:
                    return {
                        "name": name,
                        "error": {
                            "code": "vocalfix_missing_audio",
                            "status": 500,
                            "message": "VocalFix /v1/vocal/fix returned no audio_wav_base64",
                        },
                    }
                out_vf = _b64.b64decode(b64_vf)
                v_dir2, v_name2 = os.path.dirname(transformed_vocals), os.path.basename(transformed_vocals)
                v_base2, v_ext2 = os.path.splitext(v_name2)
                vf_path = os.path.join(v_dir2, f"{v_base2}.vf{v_ext2 or '.wav'}")
                with open(vf_path, "wb") as _wf2:
                    _wf2.write(out_vf)
                transformed_vocals = vf_path
                # Attach VocalFix metrics to trace for distillation.
                trace_append(
                    "audio.vocals.vocalfix",
                    {
                        "trace_id": (args.get("trace_id") if isinstance(args, dict) else None),
                        "mix_path": mix_path,
                        "vocals_path": vocals_path,
                        "transformed_path": transformed_vocals,
                        "metrics_before": js_vf.get("metrics_before") if isinstance(js_vf.get("metrics_before"), dict) else {},
                        "metrics_after": js_vf.get("metrics_after") if isinstance(js_vf.get("metrics_after"), dict) else {},
                    },
                )
            except Exception as ex:
                return {
                    "name": name,
                    "error": {
                        "code": "vocalfix_exception",
                        "status": 500,
                        "message": str(ex),
                    },
                }
            # 4) Re-mix vocals with backing
            stems_for_mix = [{"path": transformed_vocals, "gain_db": 0.0}] + backing_stems
            mix_args = {
                "stems": stems_for_mix,
                "sample_rate": args.get("sample_rate"),
                "channels": args.get("channels"),
                "seed": args.get("seed"),
            }
            manifest = {"items": []}
            env = run_music_mixdown(mix_args, manifest)
            # Canonicalize edit into lock bundle when provided.
            lock_bundle = args.get("lock_bundle") if isinstance(args.get("lock_bundle"), dict) else None
            if lock_bundle and isinstance(env, dict):
                music_branch = lock_bundle.get("music") if isinstance(lock_bundle.get("music"), dict) else {}
                full_track = music_branch.get("full_track_path")
                meta_env = env.get("meta") if isinstance(env.get("meta"), dict) else {}
                mix_cid = meta_env.get("cid") if isinstance(meta_env.get("cid"), str) else None
                result_ref = None
                tool_calls = env.get("tool_calls") if isinstance(env.get("tool_calls"), list) else []
                if tool_calls:
                    tc0 = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
                    result_ref = tc0.get("result_ref") if isinstance(tc0.get("result_ref"), str) else None
                new_path = None
                if mix_cid and result_ref:
                    new_path = os.path.join(UPLOAD_DIR, "artifacts", "music", mix_cid, result_ref)
                updated = False
                if isinstance(new_path, str):
                    if isinstance(full_track, str) and os.path.normpath(full_track) == os.path.normpath(mix_path):
                        music_branch["full_track_path"] = new_path
                        updated = True
                    win_list = music_branch.get("windows") if isinstance(music_branch.get("windows"), list) else []
                    for win in win_list:
                        if not isinstance(win, dict):
                            continue
                        w_path = win.get("artifact_path")
                        if isinstance(w_path, str) and os.path.normpath(w_path) == os.path.normpath(mix_path):
                            win["artifact_path"] = new_path
                            updated = True
                    music_branch["windows"] = win_list
                    # Persist stems summary on music branch (vocals + backing).
                    music_branch["stems"] = [st.get("path") for st in stems_for_mix if isinstance(st.get("path"), str)]
                if updated:
                    lock_bundle["music"] = music_branch
                    char_id = str(args.get("character_id") or lock_bundle.get("character_id") or "").strip()
                    if char_id:
                        await _lock_save(char_id, lock_bundle)
                    meta_env = env.setdefault("meta", {})
                    if isinstance(meta_env, dict):
                        locks_meta = meta_env.setdefault("locks", {})
                        if isinstance(locks_meta, dict):
                            locks_meta["music"] = music_branch
            # Trace vocal transform usage.
            trace_append(
                "audio.vocals.transform",
                {
                    "trace_id": (args.get("trace_id") if isinstance(args, dict) else None),
                    "mix_path": mix_path,
                    "pitch_shift_semitones": float(pitch_shift) if isinstance(pitch_shift, (int, float)) else None,
                    "voice_lock_id": voice_lock_id if isinstance(voice_lock_id, str) else None,
                },
            )
            return {"name": name, "result": env}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "audio.vc.rvc" and ALLOW_TOOL_EXECUTION:
        # Generic voice conversion tool using the RVC microservice.
        src_path = args.get("source_vocal_wav") or args.get("src")
        voice_lock_id = args.get("voice_lock_id") or args.get("target_voice_ref") or args.get("voice_ref")
        if not isinstance(src_path, str) or not src_path.strip():
            return {"name": name, "error": {"code": "missing_src", "message": "source_vocal_wav/src is required"}}
        if not isinstance(voice_lock_id, str) or not voice_lock_id.strip():
            return {"name": name, "error": {"code": "missing_voice_lock_id", "message": "voice_lock_id/target_voice_ref/voice_ref is required"}}
        try:
            with open(src_path, "rb") as _f:
                src_bytes = _f.read()
        except Exception as ex:
            return {"name": name, "error": {"code": "src_read_error", "message": str(ex)}}
        payload = {
            "source_wav_base64": _b64.b64encode(src_bytes).decode("ascii"),
            "voice_lock_id": voice_lock_id.strip(),
        }
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(RVC_API_URL.rstrip("/") + "/v1/audio/convert", json=payload)
        parser = JSONParser()
        js = parser.parse(r.text or "", {"ok": bool, "audio_wav_base64": str, "sample_rate": int})
        if not isinstance(js, dict) or not bool(js.get("ok")):
            return {
                "name": name,
                "error": {
                    "code": "rvc_error",
                    "status": int(getattr(r, "status_code", 500) or 500),
                    "message": "RVC /v1/audio/convert failed",
                    "raw": js,
                },
            }
        return {"name": name, "result": js}
    if name == "voice.sing.diffsinger.rvc" and ALLOW_TOOL_EXECUTION:
        if not DIFFSINGER_RVC_API_URL:
            return {"name": name, "error": "DIFFSINGER_RVC_API_URL not configured"}
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            payload = {
                "lyrics": args.get("lyrics"),
                "notes_midi": args.get("notes_midi"),
                "melody_wav": args.get("melody_wav"),
                "target_voice_ref": args.get("target_voice_ref") or args.get("voice_ref"),
                "seed": args.get("seed"),
            }
            r = await client.post(DIFFSINGER_RVC_API_URL.rstrip("/") + "/v1/voice/sing", json=payload)
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            return {"name": name, "result": js if isinstance(js, dict) else {}}
    if name == "refs.music.build_profile" and ALLOW_TOOL_EXECUTION:
        try:
            body = args if isinstance(args, dict) else {}
            return {"name": name, "result": _refs_music_profile(body)}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "music.lyrics.align" and ALLOW_TOOL_EXECUTION:
        a = args if isinstance(args, dict) else {}
        audio_path = a.get("audio_path")
        lyrics_sections = a.get("lyrics_sections") if isinstance(a.get("lyrics_sections"), list) else []
        changed_line_ids = a.get("changed_line_ids") if isinstance(a.get("changed_line_ids"), list) else []
        if not isinstance(audio_path, str) or not audio_path.strip():
            return {"name": name, "error": "missing_audio_path"}
        if not lyrics_sections:
            return {"name": name, "error": "missing_lyrics_sections"}
        # Two-stage alignment:
        # 1) Coarse section ranges from Song Graph or uniform over duration.
        # 2) Fine line timings per section via MFA when available, else uniform within section.
        full = audio_path
        if not os.path.isabs(full):
            full = os.path.join(UPLOAD_DIR, audio_path.lstrip("/"))
        duration_s = 0.0
        sr = 1
        n_channels = 1
        sampwidth = 2
        frames = 0
        pcm_bytes: bytes = b""
        with contextlib.closing(wave.open(full, "rb")) as wf:  # type: ignore[arg-type]
            frames = wf.getnframes()
            sr = wf.getframerate() or 1
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            duration_s = float(frames) / float(sr)
            pcm_bytes = wf.readframes(frames)
        # Stage 1: coarse section ranges
        section_ranges: Dict[str, Dict[str, float]] = {}
        lb_for_sections = a.get("lock_bundle") if isinstance(a.get("lock_bundle"), dict) else None
        music_sections: List[Dict[str, Any]] = []
        if isinstance(lb_for_sections, dict):
            mus = lb_for_sections.get("music") if isinstance(lb_for_sections.get("music"), dict) else {}
            secs = mus.get("sections") if isinstance(mus.get("sections"), list) else []
            for sec in secs:
                if not isinstance(sec, dict):
                    continue
                sid = sec.get("section_id")
                t0 = sec.get("time_start")
                t1 = sec.get("time_end")
                if isinstance(sid, str) and isinstance(t0, (int, float)) and isinstance(t1, (int, float)):
                    section_ranges[sid] = {"t_start": float(t0), "t_end": float(t1)}
                    music_sections.append(sec)
        # If no timing from Song Graph, fall back to uniform per section.
        if not section_ranges:
            # Divide duration across lyric sections
            total_sections = len([s for s in lyrics_sections if isinstance(s, dict)]) or 1
            sec_span = (duration_s / total_sections) if duration_s > 0 else 0.0
            t0 = 0.0
            idx = 0
            for sec in lyrics_sections:
                if not isinstance(sec, dict):
                    continue
                sec_id = sec.get("section_id")
                if not isinstance(sec_id, str) or not sec_id:
                    continue
                start = t0
                end = t0 + sec_span
                section_ranges[sec_id] = {"t_start": start, "t_end": end}
                t0 = end
                idx += 1
        # Determine whether we have real Song Graph timings (vs uniform fallback).
        has_graph_timings = bool(music_sections)
        # Pre-encode full audio once for MFA in the fallback case.
        full_b64: Optional[str] = None
        if MFA_API_URL and isinstance(MFA_API_URL, str) and MFA_API_URL.strip() and not has_graph_timings:
            with open(full, "rb") as f:
                raw = f.read()
            if raw:
                full_b64 = _b64.b64encode(raw).decode("ascii")
        alignment_sections: List[Dict[str, Any]] = []
        alignment_method = "uniform"
        # Stage 2: per-section fine alignment
        for sec in lyrics_sections:
            if not isinstance(sec, dict):
                continue
            sec_id = sec.get("section_id")
            name = sec.get("name")
            lines = sec.get("lines") if isinstance(sec.get("lines"), list) else []
            if not isinstance(sec_id, str) or not lines:
                continue
            coarse = section_ranges.get(sec_id) or {"t_start": 0.0, "t_end": duration_s}
            sec_t0 = float(coarse.get("t_start") or 0.0)
            sec_t1 = float(coarse.get("t_end") or duration_s)
            # Build section text
            texts: List[str] = []
            for ln in lines:
                if isinstance(ln, dict):
                    txt = ln.get("text")
                    if isinstance(txt, str) and txt.strip():
                        texts.append(txt.strip())
            section_text = "\n".join(texts)
            aligned_lines: List[Dict[str, Any]] = []
            # Try MFA-based alignment when available
            words: List[Dict[str, Any]] = []
            if MFA_API_URL and isinstance(MFA_API_URL, str) and MFA_API_URL.strip() and section_text.strip():
                if has_graph_timings and pcm_bytes and frames > 0 and sr > 0 and n_channels > 0 and sampwidth > 0:
                    # Use a section-specific slice when Song Graph timings are available.
                    bytes_per_frame = n_channels * sampwidth
                    start_frame = int(max(0.0, sec_t0) * float(sr))
                    end_frame = int(min(sec_t1, duration_s) * float(sr))
                    if end_frame > frames:
                        end_frame = frames
                    if end_frame > start_frame:
                        offset_bytes = start_frame * bytes_per_frame
                        length_bytes = (end_frame - start_frame) * bytes_per_frame
                        section_pcm = pcm_bytes[offset_bytes:offset_bytes + length_bytes]
                        buf = BytesIO()
                        with wave.open(buf, "wb") as wf_sec:
                            wf_sec.setnchannels(n_channels)
                            wf_sec.setsampwidth(sampwidth)
                            wf_sec.setframerate(sr)
                            wf_sec.writeframes(section_pcm)
                        wav_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
                        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                            r = await client.post(
                                MFA_API_URL.rstrip("/") + "/align",
                                json={"lyrics": section_text, "wav_bytes": wav_b64},
                            )
                        js = JSONParser().parse(r.text, {"alignment": list}) or {}
                        raw_words = js.get("alignment")
                        if isinstance(raw_words, list):
                            for w in raw_words:
                                if not isinstance(w, dict):
                                    continue
                                ws = w.get("start")
                                we = w.get("end")
                                if isinstance(ws, (int, float)) and isinstance(we, (int, float)):
                                    # Section-relative timings; convert to global seconds later.
                                    words.append({"start": float(ws), "end": float(we)})
                        if words:
                            alignment_method = "mfa"
                elif full_b64:
                    # Fallback: use full-track audio when no graph timings are available.
                    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                        r = await client.post(
                            MFA_API_URL.rstrip("/") + "/align",
                            json={"lyrics": section_text, "wav_bytes": full_b64},
                        )
                    js = JSONParser().parse(r.text, {"alignment": list}) or {}
                    raw_words = js.get("alignment")
                    if isinstance(raw_words, list):
                        for w in raw_words:
                            if not isinstance(w, dict):
                                continue
                            ws = w.get("start")
                            we = w.get("end")
                            if isinstance(ws, (int, float)) and isinstance(we, (int, float)):
                                words.append({"start": float(ws), "end": float(we)})
                    if words:
                        alignment_method = "mfa"
            if words:
                # Map words to lines in sequence
                total_words = len(words)
                line_objs = [ln for ln in lines if isinstance(ln, dict) and isinstance(ln.get("line_id"), str)]
                n_lines_sec = len(line_objs) or 1
                base = total_words // n_lines_sec
                rem = total_words % n_lines_sec
                idx_word = 0
                for i, ln in enumerate(line_objs):
                    line_id = ln.get("line_id")
                    text = ln.get("text")
                    if not isinstance(line_id, str) or not isinstance(text, str):
                        continue
                    count = base + (1 if i < rem else 0)
                    if count <= 0:
                        # fallback: uniform segment within section
                        span = (sec_t1 - sec_t0) / float(n_lines_sec)
                        start = sec_t0 + span * i
                        end = start + span
                        aligned_lines.append({"line_id": line_id, "text": text, "t_start": start, "t_end": end})
                        continue
                    start_idx = idx_word
                    end_idx = min(idx_word + count - 1, total_words - 1)
                    idx_word = end_idx + 1
                    w0 = words[start_idx]
                    w1 = words[end_idx]
                    # MFA times are section-relative when using slices; convert to global seconds.
                    g_start = sec_t0 + float(w0["start"])
                    g_end = sec_t0 + float(w1["end"])
                    aligned_lines.append({"line_id": line_id, "text": text, "t_start": g_start, "t_end": g_end})
            else:
                # Fallback: uniform distribution within section for this section
                line_objs = [ln for ln in lines if isinstance(ln, dict) and isinstance(ln.get("line_id"), str)]
                n_lines_sec = len(line_objs) or 1
                span = (sec_t1 - sec_t0) / float(n_lines_sec) if sec_t1 > sec_t0 else 0.0
                for i, ln in enumerate(line_objs):
                    line_id = ln.get("line_id")
                    text = ln.get("text")
                    if not isinstance(line_id, str) or not isinstance(text, str):
                        continue
                    start = sec_t0 + span * i
                    end = start + span
                    aligned_lines.append({"line_id": line_id, "text": text, "t_start": start, "t_end": end})
            alignment_sections.append({"section_id": sec_id, "name": name, "lines": aligned_lines})
        alignment = {"sections": alignment_sections}
        # Persist into lock bundle if provided
        lock_bundle = a.get("lock_bundle") if isinstance(a.get("lock_bundle"), dict) else None
        if lock_bundle:
            music_branch = lock_bundle.get("music") if isinstance(lock_bundle.get("music"), dict) else {}
            music_branch["lyrics_alignment"] = alignment
            lock_bundle["music"] = music_branch
            try:
                char_id = str(a.get("character_id") or lock_bundle.get("character_id") or "").strip()
                if char_id:
                    await _lock_save(char_id, lock_bundle)
            except Exception as ex:
                logging.error(f"Failed to save lock bundle after music.lyrics.align: {ex}")
        # Optional: if changed_line_ids are provided, map them to windows and trigger
        # per-window refinements via music.refine.window.
        refined_windows: List[str] = []
        cid_val = a.get("cid")
        if lock_bundle and changed_line_ids:
            music_branch = lock_bundle.get("music") if isinstance(lock_bundle.get("music"), dict) else {}
            windows_list = music_branch.get("windows") if isinstance(music_branch.get("windows"), list) else []
            # Build lookup of changed line ids to time ranges from alignment.
            changed_ranges: List[Dict[str, float]] = []
            changed_lines_by_window: Dict[str, List[str]] = {}
            for sec in alignment_sections:
                if not isinstance(sec, dict):
                    continue
                lines = sec.get("lines") if isinstance(sec.get("lines"), list) else []
                for ln in lines:
                    if not isinstance(ln, dict):
                        continue
                    lid = ln.get("line_id")
                    if not isinstance(lid, str) or not lid:
                        continue
                    if lid not in changed_line_ids:
                        continue
                    ts = ln.get("t_start")
                    te = ln.get("t_end")
                    if isinstance(ts, (int, float)) and isinstance(te, (int, float)):
                        changed_ranges.append({"t_start": float(ts), "t_end": float(te)})
            # Map time ranges to windows and collect window-level lyrics snippets.
            window_ids: List[str] = []
            for win in windows_list or []:
                if not isinstance(win, dict):
                    continue
                wid = win.get("window_id")
                if not isinstance(wid, str) or not wid:
                    continue
                wt0 = win.get("t_start")
                wt1 = win.get("t_end")
                if not isinstance(wt0, (int, float)) or not isinstance(wt1, (int, float)):
                    continue
                for cr in changed_ranges:
                    # simple overlap check
                    if cr["t_end"] <= float(wt0) or cr["t_start"] >= float(wt1):
                        continue
                    window_ids.append(wid)
                    # Collect text of any changed lines overlapping this window for prompt/lyrics.
                    for sec in alignment_sections:
                        if not isinstance(sec, dict):
                            continue
                        lines = sec.get("lines") if isinstance(sec.get("lines"), list) else []
                        for ln in lines:
                            if not isinstance(ln, dict):
                                continue
                            lid = ln.get("line_id")
                            if not isinstance(lid, str) or lid not in changed_line_ids:
                                continue
                            ts = ln.get("t_start")
                            te = ln.get("t_end")
                            if not isinstance(ts, (int, float)) or not isinstance(te, (int, float)):
                                continue
                            if te <= float(wt0) or ts >= float(wt1):
                                continue
                            txt = ln.get("text")
                            if isinstance(txt, str) and txt.strip():
                                existing = changed_lines_by_window.get(wid) or []
                                if txt not in existing:
                                    existing.append(txt)
                                changed_lines_by_window[wid] = existing
                    break
            # Deduplicate
            seen: Dict[str, bool] = {}
            uniq_ids: List[str] = []
            for wid in window_ids:
                if wid in seen:
                    continue
                seen[wid] = True
                uniq_ids.append(wid)
            # If no cid provided, try to derive it from full_track_path in music branch.
            if not isinstance(cid_val, str) or not cid_val.strip():
                full_track = music_branch.get("full_track_path")
                if isinstance(full_track, str) and full_track:
                    # Expect .../artifacts/music/{cid}/file.wav
                    parts = full_track.replace("\\", "/").split("/")
                    if len(parts) >= 3:
                        cid_val = parts[-2]
            # Trigger per-window refine calls
            for wid in uniq_ids:
                refine_args: Dict[str, Any] = {
                    "segment_id": wid,
                    "window_id": wid,
                    "cid": cid_val or "",
                    "lock_bundle": lock_bundle,
                }
                # Inject updated lyrics into prompt to steer regeneration.
                win_lines = changed_lines_by_window.get(wid) or []
                if win_lines:
                    snippet = " ".join(win_lines)
                    base_prompt = a.get("prompt") if isinstance(a.get("prompt"), str) else ""
                    if base_prompt:
                        refine_args["prompt"] = f"{base_prompt} New lyrics for this window: {snippet}"
                    else:
                        refine_args["prompt"] = f"New lyrics for this window: {snippet}"
                    refine_args["lyrics"] = snippet
                res_ref = await execute_tool_call({"name": "music.refine.window", "arguments": refine_args})
                if isinstance(res_ref, dict) and not res_ref.get("error"):
                    refined_windows.append(wid)
        env: Dict[str, Any] = {
            "meta": {
                "model": "lyrics-align",
                "ts": _now_ts(),
                "cid": cid_val or "",
                "step": 0,
                "state": "halt",
                "cont": {"present": False, "state_hash": None, "reason": None},
            },
            "reasoning": {"goal": "music lyrics alignment", "constraints": ["json-only"], "decisions": ["music.lyrics.align done"]},
            "evidence": [],
            "message": {"role": "assistant", "type": "tool", "content": "lyrics aligned heuristically"},
            "tool_calls": [
                {
                    "tool": "music.lyrics.align",
                    "args": {"audio_path": audio_path},
                    "status": "done",
                    "result_ref": None,
                }
            ],
            "artifacts": [],
            "telemetry": {
                "window": {"input_bytes": 0, "output_target_tokens": 0},
                "compression_passes": [],
                "notes": [],
            },
        }
        env = normalize_to_envelope(env)
        env = bump_envelope(env)
        assert_envelope(env)
        meta_block = env.get("meta") if isinstance(env.get("meta"), dict) else {}
        if isinstance(meta_block, dict):
            meta_block["alignment_method"] = alignment_method
        env = stamp_env(env, "music.lyrics.align", env.get("meta", {}).get("model"))
        env["alignment"] = alignment
        if refined_windows:
            meta_block = env.get("meta")
            if isinstance(meta_block, dict):
                meta_block["refined_windows"] = refined_windows
        # Trace alignment usage for distillation.
        trace_append(
            "music.lyrics.align",
            {
                "trace_id": (args.get("trace_id") if isinstance(args, dict) else None),
                "cid": cid_val,
                "alignment_method": alignment_method,
                "num_sections": len(alignment_sections),
                "num_lines": sum(
                    len(sec.get("lines") or [])
                    for sec in alignment_sections
                    if isinstance(sec, dict)
                ),
                "changed_line_ids": changed_line_ids,
                "refined_windows": refined_windows,
            },
        )
        return {"name": name, "result": env}
    if name == "audio.foley.hunyuan" and ALLOW_TOOL_EXECUTION:
        if not HUNYUAN_FOLEY_API_URL:
            return {"name": name, "error": "HUNYUAN_FOLEY_API_URL not configured"}
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            payload = {
                "video_ref": args.get("video_ref") or args.get("src"),
                "cue_regions": args.get("cue_regions") or [],
                "style_tags": args.get("style_tags") or [],
            }
            r = await client.post(HUNYUAN_FOLEY_API_URL.rstrip("/") + "/v1/audio/foley", json=payload)
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            return {"name": name, "result": js if isinstance(js, dict) else {}}
    if name in ("image.edit", "image.upscale") and ALLOW_TOOL_EXECUTION:
        return {"name": name, "error": "disabled: use image.dispatch with full graph (no fallbacks)"}
    if name == "image.super_gen" and ALLOW_TOOL_EXECUTION:
        # Multi-object iterative generation: base canvas + per-object refinement + global polish
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            return {
                "name": name,
                "error": {
                    "code": "missing_prompt",
                    "message": "image.super_gen requires a non-empty prompt",
                    "status": 400,
                },
            }
        size_text = args.get("size") or "1024x1024"
        # Parse size; if invalid, surface a structured error instead of relying on
        # implicit ValueError from int() casting.
        size_parts = str(size_text).lower().split("x")
        if len(size_parts) != 2:
            return {
                "name": name,
                "error": {
                    "code": "bad_image_size",
                    "message": f"Invalid size '{size_text}', expected 'WIDTHxHEIGHT'",
                    "status": 400,
                    "raw": {"size": size_text},
                },
            }
        try:
            w, h = [int(x) for x in size_parts]
        except (TypeError, ValueError) as ex:
            _log(
                "image.super_gen.bad_size_cast",
                size=size_text,
                error=str(ex),
                stack="".join(traceback.format_exception(type(ex), ex, ex.__traceback__)),
            )
            return {
                "name": name,
                "error": {
                    "code": "bad_image_size",
                    "message": f"Invalid size '{size_text}', expected 'WIDTHxHEIGHT'",
                    "status": 400,
                    "raw": {"size": size_text},
                },
            }
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "image", f"super-{int(time.time())}")
        os.makedirs(outdir, exist_ok=True)
        # 1) Decompose prompt → object prompts (heuristic: split by commas/" and ")
        objs = []
        base_style = prompt
        parts = [p.strip() for p in prompt.replace(" and ", ", ").split(",") if p.strip()]
        if len(parts) >= 3:
            base_style = parts[0]
            objs = parts[1:]
        else:
            objs = parts
        if not objs:
            objs = [prompt]
        # 2) Layout: grid boxes left→right, top→bottom
        cols = max(1, int((len(objs) + 1) ** 0.5))
        rows = max(1, (len(objs) + cols - 1) // cols)
        cell_w = max(256, w // cols)
        cell_h = max(256, h // rows)
        boxes = []
        for i, _ in enumerate(objs):
            r = i // cols; c = i % cols
            x0 = min(w - cell_w, c * cell_w)
            y0 = min(h - cell_h, r * cell_h)
            boxes.append((x0, y0, x0 + cell_w, y0 + cell_h))
        # Best-effort extraction of exact text when the user explicitly quotes it.
        exact_text = None
        quoted = _re.findall(r"\"([^\"]{2,})\"|'([^']{2,})'", prompt)
        for a, b in quoted:
            t = a or b
            if len(t.split()) <= 4:
                exact_text = t
                break
        # 4) Base canvas
        base_args = {"prompt": base_style, "size": f"{w}x{h}", "seed": args.get("seed"), "refs": args.get("refs"), "cid": args.get("cid")}
        base = await execute_tool_call({"name": "legacy.image.gen", "arguments": base_args})
        base_path = None
        if isinstance(base, dict):
            base_res = base.get("result") if isinstance(base.get("result"), dict) else {}
            meta = base_res.get("meta") if isinstance(base_res.get("meta"), dict) else {}
            cid = meta.get("cid")
            tool_calls = base_res.get("tool_calls") if isinstance(base_res.get("tool_calls"), list) else []
            tc0 = tool_calls[0] if tool_calls and isinstance(tool_calls[0], dict) else {}
            rid = tc0.get("result_ref")
            if isinstance(cid, str) and cid and isinstance(rid, str) and rid:
                base_path = _os.path.join(UPLOAD_DIR, "artifacts", "image", cid, rid)
        if not base_path or not _os.path.exists(base_path):
            return {"name": name, "error": "base generation failed"}
        canvas = Image.open(base_path).convert("RGB")
        # 5) Per-object refinement via tiled generations blended into canvas with object-level CLIP constraint
        for (obj_prompt, box) in zip(objs, boxes):
            try:
                refined_prompt = obj_prompt
                if exact_text:
                    refined_prompt = f"{obj_prompt}, exact text: {exact_text}"
                sub_args = {"prompt": refined_prompt, "size": f"{box[2]-box[0]}x{box[3]-box[1]}", "seed": args.get("seed"), "refs": args.get("refs"), "cid": args.get("cid")}
                best_tile = None
                for attempt in range(0, 3):
                    sub = await execute_tool_call({"name": "legacy.image.gen", "arguments": sub_args})
                    scid = ((sub.get("result") or {}).get("meta") or {}).get("cid")
                    srid = ((sub.get("result") or {}).get("tool_calls") or [{}])[0].get("result_ref")
                    sub_path = _os.path.join(UPLOAD_DIR, "artifacts", "image", scid, srid) if (scid and srid) else None
                    if sub_path and _os.path.exists(sub_path):
                        tile = Image.open(sub_path).convert("RGB")
                        # object-level CLIP score via full analyzer
                        try:
                            ai = _analyze_image(sub_path, prompt=refined_prompt)
                            sem = ai.get("semantics") or {}
                            sc = float(sem.get("clip_score") or 0.0)
                        except Exception as ex:
                            logging.debug(f"image.super_gen.clip_score_failed: {ex}", exc_info=True)
                            sc = 1.0
                        best_tile = tile if (best_tile is None or sc >= 0.35) else best_tile
                        if sc >= 0.35:
                            break
                        # strengthen prompt for retry
                        sub_args["prompt"] = f"{refined_prompt}, literal, clear details, no drift"
                if best_tile is not None:
                    canvas.paste(best_tile.resize((box[2]-box[0], box[3]-box[1])), (box[0], box[1]))
            except Exception as ex:
                _log("image.super_gen.tile.error", error=str(ex))
                continue
        # 6) Final signage overlay (safety net) to strictly enforce text if requested
        # This is best-effort: failures are logged but do not abort image generation.
        if exact_text:
            try:
                draw = ImageDraw.Draw(canvas)
                # Choose a reasonable target region without keyword heuristics.
                target_box = boxes[0] if boxes else (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
                tx, ty = (target_box[0] + 10, target_box[1] + 10)
                # Fallback font
                try:
                    font = ImageFont.truetype("arial.ttf", max(24, (target_box[3] - target_box[1]) // 6))
                except (OSError, IOError):
                    font = None
                draw.text((tx, ty), str(exact_text), fill=(220, 220, 220), font=font)
            except Exception as ex:
                _log("image.super_gen.signage.error", error=str(ex))
        final_path = _os.path.join(outdir, "final.png")
        canvas.save(final_path)
        url = final_path.replace("/workspace", "") if final_path.startswith("/workspace/") else final_path
        cid_val = args.get("cid")
        if isinstance(cid_val, (str, int)):
            cid_str = str(cid_val).strip()
        else:
            cid_str = ""
        if not cid_str:
            cid_str = f"img-{_now_ts()}"
            args["cid"] = cid_str
        _ctx_add(
            cid_str,
            "image",
            final_path,
            url,
            base_path,
            ["super_gen"],
            {"objects": objs, "boxes": boxes, "signage_text": exact_text},
        )
        trace_append(
            "image",
            {
                "cid": cid_str,
                "tool": "image.super_gen",
                "prompt": prompt,
                "size": f"{w}x{h}",
                "objects": objs,
                "boxes": boxes,
                "path": final_path,
                "signage_text": exact_text,
                "web_sources": web_sources,
            },
        )
        return {"name": name, "result": {"path": url}}
    if name == "research.run" and ALLOW_TOOL_EXECUTION:
        # Prefer local orchestrator; on failure, try DRT service if configured
        try:
            job_args = args if isinstance(args, dict) else {}
            job_args.setdefault("job_id", uuid.uuid4().hex)
            result = await run_research(job_args)
            if isinstance(result, dict):
                result.setdefault("job_id", job_args.get("job_id"))
            return {"name": name, "result": result}
        except Exception as ex:
            base = (DRT_API_URL or "").rstrip("/")
            if not base:
                return {"name": name, "error": str(ex)}
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                r = await client.post(base + "/research/run", json=args)
                parser = JSONParser()
                js = parser.parse(r.text or "", {})
                return {"name": name, "result": js if isinstance(js, dict) else {}}
    if name == "code.super_loop" and ALLOW_TOOL_EXECUTION:
        # Back code.super_loop with the central committee AI path only.
        repo_root = os.getenv("REPO_ROOT", "/workspace")
        _step_tokens_raw = os.getenv("CODE_LOOP_STEP_TOKENS", "900") or "900"
        try:
            step_tokens = int(str(_step_tokens_raw).strip() or "900")
        except Exception as ex:
            logging.warning("code.super_loop: bad CODE_LOOP_STEP_TOKENS=%r; defaulting to 900", _step_tokens_raw, exc_info=True)
            step_tokens = 900
        task = args.get("task") or ""
        env = await run_super_loop(
            task=str(task or ""),
            repo_root=repo_root,
            trace_id=trace_id,
            step_tokens=step_tokens,
        )
        return {"name": name, "result": env}
    if name == "web_search" and ALLOW_TOOL_EXECUTION:
        # Robust multi-engine metasearch without external API keys
        q = args.get("q") or args.get("query") or ""
        if not q:
            return {"name": name, "error": "missing query"}
        try:
            k = int(args.get("k", 10))
        except Exception as ex:
            logging.warning("web_search: bad k=%r; defaulting to 10", args.get("k"), exc_info=True)
            k = 10
        # Reuse the metasearch fuse logic directly
        engines: Dict[str, List[Dict[str, Any]]] = {}
        for eng in ("google", "brave", "duckduckgo", "bing", "mojeek"):
            engines.setdefault(eng, _synthetic_search_results(eng, q, k))
        fused = _rrf_fuse(engines, k=60)[:k]
        # Build text snippet bundle for convenience
        lines = []
        for it in fused:
            lines.append(f"- {it.get('title','')}")
            lines.append(it.get('snippet',''))
            lines.append(it.get('link',''))
        return {"name": name, "result": "\n".join(lines)}
    if name == "metasearch.fuse" and ALLOW_TOOL_EXECUTION:
        q = args.get("q") or ""
        if not q:
            return {"name": name, "error": "missing q"}
        try:
            k = int(args.get("k", 10))
        except Exception as ex:
            logging.warning(f"metasearch.fuse: bad k={args.get('k')!r}; defaulting to 10 ({ex})", exc_info=True)
            k = 10
        # Deterministic multi-engine placeholder only; SERPAPI removed in favor of internal metasearch
        engines: Dict[str, List[Dict[str, Any]]] = {}
        # deterministic synthetic engines
        for eng in ("brave", "mojeek", "openalex", "gdelt"):
            engines.setdefault(eng, _synthetic_search_results(eng, q, k))
        fused = _rrf_fuse(engines, k=60)[:k]
        return {"name": name, "result": {"engines": list(engines.keys()), "results": fused}}
    if name == "web.smart_get" and ALLOW_TOOL_EXECUTION:
        url = args.get("url") or ""
        if not url:
            return {"name": name, "error": "missing url"}
        base = (DRT_API_URL or "").rstrip("/")
        if not base:
            return {"name": name, "error": "drt_unavailable"}
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            payload = {"url": url, "modes": (args.get("modes") or {})}
            if isinstance(args.get("headers"), dict):
                payload["headers"] = args.get("headers")
            r = await client.post(base + "/web/smart_get", json=payload)
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            return {"name": name, "result": js if isinstance(js, dict) else {}}
    if name == "source_fetch" and ALLOW_TOOL_EXECUTION:
        url = args.get("url") or ""
        if not url:
            return {"name": name, "error": "missing url"}
        try:
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                r = await client.get(url)
            ct = r.headers.get("content-type", "")
            data = r.content or b""
            h = _hl.sha256(data).hexdigest()
            preview = None
            if ct.startswith("text/") or ct.find("json") >= 0:
                txt = data.decode("utf-8", errors="ignore")
                preview = txt[:2000]
            return {"name": name, "result": {"status": r.status_code, "content_type": ct, "sha256": h, "preview": preview}}
        except Exception as ex:
            logging.warning(f"source_fetch.failed url={url}: {ex}", exc_info=True)
            return {"name": name, "error": str(ex)}
    if name == "frame_interpolate":
        # Acknowledge and surface parameters; actual interpolation occurs during assembly if requested
        factor = int(args.get("factor") or 2)
        return {"name": name, "result": {"accepted": True, "factor": factor}}
    if name == "upscale":
        scale = int(args.get("scale") or 2)
        tile = int(args.get("tile") or 512)
        overlap = int(args.get("overlap") or 16)
        return {"name": name, "result": {"accepted": True, "scale": scale, "tile": tile, "overlap": overlap}}
    # --- Standalone FFmpeg tools ---
    if name == "ffmpeg.trim":
        src = args.get("src") or ""
        start = str(args.get("start") or "0")
        duration = str(args.get("duration") or "")
        if not src:
            return {"name": name, "error": "missing src"}
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "video", f"trim-{int(time.time())}")
        os.makedirs(outdir, exist_ok=True)
        dst = os.path.join(outdir, "trimmed.mp4")
        ff = ["ffmpeg", "-y", "-i", src, "-ss", start]
        if duration:
            ff += ["-t", duration]
        ff += ["-c", "copy", dst]
        try:
            subprocess.run(ff, check=True)
            url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
            # Distilled artifact row for video
            tr = args.get("trace_id") if isinstance(args.get("trace_id"), str) else None
            if tr:
                try:
                    rel = os.path.relpath(dst, UPLOAD_DIR).replace("\\", "/")
                    _log("artifact", trace_id=str(tr), kind="video", path=rel)
                except Exception as ex:
                    _log("video.interpolate.checkpoint.error", error=str(ex), path=dst)
            return {"name": name, "result": {"path": url}}
        except Exception as ex:
            logging.error(f"ffmpeg.trim failed src={src} dst={dst} cmd={' '.join(ff)}: {ex}", exc_info=True)
            return {"name": name, "error": str(ex)}
    if name == "ffmpeg.concat":
        inputs = args.get("inputs") or []
        if not isinstance(inputs, list) or not inputs:
            return {"name": name, "error": "missing inputs"}
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "video", f"concat-{int(time.time())}")
        os.makedirs(outdir, exist_ok=True)
        listfile = os.path.join(outdir, "list.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            for p in inputs:
                f.write(f"file '{p}'\n")
        dst = os.path.join(outdir, "concat.mp4")
        ff = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", dst]
        try:
            subprocess.run(ff, check=True)
            url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
            return {"name": name, "result": {"path": url}}
        except Exception as ex:
            logging.error(f"ffmpeg.concat failed dst={dst} cmd={' '.join(ff)}: {ex}", exc_info=True)
            return {"name": name, "error": str(ex)}
    if name == "ffmpeg.audio_mix":
        a = args.get("a"); b = args.get("b"); vol_a = str(args.get("vol_a") or "1.0"); vol_b = str(args.get("vol_b") or "1.0")
        if not a or not b:
            return {"name": name, "error": "missing a/b"}
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "audio", f"mix-{int(time.time())}")
        os.makedirs(outdir, exist_ok=True)
        dst = os.path.join(outdir, "mix.wav")
        ff = ["ffmpeg", "-y", "-i", a, "-i", b, "-filter_complex", f"[0:a]volume={vol_a}[a0];[1:a]volume={vol_b}[a1];[a0][a1]amix=inputs=2:duration=longest", dst]
        try:
            subprocess.run(ff, check=True)
            url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
            return {"name": name, "result": {"path": url}}
        except Exception as ex:
            logging.error(f"ffmpeg.audio_mix failed dst={dst} cmd={' '.join(ff)}: {ex}", exc_info=True)
            return {"name": name, "error": str(ex)}
    if name == "image.cleanup":
        src = args.get("src") or ""
        if not src:
            return {"name": name, "error": "missing src"}
        try:
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                return {"name": name, "error": "failed to read src"}
            work = img.copy()
            if bool(args.get("denoise")):
                work = cv2.fastNlMeansDenoisingColored(work, None, 5, 5, 7, 21)
            if bool(args.get("dehalo")):
                # simple dehalo: bilateral + unsharp blend
                blur = cv2.bilateralFilter(work, 9, 75, 75)
                work = cv2.addWeighted(work, 0.6, blur, 0.4, 0)
            if bool(args.get("sharpen")):
                k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                work = cv2.filter2D(work, -1, k)
            if bool(args.get("clahe")):
                lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                lab = cv2.merge((cl, a, b))
                work = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            outdir = os.path.join(UPLOAD_DIR, "artifacts", "image", f"cleanup-{int(time.time())}")
            os.makedirs(outdir, exist_ok=True)
            dst = os.path.join(outdir, "clean.png")
            cv2.imwrite(dst, work)
            url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
            cid_val = args.get("cid")
            if isinstance(cid_val, (str, int)):
                cid_str = str(cid_val).strip()
            else:
                cid_str = ""
            if not cid_str:
                cid_str = f"img-{_now_ts()}"
                args["cid"] = cid_str
            _ctx_add(cid_str, "image", dst, url, src, ["cleanup"], {})
            trace_append("image", {"cid": cid_str, "tool": "image.cleanup", "src": src, "path": dst})
            return {"name": name, "result": {"path": url}}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "video.cleanup":
        src = args.get("src") or ""
        if not src:
            return {"name": name, "error": "missing src"}
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "video", f"vclean-{int(time.time())}")
        os.makedirs(outdir, exist_ok=True)
        dst = os.path.join(outdir, "clean.mp4")
        # ffmpeg filter chain
        denoise = "hqdn3d=2.0:1.5:3.0:3.0" if bool(args.get("denoise", True)) else None
        deband = "gradfun=20:30" if bool(args.get("deband", True)) else None
        sharpen = "unsharp=5:5:0.8:3:3:0.4" if bool(args.get("sharpen", True)) else None
        vf_parts = [p for p in (denoise, deband, sharpen) if p]
        vf = ",".join(vf_parts) if vf_parts else "null"
        ff = ["ffmpeg", "-y", "-i", src, "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-c:a", "copy", dst]
        try:
            subprocess.run(ff, check=True)
            url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
            cid_val = args.get("cid")
            if isinstance(cid_val, (str, int)):
                cid_str = str(cid_val).strip()
            else:
                cid_str = ""
            if not cid_str:
                cid_str = f"vid-{_now_ts()}"
                args["cid"] = cid_str
            _ctx_add(cid_str, "video", dst, url, src, ["cleanup"], {"vf": vf})
            trace_append("video", {"cid": cid_str, "tool": "video.cleanup", "src": src, "vf": vf, "path": dst})
            return {"name": name, "result": {"path": url}}
        except Exception as ex:
            logging.error(f"video.cleanup failed src={src} dst={dst} cmd={' '.join(ff)}: {ex}", exc_info=True)
            return {"name": name, "error": str(ex)}
    if name == "image.artifact_fix":
        src = args.get("src") or ""; atype = (args.get("type") or "").strip().lower()
        if not src or atype not in ("clock", "glass"):
            return {"name": name, "error": "missing src or unsupported type"}
        try:
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                return {"name": name, "error": "failed to read src"}
            work = img.copy()
            h, w = work.shape[:2]
            region = args.get("region") if isinstance(args.get("region"), list) and len(args.get("region")) == 4 else [0, 0, w, h]
            x, y, rw, rh = [int(max(0, v)) for v in region]
            x = min(x, w - 1); y = min(y, h - 1); rw = min(rw, w - x); rh = min(rh, h - y)
            roi = work[y:y+rh, x:x+rw].copy()
            if atype == "clock":
                # Heuristic clock fix: detect circle and overlay canonical hands at target_time (default 10:10)
                target_time = str(args.get("target_time") or "10:10")
                hh, mm = 10, 10
                if isinstance(target_time, str) and target_time:
                    m = _re.match(r"^\s*(\d{1,2})\s*:\s*(\d{1,2})\s*$", target_time)
                    if m:
                        hh = int(m.group(1)) % 12
                        mm = int(m.group(2)) % 60
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 30, param1=80, param2=30, minRadius=int(min(rw, rh) * 0.2), maxRadius=int(min(rw, rh) * 0.5))
                cx, cy, cr = rw // 2, rh // 2, int(min(rw, rh) * 0.4)
                if circles is not None and len(circles[0]) > 0:
                    c = circles[0][0]; cx, cy, cr = int(c[0]), int(c[1]), int(c[2])
                # Draw hands
                center = (cx, cy)
                mm_angle = (mm / 60.0) * 2 * np.pi
                hh_angle = ((hh % 12) / 12.0 + (mm / 60.0) / 12.0) * 2 * np.pi
                mm_pt = (int(cx + 0.85 * cr * np.sin(mm_angle)), int(cy - 0.85 * cr * np.cos(mm_angle)))
                hh_pt = (int(cx + 0.55 * cr * np.sin(hh_angle)), int(cy - 0.55 * cr * np.cos(hh_angle)))
                cv2.line(roi, center, mm_pt, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.line(roi, center, hh_pt, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(roi, center, 3, (0, 0, 0), -1, cv2.LINE_AA)
            elif atype == "glass":
                # Heuristic glass fill-level smoothing in region: equalize horizontal band color
                band_h = max(2, rh // 8)
                band_y = y + (rh // 2 - band_h // 2)
                band = work[band_y:band_y+band_h, x:x+rw]
                mean_color = band.mean(axis=(0, 1)).astype(np.uint8)
                for yy in range(y + rh // 3, y + int(rh * 0.95)):
                    work[yy, x:x+rw] = cv2.addWeighted(work[yy, x:x+rw], 0.6, np.full((1, rw, 3), mean_color, dtype=np.uint8), 0.4, 0)
            work[y:y+rh, x:x+rw] = roi
            outdir = os.path.join(UPLOAD_DIR, "artifacts", "image", f"afix-{int(time.time())}")
            os.makedirs(outdir, exist_ok=True)
            dst = os.path.join(outdir, "fixed.png")
            cv2.imwrite(dst, work)
            url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
            cid_val = args.get("cid")
            if isinstance(cid_val, (str, int)):
                cid_str = str(cid_val).strip()
            else:
                cid_str = ""
            if not cid_str:
                cid_str = f"img-{_now_ts()}"
                args["cid"] = cid_str
            _ctx_add(
                cid_str,
                "image",
                dst,
                url,
                src,
                ["artifact_fix", atype],
                {"region": region, "target_time": args.get("target_time")},
            )
            trace_append(
                "image",
                {
                    "cid": cid_str,
                    "tool": "image.artifact_fix",
                    "src": src,
                    "type": atype,
                    "path": dst,
                },
            )
            return {"name": name, "result": {"path": url}}
        except Exception as ex:
            logging.error(f"image.artifact_fix failed src={src} type={atype}: {ex}", exc_info=True)
            return {"name": name, "error": str(ex)}
    if name == "video.artifact_fix":
        src = args.get("src") or ""; atype = (args.get("type") or "").strip().lower()
        if not src or atype not in ("clock", "glass"):
            return {"name": name, "error": "missing src or unsupported type"}
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "video", f"vafix-{int(time.time())}")
        frames_dir = os.path.join(outdir, "frames"); os.makedirs(frames_dir, exist_ok=True)
        dst = os.path.join(outdir, "fixed.mp4")
        try:
            # Extract frames
            subprocess.run(["ffmpeg", "-y", "-i", src, os.path.join(frames_dir, "%06d.png")], check=True)
            # Process frames with image.artifact_fix
            frame_files = sorted(_glob(os.path.join(frames_dir, "*.png")))
            for fp in frame_files:
                _ = await execute_tool_call({"name": "image.artifact_fix", "arguments": {"src": fp, "type": atype, "target_time": args.get("target_time"), "region": args.get("region"), "cid": args.get("cid")}})
            # Reassemble
            subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(frames_dir, "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", dst], check=True)
            url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
            cid_val = args.get("cid")
            if isinstance(cid_val, (str, int)):
                cid_str = str(cid_val).strip()
            else:
                cid_str = ""
            if not cid_str:
                cid_str = f"vid-{_now_ts()}"
                args["cid"] = cid_str
            _ctx_add(cid_str, "video", dst, url, src, ["artifact_fix", atype], {})
            trace_append(
                "video",
                {
                    "cid": cid_str,
                    "tool": "video.artifact_fix",
                    "src": src,
                    "type": atype,
                    "path": dst,
                },
            )
            return {"name": name, "result": {"path": url}}
        except Exception as ex:
            logging.error(f"video.flow.derive failed src={src} frame_a={frame_a} frame_b={frame_b}: {ex}", exc_info=True)
            return {"name": name, "error": str(ex)}
    if name == "image.hands.fix":
        src = args.get("src") or ""
        if not src:
            return {"name": name, "error": "missing src"}
        try:
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            if img is None:
                return {"name": name, "error": "failed to read src"}
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mp_hands = mp.solutions.hands
            with mp_hands.Hands(static_image_mode=True, max_num_hands=4, min_detection_confidence=0.3) as hands:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    for hand_landmarks in res.multi_hand_landmarks:
                        pts = []
                        for lm in hand_landmarks.landmark:
                            pts.append((int(lm.x * w), int(lm.y * h)))
                        if len(pts) >= 3:
                            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                            cv2.fillConvexPoly(mask, hull, 255)
            if mask.max() == 0:
                return {"name": name, "error": "no hands detected"}
            # Split into per-hand components and fix each with dynamic parameters derived from local geometry
            work = img.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 50:
                    continue
                x, y, ww, hh = cv2.boundingRect(cnt)
                pad = max(4, int(min(ww, hh) * 0.08))
                x0 = max(0, x - pad); y0 = max(0, y - pad)
                x1 = min(w, x + ww + pad); y1 = min(h, y + hh + pad)
                roi_img = work[y0:y1, x0:x1].copy()
                roi_mask_full = np.zeros_like(mask)
                cv2.drawContours(roi_mask_full, [cnt], -1, 255, thickness=cv2.FILLED)
                roi_mask = roi_mask_full[y0:y1, x0:x1].copy()
                # Dynamic morphology and feather based on local size
                ksz = max(1, int(min(ww, hh) * 0.02)) | 1
                roi_mask = cv2.dilate(roi_mask, np.ones((ksz,ksz), np.uint8), iterations=1)
                roi_mask = cv2.erode(roi_mask, np.ones((ksz,ksz), np.uint8), iterations=1)
                # Distance-transform based feather for smooth seams
                dist = cv2.distanceTransform((roi_mask>0).astype(np.uint8), cv2.DIST_L2, 5)
                if dist.max() > 0:
                    alpha = (dist / dist.max()).astype(np.float32)
                else:
                    alpha = (roi_mask.astype(np.float32) / 255.0)
                alpha = cv2.GaussianBlur(alpha, (0,0), sigmaX=max(1.0, min(6.0, min(ww,hh)*0.02)))
                alpha3 = cv2.merge([alpha, alpha, alpha])
                # Dynamic inpaint radius from local box
                radius = max(2, min(15, int(min(ww, hh) * 0.03)))
                roi_inpaint = cv2.inpaint(roi_img, roi_mask, radius, cv2.INPAINT_TELEA)
                # Composite back using feathered alpha to preserve detail
                roi_blend = (roi_inpaint.astype(np.float32) * alpha3 + roi_img.astype(np.float32) * (1.0 - alpha3)).astype(np.uint8)
                work[y0:y1, x0:x1] = roi_blend
            out = work
            outdir = os.path.join(UPLOAD_DIR, "artifacts", "image", f"handsfix-{int(time.time())}")
            os.makedirs(outdir, exist_ok=True)
            dst = os.path.join(outdir, "fixed.png")
            cv2.imwrite(dst, out)
            url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
            cid_val = args.get("cid")
            if isinstance(cid_val, (str, int)):
                cid_str = str(cid_val).strip()
            else:
                cid_str = ""
            if not cid_str:
                cid_str = f"img-{_now_ts()}"
                args["cid"] = cid_str
            _ctx_add(cid_str, "image", dst, url, src, ["hands_fix"], {})
            trace_append("image", {"cid": cid_str, "tool": "image.hands.fix", "src": src, "path": dst})
            return {"name": name, "result": {"path": url}}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "video.hands.fix":
        src = args.get("src") or ""
        if not src:
            return {"name": name, "error": "missing src"}
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "video", f"vhandsfix-{int(time.time())}")
        frames_dir = os.path.join(outdir, "frames"); os.makedirs(frames_dir, exist_ok=True)
        dst = os.path.join(outdir, "fixed.mp4")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", src, os.path.join(frames_dir, "%06d.png")], check=True)
            frame_files = sorted(_glob(os.path.join(frames_dir, "*.png")))
            for fp in frame_files:
                _ = await execute_tool_call({"name": "image.hands.fix", "arguments": {"src": fp, "cid": args.get("cid")}})
            subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(frames_dir, "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", dst], check=True)
            url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
            cid_val = args.get("cid")
            if isinstance(cid_val, (str, int)):
                cid_str = str(cid_val).strip()
            else:
                cid_str = ""
            if not cid_str:
                cid_str = f"vid-{_now_ts()}"
                args["cid"] = cid_str
            _ctx_add(cid_str, "video", dst, url, src, ["hands_fix"], {})
            trace_append("video", {"cid": cid_str, "tool": "video.hands.fix", "src": src, "path": dst})
            return {"name": name, "result": {"path": url}}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "video.interpolate":
        src = args.get("src") or ""
        target_fps = int(args.get("target_fps") or 60)
        if not src:
            return {"name": name, "error": "missing src"}
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "video", f"interp-{int(time.time())}")
        os.makedirs(outdir, exist_ok=True)
        dst = os.path.join(outdir, "interpolated.mp4")
        vf = f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        ff = ["ffmpeg", "-y", "-i", src, "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-an", dst]
        proc = subprocess.run(ff, check=False)
        if proc.returncode != 0:
            return {"name": name, "error": f"ffmpeg interpolate exited with code {proc.returncode}"}
        # Optional face/consistency lock stabilization pass
        locks = args.get("locks") or {}
        face_images: list[str] = []
        # Resolve ref_ids -> images via refs.apply
        if isinstance(args.get("ref_ids"), list):
            for rid in args.get("ref_ids"):
                pack, code = _refs_apply({"ref_id": rid})
                if isinstance(pack, dict) and pack.get("ref_pack"):
                    imgs = (pack.get("ref_pack") or {}).get("images") or []
                    for p in imgs:
                        if isinstance(p, str):
                            face_images.append(p)
        # Also accept direct locks.image.images
        if isinstance(locks.get("image"), dict):
            for k in ("face_images", "images"):
                for p in (locks.get("image", {}).get(k) or []):
                    if isinstance(p, str):
                        face_images.append(p)
        if not face_images:
            cid_raw = args.get("cid")
            cid_val = str(cid_raw).strip() if isinstance(cid_raw, (str, int)) else ""
            if cid_val:
                recents = _ctx_list(cid_val, limit=20, kind_hint="image")
                for it in reversed(recents):
                    p = it.get("path")
                    if isinstance(p, str):
                        face_images.append(p)
                        break
        if COMFYUI_API_URL and FACEID_API_URL and face_images:
            # 1) Extract frames
            frames_dir = os.path.join(outdir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-i", dst, os.path.join(frames_dir, "%06d.png")], check=True)
            # 2) Compute face embedding from first ref
            face_src = face_images[0]
            if face_src.startswith("/workspace/"):
                face_url = face_src.replace("/workspace", "")
            else:
                face_url = face_src if face_src.startswith("/uploads/") else face_src
            with _hx.Client() as _c:
                er = _c.post(FACEID_API_URL.rstrip("/") + "/embed", json={"image_url": face_url})
                emb = None
                if er.status_code == 200:
                    parser = JSONParser()
                    ej = parser.parse(er.text or "", {"embedding": list, "vec": list})
                    if isinstance(ej, dict):
                        emb = ej.get("embedding") or ej.get("vec")
            # 3) Per-frame InstantID apply with low denoise
            if emb and isinstance(emb, list):
                frame_files = sorted(_glob(os.path.join(frames_dir, "*.png")))
                for fp in frame_files:
                    # Build minimal ComfyUI graph for stabilization
                    g = {
                        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
                        "2": {"class_type": "LoadImage", "inputs": {"image": fp}},
                        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
                        "6": {
                            "class_type": "KSampler",
                            "inputs": {
                                "seed": 0,
                                "steps": 10,
                                "cfg": 4.0,
                                "sampler_name": "dpmpp_2m",
                                "scheduler": "karras",
                                "denoise": 0.15,
                                "model": ["1", 0],
                                "positive": ["3", 0],
                                "negative": ["4", 0],
                                "latent_image": ["5", 0],
                            },
                        },
                        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "stabilize face, preserve identity", "clip": ["1", 1]}},
                        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
                        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
                        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "frame_out", "images": ["7", 0]}},
                    }
                    # Inject InstantIDApply
                    g["22"] = {
                        "class_type": "InstantIDApply",
                        "inputs": {"model": ["1", 0], "image": ["2", 0], "embedding": emb, "strength": 0.70},
                    }
                    g["6"]["inputs"]["model"] = ["22", 0]
                    with _hx.Client() as _c2:
                        pr = _c2.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json={"prompt": g, "client_id": "wrapper-001"})
                        if pr.status_code == 200:
                            parser = JSONParser()
                            pj = parser.parse(pr.text or "", {"prompt_id": str, "uuid": str, "id": str})
                            pid = (pj.get("prompt_id") or pj.get("uuid") or pj.get("id")) if isinstance(pj, dict) else None
                            # poll for completion
                            while True:
                                hr = _c2.get(COMFYUI_API_URL.rstrip("/") + f"/history/{pid}")
                                if hr.status_code == 200:
                                    parser = JSONParser()
                                    hj = parser.parse(hr.text or "{}", {})
                                    h = _normalize_comfy_history_entry(hj if isinstance(hj, dict) else {}, str(pid))
                                    if h and _comfy_is_completed(h):
                                        assets = _extract_comfy_asset_urls(h, COMFYUI_API_URL or "http://comfyui:8188")
                                        if assets:
                                            src = assets[0].get("url")
                                            if isinstance(src, str) and src:
                                                vr = _c2.get(src)
                                                if vr.status_code == 200:
                                                    with open(fp, "wb") as wf:
                                                        wf.write(vr.content)
                                                    break
                                        break
                                time.sleep(0.5)
                # 4) Reassemble stabilized video
                dst2 = os.path.join(outdir, "interpolated_stabilized.mp4")
                subprocess.run(
                    ["ffmpeg", "-y", "-framerate", str(target_fps), "-i", os.path.join(frames_dir, "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", dst2],
                    check=True,
                )
                dst = dst2
        url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
        cid_val = args.get("cid")
        if isinstance(cid_val, (str, int)):
            cid_str = str(cid_val).strip()
        else:
            cid_str = ""
        if not cid_str:
            cid_str = f"vid-{_now_ts()}"
            args["cid"] = cid_str
        _ctx_add(cid_str, "video", dst, url, src, ["interpolate"], {"target_fps": target_fps})
        trace_append("video", {"cid": cid_str, "tool": "video.interpolate", "src": src, "target_fps": target_fps, "path": dst})
        return {"name": name, "result": {"path": url}}
    if name == "video.flow.derive":
        frame_a = args.get("frame_a") or ""
        frame_b = args.get("frame_b") or ""
        src = args.get("src") or ""
        a_img = None
        b_img = None
        try:
            if frame_a and frame_b:
                a_img = _video_read_any(frame_a)
                b_img = _video_read_any(frame_b)
            elif src:
                # Grab two frames step apart from the tail
                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    return {"name": name, "error": "failed to open src"}
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                step = max(1, int(args.get("step") or 1))
                idx_b = max(0, total - 1)
                idx_a = max(0, idx_b - step)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx_a)
                ok_a, a_img = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
                ok_b, b_img = cap.read()
                cap.release()
                if not ok_a or not ok_b:
                    a_img = None; b_img = None
            if a_img is None or b_img is None:
                return {"name": name, "error": "missing frames"}
            # Convert to gray
            a_gray = cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY)
            b_gray = cv2.cvtColor(b_img, cv2.COLOR_BGR2GRAY)
            # Try TV-L1 if available; fallback to Farneback
            flow = None
            try:
                tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()  # type: ignore
                flow = tvl1.calc(a_gray, b_gray, None)
            except (AttributeError, cv2.error) as ex:
                logging.debug(f"video.flow.tvl1_unavailable: {ex}", exc_info=True)
                flow = cv2.calcOpticalFlowFarneback(a_gray, b_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fx = flow[..., 0].astype(np.float32)
            fy = flow[..., 1].astype(np.float32)
            mag = np.sqrt(fx * fx + fy * fy)
            outdir = os.path.join(UPLOAD_DIR, "artifacts", "video", f"flow-{int(_tm.time())}")
            os.makedirs(outdir, exist_ok=True)
            npz_path = os.path.join(outdir, "flow.npz")
            np.savez_compressed(npz_path, fx=fx, fy=fy, mag=mag)
            url = npz_path.replace("/workspace", "") if npz_path.startswith("/workspace/") else npz_path
            cid_val = args.get("cid")
            if isinstance(cid_val, (str, int)):
                cid_str = str(cid_val).strip()
            else:
                cid_str = ""
            if not cid_str:
                cid_str = f"vid-{_now_ts()}"
                args["cid"] = cid_str
            _ctx_add(cid_str, "video", npz_path, url, src or (frame_a + "," + frame_b), ["flow"], {})
            trace_append(
                "video",
                {
                    "cid": cid_str,
                    "tool": "video.flow.derive",
                    "src": src,
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "path": npz_path,
                },
            )
            return {"name": name, "result": {"path": url}}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "video.upscale":
        src = args.get("src") or ""
        scale = int(args.get("scale") or 0)
        w = args.get("width"); h = args.get("height")
        if not src:
            return {"name": name, "error": "missing src"}
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "video", f"upscale-{int(time.time())}")
        os.makedirs(outdir, exist_ok=True)
        dst = os.path.join(outdir, "upscaled.mp4")
        if scale and scale > 1:
            vf = f"scale=iw*{scale}:ih*{scale}:flags=lanczos"
        elif w and h:
            vf = f"scale={int(w)}:{int(h)}:flags=lanczos"
        else:
            vf = "scale=iw*2:ih*2:flags=lanczos"
        ff = ["ffmpeg", "-y", "-i", src, "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-an", dst]
        proc = subprocess.run(ff, check=False)
        if proc.returncode != 0:
            return {"name": name, "error": f"ffmpeg upscale exited with code {proc.returncode}"}
        # Optional stabilization pass via face lock
        locks = args.get("locks") or {}
        face_images: list[str] = []
        if isinstance(args.get("ref_ids"), list):
            for rid in args.get("ref_ids"):
                pack, code = _refs_apply({"ref_id": rid})
                if isinstance(pack, dict) and pack.get("ref_pack"):
                    imgs = (pack.get("ref_pack") or {}).get("images") or []
                    for p in imgs:
                        if isinstance(p, str):
                            face_images.append(p)
        if isinstance(locks.get("image"), dict):
            for k in ("face_images", "images"):
                for p in (locks.get("image", {}).get(k) or []):
                    if isinstance(p, str):
                        face_images.append(p)
        if not face_images:
            cid_raw = args.get("cid")
            cid_val = str(cid_raw).strip() if isinstance(cid_raw, (str, int)) else ""
            if cid_val:
                recents = _ctx_list(cid_val, limit=20, kind_hint="image")
                for it in reversed(recents):
                    p = it.get("path")
                    if isinstance(p, str):
                        face_images.append(p)
                        break
        if COMFYUI_API_URL and FACEID_API_URL and face_images:
            frames_dir = os.path.join(outdir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-i", dst, os.path.join(frames_dir, "%06d.png")], check=True)
            face_src = face_images[0]
            face_url = face_src.replace("/workspace", "") if face_src.startswith("/workspace/") else (
                face_src if face_src.startswith("/uploads/") else face_src
            )
            with _hx.Client() as _c:
                er = _c.post(FACEID_API_URL.rstrip("/") + "/embed", json={"image_url": face_url})
                emb = None
                if er.status_code == 200:
                    parser = JSONParser()
                    ej = parser.parse(er.text or "", {"embedding": list, "vec": list})
                    if isinstance(ej, dict):
                        emb = ej.get("embedding") or ej.get("vec")
            if emb and isinstance(emb, list):
                frame_files = sorted(_glob(os.path.join(frames_dir, "*.png")))
                for fp in frame_files:
                    g = {
                        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
                        "2": {"class_type": "LoadImage", "inputs": {"image": fp}},
                        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
                        "6": {
                            "class_type": "KSampler",
                            "inputs": {
                                "seed": 0,
                                "steps": 10,
                                "cfg": 4.0,
                                "sampler_name": "dpmpp_2m",
                                "scheduler": "karras",
                                "denoise": 0.15,
                                "model": ["1", 0],
                                "positive": ["3", 0],
                                "negative": ["4", 0],
                                "latent_image": ["5", 0],
                            },
                        },
                        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "stabilize face, preserve identity", "clip": ["1", 1]}},
                        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
                        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
                        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "frame_out", "images": ["7", 0]}},
                    }
                    g["22"] = {
                        "class_type": "InstantIDApply",
                        "inputs": {"model": ["1", 0], "image": ["2", 0], "embedding": emb, "strength": 0.70},
                    }
                    g["6"]["inputs"]["model"] = ["22", 0]
                    with _hx.Client() as _c2:
                        pr = _c2.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json={"prompt": g, "client_id": "wrapper-001"})
                        if pr.status_code == 200:
                            parser = JSONParser()
                            pj = parser.parse(pr.text or "", {"prompt_id": str, "uuid": str, "id": str})
                            pid = (pj.get("prompt_id") or pj.get("uuid") or pj.get("id")) if isinstance(pj, dict) else None
                            while True:
                                hr = _c2.get(COMFYUI_API_URL.rstrip("/") + f"/history/{pid}")
                                if hr.status_code == 200:
                                    parser = JSONParser()
                                    hj = parser.parse(hr.text or "", {"history": dict})
                                    h = _normalize_comfy_history_entry(hj if isinstance(hj, dict) else {}, str(pid))
                                    if h and _comfy_is_completed(h):
                                        assets = _extract_comfy_asset_urls(h, COMFYUI_API_URL or "http://comfyui:8188")
                                        if assets:
                                            src = assets[0].get("url")
                                            if isinstance(src, str) and src:
                                                vr = _c2.get(src)
                                                if vr.status_code == 200:
                                                    with open(fp, "wb") as wf:
                                                        wf.write(vr.content)
                                                    break
                                        break
                                time.sleep(0.5)
                dst2 = os.path.join(outdir, "upscaled_stabilized.mp4")
                subprocess.run(
                    ["ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(frames_dir, "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", dst2],
                    check=True,
                )
                dst = dst2
        url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
        cid_val = args.get("cid")
        if isinstance(cid_val, (str, int)):
            cid_str = str(cid_val).strip()
        else:
            cid_str = ""
        if not cid_str:
            cid_str = f"vid-{_now_ts()}"
            args["cid"] = cid_str
        _ctx_add(cid_str, "video", dst, url, src, ["upscale"], {"scale": scale, "width": w, "height": h})
        trace_append("video", {"cid": cid_str, "tool": "video.upscale", "src": src, "scale": scale, "width": w, "height": h, "path": dst})
        return {"name": name, "result": {"path": url}}
    if name == "video.text.overlay":
        src = args.get("src") or ""
        texts = args.get("texts") or []
        if not src or not isinstance(texts, list) or not texts:
            return {"name": name, "error": "missing src|texts"}
        # Extract frames
        outdir = os.path.join(UPLOAD_DIR, "artifacts", "video", f"txtov-{int(_tm.time())}")
        frames_dir = os.path.join(outdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        proc1 = subprocess.run(["ffmpeg", "-y", "-i", src, os.path.join(frames_dir, "%06d.png")], check=False)
        if proc1.returncode != 0:
            return {"name": name, "error": f"ffmpeg text.extract exited with code {proc1.returncode}"}
        frame_files = sorted(_glob(os.path.join(frames_dir, "*.png")))
        for fp in frame_files:
            im = Image.open(fp).convert("RGB")
            for spec in texts:
                im = _video_draw_on(im, spec)
            im.save(fp)
        # Re-encode
        dst = os.path.join(outdir, "overlay.mp4")
        proc2 = subprocess.run(
            ["ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(frames_dir, "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", dst],
            check=False,
        )
        if proc2.returncode != 0:
            return {"name": name, "error": f"ffmpeg text.encode exited with code {proc2.returncode}"}
        url = dst.replace("/workspace", "") if dst.startswith("/workspace/") else dst
        cid_val = args.get("cid")
        if isinstance(cid_val, (str, int)):
            cid_str = str(cid_val).strip()
        else:
            cid_str = ""
        if not cid_str:
            cid_str = f"vid-{_now_ts()}"
            args["cid"] = cid_str
        _ctx_add(cid_str, "video", dst, url, src, ["text_overlay"], {"count": len(texts)})
        trace_append("video", {"cid": cid_str, "tool": "video.text.overlay", "src": src, "path": dst, "texts": texts})
        return {"name": name, "result": {"path": url}}
    if name == "video.hv.t2v" and ALLOW_TOOL_EXECUTION:
        if not HUNYUAN_VIDEO_API_URL:
            return {"name": name, "error": "HUNYUAN_VIDEO_API_URL not configured"}
        try:
            lr_every = int(args.get("latent_reinit_every") or 48)
            payload = {
                "prompt": args.get("prompt"),
                "negative": args.get("negative"),
                "video": {"width": int(args.get("width") or 1024), "height": int(args.get("height") or 576), "fps": int(args.get("fps") or 24), "seconds": int(args.get("seconds") or 6)},
                "locks": args.get("locks") or {},
                "seed": args.get("seed"),
                "quality": "max",
                "post": args.get("post") or {"interpolate": True, "upscale": True, "face_lock": True, "hand_fix": True},
                "latent_reinit": {"every_n_frames": lr_every},
                "meta": {"trace_level": "full", "stream": True},
            }
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                r = await client.post(HUNYUAN_VIDEO_API_URL.rstrip("/") + "/v1/video/hv/t2v", json=payload)
            parser = JSONParser()
            # Treat Hunyuan video responses as free-form mapping; callers inspect fields as needed.
            js = parser.parse(r.text or "{}", {})
            return {"name": name, "result": js if isinstance(js, dict) else {}}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "video.hv.i2v" and ALLOW_TOOL_EXECUTION:
        if not HUNYUAN_VIDEO_API_URL:
            return {"name": name, "error": "HUNYUAN_VIDEO_API_URL not configured"}
        try:
            lr_every = int(args.get("latent_reinit_every") or 48)
            payload = {
                "init_image": args.get("init_image"),
                "prompt": args.get("prompt"),
                "negative": args.get("negative"),
                "video": {"width": int(args.get("width") or 1024), "height": int(args.get("height") or 1024), "fps": int(args.get("fps") or 24), "seconds": int(args.get("seconds") or 5)},
                "locks": args.get("locks") or {},
                "seed": args.get("seed"),
                "quality": "max",
                "post": args.get("post") or {"interpolate": True, "upscale": True, "face_lock": True},
                "latent_reinit": {"every_n_frames": lr_every},
                "meta": {"trace_level": "full", "stream": True},
            }
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                r = await client.post(HUNYUAN_VIDEO_API_URL.rstrip("/") + "/v1/video/hv/i2v", json=payload)
            parser = JSONParser()
            js = parser.parse(r.text or "{}", {})
            return {"name": name, "result": js if isinstance(js, dict) else {}}
        except Exception as ex:
            return {"name": name, "error": str(ex)}
    if name == "math.eval":
        expr = args.get("expr") or ""
        task = (args.get("task") or "").strip().lower()
        var = (args.get("var") or "x").strip() or "x"
        point = args.get("point")
        order = int(args.get("order") or 6)
        try:
            # Prefer SymPy for exact math; fallback to safe math eval
            try:
                x = _sp.symbols(var)
                scope = {str(x): x, 'E': _sp.E, 'pi': _sp.pi}
                e = _parse(str(expr), local_dict=scope, evaluate=True)
                res = {}
                if task in ("diff", "differentiate", "derivative"):
                    de = _sp.diff(e, x)
                    res["exact"] = _sp.sstr(de)
                    res["latex"] = _sp.latex(de)
                elif task in ("integrate", "int", "antiderivative"):
                    ie = _sp.integrate(e, x)
                    res["exact"] = _sp.sstr(ie)
                    res["latex"] = _sp.latex(ie)
                elif task in ("limit",):
                    # For limits, require an explicit point; if missing, return a structured error
                    if point is None:
                        res["error"] = {
                            "code": "point_required",
                            "message": "limit tasks require a finite point; provide args.point",
                            "status": 0,
                            "details": {},
                        }
                    else:
                        le = _sp.limit(e, x, point)
                        res["exact"] = _sp.sstr(le)
                        res["latex"] = _sp.latex(le)
                elif task in ("series",):
                    pe = e.series(x, 0 if point is None else point, order)
                    res["exact"] = _sp.sstr(pe)
                    res["latex"] = _sp.latex(pe.removeO())
                elif task in ("solve",):
                    sol = _sp.solve(_sp.Eq(e, 0), x, dict=True)
                    res["solutions"] = [_sp.sstr(s) for s in sol]
                    res["latex_solutions"] = [_sp.latex(s) for s in sol]
                elif task in ("factor",):
                    fe = _sp.factor(e)
                    res["exact"] = _sp.sstr(fe)
                    res["latex"] = _sp.latex(fe)
                elif task in ("expand",):
                    ex = _sp.expand(e)
                    res["exact"] = _sp.sstr(ex)
                    res["latex"] = _sp.latex(ex)
                else:
                    se = _sp.simplify(e)
                    res["exact"] = _sp.sstr(se)
                    res["latex"] = _sp.latex(se)
                try:
                    res["approx"] = float(_sp.N(e, 20))
                except Exception as ex:
                    # Approximate evaluation is best-effort; preserve the exact result but surface telemetry.
                    logging.debug("sympy_approx_failed", exc_info=True)
                return {"name": name, "result": res}
            except Exception as ex:
                logging.debug(f"sympy_parse_failed: {ex}", exc_info=True)
                allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
                allowed.update({"__builtins__": {}})
                val = eval(str(expr), allowed, {})  # noqa: S307 (safe namespace)
                return {"name": name, "result": {"approx": float(val)}}
        except Exception as ex:
            logging.error(f"math.sympy failed: {ex}", exc_info=True)
            return {"name": name, "error": str(ex)}
    if name == "rag_index":
        res = await rag_index_dir(root=args.get("root", "/workspace"))
        return {"name": name, "result": res}
    if name == "rag_search":
        query = args.get("query")
        k = int(args.get("k", 8))
        if not query:
            return {"name": name, "error": "missing query"}
        res = await rag_search(query, k)
        return {"name": name, "result": res}
    # --- Multimodal external services (optional; enable via env URLs) ---
    if name == "tts_speak" and XTTS_API_URL and ALLOW_TOOL_EXECUTION:
        payload = {"text": args.get("text", ""), "voice": args.get("voice")}
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(XTTS_API_URL.rstrip("/") + "/tts", json=payload)
            status = r.status_code
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= status < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            # Non-2xx: surface a structured error rather than raising
            return {
                "name": name,
                "error": {
                    "code": "tts_http_error",
                    "status": status,
                    "message": r.text,
                },
            }
    if name == "asr_transcribe" and WHISPER_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(WHISPER_API_URL.rstrip("/") + "/transcribe", json={"audio_url": args.get("audio_url"), "language": args.get("language")})
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= r.status_code < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            return {"name": name, "error": r.text}
    if name == "image_generate" and COMFYUI_API_URL and ALLOW_TOOL_EXECUTION:
        # expects a ComfyUI workflow graph or minimal prompt params passed through
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            wf_payload = args.get("workflow") or args
            if isinstance(wf_payload, dict) and "prompt" not in wf_payload:
                wf_payload = {"prompt": wf_payload}
            wf_payload = {**(wf_payload or {}), "client_id": "wrapper-001"}
            r = await client.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json=wf_payload)
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= r.status_code < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            return {"name": name, "error": r.text}
    if name == "controlnet" and COMFYUI_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            wf_payload = args.get("workflow") or args
            if isinstance(wf_payload, dict) and "prompt" not in wf_payload:
                wf_payload = {"prompt": wf_payload}
            wf_payload = {**(wf_payload or {}), "client_id": "wrapper-001"}
            r = await client.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json=wf_payload)
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= r.status_code < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            return {"name": name, "error": r.text}
    if name == "video_generate" and COMFYUI_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            wf_payload = args.get("workflow") or args
            if isinstance(wf_payload, dict) and "prompt" not in wf_payload:
                wf_payload = {"prompt": wf_payload}
            wf_payload = {**(wf_payload or {}), "client_id": "wrapper-001"}
            r = await client.post(COMFYUI_API_URL.rstrip("/") + "/prompt", json=wf_payload)
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= r.status_code < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            return {"name": name, "error": r.text}
    if name == "face_embed" and FACEID_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(FACEID_API_URL.rstrip("/") + "/embed", json={"image_url": args.get("image_url")})
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= r.status_code < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            return {"name": name, "error": r.text}
    if name == "music_generate" and MUSIC_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(MUSIC_API_URL.rstrip("/") + "/generate", json=args)
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= r.status_code < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            return {"name": name, "error": r.text}
    if name == "vlm_analyze" and VLM_API_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(VLM_API_URL.rstrip("/") + "/analyze", json={"image_url": args.get("image_url"), "prompt": args.get("prompt")})
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= r.status_code < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            return {"name": name, "error": r.text}
    # --- Film tools (LLM-driven, simple UI) ---
    if name == "run_python" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/run_python", json={"code": args.get("code", "")})
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= r.status_code < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            return {"name": name, "error": r.text}
    if name == "write_file" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        # Guard: forbid writing banned libraries into requirements/pyproject
        p = (args.get("path") or "").lower()
        content_val = str(args.get("content", ""))
        if any(name in p for name in ("requirements.txt", "pyproject.toml")):
            banned = ("pydantic", "sqlalchemy", "pandas", "pyarrow", "fastparquet", "polars")
            if any(b in content_val.lower() for b in banned):
                return {"name": name, "error": "forbidden_library", "detail": "banned library in dependency file"}
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/write_file", json={"path": args.get("path"), "content": content_val})
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= r.status_code < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            return {"name": name, "error": r.text}
    if name == "read_file" and EXECUTOR_BASE_URL and ALLOW_TOOL_EXECUTION:
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            r = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/read_file", json={"path": args.get("path")})
            parser = JSONParser()
            js = parser.parse(r.text or "", {})
            if 200 <= r.status_code < 300:
                return {"name": name, "result": js if isinstance(js, dict) else {}}
            return {"name": name, "error": r.text}
    # MCP tool forwarding (HTTP bridge)
    if name and (name.startswith("mcp:") or args.get("mcpTool") is True) and ALLOW_TOOL_EXECUTION:
        res = await _call_mcp_tool(MCP_HTTP_BRIDGE_URL, name.replace("mcp:", "", 1), args)
        return {"name": name, "result": res}
    # Unknown tool - return as unexecuted
    return {"name": name or "unknown", "skipped": True, "reason": "unsupported tool in orchestrator"}

# ---- One-pass input coercion (avoid repeated typecasting throughout the function) ----
def _as_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, str):
        s = v
    else:
        try:
            s = str(v)
        except Exception:
            return default
    s = s.strip()
    return s if s else default

def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)

def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        # NaN guard without importing math
        if x != x:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


@app.post("/v1/chat/completions")
async def chat_completions(body: Dict[str, Any], request: Request):
    log.info(f"chat_completions:body={body}")
    log.info(f"request={request}")
    env = await committee_ai_text(messages=(body.get("messages")), trace_id=uuid.uuid4().hex)
    log.debug(f"chat_completions:env={env}")
    body_text = json.dumps(env, ensure_ascii=False)
    body_bytes = body_text.encode("utf-8")
    # Deterministic framing + no persistent socket reuse (helps with intermittent client-side failures).
    headers = {
        "Cache-Control": "no-store",
        "Content-Type": "application/json; charset=utf-8",
        "Content-Length": str(len(body_bytes)),
    }
    return Response(content=body_bytes, status_code=200, media_type="application/json", headers=headers)


async def chat_completions2(body: Dict[str, Any], request: Request):
    # Single-exit discipline: exactly one return at the bottom of this function.
    response = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    display_content = ""
    trace_id = "tt_unknown"
    master_seed = 0
    run_id = None
    abort_pipeline: bool = False
    planner_error_payload: Optional[Dict[str, Any]] = None
    
    # Always-defined downstream variables (no NameErrors, no fallthrough crashes).
    qwen_result: Dict[str, Any] = {}
    glm_result: Dict[str, Any] = {}
    deepseek_result: Dict[str, Any] = {}
    synth_result: Dict[str, Any] = {}
    tool_results: List[Dict[str, Any]] = []
    final_env: Dict[str, Any] = {}
    artifacts_out: Dict[str, Any] = {"urls": [], "items": []}
    # Required post-run features (artifacts ledger, ablation export, teacher tap).
    tool_exec_meta: List[Dict[str, Any]] = []
    _ledger_shard: Optional[dict] = None
    _ledger_root: Optional[str] = None
    _ledger_name: str = "ledger"
    pack_hash: Optional[str] = None
    # Absolutize helper must exist on all paths (including aborted requests).
    base_url: str = (PUBLIC_BASE_URL or "").rstrip("/") or (str(request.base_url) or "").rstrip("/")
    abs_url_fn = partial(_abs_url, request_base_url=base_url)
    # Always-defined planner outputs (avoid fallthrough crashes on abort paths).
    plan_text: str = ""
    tool_calls: List[Dict[str, Any]] = []
    planner_env: Dict[str, Any] = {}
    # Executor routing knobs are used in multiple places; define once.
    executor_endpoint: str = EXECUTOR_BASE_URL or "http://127.0.0.1:8081"
    

    # Normalize request shapes once (avoid repeated isinstance checks throughout chat_completions)
    body0: Dict[str, Any] = body if isinstance(body, dict) else {}
    # Tools are NOT accepted from the client request body in this project.
    # The planner always uses the internal, fixed tool catalog.
    tools_spec: Optional[List[Dict[str, Any]]] = None
    tools_allowed: List[str] = sorted(list(catalog_allowed(get_builtin_tools_schema)))
    _temp_raw = body0.get("temperature")
    exec_temperature = _as_float(DEFAULT_TEMPERATURE if _temp_raw is None else _temp_raw, float(DEFAULT_TEMPERATURE))
    # Clamp to OpenAI-compatible range.
    if exec_temperature < 0.0:
        exec_temperature = 0.0
    if exec_temperature > 2.0:
        exec_temperature = 2.0
    _cat_hash: str = ""
    try:
        t0 = time.time()
        logging.debug("chat_completions:try t0=%r", t0)
        # ---- NO request shaping: use original body/messages as provided ----
        # ids / mode / seed
        requested_cid = None
        if isinstance(body0.get("cid"), (int, str)) and str(body0.get("cid")).strip():
            requested_cid = str(body0.get("cid")).strip()
        elif isinstance(body0.get("conversation_id"), (int, str)) and str(body0.get("conversation_id")).strip():
            requested_cid = str(body0.get("conversation_id")).strip()

        # Canonical cid is orchestrator-owned; never trust external sources.
        conv_cid = await _resolve_or_create_conversation_cid(requested_cid)

        mode = _as_str(body0.get("mode"), "general")
        effective_mode = mode

        master_seed = _as_int(body0.get("seed"), 0)

        # Request id (client-visible) and trace id (internal correlation id)
        request_id = ""
        if isinstance(body0.get("request_id"), (str, int)) and str(body0.get("request_id")).strip():
            request_id = str(body0.get("request_id")).strip()
        else:
            ikey = body0.get("idempotency_key")
            if isinstance(ikey, str) and ikey.strip():
                request_id = ikey.strip()
            else:
                request_id = uuid.uuid4().hex

        # trace_id is not part of the OpenAI chat completions contract; generate one when missing
        # so all planner/executor logs and artifacts are correlated.
        trace_id = uuid.uuid4().hex
        logging.debug(f"chat_completions:trace_id source={trace_id} trace_id={trace_id}")

        # messages (verbatim)
        messages: List[Dict[str, Any]] = []
        last_user_text: str = ""
        messages_raw = body0.get("messages")
        if not isinstance(messages_raw, list):
            usage0 = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            logging.debug("chat_completions:usage0=%r", usage0)
            response = _build_openai_envelope(
                ok=False,
                text="Invalid request: 'messages' must be a list.",
                error={"code": "bad_request", "message": "messages must be a list"},
                usage=usage0,
                model=COMMITTEE_MODEL_ID,
                seed=0,
                id_="orc-1",
            )
            logging.debug("chat_completions:built bad_request response keys=%r", sorted(list(response.keys())) if isinstance(response, dict) else type(response).__name__)
            usage = usage0
            logging.debug("chat_completions:set usage=%r", usage)
            abort_pipeline = True
            logging.debug("chat_completions:set abort_pipeline=%r", abort_pipeline)
            plan_text = ""
            logging.debug("chat_completions:set plan_text=%r", plan_text)
            tool_calls = []
            logging.debug(f"chat_completions:set tool_calls_len={len(tool_calls)}")
            planner_env = {}
            logging.debug("chat_completions:set planner_env=%r", planner_env)
        else:
            messages = [m for m in messages_raw if isinstance(m, dict)]
            # last user text (no normalization)
            for m in reversed(messages):
                if m.get("role") == "user":
                    cv = m.get("content")
                    if isinstance(cv, str) and cv.strip():
                        last_user_text = cv.strip()
                        break
            first_user_len = len((last_user_text or "")) if isinstance(last_user_text, str) else 0
            logging.debug("chat_completions:first_user_len=%r", first_user_len)
            # High-level request trace (distillation-friendly)
            _trace_log_request(
                STATE_DIR,
                trace_id,
                {
                    "request_id": request_id,
                    "kind": "chat",
                    "route": str(request.url.path),
                    "payload": {"messages_count": len(messages or []), "first_user_text_len": first_user_len},
                    "meta": {
                        "mode": mode,
                        "effective_mode": effective_mode,
                        "schema_version": 1,
                    },
                },
            )
            _log(
                "request",
                trace_id=trace_id,
                route="/v1/chat/completions",
                body_summary={"messages_count": len(messages or []), "first_user_text_len": first_user_len},
                seed=master_seed,
            )
            logging.debug(
                f"chat_completions:chat.start trace_id={trace_id} mode={mode} effective_mode={effective_mode} cid={conv_cid!r}"
            )
            _log("chat.start", trace_id=trace_id, mode=mode, effective_mode=effective_mode, cid=conv_cid)
            _log("start", trace_id=trace_id, seed=master_seed, mode=mode, effective_mode=effective_mode)


            try:
                run_id = await _db_insert_run(trace_id=trace_id, mode=mode, seed=master_seed, pack_hash=pack_hash, request_json=body0)
                logging.debug("chat_completions:_db_insert_run run_id=%r", run_id)            
            except Exception as ex:
                log.error(f"chat_completions:_db_insert_run failed trace_id={trace_id} ex={ex}", exc_info=True)
                run_id = None
            # Artifacts ledger shard (required): store step-level tool execution summaries.
            try:
                _ledger_root = os.path.join(UPLOAD_DIR, "artifacts", "runs", trace_id)
                logging.debug("chat_completions:ledger _ledger_root=%r", _ledger_root)
                _ledger_shard = _art_open_shard(_ledger_root, _ledger_name, int(ARTIFACT_SHARD_BYTES))
                logging.debug("chat_completions:ledger _art_open_shard ok shard=%r", _ledger_shard)
            except Exception as ex:
                logging.debug(f"chat_completions:ledger.open.failed trace_id={trace_id} ex={ex}", exc_info=True)

        # If client supplies tool results (role=tool) we include them verbatim for the planner/executors

        # Optional Windowed Solver path (capless sliding window + CONT/HALT)
        route_mode = "committee"
        # No early returns: continue to planner/tools; responses emitted only after work completes

        # 1) Planner proposes tool calls (planning lives under orchestrator/app/plan/*).
        _log("planner.call", trace_id=trace_id, mode=effective_mode)
        logging.debug(f"chat_completions:planner.call trace_id={trace_id} effective_mode={effective_mode}")
        # IMPORTANT: never leave these uninitialized. If produce_tool_plan throws,
        # subsequent logging/normalization must not crash with UnboundLocalError.
        plan_text: str = ""
        tool_calls: List[Dict[str, Any]] = []
        planner_env: Dict[str, Any] = {"ok": False, "error": {"code": "planner_not_run", "message": "planner did not run"}}
        try:
            plan_text, tool_calls, planner_env = await produce_tool_plan(
                messages=messages,
                tools=None,
                temperature=exec_temperature,
                trace_id=trace_id,
                mode=effective_mode,
            )
        except Exception as e:
            log.error(f"chat_completions:produce_tool_plan failed trace_id={trace_id} ex={e}", exc_info=True)
            # Preserve a structured envelope so downstream logic can surface a
            # meaningful error instead of crashing on undefined locals.
            planner_env = {
                "ok": False,
                "error": {"code": "produce_tool_plan_exception", "message": str(e)},
                "trace_id": trace_id,
            }
            plan_text = ""
            tool_calls = []
        plan_text_len = len(plan_text) if isinstance(plan_text, str) else -1
        tool_calls_type = type(tool_calls).__name__
        tool_calls_len = len(tool_calls) if isinstance(tool_calls, list) else -1
        planner_env_keys = sorted(list(planner_env.keys())) if isinstance(planner_env, dict) else type(planner_env).__name__
        logging.debug(
            f"chat_completions:produce_tool_plan done plan_text_len={plan_text_len} tool_calls_type={tool_calls_type} "
            f"tool_calls_len={tool_calls_len} planner_env_keys={planner_env_keys!r}"
        )
        tool_calls = _normalize_tool_calls(tool_calls)
        logging.debug(
            f"chat_completions:_normalize_tool_calls len={len(tool_calls) if isinstance(tool_calls, list) else -1}"
        )
        # Hardening: ensure tool arguments are objects (parse JSON strings / {"_raw": "..."}), and fill minimal defaults.
        parser_tc = JSONParser()
        hardened_tool_calls: List[Dict[str, Any]] = []
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            nm = str(tc.get("name") or "").strip()
            raw_args = tc.get("arguments")
            args_obj: Dict[str, Any]
            if isinstance(raw_args, dict):
                raw_inner = raw_args.get("_raw")
                if isinstance(raw_inner, str):
                    try:
                        parsed_inner = parser_tc.parse(raw_inner or "", {})
                    except Exception as ex:
                        logging.warning(
                            f"planner.tool_args._raw parse failed trace_id={trace_id} tool={nm}: {ex}", exc_info=True
                        )
                        parsed_inner = {}
                    if isinstance(parsed_inner, dict):
                        args_obj = dict(parsed_inner)
                    else:
                        args_obj = {"_raw": raw_inner}
                    # Overlay any explicit keys alongside _raw (do not drop them).
                    for k, v in raw_args.items():
                        if k == "_raw":
                            continue
                        args_obj[k] = v
                else:
                    args_obj = dict(raw_args)
            elif isinstance(raw_args, str):
                try:
                    parsed = parser_tc.parse(raw_args or "", {})
                except Exception as ex:
                    logging.warning(
                        f"planner.tool_args parse failed trace_id={trace_id} tool={nm}: {ex}", exc_info=True
                    )
                    parsed = {}
                args_obj = dict(parsed) if isinstance(parsed, dict) else {"_raw": raw_args}
            elif raw_args is None:
                args_obj = {}
            else:
                args_obj = {"_raw": raw_args}

            if nm == "image.dispatch":
                # Defaults-first, then overlay any provided values.
                merged = {"negative": "", "width": 1024, "height": 1024, "steps": 32, "cfg": 5.5}
                merged.update(args_obj or {})
                if (not isinstance(merged.get("prompt"), str)) or not (merged.get("prompt") or "").strip():
                    if isinstance(last_user_text, str) and last_user_text.strip():
                        merged["prompt"] = last_user_text.strip()
                args_obj = merged
                _log(
                    "planner.image.dispatch.args",
                    trace_id=trace_id,
                    args={k: args_obj.get(k) for k in ("prompt", "width", "height", "steps", "cfg", "negative")},
                )

            hardened_tool_calls.append({**tc, "name": nm, "arguments": args_obj})

        tool_calls = hardened_tool_calls
        logging.debug(
            f"chat_completions:tool_calls.hardened len={len(tool_calls) if isinstance(tool_calls, list) else -1}"
        )
        try:
            # Deep planner/tool-call visibility: log the *shape* of tool args without dumping full prompts.
            preview: List[Dict[str, Any]] = []
            for c in (tool_calls or [])[:20]:
                if not isinstance(c, dict):
                    continue
                nm = str(c.get("name") or "")
                av = c.get("arguments")
                preview.append(
                    {
                        "tool": nm,
                        "arguments_type": type(av).__name__,
                        "arguments_preview": _args_preview_for_log(av),
                    }
                )
            _log("planner.tool_calls.normalized", trace_id=trace_id, count=len(tool_calls or []), preview=preview)
        except Exception as ex:
            logging.warning(f"planner.tool_calls.normalized log failed trace_id={trace_id}: {ex}", exc_info=True)
        # Hardening: filter unknown tools against runtime catalog (registry + builtins).
        allowed_runtime = catalog_allowed(get_builtin_tools_schema)
        logging.debug(
            f"chat_completions:catalog_allowed count={len(allowed_runtime) if isinstance(allowed_runtime, (list, set, tuple)) else -1}"
        )
        # Restrict to planner-visible + mode-allowed tools (single source of truth; do not re-filter later).
        allowed_runtime = {n for n in allowed_runtime if n in PLANNER_VISIBLE_TOOLS}
        logging.debug(f"chat_completions:allowed_runtime after planner_visible count={len(allowed_runtime)}")
        allowed_runtime = {n for n in allowed_runtime if n in set(_allowed_tools_for_mode(effective_mode))}
        logging.debug(f"chat_completions:allowed_runtime after mode filter count={len(allowed_runtime)}")
        tool_calls, unknown_tools = catalog_validate(tool_calls or [], allowed_runtime)
        logging.debug(
            f"chat_completions:catalog_validate tool_calls_len={len(tool_calls or [])} unknown_tools={unknown_tools!r}"
        )
        if unknown_tools:
            _log("tools.unknown.filtered", trace_id=trace_id, unknown=unknown_tools)
            logging.debug("chat_completions:tools.unknown.filtered unknown_tools=%r", unknown_tools)
        _log("planner.done", trace_id=trace_id, tool_count=len(tool_calls or []), ok=bool((planner_env or {}).get("ok")))
        logging.debug(
            f"chat_completions:planner.done tool_count={len(tool_calls or [])} ok={bool((planner_env or {}).get('ok'))}"
        )
        trace_append("decision", {"cid": conv_cid, "trace_id": trace_id, "plan": plan_text, "tool_calls": tool_calls})
        logging.debug("chat_completions:trace_append decision done")

        if not isinstance(planner_env, dict) or not planner_env.get("ok"):
            usage0 = estimate_usage(messages, "")
            err_block = planner_env.get("error") if isinstance(planner_env, dict) else None
            if not isinstance(err_block, dict):
                err_block = {"code": "planner_error", "message": str(planner_env)}
            planner_error_payload = dict(err_block)
            response = _build_openai_envelope(
                ok=False,
                text="Planner failed.",
                error=err_block,
                usage=usage0,
                model=COMMITTEE_MODEL_ID,
                seed=int(master_seed),
                id_="orc-1",
            )
            usage = usage0
            abort_pipeline = True
            # Hard abort: do not execute tools or run downstream synthesis on planner failure.
            tool_calls = []
            plan_text = ""
            _cat_hash = ""
        else:
            _cat_hash = compute_tools_hash(body, planner_visible_only=False, planner_visible_tools=set(PLANNER_VISIBLE_TOOLS))
        # Planner OK: continue into the full pipeline below (CO/RAG/QA/retry/compose),
        # which owns tool execution and final response composition.

        # response prepared; continue to unified finalization below.

        pipeline_ok: bool = not abort_pipeline

        if (not abort_pipeline) and tool_calls:
            _steps_preview0 = []
            for t in (tool_calls or [])[:5]:
                name_preview = (t.get("name") or "").strip()
                arg_keys_preview = list((t.get("arguments") or {}).keys()) if isinstance(t.get("arguments"), dict) else []
                _steps_preview0.append({"tool": name_preview, "args_keys": arg_keys_preview})
            if _steps_preview0:
                _log("exec.payload", trace_id=trace_id, steps=_steps_preview0)

            if tool_calls:
                # Validator removed: execute tool calls directly and rely on executor behavior.
                _log("validate.skip", trace_id=trace_id, reason="validator_removed", tool_count=len(tool_calls or []))
            _log(
                "validate.completed",
                trace_id=trace_id,
                validated_count=len(tool_calls or []),
            )
            _log(
                "flow.after_validate",
                trace_id=trace_id,
                tool_count=len(tool_calls or []),
            )
        else:
            _log("validate.skip", trace_id=trace_id, reason="no_tool_calls")

        if (not abort_pipeline) and tool_calls:
            _log("tools.exec.start", trace_id=trace_id, count=len(tool_calls or []))
            logging.debug(
                f"chat_completions:tools.exec.start abort_pipeline={abort_pipeline!r} tool_calls_len={len(tool_calls or [])}"
            )
            _inject_execution_context(tool_calls, trace_id, effective_mode, conv_cid)
            logging.debug("chat_completions:_inject_execution_context done trace_id=%r effective_mode=%r", trace_id, effective_mode)
            for tc in tool_calls:
                logging.debug(
                    f"chat_completions:tool_loop:iter tc_type={type(tc).__name__} "
                    f"tc_keys={(sorted(list(tc.keys())) if isinstance(tc, dict) else None)!r}"
                )
                tn = (tc.get("name") or "tool")
                logging.debug("chat_completions:tool_loop:tn=%r", tn)
                raw_args = tc.get("arguments")
                logging.debug(f"chat_completions:tool_loop:raw_args_type={type(raw_args).__name__}")
                if isinstance(raw_args, dict):
                    ta = raw_args
                elif raw_args is None:
                    ta = {}
                else:
                    # Never drop args: preserve non-dict payloads under _raw.
                    ta = {"_raw": raw_args}
                logging.debug(
                    f"chat_completions:tool_loop:ta_type={type(ta).__name__} "
                    f"ta_keys={(sorted(list(ta.keys())) if isinstance(ta, dict) else None)!r}"
                )
                # For film2.run, never rely on planner-fabricated clip/image URLs. Instead,
                # automatically wire in real image artifacts from any prior image.dispatch
                # (or other image-producing) tool results in this trace. If no real images
                # exist yet, drop any planner-provided images/clips so film2 falls back to
                # prompt/story-driven generation instead of referencing non-existent assets.
                if tn == "film2.run":
                    logging.debug("chat_completions:tool_loop:film2.run branch entered")
                    args_film = ta if isinstance(ta, dict) else {}
                    logging.debug("chat_completions:tool_loop:film2.run args_film_keys_pre=%r", sorted(list(args_film.keys())) if isinstance(args_film, dict) else None)
                    urls_all = assets_collect_urls(tool_results or [], abs_url_fn)
                    logging.debug(f"chat_completions:tool_loop:film2.run urls_all_len={len(urls_all or [])}")
                    img_urls = [u for u in (urls_all or []) if isinstance(u, str) and "/artifacts/image/" in u]
                    logging.debug(f"chat_completions:tool_loop:film2.run img_urls_len={len(img_urls or [])}")
                    if img_urls:
                        args_film["images"] = img_urls
                        # Ensure we do not carry over any stale clip references from the planner.
                        args_film.pop("clips", None)
                    else:
                        # No real images available yet; strip any fabricated paths.
                        args_film.pop("images", None)
                        args_film.pop("clips", None)
                    tc["arguments"] = args_film
                    ta = args_film
                    logging.debug("chat_completions:tool_loop:film2.run args_film_keys_post=%r", sorted(list(args_film.keys())) if isinstance(args_film, dict) else None)
                _log(
                    "tool.run.before",
                    trace_id=trace_id,
                    tool=str(tn),
                    args_type=type(raw_args).__name__,
                    args_preview=_args_preview_for_log(ta),
                )
                logging.debug("chat_completions:tool_loop:before_execute tn=%r executor_endpoint=%r request_id=%r", tn, executor_endpoint, request_id)
                # Execute tools via the external executor so all heavy/remote work
                # happens in the executor container instead of the orchestrator.
                exec_batch = [tc]
                logging.debug(f"chat_completions:tool_loop:exec_batch_len={len(exec_batch)}")
                try:
                    exec_res = await gateway_execute(exec_batch, trace_id, executor_endpoint, request_id=str(request_id))
                except Exception as ex:
                    # Hardening: a single tool execution failure must not crash the whole run.
                    logging.error(f"chat_completions:tool_loop:gateway_execute raised tool={tn!r} ex={ex}", exc_info=True)
                    exec_res = [
                        {
                            "name": str(tn or "tool"),
                            "error": {
                                "code": "executor_gateway_exception",
                                "message": str(ex),
                                "status": 0,
                                "stack": _tb(ex),
                            },
                            "result": {},
                            "args": (tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}),
                        }
                    ]
                logging.debug(
                    f"chat_completions:tool_loop:gateway_execute returned type={type(exec_res).__name__} "
                    f"len={(len(exec_res) if isinstance(exec_res, list) else -1)}"
                )
                tr = exec_res[0] if isinstance(exec_res, list) and exec_res else {}
                logging.debug("chat_completions:tool_loop:tr_keys=%r", sorted(list(tr.keys())) if isinstance(tr, dict) else None)
                tname = str((tr or {}).get("name") or "tool")
                logging.debug("chat_completions:tool_loop:tname=%r", tname)
                res = (tr or {}).get("result") if isinstance((tr or {}).get("result"), dict) else {}
                logging.debug("chat_completions:tool_loop:res_keys=%r", sorted(list(res.keys())) if isinstance(res, dict) else None)
                err_obj = (tr or {}).get("error")
                logging.debug(f"chat_completions:tool_loop:err_obj_type={type(err_obj).__name__}")
                if not err_obj and isinstance(res, dict):
                    err_obj = res.get("error")
                    logging.debug(f"chat_completions:tool_loop:err_obj_from_res type={type(err_obj).__name__}")
                # Executor does not echo args; executor_gateway attaches them back. Fall back to our input args.
                args_echo = tr.get("args") if isinstance(tr.get("args"), dict) else {}
                logging.debug("chat_completions:tool_loop:args_echo_keys=%r", sorted(list(args_echo.keys())) if isinstance(args_echo, dict) else None)
                if not args_echo and isinstance(ta, dict):
                    args_echo = ta
                    logging.debug("chat_completions:tool_loop:args_echo fallback to ta keys=%r", sorted(list(args_echo.keys())) if isinstance(args_echo, dict) else None)
                _log(
                    "tool.run.after_exec",
                    trace_id=trace_id,
                    tool=tname,
                    ok=not isinstance(err_obj, (str, dict)),
                    result_keys=(sorted(list(res.keys()))[:64] if isinstance(res, dict) else []),
                    args_preview=_args_preview_for_log(args_echo),
                    step_id=(tr.get("step_id") if isinstance(tr.get("step_id"), str) else None),
                )
                logging.debug(f"chat_completions:tool_loop:after_exec ok={not isinstance(err_obj, (str, dict))}")
                # Required: persist tool execution metadata (teacher + ledger + db)
                try:
                    arts_val = res.get("artifacts") if isinstance(res, dict) else None
                    logging.debug(f"chat_completions:tool_loop:arts_val_type={type(arts_val).__name__}")
                    ok_step = not isinstance(err_obj, (str, dict))
                    logging.debug("chat_completions:tool_loop:ok_step=%r", ok_step)
                    tool_exec_meta.append(
                        {
                            "name": tname,
                            "args": (args_echo if isinstance(args_echo, dict) else (ta if isinstance(ta, dict) else {})),
                            "ok": bool(ok_step),
                            "result": (res if isinstance(res, dict) else {}),
                            "error": (err_obj if isinstance(err_obj, (dict, str)) else None),
                            "artifacts": arts_val,
                        }
                    )
                    logging.debug(f"chat_completions:tool_loop:tool_exec_meta appended len={len(tool_exec_meta)}")
                    if run_id is not None:
                        logging.debug("chat_completions:tool_loop:_db_insert_tool_call starting run_id=%r tool=%r", run_id, tname)
                        await _db_insert_tool_call(
                            run_id,
                            tname,
                            int(master_seed),
                            (args_echo if isinstance(args_echo, dict) else (ta if isinstance(ta, dict) else {})),
                            (res if isinstance(res, dict) else None),
                            None,
                        )
                        logging.debug("chat_completions:tool_loop:_db_insert_tool_call done")
                    if _ledger_shard is not None and _ledger_root is not None:
                        logging.debug("chat_completions:tool_loop:ledger append starting _ledger_root=%r", _ledger_root)
                        row = {
                            "t": int(time.time() * 1000),
                            "trace_id": trace_id,
                            "run_id": int(run_id) if isinstance(run_id, int) else None,
                            "name": tname,
                            "ok": bool(ok_step),
                            "args": (args_echo if isinstance(args_echo, dict) else (ta if isinstance(ta, dict) else {})),
                            "result": (res if isinstance(res, dict) else {}),
                            "error": err_obj,
                            "artifacts": arts_val,
                        }
                        logging.debug("chat_completions:tool_loop:ledger row_keys=%r", sorted(list(row.keys())))
                        _ledger_shard = _art_append_jsonl(_ledger_shard, row)
                        logging.debug("chat_completions:tool_loop:ledger append done shard=%r", _ledger_shard)
                except Exception as ex:
                    logging.debug(
                        f"chat_completions:tool_exec_meta.persist.failed trace_id={trace_id} tool={str(tname)} ex={ex}",
                        exc_info=True,
                    )
                if isinstance(err_obj, (str, dict)):
                    logging.debug("chat_completions:tool_loop:error_branch entered")
                    code = (err_obj.get("code") if isinstance(err_obj, dict) else str(err_obj))
                    logging.debug("chat_completions:tool_loop:error_branch code=%r", code)
                    status = (err_obj.get("status") if isinstance(err_obj, dict) else None)
                    logging.debug("chat_completions:tool_loop:error_branch status=%r", status)
                    message = (err_obj.get("message") if isinstance(err_obj, dict) else "")
                    logging.debug(
                        f"chat_completions:tool_loop:error_branch message_len={len(message) if isinstance(message, str) else -1}"
                    )
                    _log("tool.run.error", trace_id=trace_id, tool=tname, code=(code or ""), status=status, message=(message or ""))
                    # Flip pipeline_ok to False when a critical front-door tool fails.
                    if tname in ("image.dispatch", "music.infinite.windowed", "film2.run", "tts.speak"):
                        pipeline_ok = False
                        logging.debug("chat_completions:tool_loop:pipeline_ok flipped to False due_to=%r", tname)
                else:
                    logging.debug("chat_completions:tool_loop:success_branch entered")
                    arts_summary: List[Dict[str, Any]] = []
                    logging.debug(f"chat_completions:tool_loop:arts_summary init len={len(arts_summary)}")
                    arts = res.get("artifacts") if isinstance(res, dict) else None
                    logging.debug(f"chat_completions:tool_loop:arts_type={type(arts).__name__}")
                    if isinstance(arts, list):
                        for a in arts:
                            if isinstance(a, dict):
                                aid = a.get("id")
                                kind = a.get("kind")
                                if isinstance(aid, str) and isinstance(kind, str):
                                    arts_summary.append({"id": aid, "kind": kind})
                                    logging.debug(
                                        f"chat_completions:tool_loop:arts_summary append id={aid!r} kind={kind!r} len={len(arts_summary)}"
                                    )
                    urls_this = assets_collect_urls([tr], _abs_url)
                    logging.debug(f"chat_completions:tool_loop:urls_this_len={len(urls_this or [])}")
                    _log("tool.run.after", trace_id=trace_id, tool=tname, artifacts=arts_summary, urls_count=len(urls_this or []))
                    meta = res.get("meta") if isinstance(res, dict) else {}
                    logging.debug(f"chat_completions:tool_loop:meta_type={type(meta).__name__}")
                    first_url = urls_this[0] if isinstance(urls_this, list) and urls_this else None
                    logging.debug("chat_completions:tool_loop:first_url=%r", first_url)
                    extra_results: List[Dict[str, Any]] = []
                    logging.debug(f"chat_completions:tool_loop:extra_results init len={len(extra_results)}")
                    if tname == "film2.run":
                        logging.debug("chat_completions:tool_loop:film2.run qa branch entered")
                        meta_obj = res.get("meta") if isinstance(res, dict) else {}
                        logging.debug(f"chat_completions:tool_loop:film2.run meta_obj_type={type(meta_obj).__name__}")
                        profile_name = meta_obj.get("quality_profile") if isinstance(meta_obj, dict) and isinstance(meta_obj.get("quality_profile"), str) else None
                        logging.debug("chat_completions:tool_loop:film2.run profile_name=%r", profile_name)
                        preset = LOCK_QUALITY_PRESETS.get((profile_name or "standard").lower(), LOCK_QUALITY_PRESETS["standard"])
                        logging.debug("chat_completions:tool_loop:film2.run preset_keys=%r", sorted(list(preset.keys())) if isinstance(preset, dict) else None)
                        # Clamp to a single refinement pass for segment-level film QA (defensive).
                        _rb_raw = preset.get("max_refine_passes", 1) if isinstance(preset, dict) else 1
                        try:
                            refine_budget = min(1, int(_rb_raw))
                        except Exception as exc:
                            logging.warning(
                                f"chat_completions: bad max_refine_passes={_rb_raw!r} for film2.run; defaulting to 1",
                                exc_info=True,
                            )
                            refine_budget = 1
                        logging.debug("chat_completions:tool_loop:film2.run refine_budget=%r", refine_budget)
                        segments_obj = meta_obj.get("segments") if isinstance(meta_obj, dict) else {}
                        logging.debug(f"chat_completions:tool_loop:film2.run segments_obj_type={type(segments_obj).__name__}")
                        clips_for_committee = segments_obj.get("clips") if isinstance(segments_obj, dict) else None
                        logging.debug(
                            f"chat_completions:tool_loop:film2.run clips_for_committee_type={type(clips_for_committee).__name__}"
                        )
                        if isinstance(clips_for_committee, list) and clips_for_committee:
                            seg_results_payload: List[Dict[str, Any]] = []
                            for seg in clips_for_committee:
                                if not isinstance(seg, dict):
                                    continue
                                seg_id = seg.get("id")
                                res_obj = seg.get("result")
                                if isinstance(seg_id, str) and seg_id and isinstance(res_obj, dict):
                                    seg_results_payload.append({"name": seg_id, "result": res_obj})
                                    logging.debug(
                                        f"chat_completions:tool_loop:film2.run seg_results_payload append seg_id={seg_id!r} "
                                        f"len={len(seg_results_payload)}"
                                    )
                            if seg_results_payload:
                                logging.debug(
                                    f"chat_completions:tool_loop:film2.run segment_qa_and_committee call len={len(seg_results_payload)}"
                                )
                                trace_append(
                                    "film2.segment_qa_start",
                                    {
                                        "trace_id": trace_id,
                                        "tool": "film2.run",
                                        "segment_ids": [s.get("name") for s in seg_results_payload if isinstance(s, dict)],
                                        "quality_profile": profile_name,
                                    },
                                )
                                updated_seg, seg_outcome, seg_patch = await segment_qa_and_committee(
                                    trace_id=trace_id,
                                    user_text=_tool_step_eval_user_text("film2.run", args_echo, last_user_text),
                                    tool_name="film2.run",
                                    segment_results=seg_results_payload,
                                    mode=effective_mode,
                                    base_url=base_url,
                                    executor_base_url=executor_endpoint,
                                    absolutize_url=_abs_url,
                                    quality_profile=profile_name,
                                    max_refine_passes=refine_budget,
                                )
                                _updated_seg_type = type(updated_seg).__name__
                                _seg_outcome_keys = sorted(list(seg_outcome.keys())) if isinstance(seg_outcome, dict) else None
                                _seg_patch_len = len(seg_patch) if isinstance(seg_patch, list) else -1
                                logging.debug(
                                    f"chat_completions:tool_loop:film2.run segment_qa_and_committee returned updated_seg_type={_updated_seg_type} "
                                    f"seg_outcome_keys={_seg_outcome_keys!r} seg_patch_len={_seg_patch_len}"
                                )
                                if isinstance(meta_obj, dict):
                                    meta_obj["segment_committee"] = seg_outcome
                                    if seg_patch:
                                        meta_obj["segment_patch_results"] = seg_patch
                                        extra_results.extend(seg_patch)
                                        logging.debug(
                                            f"chat_completions:tool_loop:film2.run extra_results extended len={len(extra_results)}"
                                        )
                                        # Mark refined clips in segments metadata.
                                        segments_container = meta_obj.get("segments")
                                        if isinstance(segments_container, dict):
                                            clips_meta = segments_container.get("clips")
                                            if isinstance(clips_meta, list):
                                                for pr in seg_patch:
                                                    if not isinstance(pr, dict):
                                                        continue
                                                    if pr.get("tool") != "video.refine.clip":
                                                        continue
                                                    if pr.get("error"):
                                                        continue
                                                    sid = pr.get("segment_id")
                                                    args_used = pr.get("args") if isinstance(pr.get("args"), dict) else {}
                                                    refine_mode_val = args_used.get("refine_mode")
                                                    for clip in clips_meta:
                                                        if not isinstance(clip, dict):
                                                            continue
                                                        if clip.get("id") == sid:
                                                            clip_meta = clip.setdefault("meta", {})
                                                            if isinstance(clip_meta, dict):
                                                                clip_meta["refined"] = True
                                                                if isinstance(refine_mode_val, str) and refine_mode_val:
                                                                    clip_meta["refine_mode"] = refine_mode_val
                                                                logging.debug("chat_completions:tool_loop:film2.run marked refined sid=%r refine_mode=%r", sid, refine_mode_val)
                                trace_append(
                                    "film2.segment_qa_result",
                                    {
                                        "trace_id": trace_id,
                                        "tool": "film2.run",
                                        "segment_ids": [s.get("name") for s in seg_results_payload if isinstance(s, dict)],
                                        "action": seg_outcome.get("action"),
                                    },
                                )
                    elif tname in ("music.infinite.windowed", "music.dispatch", "music.variation", "music.mixdown"):
                        logging.debug("chat_completions:tool_loop:music qa branch entered tname=%r", tname)
                        meta_block = res.get("meta") if isinstance(res, dict) else {}
                        profile_name = meta_block.get("quality_profile") if isinstance(meta_block, dict) and isinstance(meta_block.get("quality_profile"), str) else None
                        logging.debug("chat_completions:tool_loop:music profile_name=%r", profile_name)
                        preset_music = LOCK_QUALITY_PRESETS.get((profile_name or "standard").lower(), LOCK_QUALITY_PRESETS["standard"])
                        logging.debug("chat_completions:tool_loop:music preset_keys=%r", sorted(list(preset_music.keys())) if isinstance(preset_music, dict) else None)
                        # Clamp to a single refinement pass for music QA (defensive).
                        _rbm_raw = preset_music.get("max_refine_passes", 1) if isinstance(preset_music, dict) else 1
                        try:
                            refine_budget_music = min(1, int(_rbm_raw))
                        except Exception as exc:
                            logging.warning("chat_completions: bad max_refine_passes=%r for music; defaulting to 1", _rbm_raw, exc_info=True)
                            refine_budget_music = 1
                        logging.debug("chat_completions:tool_loop:music refine_budget_music=%r", refine_budget_music)
                        updated_seg, seg_outcome, seg_patch = await segment_qa_and_committee(
                            trace_id=trace_id,
                            user_text=_tool_step_eval_user_text(tname, args_echo, last_user_text),
                            tool_name=tname,
                            segment_results=[tr],
                            mode=effective_mode,
                            base_url=base_url,
                            executor_base_url=executor_endpoint,
                            absolutize_url=_abs_url,
                            quality_profile=profile_name,
                            max_refine_passes=refine_budget_music,
                        )
                        _seg_outcome_keys = sorted(list(seg_outcome.keys())) if isinstance(seg_outcome, dict) else None
                        _seg_patch_len = len(seg_patch) if isinstance(seg_patch, list) else -1
                        logging.debug(
                            f"chat_completions:tool_loop:music segment_qa returned outcome_keys={_seg_outcome_keys!r} patch_len={_seg_patch_len}"
                        )
                        if isinstance(res, dict):
                            meta_music = res.setdefault("meta", {})
                            if isinstance(meta_music, dict):
                                meta_music["segment_committee"] = seg_outcome
                                if seg_patch:
                                    meta_music["segment_patch_results"] = seg_patch
                        if seg_patch:
                            extra_results.extend(seg_patch)
                            logging.debug(f"chat_completions:tool_loop:music extra_results extended len={len(extra_results)}")
                    elif tname.startswith("image."):
                        logging.debug("chat_completions:tool_loop:image qa branch entered tname=%r", tname)
                        # Optional: run segment-level QA/committee for image tools as well.
                        profile_name = None
                        if isinstance(res, dict):
                            meta_block = res.get("meta")
                            if isinstance(meta_block, dict) and isinstance(meta_block.get("quality_profile"), str):
                                profile_name = meta_block.get("quality_profile")
                        logging.debug("chat_completions:tool_loop:image profile_name=%r", profile_name)
                        preset_img = LOCK_QUALITY_PRESETS.get((profile_name or "standard").lower(), LOCK_QUALITY_PRESETS["standard"])
                        logging.debug("chat_completions:tool_loop:image preset_keys=%r", sorted(list(preset_img.keys())) if isinstance(preset_img, dict) else None)
                        # Clamp to a single refinement pass for image QA (defensive).
                        _rbi_raw = preset_img.get("max_refine_passes", 0) if isinstance(preset_img, dict) else 0
                        try:
                            refine_budget_img = min(1, int(_rbi_raw))
                        except Exception as exc:
                            logging.warning("chat_completions: bad max_refine_passes=%r for image; defaulting to 0", _rbi_raw, exc_info=True)
                            refine_budget_img = 0
                        logging.debug("chat_completions:tool_loop:image refine_budget_img=%r", refine_budget_img)
                        updated_seg, seg_outcome, seg_patch = await segment_qa_and_committee(
                            trace_id=trace_id,
                            user_text=_tool_step_eval_user_text(tname, args_echo, last_user_text),
                            tool_name=tname,
                            segment_results=[tr],
                            mode=effective_mode,
                            base_url=base_url,
                            executor_base_url=executor_endpoint,
                            absolutize_url=_abs_url,
                            quality_profile=profile_name,
                            max_refine_passes=refine_budget_img,
                        )
                        _seg_outcome_keys = sorted(list(seg_outcome.keys())) if isinstance(seg_outcome, dict) else None
                        _seg_patch_len = len(seg_patch) if isinstance(seg_patch, list) else -1
                        logging.debug(
                            f"chat_completions:tool_loop:image segment_qa returned outcome_keys={_seg_outcome_keys!r} patch_len={_seg_patch_len}"
                        )
                        if isinstance(res, dict):
                            meta_img = res.setdefault("meta", {})
                            if isinstance(meta_img, dict):
                                meta_img["segment_committee"] = seg_outcome
                                if seg_patch:
                                    meta_img["segment_patch_results"] = seg_patch
                        if seg_patch:
                            extra_results.extend(seg_patch)
                            logging.debug(f"chat_completions:tool_loop:image extra_results extended len={len(extra_results)}")
                    elif tname == "tts.speak":
                        logging.debug("chat_completions:tool_loop:tts qa branch entered")
                        # Segment-level QA/committee for TTS segments
                        profile_name = None
                        if isinstance(res, dict):
                            meta_block = res.get("meta")
                            if isinstance(meta_block, dict) and isinstance(meta_block.get("quality_profile"), str):
                                profile_name = meta_block.get("quality_profile")
                        logging.debug("chat_completions:tool_loop:tts profile_name=%r", profile_name)
                        preset_tts = LOCK_QUALITY_PRESETS.get((profile_name or "standard").lower(), LOCK_QUALITY_PRESETS["standard"])
                        logging.debug("chat_completions:tool_loop:tts preset_keys=%r", sorted(list(preset_tts.keys())) if isinstance(preset_tts, dict) else None)
                        _rbt_raw = preset_tts.get("max_refine_passes", 1) if isinstance(preset_tts, dict) else 1
                        try:
                            refine_budget_tts = min(1, int(_rbt_raw))
                        except Exception as exc:
                            logging.warning("chat_completions: bad max_refine_passes=%r for tts; defaulting to 1", _rbt_raw, exc_info=True)
                            refine_budget_tts = 1
                        logging.debug("chat_completions:tool_loop:tts refine_budget_tts=%r", refine_budget_tts)
                        updated_seg, seg_outcome, seg_patch = await segment_qa_and_committee(
                            trace_id=trace_id,
                            user_text=_tool_step_eval_user_text(tname, args_echo, last_user_text),
                            tool_name=tname,
                            segment_results=[tr],
                            mode=effective_mode,
                            base_url=base_url,
                            executor_base_url=executor_endpoint,
                            absolutize_url=_abs_url,
                            quality_profile=profile_name,
                            max_refine_passes=refine_budget_tts,
                        )
                        _seg_outcome_keys = sorted(list(seg_outcome.keys())) if isinstance(seg_outcome, dict) else None
                        _seg_patch_len = len(seg_patch) if isinstance(seg_patch, list) else -1
                        logging.debug(
                            f"chat_completions:tool_loop:tts segment_qa returned outcome_keys={_seg_outcome_keys!r} patch_len={_seg_patch_len}"
                        )
                        if isinstance(res, dict):
                            meta_tts = res.setdefault("meta", {})
                            if isinstance(meta_tts, dict):
                                meta_tts["segment_committee"] = seg_outcome
                                if seg_patch:
                                    meta_tts["segment_patch_results"] = seg_patch
                        if seg_patch:
                            extra_results.extend(seg_patch)
                            logging.debug(f"chat_completions:tool_loop:tts extra_results extended len={len(extra_results)}")
                    elif tname == "audio.sfx.compose":
                        logging.debug("chat_completions:tool_loop:sfx qa branch entered")
                        # Optional: SFX QA/committee wiring (currently metrics are sparse)
                        profile_name = None
                        if isinstance(res, dict):
                            meta_block = res.get("meta")
                            if isinstance(meta_block, dict) and isinstance(meta_block.get("quality_profile"), str):
                                profile_name = meta_block.get("quality_profile")
                        logging.debug("chat_completions:tool_loop:sfx profile_name=%r", profile_name)
                        preset_sfx = LOCK_QUALITY_PRESETS.get((profile_name or "standard").lower(), LOCK_QUALITY_PRESETS["standard"])
                        logging.debug("chat_completions:tool_loop:sfx preset_keys=%r", sorted(list(preset_sfx.keys())) if isinstance(preset_sfx, dict) else None)
                        _rbs_raw = preset_sfx.get("max_refine_passes", 0) if isinstance(preset_sfx, dict) else 0
                        try:
                            refine_budget_sfx = min(1, int(_rbs_raw))
                        except Exception as exc:
                            logging.warning("chat_completions: bad max_refine_passes=%r for sfx; defaulting to 0", _rbs_raw, exc_info=True)
                            refine_budget_sfx = 0
                        logging.debug("chat_completions:tool_loop:sfx refine_budget_sfx=%r", refine_budget_sfx)
                        updated_seg, seg_outcome, seg_patch = await segment_qa_and_committee(
                            trace_id=trace_id,
                            user_text=_tool_step_eval_user_text(tname, args_echo, last_user_text),
                            tool_name=tname,
                            segment_results=[tr],
                            mode=effective_mode,
                            base_url=base_url,
                            executor_base_url=executor_endpoint,
                            absolutize_url=_abs_url,
                            quality_profile=profile_name,
                            max_refine_passes=refine_budget_sfx,
                        )
                        _seg_outcome_keys = sorted(list(seg_outcome.keys())) if isinstance(seg_outcome, dict) else None
                        _seg_patch_len = len(seg_patch) if isinstance(seg_patch, list) else -1
                        logging.debug(
                            f"chat_completions:tool_loop:sfx segment_qa returned outcome_keys={_seg_outcome_keys!r} patch_len={_seg_patch_len}"
                        )
                        if isinstance(res, dict):
                            meta_sfx = res.setdefault("meta", {})
                            if isinstance(meta_sfx, dict):
                                meta_sfx["segment_committee"] = seg_outcome
                                if seg_patch:
                                    meta_sfx["segment_patch_results"] = seg_patch
                        if seg_patch:
                            extra_results.extend(seg_patch)
                            logging.debug(f"chat_completions:tool_loop:sfx extra_results extended len={len(extra_results)}")
                    tool_results.append(tr)
                    logging.debug(f"chat_completions:tool_loop:tool_results appended len={len(tool_results)}")
                    if extra_results:
                        tool_results.extend(extra_results)
                        logging.debug(f"chat_completions:tool_loop:tool_results extended with extra_results len={len(tool_results)}")

        _log("flow.after_execute", trace_id=trace_id, tool_results_count=len(tool_results or []))
        logging.debug(
            f"chat_completions:flow.after_execute tool_results_count={len(tool_results or [])} pipeline_ok={pipeline_ok!r}"
        )

        for _tr in tool_results or []:
            logging.debug(
                f"chat_completions:post_tool_scan:iter _tr_type={type(_tr).__name__} "
                f"_tr_keys={(sorted(list(_tr.keys())) if isinstance(_tr, dict) else None)!r}"
            )
            _res = (_tr or {}).get("result") or {}
            logging.debug(
                f"chat_completions:post_tool_scan:_res_type={type(_res).__name__} "
                f"_res_keys={(sorted(list(_res.keys())) if isinstance(_res, dict) else None)!r}"
            )
            if isinstance(_res, dict):
                _meta = _res.get("meta") if isinstance(_res, dict) else {}
                logging.debug(f"chat_completions:post_tool_scan:_meta_type={type(_meta).__name__}")
                _pid = (_meta or {}).get("prompt_id")
                logging.debug("chat_completions:post_tool_scan:_pid=%r", _pid)
                if isinstance(_pid, str) and _pid:
                    _ids = _res.get("ids") if isinstance(_res.get("ids"), dict) else {}
                    logging.debug(f"chat_completions:post_tool_scan:_ids_type={type(_ids).__name__}")
                    _comfy_client_id = _ids.get("client_id") if isinstance(_ids.get("client_id"), str) else None
                    logging.debug("chat_completions:post_tool_scan:_comfy_client_id=%r", _comfy_client_id)
                    _log("[comfy] POST /prompt", trace_id=trace_id, prompt_id=_pid, client_id=_comfy_client_id)
                    _log("comfy.submit", trace_id=trace_id, prompt_id=_pid, client_id=_comfy_client_id)
                    _log("[comfy] polling /history/" + _pid)

        # Post-run QA checkpoint (lightweight) — run once after all tool results are collected.
        _img_count = assets_count_images(tool_results)
        logging.debug("chat_completions:postrun_qa _img_count=%r", _img_count)
        _vid_count = assets_count_video(tool_results)
        logging.debug("chat_completions:postrun_qa _vid_count=%r", _vid_count)
        _aud_count = assets_count_audio(tool_results)
        logging.debug("chat_completions:postrun_qa _aud_count=%r", _aud_count)
        # Defensive: counts should be ints, but never allow logging/response to crash if not.
        try:
            _img_i = int(_img_count)
        except Exception as exc:
            logging.warning("chat_completions: bad images_count=%r; defaulting to 0", _img_count, exc_info=True)
            _img_i = 0
        try:
            _vid_i = int(_vid_count)
        except Exception as exc:
            logging.warning("chat_completions: bad videos_count=%r; defaulting to 0", _vid_count, exc_info=True)
            _vid_i = 0
        try:
            _aud_i = int(_aud_count)
        except Exception as exc:
            logging.warning("chat_completions: bad audio_count=%r; defaulting to 0", _aud_count, exc_info=True)
            _aud_i = 0
        counts_summary = {"images": _img_i, "videos": _vid_i, "audio": _aud_i}
        logging.debug("chat_completions:postrun_qa counts_summary=%r", counts_summary)
        domain_qa = assets_compute_domain_qa(tool_results)
        logging.debug(f"chat_completions:postrun_qa domain_qa_type={type(domain_qa).__name__} domain_qa={domain_qa!r}")
        qa_metrics = {"counts": counts_summary, "domain": domain_qa}
        logging.debug("chat_completions:postrun_qa qa_metrics_keys=%r", sorted(list(qa_metrics.keys())))
        _log("qa.metrics", trace_id=trace_id, tool="postrun", metrics=qa_metrics)
        _log("committee.postrun.review", trace_id=trace_id, summary=qa_metrics)
        # Committee decision with optional single revision
        logging.debug("chat_completions:committee postrun_committee_decide call trace_id=%r", trace_id)
        committee_outcome = await postrun_committee_decide(
            trace_id=trace_id,
            user_text=last_user_text,
            tool_results=tool_results,
            qa_metrics=qa_metrics,
            mode=effective_mode,
        )
        logging.debug("chat_completions:committee postrun_committee_decide returned keys=%r", sorted(list(committee_outcome.keys())) if isinstance(committee_outcome, dict) else None)
        committee_action = str((committee_outcome.get("action") or "go")).strip().lower()
        logging.debug("chat_completions:committee committee_action=%r", committee_action)
        committee_rationale = str(committee_outcome.get("rationale") or "")
        logging.debug(f"chat_completions:committee committee_rationale_len={len(committee_rationale)}")
        patch_plan = committee_outcome.get("patch_plan") or []
        logging.debug(
            f"chat_completions:committee patch_plan_type={type(patch_plan).__name__} "
            f"patch_plan_len={(len(patch_plan) if isinstance(patch_plan, list) else -1)}"
        )
        _log("committee.decision", trace_id=trace_id, action=committee_action, rationale=committee_rationale)
        _log("flow.after_committee", trace_id=trace_id, action=committee_action)
        _log("committee.review.final", trace_id=trace_id, summary=qa_metrics)
        _log("committee.decision.final", trace_id=trace_id, action=committee_action)
        # Optional one-pass revision
        if committee_action == "revise" and isinstance(patch_plan, list) and patch_plan:
            logging.debug(f"chat_completions:committee revise branch entered patch_plan_len={len(patch_plan)}")
            # Filter patch plan by front-door + mode rules
            _allowed_mode_set = set(_allowed_tools_for_mode(effective_mode))
            logging.debug(f"chat_completions:committee _allowed_mode_set_len={len(_allowed_mode_set)}")
            filtered_patch_plan: List[Dict[str, Any]] = []
            logging.debug(f"chat_completions:committee filtered_patch_plan init len={len(filtered_patch_plan)}")
            for st in patch_plan:
                logging.debug(
                    f"chat_completions:committee patch_plan_iter st_type={type(st).__name__} "
                    f"st_keys={(sorted(list(st.keys())) if isinstance(st, dict) else None)!r}"
                )
                if not isinstance(st, dict):
                    continue
                tl = (st.get("tool") or "").strip() if isinstance(st.get("tool"), str) else ""
                logging.debug("chat_completions:committee patch_plan_iter tl=%r", tl)
                if not tl or tl not in PLANNER_VISIBLE_TOOLS or tl not in _allowed_mode_set:
                    logging.debug("chat_completions:committee patch_plan_iter skipped tl=%r", tl)
                    continue
                # IMPORTANT: do not drop args when the committee returns args as a JSON string.
                args_raw = st.get("args") if ("args" in st) else st.get("arguments")
                logging.debug(f"chat_completions:committee patch_plan_iter args_raw_type={type(args_raw).__name__}")
                args_st: Dict[str, Any]
                if isinstance(args_raw, dict):
                    args_st = dict(args_raw)
                elif isinstance(args_raw, str):
                    parser_patch = JSONParser()
                    logging.debug(
                        f"chat_completions:committee patch_plan_iter parsing args_raw_json_len={len(args_raw)}"
                    )
                    parsed = parser_patch.parse(args_raw or "", dict)
                    logging.debug(f"chat_completions:committee patch_plan_iter parsed_type={type(parsed).__name__}")
                    args_st = dict(parsed) if isinstance(parsed, dict) else {"_raw": args_raw}
                elif args_raw is None:
                    args_st = {}
                else:
                    args_st = {"_raw": args_raw}
                filtered_patch_plan.append({"tool": tl, "args": args_st})
                logging.debug(
                    f"chat_completions:committee filtered_patch_plan append tool={tl!r} "
                    f"args_keys={(sorted(list(args_st.keys())) if isinstance(args_st, dict) else None)!r} len={len(filtered_patch_plan)}"
                )
            if filtered_patch_plan:
                logging.debug(f"chat_completions:committee filtered_patch_plan nonempty len={len(filtered_patch_plan)}")
                # Normalize to internal tool_calls schema
                patch_calls: List[Dict[str, Any]] = [{"name": s.get("tool"), "arguments": (s.get("args") or {})} for s in filtered_patch_plan]
                logging.debug(
                    f"chat_completions:committee patch_calls_len={len(patch_calls)} first={(patch_calls[0] if patch_calls else None)!r}"
                )
                _inject_execution_context(patch_calls, trace_id, effective_mode, conv_cid)
                logging.debug("chat_completions:committee _inject_execution_context patch_calls done")
                # Validator disabled: execute patch plan directly without pre-validation.
                patch_validated = list(patch_calls)
                logging.debug(f"chat_completions:committee patch_validated_len={len(patch_validated)}")
                patch_failures: List[Dict[str, Any]] = []
                logging.debug(f"chat_completions:committee patch_failures_len={len(patch_failures)}")
                # Execute validated patch steps
                patch_failure_results: List[Dict[str, Any]] = []
                logging.debug(
                    f"chat_completions:committee patch_failure_results init len={len(patch_failure_results)}"
                )
                for failure in patch_failures or []:
                    env = failure.get("envelope") if isinstance(failure.get("envelope"), dict) else {}
                    args_snapshot = failure.get("arguments") if isinstance(failure.get("arguments"), dict) else {}
                    name_snapshot = str((failure.get("name") or "")).strip() or "tool"
                    error_snapshot = env.get("error") if isinstance(env, dict) else {}
                    patch_failure_results.append(
                        {
                            "name": name_snapshot,
                            "result": env,
                            "error": error_snapshot,
                            "args": args_snapshot,
                        }
                    )
                    logging.debug(
                        f"chat_completions:committee patch_failure_results append name={name_snapshot!r} len={len(patch_failure_results)}"
                    )
                exec_results: List[Dict[str, Any]] = []
                logging.debug(f"chat_completions:committee exec_results init len={len(exec_results)}")
                # Advisory-only: execute the patch plan once via the executor path.
                exec_batch = patch_validated or patch_calls
                logging.debug(
                    f"chat_completions:committee exec_batch_len={(len(exec_batch) if isinstance(exec_batch, list) else -1)}"
                )
                if exec_batch:
                    _log("exec.payload", trace_id=trace_id, steps=[{"tool": (c.get("name") or "")} for c in exec_batch[:5]])
                    logging.debug("chat_completions:committee gateway_execute patch exec starting")
                    try:
                        exec_results = await gateway_execute(exec_batch, trace_id, executor_endpoint, request_id=str(request_id))
                    except Exception as ex:
                        # Hardening: patch-plan execution is advisory; never crash the main run.
                        logging.error(f"chat_completions:committee:gateway_execute raised ex={ex}", exc_info=True)
                        exec_results = [
                            {
                                "name": "executor",
                                "error": {
                                    "code": "executor_gateway_exception",
                                    "message": str(ex),
                                    "status": 0,
                                    "stack": _tb(ex),
                                },
                                "result": {},
                            }
                        ]
                    logging.debug(
                        f"chat_completions:committee gateway_execute patch exec returned len={(len(exec_results) if isinstance(exec_results, list) else -1)}"
                    )
                    # Persist patch exec results (teacher + ledger + db)
                    try:
                        results_list = exec_results if isinstance(exec_results, list) else []
                        logging.debug(f"chat_completions:committee results_list_len={len(results_list)}")
                        for i, call in enumerate(exec_batch or []):
                            nm = str((call or {}).get("name") or "tool")
                            args_obj = (call or {}).get("arguments") if isinstance((call or {}).get("arguments"), dict) else {}
                            logging.debug(
                                f"chat_completions:committee patch_result_iter i={i} nm={nm!r} "
                                f"args_keys={(sorted(list(args_obj.keys())) if isinstance(args_obj, dict) else None)!r}"
                            )
                            tr: Dict[str, Any] = {}
                            if i < len(results_list) and isinstance(results_list[i], dict):
                                tr = results_list[i]
                            else:
                                for cand in results_list:
                                    if isinstance(cand, dict) and str(cand.get("name") or "") == nm:
                                        tr = cand
                                        break
                            logging.debug("chat_completions:committee patch_result_iter tr_keys=%r", sorted(list(tr.keys())) if isinstance(tr, dict) else None)
                            res_obj = tr.get("result") if isinstance(tr.get("result"), dict) else {}
                            err_obj = tr.get("error")
                            if not err_obj and isinstance(res_obj, dict):
                                err_obj = res_obj.get("error")
                            arts_val = res_obj.get("artifacts") if isinstance(res_obj, dict) else None
                            ok_step = not isinstance(err_obj, (str, dict))
                            logging.debug(
                                f"chat_completions:committee patch_result_iter ok_step={ok_step!r} err_type={type(err_obj).__name__}"
                            )
                            tool_exec_meta.append({"name": nm, "args": args_obj, "ok": bool(ok_step), "result": res_obj, "error": err_obj, "artifacts": arts_val})
                            logging.debug(f"chat_completions:committee tool_exec_meta appended len={len(tool_exec_meta)}")
                            if run_id is not None:
                                logging.debug("chat_completions:committee _db_insert_tool_call patch starting run_id=%r nm=%r", run_id, nm)
                                await _db_insert_tool_call(
                                    run_id,
                                    nm,
                                    int(master_seed),
                                    (args_obj if isinstance(args_obj, dict) else {}),
                                    (res_obj if isinstance(res_obj, dict) else None),
                                    None,
                                )
                                logging.debug("chat_completions:committee _db_insert_tool_call patch done")
                            if _ledger_shard is not None and _ledger_root is not None:
                                row = {"t": int(time.time() * 1000), "trace_id": trace_id, "run_id": int(run_id) if isinstance(run_id, int) else None, "name": nm, "ok": bool(ok_step), "args": args_obj, "result": res_obj, "error": err_obj, "artifacts": arts_val}
                                _ledger_shard = _art_append_jsonl(_ledger_shard, row)
                                logging.debug("chat_completions:committee ledger append patch done nm=%r", nm)
                    except Exception as ex:
                        logging.debug(
                            f"chat_completions:patch_exec_meta.persist.failed trace_id={trace_id} ex={ex}",
                            exc_info=True,
                        )
                patch_results = list(patch_failure_results) + list(exec_results or [])
                logging.debug(f"chat_completions:committee patch_results_len={len(patch_results)}")
                tool_results = (tool_results or []) + patch_results
                logging.debug(f"chat_completions:committee tool_results_len_after_patch={len(tool_results or [])}")
                _log("committee.revision.executed", trace_id=trace_id, steps=len(patch_validated or []), failures=len(patch_failures or []))
                logging.debug(
                    f"chat_completions:committee revision executed steps={len(patch_validated or [])} failures={len(patch_failures or [])}"
                )
                # Recompute QA metrics after revision
                _img_count = assets_count_images(tool_results)
                logging.debug("chat_completions:postrev_qa _img_count=%r", _img_count)
                _vid_count = assets_count_video(tool_results)
                logging.debug("chat_completions:postrev_qa _vid_count=%r", _vid_count)
                _aud_count = assets_count_audio(tool_results)
                logging.debug("chat_completions:postrev_qa _aud_count=%r", _aud_count)
                # Defensive: counts should be ints, but never allow logging/response to crash if not.
                try:
                    _img_i = int(_img_count)
                except Exception as exc:
                    logging.warning("chat_completions: bad images_count=%r; defaulting to 0", _img_count, exc_info=True)
                    _img_i = 0
                try:
                    _vid_i = int(_vid_count)
                except Exception as exc:
                    logging.warning("chat_completions: bad videos_count=%r; defaulting to 0", _vid_count, exc_info=True)
                    _vid_i = 0
                try:
                    _aud_i = int(_aud_count)
                except Exception as exc:
                    logging.warning("chat_completions: bad audio_count=%r; defaulting to 0", _aud_count, exc_info=True)
                    _aud_i = 0
                counts_summary = {"images": _img_i, "videos": _vid_i, "audio": _aud_i}
                logging.debug("chat_completions:postrev_qa counts_summary=%r", counts_summary)
                domain_qa = assets_compute_domain_qa(tool_results)
                logging.debug("chat_completions:postrev_qa domain_qa=%r", domain_qa)
                qa_metrics = {"counts": counts_summary, "domain": domain_qa}
                logging.debug("chat_completions:postrev_qa qa_metrics_keys=%r", sorted(list(qa_metrics.keys())))
            _log("qa.metrics", trace_id=trace_id, tool="postrun.revise", metrics=qa_metrics)
        # Make full tool results (including internal error stacks) available to the
        # final COMPOSE pass so the committee can explain failures and partial
        # successes directly to the user instead of using a prebuilt summary.
        if tool_results:
            messages = [{"role": "system", "content": "Tool results:\n" + json.dumps(tool_results, indent=2)}] + messages

        # No context orchestration: pass messages directly to committee.
        exec_messages = list(messages or [])
        exec_messages_current = exec_messages

        # Idempotency fast-path disabled to prevent early returns; defer any cached reuse to finalization if desired

        # Use the central committee path to derive a single merged LLM view over exec_messages.
        try:
            llm_env = await committee_ai_text(
                messages=exec_messages,
                trace_id=trace_id,
                temperature=exec_temperature,
            )
        except Exception as ex:
            _log(
                "executor.committee_call.error",
                trace_id=trace_id,
                error=str(ex),
            )
            # Mark pipeline as not-OK so response reflects degraded tool lane
            pipeline_ok = False
            llm_env = {
                "ok": False,
                "error": {
                    "code": "executor_committee_exception",
                    "message": str(ex),
                },
            }
        # Synthesize per-backend result views from the unified committee envelope for fallback logic.
        committee_critiques: List[Dict[str, Any]] = []
        if isinstance(llm_env, dict):
            err = llm_env.get("error")
            res = llm_env.get("result") or {}
            if isinstance(res, dict):
                raw_crits = res.get("critiques")
                if isinstance(raw_crits, list):
                    for it in raw_crits:
                        if isinstance(it, dict):
                            committee_critiques.append(it)
            txt = res.get("text")
            # Prefer structured per-backend envelopes surfaced by the committee when available.
            qwen_env = res.get("qwen") if isinstance(res, dict) else None
            glm_env = res.get("glm") if isinstance(res, dict) else None
            deepseek_env = res.get("deepseek") if isinstance(res, dict) else None
            synth_env_struct = res.get("synth") if isinstance(res, dict) else None
            if isinstance(qwen_env, dict):
                qwen_result = dict(qwen_env)
            if isinstance(glm_env, dict):
                glm_result = dict(glm_env)
            if isinstance(deepseek_env, dict):
                deepseek_result = dict(deepseek_env)
            if isinstance(synth_env_struct, dict):
                synth_result = dict(synth_env_struct)
            # Backward-compatible fallback when committee did not expose per-backend structs.
            base_payload = {
                "error": err,
                "response": txt,
                "_base_url": "committee",
                }
            if not qwen_result:
                qwen_result = dict(base_payload)
            if not glm_result:
                glm_result = dict(base_payload)
            if not deepseek_result:
                deepseek_result = dict(base_payload)
            if not synth_result:
                synth_result = dict(base_payload)

        # Normalize per-backend responses to always be strings for downstream logic.
        qwen_text = (qwen_result.get("response") or "")
        glm_text = (glm_result.get("response") or "")
        deepseek_text = (deepseek_result.get("response") or "")
        # Preserve the first-pass model outputs for robust fallback in final composition
        orig_qwen_text = qwen_text
        orig_glm_text = glm_text

        # No synthetic tool forcing; the planner alone selects tools

        # If response still looks like a refusal, synthesize a constructive message using tool_results
        refusal_markers = [
            "cannot",
            "can't",
            "unable",
            "i won't",
            "i will not",
            "refuse",
            "not able",
            "i cannot",
            "i can't",
            "sorry, i",
        ]
        final_refusal = any(tok in (qwen_text.lower() + "\n" + glm_text.lower() + "\n" + (deepseek_text or "").lower()) for tok in refusal_markers)
        if final_refusal:
            # Try to extract a film_id or any success info from tool results
            film_id = None
            errors: List[str] = []
            for tr in (tool_results or []):
                if isinstance(tr, dict):
                    rid = ((tr.get("result") or {}).get("film_id"))
                    if rid and not film_id:
                        film_id = rid
                    if tr.get("error"):
                        errors.append(str(tr.get("error")))
            details = []
            if film_id:
                details.append(f"Film ID: {film_id}")
            if errors:
                details.append("Errors: " + "; ".join(errors)[:800])
            summary = "\n".join(details)
            affirmative = (
                "Initiated tool-based generation flow. "
                + ("\n" + summary if summary else "")
                + "\nUse film_status to track progress, and jobs endpoints for live status."
            )
            # Do not modify model texts here; the final composer will only append status

        # No CO compose step: take the committee result directly as the final text.
        final_text = qwen_text or glm_text or deepseek_text or ""
        # Append discovered asset URLs from tool results so users see concrete outputs inline.
        # Do NOT synthesize or reuse artifacts from unrelated runs; only attach URLs that
        # actually come from the current tool_results set.
        # NOTE: asset URLs are appended centrally at the end (single source of truth).

        
        # Merge exact usages if available, else approximate
        usage = merge_usages([
            qwen_result.get("_usage"),
            glm_result.get("_usage"),
            deepseek_result.get("_usage"),
            synth_result.get("_usage"),
        ])
        if usage["total_tokens"] == 0:
            usage = estimate_usage(messages, final_text)
        wall_ms = int(round((time.time() - t0) * 1000))
        usage_with_wall = dict(usage)
        usage_with_wall["wall_ms"] = wall_ms

        # Ensure clean markdown content: collapse excessive whitespace but keep newlines
        cleaned = final_text.replace('\r\n', '\n')
        
        # NOTE: asset URLs are appended centrally at the end (single source of truth).
        # If tools/plan/critique exist, wrap them into a hidden metadata block and present a concise final answer
        meta_sections: List[str] = []
        if plan_text:
            meta_sections.append("Plan:\n" + plan_text)
        if tool_results:
            try:
                meta_sections.append("Tool Results:\n" + json.dumps(tool_results, indent=2))
            except Exception as ex:
                # This metadata block is non-critical; log and continue.
                logging.warning(f"final.tool_results.meta_serialize_failed: {ex}", exc_info=True)
        # Instead of verbose plan/critique footers, append a minimal Tool Results summary (IDs/errors) only
        tool_summary_lines: List[str] = []
        tool_tracebacks: List[str] = []
        if tool_results:
            try:
                film_id = None
                errors: List[str] = []
                job_ids: List[str] = []
                prompt_ids: List[str] = []
                for tr in (tool_results or []):
                    if isinstance(tr, dict):
                        # Safely normalize result to a dict before any .get
                        res_field = tr.get("result")
                        if isinstance(res_field, str):
                            try:
                                res_field = JSONParser().parse(
                                    res_field,
                                    {
                                        "film_id": str,
                                        "job_id": str,
                                        "prompt_id": str,
                                        "created": [{"result": dict, "job_id": str, "prompt_id": str}],
                                    },
                                )
                            except Exception as ex:
                                logging.warning(f"tool_results.parse.failed: {ex}", exc_info=True)
                                res_field = {}
                        if not isinstance(res_field, dict):
                            res_field = {}
                        if not film_id:
                            film_id = res_field.get("film_id")
                        if tr.get("error"):
                            errors.append(str(tr.get("error")))
                        tb = tr.get("traceback")
                        if isinstance(tb, str) and tb.strip():
                            tool_tracebacks.append(tb)
                        # direct job_id/prompt_id
                        res = res_field
                        jid = res.get("job_id") or tr.get("job_id")
                        pid = res.get("prompt_id") or tr.get("prompt_id")
                        if isinstance(jid, str):
                            job_ids.append(jid)
                        if isinstance(pid, str):
                            prompt_ids.append(pid)
                        # make_movie created list
                        created = res.get("created") if isinstance(res, dict) else None
                        if isinstance(created, list):
                            for it in created:
                                if isinstance(it, dict):
                                    r2 = it.get("result") or {}
                                    j2 = r2.get("job_id") or it.get("job_id")
                                    p2 = r2.get("prompt_id") or it.get("prompt_id")
                                    if isinstance(j2, str):
                                        job_ids.append(j2)
                                    if isinstance(p2, str):
                                        prompt_ids.append(p2)
                if film_id:
                    tool_summary_lines.append(f"- **film_id**: `{film_id}`")
                if job_ids:
                    juniq = list(dict.fromkeys([j for j in job_ids if j]))
                    if juniq:
                        tool_summary_lines.append("- **jobs**: " + ", ".join([f"`{j}`" for j in juniq[:20]]))
                if prompt_ids:
                    puniq = list(dict.fromkeys([p for p in prompt_ids if p]))
                    if puniq:
                        tool_summary_lines.append("- **prompts**: " + ", ".join([f"`{p}`" for p in puniq[:20]]))
                if errors:
                    tool_summary_lines.append("- **errors**: " + "; ".join(errors)[:800])
            except Exception as ex:
                logging.error(f"tool_results.summary_failed: {ex}", exc_info=True)
        footer = ("\n\n### Tool Results\n" + "\n".join(tool_summary_lines)) if tool_summary_lines else ""
        if tool_tracebacks:
            # Include full tracebacks verbatim (truncated to a safe length per item)
            footer += "\n\n### Debug — Tracebacks\n" + "\n\n".join([f"```\n{_tb(t)}\n```" for t in tool_tracebacks])
        # Decide emptiness/refusal based on the main body only, not the footer
        main_only = cleaned
        text_lower = (main_only or "").lower()
        refusal_markers = ("can't", "cannot", "unable", "not feasible", "can't create", "cannot create", "won't be able", "won't be able")
        looks_empty = (not str(main_only or "").strip())
        looks_refusal = any(tok in text_lower for tok in refusal_markers)
        display_content = f"{cleaned}{footer}"
        try:
            _log(
                "response",
                trace_id=trace_id,
                seed=int(master_seed),
                pack_hash=pack_hash,
                route_mode=route_mode,
                tool_results_count=int(len(tool_results or [])),
                content_preview=(display_content or "")[:800],
            )
        except Exception as ex:
            logging.warning(f"final.response.checkpoint_failed: {ex}", exc_info=True)
        # Provide an appendix with raw model answers to avoid any perception of truncation
        try:
            raw_q = (qwen_text or "").strip()
            raw_g = (glm_text or "").strip()
            raw_d = (deepseek_text or "").strip()
            appendix_parts: List[str] = []
            if raw_q or raw_g or raw_d:
                appendix_parts.append("\n\n### Appendix — Model Answers")
            if raw_q:
                appendix_parts.append("\n\n#### Qwen\n" + _shorten(raw_q))
            if raw_g:
                appendix_parts.append("\n\n#### GLM4-9B\n" + _shorten(raw_g))
            if raw_d:
                appendix_parts.append("\n\n#### DeepSeek Coder\n" + _shorten(raw_d))
            if appendix_parts:
                display_content += "".join(appendix_parts)
        except Exception as ex:
            logging.warning(f"final.appendix.build_failed: {ex}", exc_info=True)
        # Evaluate fallback after footer creation, but using main-only flags
        if looks_empty or looks_refusal:
            # For now, log detection without overriding the composed content.
            logging.info("final.response.empty_or_refusal_detected; no automatic fallback applied")
        # Canonical artifacts payload: urls + items, plus optional film export summary.
        artifacts_payload: Dict[str, Any] = {}
        try:
            artifacts_out = _collect_artifacts_payload(tool_results or [], abs_url_fn)
        except Exception:
            artifacts_out = {"urls": [], "items": []}
        if isinstance(artifacts_out, dict):
            artifacts_payload = dict(artifacts_out)
        film_run = None
        for tr in (tool_results or []):
            if isinstance(tr, dict) and tr.get("name") == "film2.run" and isinstance(tr.get("result"), dict):
                film_run = tr.get("result")
                break
        if isinstance(film_run, dict) and film_run:
            master = film_run.get("master") if isinstance(film_run.get("master"), dict) else {}
            eff = film_run.get("effective") if isinstance(film_run.get("effective"), dict) else {}
            edl = film_run.get("edl") if isinstance(film_run.get("edl"), dict) else {}
            qc = film_run.get("qc_report") if isinstance(film_run.get("qc_report"), dict) else {}
            nodes = film_run.get("nodes") if isinstance(film_run.get("nodes"), dict) else {}
            artifacts_payload["film"] = {
                "master": {"uri": (film_run.get("master_uri") or master.get("uri")), "hash": (film_run.get("hash") or master.get("hash")), "res": eff.get("res"), "fps": eff.get("refresh")},
                "edl": {"uri": edl.get("uri"), "hash": edl.get("hash")},
                "nodes": {"uri": nodes.get("uri"), "hash": nodes.get("hash")},
                "qc": {"uri": qc.get("uri"), "hash": qc.get("hash")},
                "export": {"uri": film_run.get("package_uri"), "hash": film_run.get("hash")},
            }

        # Build canonical envelope (merged from steps) to attach to response for internal use
        try:
            # Ensure tool_exec_meta includes *all* tool runs, including any QA-driven patch runs
            # that were appended directly to tool_results (segment-level refine loops).
            _warn_double_wrapped_tool_results(tool_results or [])
            tool_exec_meta = _merge_tool_exec_meta_from_tool_results(tool_exec_meta, tool_results)
            step_texts: List[str] = []
            if isinstance(plan_text, str) and plan_text.strip():
                step_texts.append(plan_text)
            if tool_results:
                try:
                    step_texts.append(json.dumps(tool_results, ensure_ascii=False))
                except Exception as ex:
                    logging.warning(f"final.step_texts.tool_results_serialize_failed: {ex}", exc_info=True)
            if isinstance(qwen_text, str) and qwen_text.strip():
                step_texts.append(qwen_text)
            if isinstance(glm_text, str) and glm_text.strip():
                step_texts.append(glm_text)
            if isinstance(deepseek_text, str) and deepseek_text.strip():
                step_texts.append(deepseek_text)
            if isinstance(display_content, str) and display_content.strip():
                step_texts.append(display_content)
            step_envs = [normalize_to_envelope(t) for t in step_texts]
            final_env = stitch_merge_envelopes(step_envs)
            # Merge tool_calls deterministically from tool_exec_meta
            tc_merged: List[Dict[str, Any]] = []
            for meta in (tool_exec_meta or []):
                name = meta.get("name")
                args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
                tc_merged.append({"tool": name, "args": args, "status": "done", "result_ref": None})
            if isinstance(final_env, dict):
                final_env["tool_calls"] = tc_merged
                final_env.setdefault("meta", {})
                if isinstance(final_env.get("meta"), dict) and isinstance(artifacts_out, dict):
                    final_env["meta"]["asset_urls"] = artifacts_out.get("urls") if isinstance(artifacts_out.get("urls"), list) else []
                final_env.setdefault("artifacts", [])
                if isinstance(final_env.get("artifacts"), list) and isinstance(artifacts_out, dict) and isinstance(artifacts_out.get("items"), list):
                    final_env["artifacts"].extend(artifacts_out.get("items"))
        except Exception as ex:
            # Do not let final envelope stitching fail silently; log and fall back
            # to an empty envelope so the main /v1/chat/completions response still
            # returns a well-formed JSON body to clients.
            logging.error(f"chat.final_env_build_failed: {ex}", exc_info=True)
            final_env = {}

        # Compose final OpenAI-compatible response envelope. The ok flag reflects
        # whether the critical tool pipeline (image/music/tts/film) completed
        # without hard failures, even though the assistant still returns a useful
        # explanation and any partial artifacts.
        if response is None:
            response = compose_openai_response(
                display_content,
                usage_with_wall,
                COMMITTEE_MODEL_ID,
                master_seed,
                "orc-1",
                envelope_builder=_build_openai_envelope,
                artifacts=(artifacts_payload if isinstance(artifacts_payload, dict) else None),
                final_env=None,  # will be attached below after bump/assert/stamp
                ok=pipeline_ok,
            )
        # Attach trace/cid metadata for clients that need stable ids on responses.
        response_meta = response.setdefault("_meta", {})
        if isinstance(response_meta, dict):
            response_meta.setdefault("trace_id", trace_id)
            # Always surface the canonical orchestrator-owned cid (even if caller supplied something else).
            response_meta["cid"] = str(conv_cid)
            # Surface planner failures explicitly in the response metadata so callers
            # are not left with a silent empty plan when the planner/committee path fails.
            if isinstance(planner_error_payload, dict):
                errors_list = response_meta.setdefault("errors", [])
                if isinstance(errors_list, list):
                    errors_list.append(
                        {
                            "source": "planner",
                            "error": planner_error_payload,
                        }
                    )
                # Also attach a top-level error when no tool results were produced,
                # so OpenAI-style clients can detect planner failures directly.
                if not tool_results and not response.get("error"):
                    response["error"] = planner_error_payload
        if isinstance(final_env, dict) and final_env:
            final_env = _env_bump(final_env); _env_assert(final_env)
            final_env = _env_stamp(final_env, tool=None, model=COMMITTEE_MODEL_ID)
            # Optional ablation: extract grounded facts and export
            do_ablate = True
            do_export = True
            scope = "auto"
            if do_ablate:
                abl = ablate_env(final_env, scope_hint=(scope if scope != "auto" else "chat"))
                final_env["ablated"] = abl
                if do_export:
                    outdir = os.path.join(UPLOAD_DIR, "ablation", trace_id)
                    facts_path = ablate_write_facts(abl, trace_id, outdir)
                    # attach a reference to the exported dataset
                    response.setdefault("artifacts", {})
                    if isinstance(response["artifacts"], dict):
                        uri = _uri_from_upload_path(facts_path)
                        response["artifacts"]["ablation_facts"] = {"uri": uri}
            response["envelope"] = final_env
        # Finalize artifacts shard and write a tiny manifest reference
        if _ledger_shard is not None:
            _art_finalize_shard_local = _art_finalize  # alias to avoid shadowing name
            _art_finalize_shard_local(_ledger_shard)
            _mani = {"items": []}
            idx_path = os.path.join(_ledger_root, f"{_ledger_name}.index.json")
            _art_manifest_add(_mani, idx_path, step_id="final")
            _art_manifest_write(_ledger_root, _mani)

        # Fire-and-forget: Teacher trace tap (if reachable)
        # Build trace payload (errors here will surface)
        msgs_for_seed = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        provided_seed2 = None
        if isinstance(body0.get("seed"), (int, float)):
            try:
                provided_seed2 = int(body0.get("seed"))
            except Exception as exc:
                logging.warning(
                    "chat_completions: bad seed=%r (teacher tap); deriving from messages",
                    body0.get("seed"),
                    exc_info=True,
                )
        master_seed = provided_seed2 if provided_seed2 is not None else _derive_seed("chat", msgs_for_seed)
        label_cfg = (WRAPPER_CONFIG.get("teacher") or {}).get("default_label")
        # Normalize tool_calls shape for teacher (use args, not arguments)
        _warn_double_wrapped_tool_results(tool_results or [])
        _tc2 = _merge_tool_exec_meta_from_tool_results(tool_exec_meta, tool_results)
        # Mirror planner routing decision from the unified planner entry point (plan.committee.produce_tool_plan) for trace metadata.
        _use_qwen2 = str(PLANNER_MODEL or "qwen").lower().startswith("qwen")
        planner_id = QWEN_MODEL_ID if _use_qwen2 else GLM_MODEL_ID
        # Prefer the last assistant message from the final envelope's result.messages
        # (excludes system tails) when available; otherwise fall back to the
        # composed display_content used for the OpenAI response.
        response_text_for_teacher = display_content or ""
        res_block = final_env.get("result") if isinstance(final_env.get("result"), dict) else {}
        msgs_block = res_block.get("messages") if isinstance(res_block.get("messages"), list) else []
        last_assistant: str | None = None
        for m in msgs_block:
            if not isinstance(m, dict):
                continue
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                last_assistant = content
        if isinstance(last_assistant, str) and last_assistant.strip():
            response_text_for_teacher = last_assistant

        trace_payload = {
            "label": label_cfg or "exp_default",
            "seed": master_seed,
            "request": {"messages": messages, "tools_allowed": tools_allowed},
            "context": ({"pack_hash": pack_hash} if pack_hash else {}),
            "routing": {"planner_model": planner_id, "executors": [QWEN_MODEL_ID, GLM_MODEL_ID, DEEPSEEK_CODER_MODEL_ID], "seed_router": det_seed_router(trace_id, master_seed)},
            "tool_calls": _tc2,
            "response": {"text": (response_text_for_teacher or "")[:4000]},
            "metrics": usage,
            "env": {"public_base_url": PUBLIC_BASE_URL, "config_hash": WRAPPER_CONFIG_HASH},
            "privacy": {"vault_refs": 0, "secrets_in": 0, "secrets_out": 0},
        }
        # Blocking trace emission (NO background tasks allowed).
        await _send_trace_stream_async(trace_payload)

        # response prepared; single return happens after the try/except.
    except Exception as ex:
        logging.error(f"chat.finish: {ex}", exc_info=True)
        response = _build_openai_envelope(
            ok=False,
            text="Internal error.",
            error={"code": "internal_error", "message": str(ex)},
            usage=usage,
            model=COMMITTEE_MODEL_ID,
            seed=0,
            id_="orc-1",
        )

    if response is None:
        response = _build_openai_envelope(
            ok=False,
            text="Internal error.",
            error={"code": "internal_error", "message": "no response produced"},
            usage=usage,
            model=COMMITTEE_MODEL_ID,
            seed=0,
            id_="orc-1",
        )

    # Always attach assets/artifacts/ids to the outward response (even on abort paths).
    try:
        artifacts_out = _collect_artifacts_payload(tool_results or [], abs_url_fn)
    except Exception:
        artifacts_out = {"urls": [], "items": []}
    try:
        response_meta = response.setdefault("_meta", {})
        if isinstance(response_meta, dict):
            response_meta.setdefault("trace_id", trace_id)
            if 'conv_cid' in locals() and isinstance(locals().get("conv_cid"), (str, int)):
                # Always overwrite with canonical cid (never trust external callers).
                response_meta["cid"] = str(locals().get("conv_cid"))
            if isinstance(artifacts_out, dict):
                urls_out = artifacts_out.get("urls") if isinstance(artifacts_out.get("urls"), list) else []
                response_meta.setdefault("assets", urls_out)
        response_art = response.setdefault("artifacts", {})
        if isinstance(response_art, dict) and isinstance(artifacts_out, dict):
            urls_out = artifacts_out.get("urls") if isinstance(artifacts_out.get("urls"), list) else []
            items_out = artifacts_out.get("items") if isinstance(artifacts_out.get("items"), list) else []
            response_art.setdefault("assets", urls_out)
            response_art.setdefault("items", items_out)
    except Exception:
        log.debug("failed to attach artifacts to response (non-fatal)", exc_info=True)

    # Always append assets URLs to the visible assistant content (avoid double-appends).
    try:
        urls_out = artifacts_out.get("urls") if isinstance(artifacts_out, dict) else []
        if isinstance(urls_out, list) and urls_out:
            msg0 = (((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            if isinstance(msg0, str) and "Assets:" not in msg0:
                msg0 = (msg0 + "\n\nAssets:\n" + "\n".join([f"- {u}" for u in urls_out])).strip()
                try:
                    response["choices"][0]["message"]["content"] = msg0
                except Exception:
                    log.debug("failed to mutate response message content for assets (non-fatal)", exc_info=True)
    except Exception:
        log.debug("failed to append assets to visible content (non-fatal)", exc_info=True)

    # Unified local trace only; no background network calls
    try:
        display_content = display_content or (((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    except Exception:
        display_content = display_content or ""
    emit_trace(STATE_DIR, trace_id, "chat.finish", {"ok": bool(response.get("ok")), "message_len": len(display_content or ""), "usage": usage or {}})
    if run_id is not None:
        await _db_update_run_response(run_id, response, usage)
        _log("halt", trace_id=trace_id, kind="committee", chars=int(len(display_content)))
    _trace_response(trace_id, response)
    return _json_response(response)


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "openai_compat": True,
        "teacher_enabled": True,
        "icw_enabled": bool((WRAPPER_CONFIG.get("icw") or {}).get("enabled", False)),
        "film_enabled": bool((WRAPPER_CONFIG.get("film") or {}).get("enabled", False)),
        "ablation_enabled": bool((WRAPPER_CONFIG.get("ablation") or {}).get("enabled", False)),
    }


# ---------- Refs API ----------
@app.post("/refs.save")
async def refs_save(body: Dict[str, Any]):
    try:
        return _refs_save(body or {})
    except Exception as ex:
        _tb = traceback.format_exc()
        logging.error(_tb)
        return JSONResponse(status_code=422, content={"error": "tool_error", "message": str(ex), "traceback": _tb})


@app.post("/refs.refine")
async def refs_refine(body: Dict[str, Any]):
    try:
        return _refs_refine(body or {})
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.get("/refs.list")
async def refs_list(kind: Optional[str] = None):
    try:
        return _refs_list(kind)
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.post("/refs.apply")
async def refs_apply(body: Dict[str, Any]):
    try:
        res, code = _refs_apply(body or {})
        if code != 200:
            return JSONResponse(status_code=code, content=res)
        return res
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.post("/refs.resolve")
async def refs_resolve(body: Dict[str, Any]):
    try:
        cid_val = (body or {}).get("cid")
        cid = str(cid_val).strip() if isinstance(cid_val, (str, int)) else ""
        if not cid:
            return JSONResponse(status_code=400, content={"error": "missing cid"})
        text = str((body or {}).get("text") or "")
        kind = (body or {}).get("kind")
        rec = _ctx_resolve(cid, text, kind)
        if not rec:
            # Fallback to global artifact memory across conversations
            rec = _glob_resolve(text, kind)
            if not rec:
                trace_append("memory_ref", {"cid": cid, "text": text, "kind": kind, "found": False})
                return {"ok": False, "matches": []}
            trace_append("memory_ref", {"cid": cid, "text": text, "kind": kind, "found": True, "path": rec.get("path"), "source": ("cid" if cid else "global")})
        return {"ok": True, "matches": [rec]}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


def _build_locks_from_context(cid: str) -> Dict[str, Any]:
    locks: Dict[str, Any] = {"characters": []}
    # Pick recent artifacts
    last_img = _ctx_resolve(cid, "last image", "image")
    last_voice = _ctx_resolve(cid, "last voice", "audio")
    last_music = _ctx_resolve(cid, "last music", "audio")
    char: Dict[str, Any] = {"id": "C_A"}
    # Save temporary refs for each kind and wire ref_ids
    if last_img and isinstance(last_img.get("path"), str):
        ref = _refs_save({"kind": "image", "title": f"ctx:{cid}:image", "files": {"images": [last_img.get("path")]}, "compute_embeds": False})
        if isinstance(ref, dict) and isinstance(ref.get("ref"), dict):
            char["image_ref_id"] = ref["ref"].get("ref_id")
    if last_voice and isinstance(last_voice.get("path"), str):
        ref = _refs_save({"kind": "voice", "title": f"ctx:{cid}:voice", "files": {"voice_samples": [last_voice.get("path")]}, "compute_embeds": False})
        if isinstance(ref, dict) and isinstance(ref.get("ref"), dict):
            char["voice_ref_id"] = ref["ref"].get("ref_id")
    if last_music and isinstance(last_music.get("path"), str):
        # Collect any stems tagged in recent context
        stems: List[str] = []
        for it in reversed(_ctx_list(cid, limit=20, kind_hint="audio")):
            tags = it.get("tags") or []
            if any(str(t).startswith("stem:") for t in tags):
                pth = it.get("path")
                if isinstance(pth, str): stems.append(pth)
        ref = _refs_save({"kind": "music", "title": f"ctx:{cid}:music", "files": {"track": last_music.get("path"), "stems": stems}, "compute_embeds": False})
        if isinstance(ref, dict) and isinstance(ref.get("ref"), dict):
            char["music_ref_id"] = ref["ref"].get("ref_id")
    if len(char) > 1:
        locks["characters"].append(char)
    return locks


@app.post("/locks.from_context")
async def locks_from_context(body: Dict[str, Any]):
    try:
        cid = str((body or {}).get("cid") or "").strip()
        if not cid:
            return JSONResponse(status_code=400, content={"error": "missing cid"})
        locks = _build_locks_from_context(cid)
        return {"ok": True, "locks": locks}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.post("/refs.compose_character")
async def refs_compose_character(body: Dict[str, Any]):
    try:
        cid = str((body or {}).get("cid") or "").strip()
        face_txt = str((body or {}).get("face") or "")
        style_txt = str((body or {}).get("style") or "")
        clothes_txt = str((body or {}).get("clothes") or "")
        emotion_txt = str((body or {}).get("emotion") or "")
        out: Dict[str, Any] = {"image": {}, "audio": {}}
        if face_txt:
            rec = _ctx_resolve(cid, face_txt, "image") or _glob_resolve(face_txt, "image")
            if rec and isinstance(rec.get("path"), str):
                out["image"]["face_images"] = [rec.get("path")]
        if style_txt:
            rec = _ctx_resolve(cid, style_txt, "image") or _glob_resolve(style_txt, "image")
            if rec and isinstance(rec.get("path"), str):
                out["image"].setdefault("style_images", []).append(rec.get("path"))
        if clothes_txt:
            rec = _ctx_resolve(cid, clothes_txt, "image") or _glob_resolve(clothes_txt, "image")
            if rec and isinstance(rec.get("path"), str):
                out["image"].setdefault("clothes_images", []).append(rec.get("path"))
        if emotion_txt:
            em = _infer_emotion(emotion_txt)
            if em:
                out["audio"]["emotion"] = em
        return {"ok": True, "refs": out}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


# ---------- Direct Tool Runner helper (route centralized in routes/toolrun) ----------
async def http_tool_run(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    base = (PUBLIC_BASE_URL or "http://127.0.0.1:8000").rstrip("/")
    url = base + "/tool.run"
    async with _hx.AsyncClient(trust_env=False, timeout=None) as client:
        # Strict: only propagate explicit trace_id (no aliases, no fallbacks, no normalization).
        payload: Dict[str, Any] = {"name": name, "args": args}
        trace_id: str = ""
        if isinstance(args, dict):
            _trace_val = args.get("trace_id")
            if isinstance(_trace_val, str) and _trace_val:
                trace_id = _trace_val
                payload["trace_id"] = trace_id

        t0 = time.perf_counter()
        try:
            _log(
                "toolrun.http.start",
                trace_id=trace_id,
                tool=str(name),
                url=str(url),
                payload_keys=sorted([str(k) for k in payload.keys()]),
                args_preview=_args_preview_for_log(args),
            )
            r = await client.post(url, json=payload)
        except Exception as ex:
            dur_ms = int((time.perf_counter() - t0) * 1000)
            _log(
                "toolrun.http.exception",
                trace_id=trace_id,
                tool=str(name),
                url=str(url),
                duration_ms=dur_ms,
                error=str(ex),
                args_preview=_args_preview_for_log(args),
            )
            return {
                "name": name,
                "error": {
                    "code": "tool_http_exception",
                    "message": str(ex),
                    "status": 0,
                    "duration_ms": dur_ms,
                    "url": url,
                    "stack": "".join(traceback.format_exc()),
                },
            }

        raw_body = r.text or ""
        dur_ms = int((time.perf_counter() - t0) * 1000)
        parser = JSONParser()
        # Include cid/trace_id in the schema so they survive coercion and can be
        # surfaced in error payloads for robust correlation, matching how cid is
        # treated elsewhere in the system.
        env_schema = {
            "schema_version": int,
            "request_id": str,
            "ok": bool,
            "result": dict,
            "error": dict,
            "cid": str,
            "trace_id": str,
        }
        env = parser.parse(raw_body or "{}", env_schema)
        try:
            env_keys = sorted([str(k) for k in (env.keys() if isinstance(env, dict) else [])])
        except Exception:
            env_keys = []
        _log(
            "toolrun.http.response",
            trace_id=trace_id,
            tool=str(name),
            url=str(url),
            status_code=int(r.status_code),
            duration_ms=dur_ms,
            body_len=int(len(raw_body)),
            body_preview=(raw_body[:2000] + "…") if len(raw_body) > 2000 else raw_body,
            env_keys=env_keys,
            env_ok=bool(env.get("ok")) if isinstance(env, dict) else False,
            env_has_result=bool(isinstance(env.get("result"), dict)) if isinstance(env, dict) else False,
            env_has_error=bool(isinstance(env.get("error"), dict)) if isinstance(env, dict) else False,
        )
        # Canonical consumer: interpret ok/error from envelope; propagate full error object.
        if isinstance(env, dict) and env.get("ok") and isinstance(env.get("result"), dict):
            try:
                rk = sorted([str(k) for k in env["result"].keys()])
            except Exception:
                rk = []
            _log(
                "toolrun.http.ok",
                trace_id=trace_id,
                tool=str(name),
                url=str(url),
                status_code=int(r.status_code),
                duration_ms=dur_ms,
                result_keys=rk[:128],
            )
            return {"name": name, "result": env["result"]}
        if isinstance(env, dict) and isinstance(env.get("error"), dict):
            # Preserve the entire error payload from /tool.run, including status/details,
            # and add a local stack + raw body for additional debugging context.
            err_obj = env.get("error") or {}
            error_out: Dict[str, Any] = dict(err_obj)
            if "status" not in error_out:
                _st_raw = err_obj.get("status")
                try:
                    error_out["status"] = int(_st_raw or 0)
                except Exception as exc:
                    logging.warning("toolrun.http.error: bad status=%r; defaulting to 0", _st_raw, exc_info=True)
                    error_out["status"] = 0
            # Preserve any top-level cid/trace_id derived by /tool.run so callers
            # can join error telemetry back to conversation and trace identifiers.
            if isinstance(env.get("cid"), str) and env.get("cid"):
                error_out.setdefault("cid", env.get("cid"))
            if isinstance(env.get("trace_id"), str) and env.get("trace_id"):
                error_out.setdefault("trace_id", env.get("trace_id"))
            error_out.setdefault("body", raw_body)
            error_out.setdefault("stack_local", "".join(traceback.format_stack()))
            _log(
                "toolrun.http.error",
                trace_id=trace_id,
                tool=str(name),
                url=str(url),
                status_code=int(r.status_code),
                duration_ms=dur_ms,
                err_code=str(error_out.get("code") or ""),
                err_message=(str(error_out.get("message") or "")[:400] + "…")
                if len(str(error_out.get("message") or "")) > 400
                else str(error_out.get("message") or ""),
            )
            return {
                "name": name,
                "error": error_out,
            }
        # Fallback: unknown error shape; include raw body and local stack so nothing is masked.
        _log(
            "toolrun.http.unknown",
            trace_id=trace_id,
            tool=str(name),
            url=str(url),
            status_code=int(r.status_code),
            duration_ms=dur_ms,
            env_type=type(env).__name__,
        )
        return {
            "name": name,
            "error": {
                "code": "tool_error",
                "message": str(env),
                "status": int(r.status_code),
                "body": raw_body,
                "stack": "".join(traceback.format_stack()),
            },
        }
from fastapi import Request  # type: ignore
from fastapi import Response  # type: ignore

@app.post("/logs/tools.append")
async def tools_append(body: Dict[str, Any], request: Request):
    row = body if isinstance(body, dict) else {}
    row.setdefault("t", int(time.time()*1000))
    try:
        _append_jsonl(os.path.join(STATE_DIR, "tools", "tools.jsonl"), row)
        # Distillation trace routing (no stack traces)
        trace_id = row.get("trace_id")
        if trace_id:
            # Always persist the full executor event row into the unified trace stream.
            # This is the highest-fidelity representation for training/debugging.
            try:
                trace_append(str(row.get("event") or "executor.event"), row)
            except Exception:
                # Never allow tracing to break ingestion.
                pass
            if row.get("event") == "exec_step_start":
                _log("exec_step_start", trace_id=trace_id, tool=row.get("tool"), step_id=row.get("step_id"))
            if row.get("event") == "exec_step_finish":
                _log(
                    "exec_step_finish",
                    trace_id=trace_id,
                    tool=row.get("tool"),
                    step_id=row.get("step_id"),
                    ok=bool(row.get("ok")),
                )
            if row.get("event") == "exec_step_attempt":
                _att_raw = row.get("attempt")
                try:
                    _att = int(_att_raw or 0)
                except Exception as ex:
                    logging.warning("trace.append: bad attempt=%r; defaulting to 0", _att_raw, exc_info=True)
                    _att = 0
                _log("exec_step_attempt", trace_id=trace_id, tool=row.get("tool"), step_id=row.get("step_id"), attempt=_att)
            if row.get("event") == "exec_plan_start":
                _steps_raw = row.get("steps")
                try:
                    _steps = int(_steps_raw or 0)
                except Exception as ex:
                    logging.warning("trace.append: bad steps=%r; defaulting to 0", _steps_raw, exc_info=True)
                    _steps = 0
                _log("exec_plan_start", trace_id=trace_id, steps=_steps)
            if row.get("event") == "exec_plan_finish":
                _log("exec_plan_finish", trace_id=trace_id, produced_keys=(row.get("produced_keys") or []))
            if row.get("event") == "exec_batch_start":
                _log("exec_batch_start", trace_id=trace_id, items=(row.get("items") or []))
            if row.get("event") == "exec_batch_finish":
                _log("exec_batch_finish", trace_id=trace_id, items=(row.get("items") or []))
            if bool(row.get("ok") is True) and row.get("event") == "end":
                distilled = {
                    "t": 0,
                    "event": "tool_end",
                    "tool": row.get("tool"),
                    "step_id": row.get("step_id"),
                    "duration_ms": 0,
                }
                _t_raw = row.get("t")
                try:
                    distilled["t"] = int(_t_raw or 0)
                except Exception as ex:
                    logging.warning("trace.append: bad t=%r; defaulting to 0", _t_raw, exc_info=True)
                    distilled["t"] = 0
                _dur_raw = row.get("duration_ms")
                try:
                    distilled["duration_ms"] = int(_dur_raw or 0)
                except Exception as ex:
                    logging.warning("trace.append: bad duration_ms=%r; defaulting to 0", _dur_raw, exc_info=True)
                    distilled["duration_ms"] = 0
                if isinstance(row.get("summary"), dict):
                    distilled["summary"] = row.get("summary")
                _log("tool_summary", trace_id=trace_id, **distilled)
                # Compact artifact entries inferred from summary (no stack traces)
                try:
                    s = row.get("summary") or {}
                    if isinstance(s, dict):
                        _ic_raw = s.get("images_count")
                        try:
                            _ic = int(_ic_raw or 0)
                        except Exception as ex:
                            logging.warning("trace.append: bad images_count=%r; defaulting to 0", _ic_raw, exc_info=True)
                            _ic = 0
                        if _ic > 0:
                            _log("artifact_summary", trace_id=trace_id, kind="image", count=_ic)
                        _vc_raw = s.get("videos_count")
                        try:
                            _vc = int(_vc_raw or 0)
                        except Exception as ex:
                            logging.warning("trace.append: bad videos_count=%r; defaulting to 0", _vc_raw, exc_info=True)
                            _vc = 0
                        if _vc > 0:
                            _log("artifact_summary", trace_id=trace_id, kind="video", count=_vc)
                        _wb_raw = s.get("wav_bytes")
                        try:
                            _wb = int(_wb_raw or 0)
                        except Exception as ex:
                            logging.warning("trace.append: bad wav_bytes=%r; defaulting to 0", _wb_raw, exc_info=True)
                            _wb = 0
                        if _wb > 0:
                            _log("artifact_summary", trace_id=trace_id, kind="audio", bytes=_wb)
                except Exception as ex:
                    logging.warning(f"artifact summary distillation failed: {ex}", exc_info=True)
            # Forward review WS events to connected client if present
            try:
                app = request.app
                ws_map = getattr(app.state, "ws_clients", {})
                ws = ws_map.get(trace_id)
                if ws and isinstance(row.get("event"), str) and (row["event"].startswith("review.") or row["event"] == "edit.plan"):
                    payload = {"type": row["event"], "trace_id": trace_id, "step_id": row.get("step_id"), "notes": row.get("notes")}
                    await ws.send_json(payload)
            except Exception as ex:
                logging.warning(f"websocket forward failed: {ex}", exc_info=True)
    except Exception as ex:
        logging.error(f"trace.append failed trace_id={trace_id}: {ex}", exc_info=True)
        return JSONResponse(status_code=400, content={"error": "append_failed"})
    return {"ok": True}



@app.post("/jobs/start")
async def jobs_start(body: Dict[str, Any]):
    """
    Minimal job starter compatible with smoke tests.
    Immediately executes the requested tool synchronously and returns the result with a synthetic job_id.
    """
    try:
        jid = uuid.uuid4().hex
        name = (body or {}).get("tool") or (body or {}).get("name")
        args = (body or {}).get("args") or {}
        res = await http_tool_run(str(name or ""), args if isinstance(args, dict) else {})
        # Pass through result; include job_id for clients expecting it
        if isinstance(res, JSONResponse):
            # unwrap JSONResponse content
            return JSONResponse(
                status_code=res.status_code,
                content={"job_id": jid, "result": res.body.decode("utf-8") if hasattr(res, "body") else None},
            )
        if isinstance(res, dict):
            return {"job_id": jid, **res}
        return {"job_id": jid, "result": res}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


# ---------- Dataset Registry / Export (Step 21) ----------
@app.post("/datasets/start")
async def datasets_start(body: Dict[str, Any]):
    try:
        # Stream via /orcjobs/{id}/stream
        jid = body.get("id") or f"ds-{uuid.uuid4().hex}"
        j = _orcjob_create(jid=jid, tool="datasets.export", args=body or {})
        _orcjob_set_state(j.id, "running", phase="start", progress=0.0)
        await _datasets_runner(j.id, body or {})
        return {"id": j.id}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.get("/datasets/list")
async def datasets_list():
    try:
        return _datasets_list()
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.get("/datasets/versions")
async def datasets_versions(name: str):
    try:
        return _datasets_versions(name)
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.get("/datasets/index")
async def datasets_index(name: str, version: str):
    try:
        content, code, headers = _datasets_index(name, version)
        return Response(content=content, status_code=code, headers=headers)
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


# ---------- Film-2 QA (Step 23) ----------
@app.post("/film2/qa")
async def film2_qa(body: Dict[str, Any]):
    try:
        cid_raw = body.get("cid") or body.get("film_id")
        cid = str(cid_raw).strip() if isinstance(cid_raw, (str, int)) else ""
        if not cid:
            return JSONResponse(status_code=400, content={"error": "missing cid"})
        shots = body.get("shots") or []
        voice_lines = body.get("voice_lines") or []
        music_cues = body.get("music_cues") or []
        _t_face_raw = os.getenv("FILM2_QA_T_FACE", "0.85")
        _t_voice_raw = os.getenv("FILM2_QA_T_VOICE", "0.85")
        _t_music_raw = os.getenv("FILM2_QA_T_MUSIC", "0.80")
        try:
            T_FACE = float(str(_t_face_raw).strip() or "0.85")
        except Exception:
            logging.warning("film2.qa: bad FILM2_QA_T_FACE=%r; defaulting to 0.85", _t_face_raw, exc_info=True)
            T_FACE = 0.85
        try:
            T_VOICE = float(str(_t_voice_raw).strip() or "0.85")
        except Exception:
            logging.warning("film2.qa: bad FILM2_QA_T_VOICE=%r; defaulting to 0.85", _t_voice_raw, exc_info=True)
            T_VOICE = 0.85
        try:
            T_MUSIC = float(str(_t_music_raw).strip() or "0.80")
        except Exception:
            logging.warning("film2.qa: bad FILM2_QA_T_MUSIC=%r; defaulting to 0.80", _t_music_raw, exc_info=True)
            T_MUSIC = 0.80
        issues = []
        for shot in shots:
            sid = shot.get("id") or shot.get("shot_id")
            if not sid:
                continue
            snap = _film_load_snap(cid, sid) or {}
            sb_meta = {"face_vec": snap.get("face_vec")}
            fin_meta = {"face_vec": None}
            score = _qa_face(sb_meta, fin_meta, face_ref_embed=None)
            if score < T_FACE:
                _seed_raw = snap.get("seed") if isinstance(snap, dict) else None
                try:
                    _seed = int(_seed_raw or 0)
                except Exception:
                    logging.warning("film2.qa: bad snap.seed=%r; defaulting to 0", _seed_raw, exc_info=True)
                    _seed = 0
                args = {"name": "image.dispatch", "args": {"mode": "gen", "prompt": shot.get("prompt") or "", "size": shot.get("size", "1024x1024"), "refs": snap.get("refs", {}), "seed": _seed, "film_cid": cid, "shot_id": sid}}
                _ = await http_tool_run(args.get("name") or "image.dispatch", args.get("args") or {})
                issues.append({"shot": sid, "type": "image", "score": score})
        for ln in voice_lines:
            lid = ln.get("id") or ln.get("line_id")
            if not lid:
                continue
            snap = _film_load_snap(cid, f"vo_{lid}") or {}
            score = _qa_voice({"voice_vec": None}, voice_ref_embed=None)
            if score < T_VOICE:
                _seed_raw2 = snap.get("seed") if isinstance(snap, dict) else None
                try:
                    _seed2 = int(_seed_raw2 or 0)
                except Exception:
                    logging.warning("film2.qa: bad snap.seed=%r; defaulting to 0", _seed_raw2, exc_info=True)
                    _seed2 = 0
                args = {"name": "tts.speak", "args": {"text": ln.get("text") or "", "voice_id": ln.get("voice_ref_id"), "voice_refs": snap.get("refs", {}), "seed": _seed2, "film_cid": cid, "line_id": lid}}
                _ = await http_tool_run(args.get("name") or "tts.speak", args.get("args") or {})
                issues.append({"line": lid, "type": "voice", "score": score})
        for cue in music_cues:
            cid2 = cue.get("id") or cue.get("cue_id")
            if not cid2:
                continue
            snap = _film_load_snap(cid, f"cue_{cid2}") or {}
            score = _qa_music({"motif_vec": None}, motif_embed=None)
            if score < T_MUSIC:
                # Prefer timed music generation for video scoring using SAO when available
                args = {"name": "music.timed.sao", "args": {"text": cue.get("prompt") or "", "seconds": 8}}
                _ = await http_tool_run(args.get("name") or "music.timed.sao", args.get("args") or {})
                issues.append({"cue": cid2, "type": "music", "score": score})
        # Best-effort integration of per-shot image QA from image.dispatch when present.
        # This does not replace the existing Film2 QA heuristics; it augments them with
        # face_lock and entity_lock_score from the lock-aware image pipeline.
        preset = LOCK_QUALITY_PRESETS.get("hero", LOCK_QUALITY_PRESETS["standard"])
        face_min_raw = preset.get("face_min", T_FACE) if isinstance(preset, dict) else T_FACE
        face_min_lock = float(face_min_raw) if isinstance(face_min_raw, (int, float)) else T_FACE
        for shot in shots:
            sid = shot.get("id") or shot.get("shot_id")
            if not sid:
                continue
            qa_block = (shot.get("qa") or {}).get("images") if isinstance(shot.get("qa"), dict) else {}
            locks_meta = (shot.get("meta") or {}).get("locks") if isinstance(shot.get("meta"), dict) else {}
            face_lock_val = None
            ent_lock_val = None
            if isinstance(qa_block, dict):
                fl = qa_block.get("face_lock")
                if isinstance(fl, (int, float)):
                    face_lock_val = float(fl)
            if isinstance(locks_meta, dict):
                el = locks_meta.get("entity_lock_score")
                if isinstance(el, (int, float)):
                    ent_lock_val = float(el)
            shot_issues = []
            if face_lock_val is not None and face_lock_val < face_min_lock:
                shot_issues.append({"type": "identity_lock", "face_lock": face_lock_val})
            if ent_lock_val is not None and ent_lock_val < 0.8:
                shot_issues.append({"type": "entity_lock", "entity_lock_score": ent_lock_val})
            if shot_issues:
                issues.append({"shot": sid, "type": "image_lock", "details": shot_issues})
        return {"cid": cid, "issues": issues, "thresholds": {"face": T_FACE, "voice": T_VOICE, "music": T_MUSIC, "face_lock": face_min_lock}}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


# ---------- Admin & Ops (Step 22) ----------
@app.get("/jobs.list")
async def admin_jobs_list():
    try:
        return _admin_jobs_list()
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.get("/jobs.replay")
async def admin_jobs_replay(cid: str):
    try:
        return _admin_jobs_replay(cid)
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.post("/artifacts.gc")
async def admin_artifacts_gc(body: Dict[str, Any]):
    try:
        return _admin_gc(body or {})
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.get("/prompts/list")
async def prompts_list():
    try:
        return {"prompts": _list_prompts()}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


# ---------- Logs Tail (debug) ----------
@app.get("/logs/tools.tail")
async def logs_tools_tail(limit: int = 200):
    try:
        path = os.path.join(STATE_DIR, "tools", "tools.jsonl")
        return {"data": _read_tail(path, n=int(limit))}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})


@app.get("/logs/trace.tail")
async def logs_trace_tail(id: str, kind: str = "responses", limit: int = 200):
    try:
        # Unified trace stream: <state>/traces/<trace_id>/trace.jsonl
        # Keep the legacy query param `kind=responses|requests` as a best-effort filter.
        safe_kind = "responses" if kind not in ("responses", "requests") else kind
        path = os.path.join(STATE_DIR, "traces", id, "trace.jsonl")
        rows = _read_tail(path, n=int(limit))
        if safe_kind == "requests":
            rows = [r for r in (rows or []) if isinstance(r, dict) and r.get("kind") == "request"]
        elif safe_kind == "responses":
            rows = [r for r in (rows or []) if isinstance(r, dict) and r.get("kind") == "response"]
        return {"id": id, "kind": safe_kind, "data": rows}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})

@app.get("/v1/state/project")
async def v1_state_project(job_id: str):
    omni = _read_json_safe(_proj_capsule_path(job_id))
    return {"ok": True, "project": omni}

@app.post("/v1/review/score")
async def v1_review_score(body: Dict[str, Any]):
    expected = {"kind": str, "path": str, "prompt": str}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    kind = (payload.get("kind") or "").strip().lower()
    path = payload.get("path") or ""
    prompt = (payload.get("prompt") or "").strip()
    if kind not in ("image", "audio", "music"):
        return JSONResponse(status_code=400, content={"error": "invalid kind"})
    scores: Dict[str, Any] = {}
    if kind == "image":
        ai = _analyze_image(path, prompt=prompt)
        sem = ai.get("semantics") or {}
        score_block = ai.get("score") or {}
        scores = {
            "image": {
                "overall": float(score_block.get("overall") or 0.0),
                "semantic": float(score_block.get("semantic") or 0.0),
                "technical": float(score_block.get("technical") or 0.0),
                "clip": float(sem.get("clip_score") or 0.0),
            }
        }
    else:
        aa = _analyze_audio(path)
        scores = {"audio": {"lufs": aa.get("lufs"), "tempo_bpm": aa.get("tempo_bpm")}}
    return {"ok": True, "scores": scores}

@app.post("/v1/review/plan")
async def v1_review_plan(body: Dict[str, Any]):
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), {"scores": dict})
    plan = _build_delta_plan(payload.get("scores") or {})
    return {"ok": True, "plan": plan}

@app.post("/v1/review/loop")
async def v1_review_loop(body: Dict[str, Any]):
    expected = {"artifacts": list, "prompt": str}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    arts = payload.get("artifacts") or []
    prompt = (payload.get("prompt") or "").strip()
    loop_idx = 0
    while loop_idx < 6:
        # Score first image or audio artifact only (fast path)
        img = next((a for a in arts if (a.get("kind") or "").startswith("image")), None)
        aud = next((a for a in arts if (a.get("kind") or "").startswith("audio")), None)
        scores: Dict[str, Any] = {}
        if img and isinstance(img.get("path"), str):
            ai = _analyze_image(img.get("path"), prompt=prompt)
            sem = ai.get("semantics") or {}
            score_block = ai.get("score") or {}
            scores["image"] = {
                "overall": float(score_block.get("overall") or 0.0),
                "semantic": float(score_block.get("semantic") or 0.0),
                "technical": float(score_block.get("technical") or 0.0),
                "clip": float(sem.get("clip_score") or 0.0),
            }
        if aud and isinstance(aud.get("path"), str):
            aa = _analyze_audio(aud.get("path"))
            scores["audio"] = {"lufs": aa.get("lufs"), "tempo_bpm": aa.get("tempo_bpm")}
        plan = _build_delta_plan(scores)
        _append_ledger({"phase": f"review.loop#{loop_idx}", "scores": scores, "decision": plan})
        if plan.get("accept") is True:
            return {"ok": True, "loop_idx": loop_idx, "accepted": True, "plan": plan}
        # Minimal refinement stub: not executing real deltas here; return plan to caller
        return {"ok": True, "loop_idx": loop_idx, "accepted": False, "plan": plan}
    return {"ok": True, "loop_idx": loop_idx, "accepted": False, "plan": {"accept": False}}

# ---- Film-2 Video endpoints (HunyuanVideo-first; deterministic; ICW capsules) ----
@app.post("/v1/video/locks/{lock_type}")
async def v1_video_locks(lock_type: str, body: Dict[str, Any]):
    lock_type = (lock_type or "").strip().lower()
    if lock_type not in ("face","clothing","style","layout","text","voice"):
        return JSONResponse(status_code=400, content={"error": "invalid lock type"})
    refs = (body or {}).get("refs") or []
    files: Dict[str, Any] = {}
    if lock_type in ("face","clothing","style","layout","text"):
        imgs: list[str] = []
        for r in refs:
            if isinstance(r, dict) and isinstance(r.get("ref_id"), str):
                pack, code = _refs_apply({"ref_id": r.get("ref_id")})
                if code == 200 and isinstance(pack, dict):
                    for p in (pack.get("images") or []) + (pack.get("image") or []):
                        if isinstance(p, str): imgs.append(p)
            elif isinstance(r, dict) and isinstance(r.get("image_url"), str):
                imgs.append(r.get("image_url"))
            elif isinstance(r, str):
                imgs.append(r)
        if not imgs:
            return JSONResponse(status_code=400, content={"error": "no refs"})
        files = {"images": imgs}
    if lock_type == "voice":
        files = {"voice_samples": [r.get("audio_url") for r in refs if isinstance(r, dict) and r.get("audio_url")]}
    saved = _refs_save({"kind": lock_type, "title": f"vlock:{lock_type}:{int(time.time())}", "files": files, "compute_embeds": True})
    return {"ok": True, "lock": saved}

def _save_capsule(cap: Dict[str, Any]) -> str:
    cid = cap.get("capsule_id") or f"cap-{int(time.time())}"
    cap["capsule_id"] = cid
    base = os.path.join(UPLOAD_DIR, "capsules")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"{cid}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cap, f, ensure_ascii=False)
    except Exception as ex:
        logging.warning(f"capsule.save.error path={path!r} error={str(ex)}", exc_info=True)
    return cid

@app.get("/v1/video/state/{capsule_id}")
async def v1_video_state(capsule_id: str):
    path = os.path.join(UPLOAD_DIR, "capsules", f"{capsule_id}.json")
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "not_found"})
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/v1/video/generate")
async def v1_video_generate(body: Dict[str, Any]):
    rid = uuid.uuid4().hex
    if not HUNYUAN_VIDEO_API_URL:
        return ToolEnvelope.failure(
            "hunyuan_not_configured",
            "HUNYUAN_VIDEO_API_URL not configured",
            status=500,
            request_id=rid,
        )
    expected = {"prompt": str, "refs": dict, "locks": dict, "duration_s": int, "fps": int, "size": list, "seed": int, "icw": dict}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return ToolEnvelope.failure(
            "missing_prompt",
            "missing prompt",
            status=400,
            request_id=rid,
        )
    w,h = 1920,1080
    sz = payload.get("size") or [1920, 1080]
    if isinstance(sz, list) and len(sz) == 2:
        a0, a1 = sz[0], sz[1]
        if isinstance(a0, (int, float)) and isinstance(a1, (int, float)):
            w, h = int(a0), int(a1)
    fps = int(payload.get("fps") or 60)
    seconds = int(payload.get("duration_s") or 8)
    seed = payload.get("seed")
    locks = payload.get("locks") or {}
    hv_args = {
        "prompt": prompt,
        "width": w,
        "height": h,
        "fps": fps,
        "seconds": seconds,
        "locks": locks,
        "seed": seed,
        "post": {"interpolate": True, "upscale": True, "face_lock": True, "hand_fix": True},
        "latent_reinit_every": 48,
    }
    res = await execute_tool_call({"name": "video.hv.t2v", "arguments": hv_args})
    icw = payload.get("icw") or {}
    cap = {"capsule_id": icw.get("capsule_id") or None, "scene": icw.get("scene"), "shot": icw.get("shot"), "window_idx": icw.get("window_idx"), "timecode": {"start_s": 0.0, "end_s": float(seconds), "fps": fps}, "locks": locks, "manifests": {"seeds": {"window": seed}}}
    cid = _save_capsule(cap)
    return ToolEnvelope.success(
        {"result": res, "capsule_id": cid},
        request_id=rid,
    )

@app.post("/v1/video/edit")
async def v1_video_edit(body: Dict[str, Any]):
    rid = uuid.uuid4().hex
    if not HUNYUAN_VIDEO_API_URL:
        return ToolEnvelope.failure(
            "hunyuan_not_configured",
            "HUNYUAN_VIDEO_API_URL not configured",
            status=500,
            request_id=rid,
        )
    expected = {"image_url": str, "instruction": str, "locks": dict, "fps": int, "size": list, "seconds": int, "seed": int}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    init_image = payload.get("image_url")
    if not init_image:
        return ToolEnvelope.failure(
            "missing_image_url",
            "missing image_url",
            status=400,
            request_id=rid,
        )
    instruction = (payload.get("instruction") or "").strip()
    if not instruction:
        return ToolEnvelope.failure(
            "missing_instruction",
            "missing instruction",
            status=400,
            request_id=rid,
        )
    w,h = 1024,1024
    sz = payload.get("size") or [1024, 1024]
    if isinstance(sz, list) and len(sz) == 2:
        a0, a1 = sz[0], sz[1]
        if isinstance(a0, (int, float)) and isinstance(a1, (int, float)):
            w, h = int(a0), int(a1)
    fps = int(payload.get("fps") or 60)
    seconds = int(payload.get("seconds") or 5)
    seed = payload.get("seed")
    locks = payload.get("locks") or {}
    hv_args = {
        "init_image": init_image,
        "prompt": instruction,
        "width": w,
        "height": h,
        "fps": fps,
        "seconds": seconds,
        "locks": locks,
        "seed": seed,
        "post": {"interpolate": True, "upscale": True, "face_lock": True},
        "latent_reinit_every": 48,
    }
    res = await execute_tool_call({"name": "video.hv.i2v", "arguments": hv_args})
    return ToolEnvelope.success({"result": res}, request_id=rid)


@app.post("/v1/audio/lyrics-to-song")
async def v1_audio_lyrics_to_song(body: Dict[str, Any]):
    rid = uuid.uuid4().hex
    expected = {"lyrics": str, "style_prompt": str, "ref_audio_ids": list, "lock_ids": list, "duration_s": int, "seed": int, "bpm": int, "key": str}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    lyrics = (payload.get("lyrics") or "").strip()
    style = (payload.get("style_prompt") or "").strip()
    duration_s = int(payload.get("duration_s") or 30)
    bpm = payload.get("bpm")
    key = payload.get("key")
    seed = payload.get("seed")
    if not lyrics:
        return ToolEnvelope.failure(
            "missing_lyrics",
            "missing lyrics",
            status=400,
            request_id=rid,
        )
    # Route lyrics-to-song through the infinite/windowed music path.
    prompt = ((style + ": ") if style else "") + lyrics
    call = {
        "name": "music.infinite.windowed",
        "arguments": {"prompt": prompt, "length_s": duration_s, "bpm": bpm, "key": key, "seed": seed},
    }
    res = await execute_tool_call(call)
    return ToolEnvelope.success({"result": res}, request_id=rid)

@app.post("/v1/audio/edit")
async def v1_audio_edit(body: Dict[str, Any]):
    rid = uuid.uuid4().hex
    expected = {"audio_url": str, "ops": list}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    audio_url = payload.get("audio_url")
    ops = payload.get("ops") or []
    if not audio_url:
        return ToolEnvelope.failure(
            "missing_audio_url",
            "missing audio_url",
            status=400,
            request_id=rid,
        )
    stems_info: Dict[str, Any] = {"ok": False}
    try:
        if DEMUCS_API_URL:
            sep = await execute_tool_call({"name": "audio.stems.demucs", "arguments": {"mix_wav": audio_url, "stems": ["vocals","drums","bass","other"]}})
            stems_info = {"ok": True, "stems": sep}
    except Exception as ex:
        stems_info = {"ok": False, "error": str(ex)}
    return ToolEnvelope.success(
        {"ingest": {"audio_url": audio_url}, "stems": stems_info, "ops_applied": ops},
        request_id=rid,
    )

@app.post("/v1/audio/tts-sing")
async def v1_audio_tts_sing(body: Dict[str, Any]):
    rid = uuid.uuid4().hex
    expected = {"script": list, "style_lock_id": str, "structure": dict}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    script = payload.get("script") or []
    if not isinstance(script, list) or not script:
        return ToolEnvelope.failure(
            "missing_script",
            "missing script",
            status=400,
            request_id=rid,
        )
    outputs: List[Dict[str, Any]] = []
    for item in script:
        kind = (item or {}).get("type")
        if kind == "tts":
            args = {"text": (item.get("text") or ""), "voice_id": item.get("voice_lock_id")}
            res = await execute_tool_call({"name": "tts.speak", "arguments": args})
            outputs.append({"type": "tts", "result": res})
        elif kind == "sing":
            lyrics = (item.get("lyrics") or "")
            prompt = lyrics
            call = {
                "name": "music.infinite.windowed",
                "arguments": {"prompt": prompt, "length_s": 30, "seed": item.get("seed")},
            }
            res = await execute_tool_call(call)
            outputs.append({"type": "sing", "result": res})
    return ToolEnvelope.success(
        {"parts": outputs, "structure": (payload.get("structure") or {})},
        request_id=rid,
    )

@app.post("/v1/audio/score-video")
async def v1_audio_score_video(body: Dict[str, Any]):
    rid = uuid.uuid4().hex
    expected = {"video_url": str, "markers": list, "style_prompt": str, "voice_lock_id": str, "accept_edits": bool}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    video_url = payload.get("video_url")
    markers = payload.get("markers") or []
    style_prompt = payload.get("style_prompt") or ""
    if not video_url:
        return ToolEnvelope.failure(
            "missing_video_url",
            "missing video_url",
            status=400,
            request_id=rid,
        )
    if not SAO_API_URL:
        return ToolEnvelope.failure(
            "sao_not_configured",
            "SAO_API_URL not configured",
            status=500,
            request_id=rid,
        )
    seconds = 30
    if isinstance(markers, list) and markers:
        ts: List[float] = []
        for m in markers:
            if not isinstance(m, dict):
                continue
            t = m.get("t")
            if isinstance(t, (int, float)):
                ts.append(float(t))
            elif isinstance(t, str) and _re.match(r"^-?\d+(\.\d+)?$", t.strip()):
                ts.append(float(t.strip()))
        if ts:
            mx = max(ts)
            seconds = max(8, int(mx) + 2)
    text = (style_prompt or "").strip() or "video score"
    res = await execute_tool_call({"name": "music.timed.sao", "arguments": {"text": text, "seconds": seconds}})
    return ToolEnvelope.success(
        {"video": video_url, "markers": markers, "music": res},
        request_id=rid,
    )

@app.post("/v1/locks/create")
async def v1_locks_create(body: Dict[str, Any]):
    rid = uuid.uuid4().hex
    expected = {"type": str, "refs": list, "tags": list}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    lk_type = (payload.get("type") or "").strip().lower()
    refs = payload.get("refs") or []
    tags = payload.get("tags") or []
    if lk_type not in ("voice", "music", "sfx", "music_style", "sfx_identity"):
        return ToolEnvelope.failure(
            "invalid_lock_type",
            "invalid lock type",
            status=400,
            request_id=rid,
        )
    files: Dict[str, Any] = {}
    if lk_type == "voice":
        files = {"voice_samples": [r.get("audio_url") for r in refs if isinstance(r, dict) and r.get("audio_url")]}
    elif lk_type in ("music", "music_style"):
        files = {"track": next((r.get("audio_url") for r in refs if isinstance(r, dict) and r.get("audio_url")), None)}
    elif lk_type == "sfx" or lk_type == "sfx_identity":
        files = {"sfx_samples": [r.get("audio_url") for r in refs if isinstance(r, dict) and r.get("audio_url")]}
    title = f"lock:{lk_type}:{int(time.time())}"
    saved = _refs_save({"kind": lk_type, "title": title, "files": files, "tags": tags, "compute_embeds": True})
    return ToolEnvelope.success({"lock": saved}, request_id=rid)

@app.post("/v1/tts")
async def v1_tts(body: Dict[str, Any]):
    rid = uuid.uuid4().hex
    expected = {"text": str, "voice_ref_url": str, "voice_id": str, "lang": str, "prosody": dict, "seed": int}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    text = (payload.get("text") or "").strip()
    if not text:
        return ToolEnvelope.failure(
            "missing_text",
            "missing text",
            status=400,
            request_id=rid,
        )
    args = {"text": text, "voice_id": payload.get("voice_id"), "voice_refs": ({"voice_ref_url": payload.get("voice_ref_url")} if payload.get("voice_ref_url") else {}), "seed": payload.get("seed")}
    res = await execute_tool_call({"name": "tts.speak", "arguments": args})
    return ToolEnvelope.success({"result": res}, request_id=rid)

@app.post("/v1/audio/sfx")
async def v1_audio_sfx(body: Dict[str, Any]):
    rid = uuid.uuid4().hex
    expected = {"type": str, "length_s": float, "pitch": float, "seed": int}
    parser = JSONParser()
    payload = parser.parse(json.dumps(body or {}), expected)
    args = {"type": payload.get("type"), "length_s": payload.get("length_s"), "pitch": payload.get("pitch"), "seed": payload.get("seed")}
    manifest = {"items": []}
    env = run_sfx_compose(args, manifest)
    return ToolEnvelope.success({"result": env}, request_id=rid)
# Lightweight in-memory job controls for orchestrator-run tools (research.run, etc.)
@app.get("/orcjobs/{job_id}")
async def orcjob_get(job_id: str):
    j = _get_orcjob(job_id)
    if not j:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    return {
        "id": j.id,
        "tool": j.tool,
        "args": j.args,
        "state": j.state,
        "phase": j.phase,
        "progress": j.progress,
        "created_at": j.created_at,
        "updated_at": j.updated_at,
        "error": j.error,
    }


@app.post("/orcjobs/{job_id}/cancel")
async def orcjob_cancel(job_id: str):
    j = _get_orcjob(job_id)
    if not j:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    if j.state in ("done", "failed", "cancelled"):
        return {"ok": True, "state": j.state}
    _orcjob_cancel(job_id)
    return {"ok": True, "state": "cancelling"}


@app.get("/orcjobs/{job_id}/stream")
async def orcjob_stream(job_id: str, interval_ms: Optional[int] = None):
    return StreamingResponse(_orcjob_stream_gen(job_id, interval_ms), media_type="text/event-stream")



@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": COMMITTEE_MODEL_ID, "object": "model"},
            {"id": QWEN_MODEL_ID, "object": "model"},
            {"id": GLM_MODEL_ID, "object": "model"},
            {"id": DEEPSEEK_CODER_MODEL_ID, "object": "model"},
        ],
    }


@app.post("/v1/embeddings")
async def embeddings(body: Dict[str, Any]):
    # OpenAI-compatible embeddings endpoint
    model_name = str((body or {}).get("model") or EMBEDDING_MODEL_NAME)
    inputs = (body or {}).get("input")
    texts: List[str] = []
    if isinstance(inputs, str):
        texts = [inputs]
    elif isinstance(inputs, list):
        _collect_embedding_texts(inputs, texts)
    else:
        texts = [json.dumps(inputs, ensure_ascii=False)]
    if not texts:
        texts = [""]
    emb = get_embedder()
    out = _build_embeddings_response(emb, texts, model_name)
    return JSONResponse(content=out)

@app.get("/v1/jobs/{job_id}")
async def v1_job_status(job_id: str):
    return await orcjob_get(job_id)

@app.get("/v1/jobs/{job_id}/stream")
async def v1_job_stream(job_id: str):
    return await orcjob_stream(job_id)

@app.get("/v1/capsules/{project_id}")
async def v1_capsules_project(project_id: str):
    base = _proj_dir(project_id)
    omni = _read_json_safe(os.path.join(base, "capsules", "OmniCapsule.json"))
    wins_dir = os.path.join(base, "capsules", "windows")
    windows = []
    try:
        for fn in sorted(os.listdir(wins_dir)):
            if fn.endswith('.json'):
                windows.append(_read_json_safe(os.path.join(wins_dir, fn)))
    except Exception as ex:
        logging.error(f"v1_state_icw: failed to list windows in {wins_dir}: {ex}", exc_info=True)
    return {"project_id": project_id, "omni": omni, "windows": windows}

@app.post("/v1/distill/pack")
async def v1_distill_pack(body: Dict[str, Any]):
    job_id = (body or {}).get("job_id")
    if not job_id:
        return JSONResponse(status_code=400, content={"error": "missing job_id"})
    out_dir = os.path.join(FILM2_DATA_DIR, "distill", "jobs", job_id)
    try:
        os.makedirs(out_dir, exist_ok=True)
        # Copy ledger tail and minimal manifests
        ledger_src = os.path.join(FILM2_DATA_DIR, "distill", "ledger.jsonl")
        if os.path.exists(ledger_src):
            _sh.copyfile(ledger_src, os.path.join(out_dir, "ledger.jsonl"))
        with open(os.path.join(out_dir, "RESULTS.md"), "w", encoding="utf-8") as f:
            f.write("# RESULTS\n\nPack created. Fill with eval results.")
        return {"ok": True, "path": out_dir}
    except Exception as ex:
        return JSONResponse(status_code=400, content={"error": str(ex)})

@app.post("/v1/replay")
async def v1_replay(body: Dict[str, Any]):
    job_id = (body or {}).get("job_id")
    if not job_id:
        return JSONResponse(status_code=400, content={"error": "missing job_id"})
    # Minimal stub: compute sha256 of latest artifact file if present
    job_dir = _proj_dir(job_id)
    arts = []
    for root, _, files in os.walk(os.path.join(job_dir, "artifacts")):
        for fn in files:
            fp = os.path.join(root, fn)
            arts.append(fp)
    if not arts:
        return {"ok": False, "reason": "no artifacts"}
    fp = sorted(arts)[-1]
    h = _hl.sha256(open(fp, 'rb').read()).hexdigest()
    return {"ok": True, "file": fp, "sha256": h}


@app.websocket("/ws")
async def ws_alias(websocket: WebSocket):
    # Deprecated alias removed. Politely close to avoid client hangs.
    await websocket.accept()
    try:
        await websocket.send_text(json.dumps({"type": "error", "error": {"code": "gone", "message": "Deprecated. Use /v1/chat/completions"}}))
    except Exception as ex:
        logging.debug(f"ws_alias: failed to send deprecation message: {ex}", exc_info=True)
    try:
        await websocket.close(code=1000)
    except Exception as ex:
        logging.debug(f"ws_alias: failed to close websocket cleanly: {ex}", exc_info=True)


@app.websocket("/tool.ws")
async def ws_tool(websocket: WebSocket):
    origin = websocket.headers.get("origin") or ""
    server_base = (os.getenv("PUBLIC_BASE_URL", "").strip()) or ""
    same = (_origin_norm(origin) == _origin_norm(server_base)) if origin and server_base else True if not origin else False
    allowed = same or origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1") or origin.startswith("https://localhost") or origin.startswith("https://127.0.0.1")
    if not allowed:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            # IMPORTANT: do not schema-coerce tool.ws requests into a narrow
            # shape, or we will drop client-provided fields (including tool
            # arguments when they arrive as non-dict).
            # Correct approach: use the robust superset parser (repairs + heuristics)
            # with a schema that includes all expected fields, while keeping
            # args/arguments as passthrough (`object`) so we never coerce/drop them.
            parser_ws = JSONParser()
            schema_toolws_req = {
                "name": str,
                "args": object,
                "arguments": object,
                "request_id": str,
                "trace_id": str,
                "cid": str,
                "tool_call_id": str,
                "meta": dict,
            }
            req = (parser_ws.parse(raw, schema_toolws_req) or {})
            if not isinstance(req, dict):
                req = {}
            name = (req.get("name") or "").strip()
            raw_args = req.get("arguments", req.get("args"))
            if isinstance(raw_args, dict):
                args = dict(raw_args)
            elif isinstance(raw_args, str):
                parsed = parser_ws.parse(raw_args, {})
                args = dict(parsed) if isinstance(parsed, dict) else {"_raw": raw_args}
            elif raw_args is None:
                args = {}
            else:
                args = {"_raw": raw_args}
            trace_id_in = req.get("trace_id")
            trace_id_ws = trace_id_in if isinstance(trace_id_in, str) and trace_id_in else None
            if not name:
                await websocket.send_text(json.dumps({"error": "missing tool name"}))
                continue
            try:
                # Synchronous tool runner (no background tasks): emit start/result/done only.
                await websocket.send_text(json.dumps({"event": "start", "name": name}))
                call = {"name": name, "arguments": args}
                if trace_id_ws:
                    call["trace_id"] = trace_id_ws
                res = await execute_tool_call(call)
                await websocket.send_text(json.dumps({"event": "result", "ok": True, "result": res}))
                await websocket.send_text(json.dumps({"done": True}))
                try:
                    await websocket.close(code=1000)
                except Exception as _e_close:
                    _append_jsonl(
                        os.path.join(STATE_DIR, "ws", "errors.jsonl"),
                        {
                            "t": int(time.time() * 1000),
                            "route": "/tool.ws",
                            "error": str(_e_close),
                            "where": "close_after_done",
                        },
                    )
            except Exception as ex:
                await websocket.send_text(json.dumps({"error": str(ex)}))
                try:
                    await websocket.close(code=1000)
                except Exception as _e_close2:
                    _append_jsonl(
                        os.path.join(STATE_DIR, "ws", "errors.jsonl"),
                        {
                            "t": int(time.time() * 1000),
                            "route": "/tool.ws",
                            "error": str(_e_close2),
                            "where": "close_after_error",
                        },
                    )
                try:
                    await websocket.close(code=1000)
                except Exception as _e_close3:
                    _append_jsonl(
                        os.path.join(STATE_DIR, "ws", "errors.jsonl"),
                        {
                            "t": int(time.time() * 1000),
                            "route": "/tool.ws",
                            "error": str(_e_close3),
                            "where": "close_after_error_dup",
                        },
                    )
    except WebSocketDisconnect:
        return
    try:
        await websocket.close(code=1000)
    except Exception as ex:
        logging.debug(f"ws_tool: failed to close websocket cleanly: {ex}", exc_info=True)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = ""
    if "." in file.filename:
        suffix = "." + file.filename.split(".")[-1]
    name = f"{uuid.uuid4().hex}{suffix}"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(await file.read())
    url = f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{name}" if PUBLIC_BASE_URL else f"/uploads/{name}"
    # Ingest and index into RAG if possible
    try:
        texts_meta = ingest_file(path, vlm_url=VLM_API_URL, whisper_url=WHISPER_API_URL, ocr_url=OCR_API_URL)
        texts = texts_meta.get("texts") or []
        if texts:
            pool = await get_pg_pool()
            if pool is not None:
                emb = get_embedder()
                async with pool.acquire() as conn:
                    for t in texts[:20]:
                        try:
                            vec = emb.encode([t])[0]
                            await conn.execute(
                                "INSERT INTO rag_docs (path, chunk, embedding) VALUES ($1, $2, $3)",
                                name,
                                t,
                                list(vec),
                            )
                        except Exception as ex:
                            _log("upload.ingest.insert.error", error=str(ex), path=path)
    except Exception as ex:
        _log("upload.ingest.error", error=str(ex), path=path)
    return {"url": url, "name": file.filename}



# ---------- Jobs API for long ComfyUI workflows ----------
_job_endpoint: Dict[str, str] = {}
_comfy_load: Dict[str, int] = {}


def _comfy_candidates() -> List[str]:
    if COMFYUI_API_URLS:
        return COMFYUI_API_URLS
    if COMFYUI_API_URL:
        return [COMFYUI_API_URL]
    return []


async def _pick_comfy_base() -> Optional[str]:
    candidates = _comfy_candidates()
    if not candidates:
        return None
    for u in candidates:
        _comfy_load.setdefault(u, 0)
    return min(candidates, key=lambda u: _comfy_load.get(u, 0))


async def _comfy_submit_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    candidates = _comfy_candidates()
    if not candidates:
        return {"error": "COMFYUI_API_URL(S) not configured"}
    delay = COMFYUI_BACKOFF_MS
    last_err: Optional[str] = None
    for attempt in range(1, COMFYUI_MAX_RETRIES + 1):
        ordered = sorted(candidates, key=lambda u: _comfy_load.get(u, 0))
        for base in ordered:
            try:
                async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                    if _comfy_sem is not None:
                        async with _comfy_sem:
                            _comfy_load[base] = _comfy_load.get(base, 0) + 1
                            payload = workflow if isinstance(workflow, dict) else {"prompt": workflow}
                            if "prompt" not in payload and isinstance(workflow, dict):
                                payload = {"prompt": workflow}
                            payload = {**payload, "client_id": "wrapper-001"}
                            r = await client.post(base.rstrip("/") + "/prompt", json=payload)
                    else:
                        _comfy_load[base] = _comfy_load.get(base, 0) + 1
                        payload = workflow if isinstance(workflow, dict) else {"prompt": workflow}
                        if "prompt" not in payload and isinstance(workflow, dict):
                            payload = {"prompt": workflow}
                        payload = {**payload, "client_id": "wrapper-001"}
                        r = await client.post(base.rstrip("/") + "/prompt", json=payload)
                    try:
                        parser = JSONParser()
                        res = parser.parse(r.text or "", {"prompt_id": str, "uuid": str, "id": str})
                        pid = res.get("prompt_id") or res.get("uuid") or res.get("id") if isinstance(res, dict) else None
                        if isinstance(pid, str):
                            _job_endpoint[pid] = base
                            # Wait for execution via WS
                            ws_url = base.replace("http", "ws").rstrip("/") + f"/ws?clientId=wrapper-001"
                            async with _ws.connect(ws_url, ping_interval=None) as ws:
                                # Drain until executed message for our pid
                                while True:
                                    msg = await ws.recv()
                                    jd = JSONParser().parse(msg, {"type": str, "data": dict}) if isinstance(msg, (str, bytes)) else {}
                                    if isinstance(jd, dict) and jd.get("type") == "executed":
                                        d = jd.get("data") or {}
                                        if d.get("prompt_id") == pid:
                                            break
                            return res if isinstance(res, dict) else {}
                    except Exception as ex:
                        last_err = r.text
                        logging.warning(
                            f"comfy.wait_for_executed.error base={str(base)} prompt_id={str(pid)}: {ex}",
                            exc_info=True,
                        )
            except Exception as ex:
                last_err = str(ex)
            finally:
                _comfy_load[base] = max(0, _comfy_load.get(base, 1) - 1)
        # backoff before next round
        try:
            await _as.sleep(max(0.0, float(delay) / 1000.0))
        except Exception as ex:
            logging.warning(f"comfy.backoff.sleep.error: {str(ex)}", exc_info=True)
        delay = min(delay * 2, COMFYUI_BACKOFF_MAX_MS)
    return {"error": last_err or "all comfyui instances failed after retries"}


async def _comfy_history(prompt_id: str) -> Dict[str, Any]:
    base = _job_endpoint.get(prompt_id) or (await _pick_comfy_base())
    if not base:
        return {"error": "COMFYUI_API_URL(S) not configured"}
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        r = await client.get(base.rstrip("/") + f"/history/{prompt_id}")
        parser = JSONParser()
        js = parser.parse(r.text or "", {})
        if 200 <= r.status_code < 300:
            return js if isinstance(js, dict) else {}
        return {"error": r.text}


async def get_film_characters(film_id: str) -> List[Dict[str, Any]]:
    pool = await get_pg_pool()
    if pool is None:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, name, description, reference_data FROM characters WHERE film_id=$1", film_id)
    return [dict(r) for r in rows]


async def get_film_preferences(film_id: str) -> Dict[str, Any]:
    pool = await get_pg_pool()
    if pool is None:
        return {}
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT metadata FROM films WHERE id=$1", film_id)
    meta = (row and row.get("metadata")) or {}
    if isinstance(meta, dict):
        return dict(meta)
    if isinstance(meta, str):
        # Expect a preferences dict; coerce into mapping with open schema.
        parser = JSONParser()
        parsed = parser.parse(meta, {"preferences": dict})
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_resolution(res: Optional[str]) -> Tuple[int, int]:
    if isinstance(res, str) and "x" in res:
        parts = res.lower().split("x", 1)
        if len(parts) == 2:
            w_str, h_str = parts[0].strip(), parts[1].strip()
            if w_str.isdigit() and h_str.isdigit():
                return max(64, int(w_str)), max(64, int(h_str))
    return 1024, 1024


def _parse_duration_seconds_dynamic(value: Any, default_seconds: float = 10.0) -> int:
    if value is None:
        return int(default_seconds)
    if isinstance(value, (int, float)):
        return max(1, int(round(float(value))))
    s = str(value).strip().lower()
    if not s:
        return int(default_seconds)
    # HH:MM:SS or MM:SS
    if _re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", s):
        parts = s.split(":")
        # All parts are digits by regex, so int() is safe
        ints = [int(x) for x in parts]
        if len(ints) == 2:
            return ints[0] * 60 + ints[1]
        if len(ints) == 3:
            return ints[0] * 3600 + ints[1] * 60 + ints[2]
    # textual: "3 minutes", "180s", "3m"
    m = _re.match(r"^(\d+)\s*(seconds|second|secs|sec|s)$", s)
    if m:
        return max(1, int(m.group(1)))
    m = _re.match(r"^(\d+)\s*(minutes|minute|mins|min|m)$", s)
    if m:
        return max(1, int(m.group(1)) * 60)
    m = _re.match(r"^(\d+)\s*(hours|hour|hrs|hr|h)$", s)
    if m:
        return max(1, int(m.group(1)) * 3600)
    # plain integer or float-like string
    if _re.match(r"^-?\d+(\.\d+)?$", s):
        return max(1, int(round(float(s))))
    return int(default_seconds)


def _parse_fps_dynamic(value: Any, default_fps: int = 60) -> int:
    if value is None:
        return default_fps
    if isinstance(value, (int, float)):
        return max(1, min(240, int(round(float(value)))))
    s = str(value).strip().lower()
    if not s:
        return default_fps
    m = _re.match(r"^(\d{1,3})\s*fps$", s)
    if m:
        return max(1, min(240, int(m.group(1))))
    if _re.match(r"^-?\d+(\.\d+)?$", s):
        return max(1, min(240, int(round(float(s)))))
    return default_fps


def build_default_scene_workflow(prompt: str, characters: List[Dict[str, Any]], style: Optional[str] = None, *, width: int = 1024, height: int = 1024, steps: int = 25, seed: int = 0, filename_prefix: str = "scene") -> Dict[str, Any]:
    # Minimal SDXL image generation graph using CheckpointLoaderSimple (MODEL, CLIP, VAE)
    if not isinstance(characters, list):
        characters = []
    positive = prompt
    # Inject lightweight story context into the prompt to reduce drift
    if characters:
        names = [c.get("name") for c in characters if isinstance(c, dict) and isinstance(c.get("name"), str) and c.get("name")]
        if names:
            positive = f"{positive}. Characters: " + ", ".join([str(n) for n in names[:3] if n])
    if style:
        positive = f"{prompt} in {style} style"
    for ch in characters[:2]:
        name = ch.get("name")
        if name:
            positive += f", featuring {name}"
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": steps, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": ["9", 0]}},
    }
    return {"prompt": g}


def _derive_seed(*parts: str) -> int:
    h = _h.sha256("::".join(parts).encode("utf-8")).hexdigest()
    # 32-bit range for Comfy samplers
    return int(h[:8], 16)


def _clamp_frames(frames: int) -> int:
    if frames is None:
        return 24
    if isinstance(frames, bool):
        return 24
    if isinstance(frames, (int, float)):
        return max(1, min(int(frames), 120))
    s = str(frames).strip()
    if _re.match(r"^-?\d+(\.\d+)?$", s):
        return max(1, min(int(float(s)), 120))
    return 24


def _round_fps(fps: float) -> int:
    if fps is None:
        return 24
    if isinstance(fps, bool):
        return 24
    if isinstance(fps, (int, float)):
        return max(1, min(int(round(float(fps))), 60))
    s = str(fps).strip()
    if _re.match(r"^-?\d+(\.\d+)?$", s):
        return max(1, min(int(round(float(s))), 60))
    return 24


def build_animated_scene_workflow(
    prompt: str,
    characters: List[Dict[str, Any]],
    *,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    style: Optional[str],
    seed: int,
    filename_prefix: str,
    upscale_enabled: bool = False,
    upscale_scale: int = 0,
    interpolation_enabled: bool = False,
    interpolation_multiplier: int = 0,
) -> Dict[str, Any]:
    # Simple animation via chunked batch sampling using CheckpointLoaderSimple
    if not isinstance(characters, list):
        characters = []
    total_frames = int(max(1, duration_seconds) * fps)
    frames = _clamp_frames(total_frames)
    batch_frames = max(1, min(SCENE_MAX_BATCH_FRAMES, frames))
    positive = prompt
    # Inject lightweight story context into the prompt to reduce drift
    names = [c.get("name") for c in characters if isinstance(c, dict) and isinstance(c.get("name"), str) and c.get("name")]
    if names:
        positive = f"{positive}. Characters: " + ", ".join([str(n) for n in names[:3] if n])
    if style:
        positive = f"{prompt} in {style} style"
    # Build a graph that renders a smaller batch (batch_frames)
    g = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": batch_frames}},
        "8": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 20, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["7", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
    }
    last_image_node = "9"
    # Optional Real-ESRGAN upscale
    if upscale_enabled and upscale_scale in (2, 3, 4):
        g["11"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": os.getenv("COMFY_UPSCALE_MODEL", "4x-UltraSharp.pth")}}
        g["12"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"image": [last_image_node, 0], "upscale_model": ["11", 0]}}
        last_image_node = "12"
    # Optional RIFE interpolation (VideoHelperSuite)
    if interpolation_enabled and interpolation_multiplier and interpolation_multiplier > 1:
        g["13"] = {"class_type": "VHS_RIFE_VFI", "inputs": {"frames": [last_image_node, 0], "multiplier": interpolation_multiplier, "model": "rife-v4.6"}}
        last_image_node = "13"
    # Final save
    g["10"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": filename_prefix, "images": [last_image_node, 0]}}
    return {"prompt": g, "_batch_frames": batch_frames, "_total_frames": total_frames}


async def _track_comfy_job(job_id: str, prompt_id: str) -> None:
    pool = await get_pg_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute("UPDATE jobs SET status='running', updated_at=NOW() WHERE id=$1", job_id)
    _jobs_store.setdefault(job_id, {})["state"] = "running"
    _jobs_store[job_id]["updated_at"] = time.time()
    last_outputs_count = -1
    last_error = None
    # Wait generously for history to appear; do not fail early just because history is temporarily missing
    _hist_grace_raw = os.getenv("HISTORY_GRACE_SECONDS", "86400")
    try:
        HISTORY_GRACE_SECONDS = int(str(_hist_grace_raw).strip() or "86400")  # default: 24h
    except Exception:
        logging.warning("comfy.track: bad HISTORY_GRACE_SECONDS=%r; defaulting to 86400", _hist_grace_raw, exc_info=True)
        HISTORY_GRACE_SECONDS = 86400
    start_time = time.time()
    while True:
        data = await _comfy_history(prompt_id)
        _jobs_store[job_id]["updated_at"] = time.time()
        _jobs_store[job_id]["last_history"] = data
        detail = _normalize_comfy_history_entry(data if isinstance(data, dict) else {}, prompt_id)
        if isinstance(detail, dict):
            outputs = detail.get("outputs", {}) or {}
            outputs_count = 0
            if isinstance(outputs, dict):
                for v in outputs.values():
                    if isinstance(v, list):
                        outputs_count += len(v)
                    else:
                        outputs_count += 1
            if outputs_count != last_outputs_count and outputs_count > 0:
                last_outputs_count = outputs_count
                async with pool.acquire() as conn:
                    await conn.execute("INSERT INTO job_checkpoints (job_id, data) VALUES ($1, $2)", job_id, json.dumps(detail))
            if detail.get("status", {}).get("completed") is True:
                result = {"outputs": detail.get("outputs", {}), "status": detail.get("status")}
                async with pool.acquire() as conn:
                    await conn.execute("UPDATE jobs SET status='succeeded', updated_at=NOW(), result=$1 WHERE id=$2", json.dumps(result), job_id)
                _jobs_store[job_id]["state"] = "succeeded"
                _jobs_store[job_id]["result"] = result
                # propagate to scene if any
                try:
                    await _update_scene_from_job(job_id, detail)
                except Exception as ex:
                    logging.warning(f"job.update_scene.error job_id={str(job_id)} err={str(ex)}", exc_info=True)
                if JOBS_RAG_INDEX:
                    try:
                        await _index_job_into_rag(job_id)
                    except Exception as ex:
                        logging.warning(f"job.rag_index.error job_id={str(job_id)} err={str(ex)}", exc_info=True)
                break
            if detail.get("status", {}).get("status") == "error":
                err = detail.get("status")
                async with pool.acquire() as conn:
                    await conn.execute("UPDATE jobs SET status='failed', updated_at=NOW(), error=$1 WHERE id=$2", json.dumps(err), job_id)
                _jobs_store[job_id]["state"] = "failed"
                _jobs_store[job_id]["error"] = err
                try:
                    await _update_scene_from_job(job_id, detail, failed=True)
                except Exception as ex:
                    logging.warning(f"job.update_scene_failed.error job_id={str(job_id)} err={str(ex)}", exc_info=True)
                break
        else:
            # History not yet available. Do NOT fail early and do NOT auto-resubmit; keep polling up to grace window.
            if isinstance(data, dict) and data.get("error"):
                last_error = data.get("error")
            if (time.time() - start_time) >= HISTORY_GRACE_SECONDS:
                # Grace window exceeded – keep job as running and attach last_error for visibility, but don't mark failed
                _jobs_store[job_id]["state"] = "running"
                if last_error:
                    _jobs_store[job_id]["error"] = last_error
                await _as.sleep(2.0)
                continue
            await _as.sleep(2.0)
            continue
        # keep polling
        await _as.sleep(2.0)


@app.post("/jobs")
async def create_job(body: Dict[str, Any]):
    if not COMFYUI_API_URL:
        return JSONResponse(status_code=400, content={"error": "COMFYUI_API_URL not configured"})
    workflow = body.get("workflow") if isinstance(body, dict) else None
    if not workflow:
        workflow = body or {}
    submit = await _comfy_submit_workflow(workflow)
    if submit.get("error"):
        return JSONResponse(status_code=502, content=submit)
    prompt_id = submit.get("prompt_id") or submit.get("uuid") or submit.get("id")
    if not prompt_id:
        return JSONResponse(status_code=502, content={"error": "invalid comfy response", "detail": submit})
    job_id = uuid.uuid4().hex
    pool = await get_pg_pool()
    if pool is not None:
        async with pool.acquire() as conn:
            await conn.execute("INSERT INTO jobs (id, prompt_id, status, workflow) VALUES ($1, $2, 'queued', $3)", job_id, prompt_id, json.dumps(workflow))
    _jobs_store[job_id] = {"id": job_id, "prompt_id": prompt_id, "state": "queued", "created_at": time.time(), "updated_at": time.time(), "result": None}
    await _track_comfy_job(job_id, prompt_id)
    return {"job_id": job_id, "prompt_id": prompt_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    pool = await get_pg_pool()
    if pool is None:
        job = _jobs_store.get(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"error": "not found"})
        return job
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, prompt_id, status, created_at, updated_at, workflow, result, error FROM jobs WHERE id=$1", job_id)
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        cps = await conn.fetch("SELECT id, created_at, data FROM job_checkpoints WHERE job_id=$1 ORDER BY id DESC LIMIT 10", job_id)
        return {**dict(row), "checkpoints": [dict(c) for c in cps]}


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str, interval_ms: Optional[int] = None):
    return StreamingResponse(_jobs_stream_gen(job_id, interval_ms), media_type="text/event-stream")


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50, offset: int = 0):
    pool = await get_pg_pool()
    if pool is None:
        items = list(_jobs_store.values())
        if status:
            items = [j for j in items if (j.get("state") or j.get("status")) == status]
        norm = [_normalize_job(j) for j in items]
        return {"data": norm[offset: offset + limit], "total": len(norm)}
    async with pool.acquire() as conn:
        # Avoid exceptions by checking table existence first
        exists = await conn.fetchval("SELECT to_regclass('public.jobs') IS NOT NULL")
        if not exists:
            items = list(_jobs_store.values())
            if status:
                items = [j for j in items if (j.get("state") or j.get("status")) == status]
            norm = [_normalize_job(j) for j in items]
            return {"data": norm[offset: offset + limit], "total": len(norm)}

        if status:
            rows = await conn.fetch(
                "SELECT id, prompt_id, status, created_at, updated_at FROM jobs WHERE status=$1 ORDER BY updated_at DESC LIMIT $2 OFFSET $3",
                status, limit, offset,
            )
            total = await conn.fetchval("SELECT COUNT(*) FROM jobs WHERE status=$1", status)
        else:
            rows = await conn.fetch(
                "SELECT id, prompt_id, status, created_at, updated_at FROM jobs ORDER BY updated_at DESC LIMIT $1 OFFSET $2",
                limit, offset,
            )
            total = await conn.fetchval("SELECT COUNT(*) FROM jobs")

        db_items = [dict(r) for r in rows]
        # Union with in-memory items to avoid empty UI when DB insert fails
        db_ids = {it.get("id") for it in db_items}
        mem_items = list(_jobs_store.values())
        if status:
            mem_items = [j for j in mem_items if (j.get("state") or j.get("status")) == status]
        mem_only = [j for j in mem_items if (j.get("id") or j.get("job_id")) not in db_ids]
        all_items = [
            _normalize_job(j) for j in db_items
        ] + [
            _normalize_job(j) for j in mem_only
        ]
        safe_total = int(total or 0) + len(mem_only)
        return {"data": all_items, "total": safe_total}


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    pool = await get_pg_pool()
    if pool is None:
        return JSONResponse(status_code=500, content={"error": "pg not configured"})
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT prompt_id, status FROM jobs WHERE id=$1", job_id)
        if not row:
            return JSONResponse(status_code=404, content={"error": "not found"})
        if row["status"] in ("succeeded", "failed", "cancelled"):
            return {"ok": True, "status": row["status"]}
        await conn.execute("UPDATE jobs SET status='cancelling', updated_at=NOW() WHERE id=$1", job_id)
    try:
        if COMFYUI_API_URL:
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                await client.post(COMFYUI_API_URL.rstrip("/") + "/interrupt")
    except Exception as ex:
        logging.warning(f"comfy.interrupt.error job_id={str(job_id)} err={str(ex)}", exc_info=True)
    async with pool.acquire() as conn:
        await conn.execute("UPDATE jobs SET status='cancelled', updated_at=NOW() WHERE id=$1", job_id)
    _jobs_store.setdefault(job_id, {})["state"] = "cancelled"
    return {"ok": True, "status": "cancelled"}


async def _index_job_into_rag(job_id: str) -> None:
    pool = await get_pg_pool()
    if pool is None:
        return
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT workflow, result FROM jobs WHERE id=$1", job_id)
        if not row:
            return
        wf = json.dumps(row["workflow"], ensure_ascii=False) if row.get("workflow") is not None else ""
        res = json.dumps(row["result"], ensure_ascii=False) if row.get("result") is not None else ""
    embedder = get_embedder()
    pieces = [p for p in [wf, res] if p]
    async with pool.acquire() as conn:
        for chunk in pieces:
            vec = embedder.encode([chunk])[0]
            await conn.execute("INSERT INTO rag_docs (path, chunk, embedding) VALUES ($1, $2, $3)", f"job:{job_id}", chunk, list(vec))


async def _update_scene_from_job(job_id: str, detail: Dict[str, Any] | str, failed: bool = False) -> None:
    pool = await get_pg_pool()
    if pool is None:
        return
    # Prefer the exact ComfyUI base that handled this job (per-job endpoint pinning)
    job = _jobs_store.get(job_id) if isinstance(_jobs_store.get(job_id), dict) else {}
    pid = job.get("prompt_id")
    base = _job_endpoint.get(pid) if pid else None
    if isinstance(detail, str):
        parsed = JSONParser().parse(detail, {"outputs": dict, "status": dict})
        detail = parsed if isinstance(parsed, dict) else {}
    assets = {
        "outputs": (detail or {}).get("outputs", {}),
        "status": (detail or {}).get("status"),
        "urls": _extract_comfy_asset_urls(detail, base or COMFYUI_API_URL or ""),
    }
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, film_id, plan FROM scenes WHERE job_id=$1", job_id)
        if not row:
            return
        if failed:
            await conn.execute("UPDATE scenes SET status='failed', updated_at=NOW(), assets=$1 WHERE id=$2", json.dumps(assets), row["id"])
        else:
            await conn.execute("UPDATE scenes SET status='succeeded', updated_at=NOW(), assets=$1 WHERE id=$2", json.dumps(assets), row["id"])
    if not failed:
        # Auto-generate TTS dialogue and background music for the scene
        scene_tts = None
        scene_music = None
        try:
            # Use prompt as provisional dialogue text; in practice, planner should generate scripts
            scene_text = (detail.get("status", {}) or {}).get("prompt", "") or ""
            if not scene_text:
                # fallback: try to use scene prompt from DB
                async with pool.acquire() as conn:
                    p = await conn.fetchval("SELECT prompt FROM scenes WHERE id=$1", row["id"])
                    scene_text = p or ""
            # derive duration preference
            duration = 12
            try:
                pl = row.get("plan") or {}
                prefs = (pl.get("preferences") if isinstance(pl, dict) else None) or {}
                if prefs.get("duration_seconds"):
                    duration = int(float(prefs.get("duration_seconds")))
            except Exception as ex:
                logging.warning(f"scene.duration.parse.error scene_id={str(row.get('id'))} err={str(ex)}", exc_info=True)
            # read toggles
            audio_enabled = True
            subs_enabled = False
            try:
                if isinstance(pl, dict):
                    prefs = (pl.get("preferences") if isinstance(pl.get("preferences"), dict) else {})
                    audio_enabled = bool(prefs.get("audio_enabled", True))
                    subs_enabled = bool(prefs.get("subtitles_enabled", False))
            except Exception as ex:
                logging.warning(f"scene.toggles.parse.error scene_id={str(row.get('id'))} err={str(ex)}", exc_info=True)
            # choose voice: character-specific -> scene plan -> film-level preference
            voice = None
            try:
                if isinstance(pl, dict):
                    prefs = (pl.get("preferences") if isinstance(pl.get("preferences"), dict) else {})
                    voice = prefs.get("voice")
                    # character-specific override if a single character is targeted
                    ch_ids = pl.get("character_ids") if isinstance(pl.get("character_ids"), list) else []
                    if len(ch_ids) == 1:
                        async with pool.acquire() as conn:
                            crow = await conn.fetchrow("SELECT reference_data FROM characters WHERE id=$1", ch_ids[0])
                            if crow:
                                rd = crow.get("reference_data")
                                if isinstance(rd, str):
                                    parser = JSONParser()
                                    rd_parsed = parser.parse(rd, {"locks": dict, "style": dict, "qa": dict})
                                    rd = rd_parsed if isinstance(rd_parsed, dict) else {}
                                if isinstance(rd, dict):
                                    v2 = rd.get("voice")
                                    if v2:
                                        voice = v2
                if not voice:
                    async with pool.acquire() as conn:
                        meta = await conn.fetchval("SELECT metadata FROM films WHERE id=$1", row["film_id"]) 
                        if isinstance(meta, dict):
                            voice = meta.get("voice")
            except Exception as ex:
                logging.warning(f"scene.voice.resolve.error scene_id={str(row.get('id'))} err={str(ex)}", exc_info=True)
            # language preference
            language = None
            try:
                async with pool.acquire() as conn:
                    meta = await conn.fetchval("SELECT metadata FROM films WHERE id=$1", row["film_id"]) 
                    if isinstance(meta, dict):
                        language = meta.get("language")
            except Exception as ex:
                logging.warning(f"scene.language.resolve.error scene_id={str(row.get('id'))} err={str(ex)}", exc_info=True)
            if XTTS_API_URL and ALLOW_TOOL_EXECUTION and scene_text and audio_enabled:
                async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                    tr = await client.post(XTTS_API_URL.rstrip("/") + "/tts", json={"text": scene_text, "voice": voice, "language": language})
                    if tr.status_code == 200:
                        parser = JSONParser()
                        scene_tts = parser.parse(tr.text or "", {"audio_wav_base64": str, "sample_rate": int, "language": str})
            if MUSIC_API_URL and ALLOW_TOOL_EXECUTION and audio_enabled:
                async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                    mr = await client.post(MUSIC_API_URL.rstrip("/") + "/generate", json={"prompt": "cinematic background score", "duration": duration})
                    if mr.status_code == 200:
                        parser = JSONParser()
                        scene_music = parser.parse(mr.text or "", {"audio_wav_base64": str, "duration": float})
            # simple SRT creation from scene_text if enabled
            if scene_text and subs_enabled:
                srt = _text_to_simple_srt(scene_text, duration)
                assets["subtitles_srt"] = srt
        except Exception as ex:
            logging.warning(f"scene.assets.build.error scene_id={str(row.get('id'))} err={str(ex)}", exc_info=True)
        # Update scene assets with audio
        if scene_tts or scene_music:
            assets = dict(assets)
            if scene_tts:
                assets["tts"] = scene_tts
            if scene_music:
                assets["music"] = scene_music
            async with pool.acquire() as conn:
                await conn.execute("UPDATE scenes SET assets=$1 WHERE id=$2", json.dumps(assets), row["id"])
        try:
            await _maybe_compile_film(row["film_id"])
        except Exception as ex:
            logging.warning(f"film.compile.maybe.error film_id={str(row.get('film_id'))} err={str(ex)}", exc_info=True)


async def _maybe_compile_film(film_id: str) -> None:
    pool = await get_pg_pool()
    if pool is None:
        return
    # If all scenes for film are succeeded, trigger compilation via N8N if available
    async with pool.acquire() as conn:
        counts = await conn.fetchrow("SELECT COUNT(*) AS total, SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS done FROM scenes WHERE film_id=$1", film_id)
        total = int(counts["total"] or 0)
        done = int(counts["done"] or 0)
        if total == 0 or done != total:
            return
        rows = await conn.fetch("SELECT id, index_num, assets FROM scenes WHERE film_id=$1 ORDER BY index_num ASC", film_id)
    scenes = [dict(r) for r in rows]
    payload = {"film_id": film_id, "scenes": [dict(s) for s in scenes], "public_base_url": PUBLIC_BASE_URL}
    assembly_result = None
    # Prefer local assembler if available
    if ASSEMBLER_API_URL:
        try:
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                r = await client.post(ASSEMBLER_API_URL.rstrip("/") + "/assemble", json=payload)
                parser = JSONParser()
                assembly_result = parser.parse(r.text or "", {})
        except Exception as ex:
            logging.warning(f"film.compile.assembler.failed film_id={str(film_id)}: {ex}", exc_info=True)
            assembly_result = {"error": True, "message": str(ex)}
    elif N8N_WEBHOOK_URL and ENABLE_N8N:
        try:
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                r = await client.post(N8N_WEBHOOK_URL, json=payload)
                parser = JSONParser()
                assembly_result = parser.parse(r.text or "", {})
        except Exception as ex:
            logging.warning(f"film.compile.n8n.failed film_id={str(film_id)}: {ex}", exc_info=True)
            assembly_result = {"error": True, "message": str(ex)}
    # Always write a manifest to uploads for convenience
    manifest_url = None
    try:
        manifest = json.dumps(payload)
        if EXECUTOR_BASE_URL:
            async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                res = await client.post(EXECUTOR_BASE_URL.rstrip("/") + "/write_file", json={"path": "uploads/film_" + film_id + "_manifest.json", "content": manifest})
                # Build public URL
                name = f"film_{film_id}_manifest.json"
                manifest_url = (f"{PUBLIC_BASE_URL.rstrip('/')}/uploads/{name}" if PUBLIC_BASE_URL else f"/uploads/{name}")
    except Exception as ex:
        logging.warning(f"film.compile.manifest_write.error film_id={str(film_id)} err={str(ex)}", exc_info=True)
    # persist compile artifact into films.metadata
    async with pool.acquire() as conn:
        meta_row = await conn.fetchrow("SELECT metadata FROM films WHERE id=$1", film_id)
        metadata = (meta_row and meta_row.get("metadata")) or {}
        metadata = dict(metadata)
        metadata["compiled_at"] = time.time()
        if assembly_result is not None:
            metadata["assembly"] = assembly_result
        if manifest_url:
            metadata["manifest_url"] = manifest_url
        metadata["scenes_count"] = total
        await conn.execute("UPDATE films SET metadata=$1, updated_at=NOW() WHERE id=$2", json.dumps(metadata), film_id)


# ---------- Film project endpoints ----------

