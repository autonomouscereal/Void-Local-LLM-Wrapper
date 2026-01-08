#!/usr/bin/env python3
from __future__ import annotations

"""
bootstrap models script (HunyuanVideo-1.5 + official 1080p SR)

Key behaviors:
- OFFLINE runtime: services must load from /opt/models/*
- Hunyuan dirs are VERSIONED:
  - If existing content != expected repo/patterns => DELETE and re-download
- Other model dirs remain idempotent "if non-empty => skip" (no forced redownload)

CRITICAL FIXES (for your current failure):
- Stop trimming Tencent weights repo. Download full tencent/HunyuanVideo-1.5.
- Ensure Glyph-SDXL-v2 + byt5-small are present under /opt/models/hunyuan/text_encoder/...
- Keep hunyuan/text_encoder/Glyph-SDXL-v2 as VERSIONED so it re-verifies (your repo layout changed before).
"""

import os
import sys
import json
import subprocess
import shutil
import time
from typing import Any


MODELS_DIR = os.environ.get("FILM2_MODELS", "/opt/models")
HF_HOME = os.environ.get("HF_HOME", os.path.join(MODELS_DIR, ".hf"))
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "").strip() or None

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)

os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["TORCH_HOME"] = os.path.join(MODELS_DIR, ".torch")
os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "info")
os.environ.setdefault("HF_HUB_ENABLE_XET", "0")

HF_MAX_WORKERS = int(os.environ.get("HF_MAX_WORKERS", "4"))

STATUS_PATH = os.path.join(MODELS_DIR, "manifests", "bootstrap_status.json")
STATUS: dict[str, dict[str, dict[str, str]]] = {"hf": {}, "git": {}}

META_FILE = ".bootstrap_meta.json"


def write_status() -> None:
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(STATUS, f, indent=2)


def log(*a: object) -> None:
    print("[bootstrap]", *a, flush=True)


HUNYUAN15_DIFFUSERS_720P_T2V = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
HUNYUAN15_TENCENT = "tencent/HunyuanVideo-1.5"

# Version-enforced keys: if mismatch => delete + redownload
VERSIONED_KEYS = {
    "hunyuan",
    "hunyuan_diffusers",
    "hunyuan/text_encoder/Glyph-SDXL-v2",
}

HF_MODELS: list[tuple[str, str, list[str] | None]] = [
    # 1) Tencent core weights repo: download FULL repo now (no allow_patterns).
    (HUNYUAN15_TENCENT, "hunyuan", None),

    # 2) Glyph encoder assets
    ("multimodalart/glyph-sdxl-v2-byt5-small", "hunyuan/text_encoder/Glyph-SDXL-v2", None),

    # 3) Base ByT5 weights (some code paths reference byt5-small)
    ("google/byt5-small", "hunyuan/text_encoder/byt5-small", None),

    # 4) Optional diffusers-format checkpoint (not required for Tencent pipeline, but you keep it)
    (HUNYUAN15_DIFFUSERS_720P_T2V, "hunyuan_diffusers", None),

    # 5) Rest of your ecosystem models (idempotent)
    ("Lightricks/LTX-Video",                  "ltx_video",      None),
    ("InstantX/InstantID",                    "instantid",      None),
    ("h94/IP-Adapter",                        "ip_adapter",     None),
    ("lllyasviel/ControlNet-v1-1",            "controlnet",     None),
    ("facebook/sam2-hiera-large",             "sam",            None),  # no trimming
    ("depth-anything/Depth-Anything-V2-Base", "depth_anything", None),
    ("openai/clip-vit-large-patch14-336",     "clip",           None),
    ("laion/clap-htsat-unfused",              "clap",           None),
    ("Salesforce/blip2-flan-t5-xl",           "blip2",          None),
    ("openai/whisper-large-v3",               "whisper",        None),
]

GIT_REPOS: list[tuple[str, str]] = [
    ("https://github.com/hpcaitech/Open-Sora.git",                   "opensora"),
    ("https://github.com/princeton-vl/RAFT.git",                     "raft"),
    ("https://github.com/ckkelvinchan/BasicVSR_PlusPlus.git",        "basicvsrpp"),
    ("https://github.com/xinntao/EDVR.git",                          "edvr"),
    ("https://github.com/TencentARC/GFPGAN.git",                     "gfpgan"),
    ("https://github.com/sczhou/CodeFormer.git",                     "codeformer"),
    ("https://github.com/MegEngine/ECCV2022-RIFE.git",               "rife"),
    ("https://github.com/CMU-Perceptual-Computing-Lab/openpose.git", "openpose"),
]

MANDATORY_DIRS = [
    "hunyuan",
    "hunyuan/text_encoder/Glyph-SDXL-v2",
    "hunyuan/text_encoder/byt5-small",
    "hunyuan_diffusers",
    "ltx_video",
    "instantid",
    "ip_adapter",
    "controlnet",
    "sam",
    "depth_anything",
    "clip",
    "clap",
    "blip2",
    "whisper",
    "rvc_titan",
    "rife_vfi",
    "realesrgan",
]


def ensure_pkg() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub>=0.23.0"])


def _meta_expected(repo_id: str, allow_patterns: list[str] | None) -> dict:
    return {"repo_id": repo_id, "allow_patterns": allow_patterns or None}


def _read_meta(tgt: str) -> dict | None:
    p = os.path.join(tgt, META_FILE)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_meta(tgt: str, meta: dict) -> None:
    p = os.path.join(tgt, META_FILE)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as ex:
        log("WARN: failed to write meta", p, ex)


def _rm_tree(path: str) -> None:
    if not path or path in ("/", "/opt", "/opt/models"):
        raise RuntimeError(f"refusing to delete unsafe path: {path!r}")
    shutil.rmtree(path, ignore_errors=True)


def snapshot(repo_id: str, local_key: str, allow_patterns: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    tgt = os.path.join(MODELS_DIR, local_key)

    # Guard: skip invalid repo ids
    if not isinstance(repo_id, str) or "=" in repo_id or repo_id.strip().startswith(("/", "./", "../")):
        log("skip-invalid-repo-id", repo_id, "->", tgt)
        return

    # Versioned behavior: mismatch => delete + redownload
    if local_key in VERSIONED_KEYS and os.path.isdir(tgt) and os.listdir(tgt):
        expected = _meta_expected(repo_id, allow_patterns)
        meta = _read_meta(tgt)
        if meta != expected:
            log("VERSION-MISMATCH", local_key)
            log("  expected:", expected)
            log("  found   :", meta)
            log("  deleting:", tgt)
            _rm_tree(tgt)
        else:
            log("exists", local_key, "->", tgt, "(version ok)")
            return

    # Non-versioned behavior: if non-empty, skip
    if local_key not in VERSIONED_KEYS:
        if os.path.isdir(tgt) and os.listdir(tgt):
            log("exists", local_key, "->", tgt)
            return

    log("START-HF", repo_id, "->", tgt)
    os.makedirs(tgt, exist_ok=True)
    STATUS.setdefault("hf", {})[local_key] = {"repo": repo_id, "state": "downloading"}
    write_status()

    kw: dict[str, Any] = dict(
        repo_id=repo_id,
        local_dir=tgt,
        local_dir_use_symlinks=False,
        max_workers=HF_MAX_WORKERS,
    )

    if allow_patterns:
        kw["allow_patterns"] = allow_patterns

    if HF_TOKEN:
        kw["token"] = HF_TOKEN

    snapshot_download(**kw)

    STATUS["hf"][local_key]["state"] = "done"
    write_status()

    if local_key in VERSIONED_KEYS:
        _write_meta(tgt, _meta_expected(repo_id, allow_patterns))

    log("DONE-HF", repo_id, "->", tgt)


def git_clone(url: str, local_key: str) -> None:
    tgt = os.path.join(MODELS_DIR, local_key)
    if os.path.exists(os.path.join(tgt, ".git")) or (os.path.isdir(tgt) and os.listdir(tgt)):
        log("exists", local_key, "->", tgt)
        return
    log("START-GIT", url, "->", tgt)
    STATUS.setdefault("git", {})[local_key] = {"repo": url, "state": "cloning"}
    write_status()
    subprocess.check_call(["git", "clone", "--depth=1", url, tgt])
    STATUS["git"][local_key]["state"] = "done"
    write_status()
    log("DONE-GIT", url, "->", tgt)


def ensure_rvc_titan() -> None:
    from huggingface_hub import hf_hub_download
    target_dir = os.path.join(MODELS_DIR, "rvc_titan")
    os.makedirs(target_dir, exist_ok=True)

    d_name = "D-f048k-TITAN-Medium.pth"
    g_name = "G-f048k-TITAN-Medium.pth"
    d_path = os.path.join(target_dir, d_name)
    g_path = os.path.join(target_dir, g_name)

    if os.path.isfile(d_path) and os.path.getsize(d_path) > 0 and os.path.isfile(g_path) and os.path.getsize(g_path) > 0:
        log("exists", "rvc_titan.DG48k", "->", target_dir)
        return

    log("START-RVC-TITAN-DG48k", "blaise-tk/TITAN", "->", target_dir)
    d_remote = "models/medium/48k/pretrained/" + d_name
    g_remote = "models/medium/48k/pretrained/" + g_name

    hf_hub_download(repo_id="blaise-tk/TITAN", filename=d_remote, local_dir=target_dir, token=HF_TOKEN)
    hf_hub_download(repo_id="blaise-tk/TITAN", filename=g_remote, local_dir=target_dir, token=HF_TOKEN)
    log("DONE-RVC-TITAN-DG48k", "->", target_dir)


def ensure_rife_vfi() -> None:
    from huggingface_hub import hf_hub_download
    import zipfile

    target_root = os.path.join(MODELS_DIR, "rife_vfi")
    train_log_dir = os.path.join(target_root, "train_log")
    os.makedirs(train_log_dir, exist_ok=True)

    if os.path.isdir(train_log_dir) and os.listdir(train_log_dir):
        log("exists", "rife_vfi.train_log", "->", train_log_dir)
        return

    assets_dir = os.path.join(target_root, "_assets")
    os.makedirs(assets_dir, exist_ok=True)

    zip_name = "RIFEv4.26_0921.zip"
    log("START-RIFE-VFI", "hzwer/RIFE", zip_name, "->", assets_dir)

    zip_path = hf_hub_download(repo_id="hzwer/RIFE", filename=zip_name, local_dir=assets_dir, token=HF_TOKEN)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for n in zf.namelist():
            base = os.path.basename(n)
            if not base:
                continue
            if base.endswith(".py") or base.endswith(".pkl"):
                out_path = os.path.join(train_log_dir, base)
                with zf.open(n, "r") as src:
                    with open(out_path, "wb") as dst:
                        dst.write(src.read())

    if not (os.path.isdir(train_log_dir) and os.listdir(train_log_dir)):
        log("ERROR-RIFE-VFI", "train_log empty after unzip", "->", train_log_dir)
        sys.exit(3)

    log("DONE-RIFE-VFI", "->", train_log_dir)


def ensure_realesrgan_weights() -> None:
    import urllib.request

    target_root = os.path.join(MODELS_DIR, "realesrgan")
    weights_dir = os.path.join(target_root, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    urls = [
        ("RealESRGAN_x4plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
        ("RealESRGAN_x4plus_anime_6B.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
        ("realesr-general-x4v3.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"),
        ("realesr-general-wdn-x4v3.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"),
    ]

    for fname, url in urls:
        dst = os.path.join(weights_dir, fname)
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            log("exists", "realesrgan", fname, "->", dst)
            continue
        log("START-REALESRGAN", url, "->", dst)
        with urllib.request.urlopen(url) as resp:
            with open(dst, "wb") as f:
                f.write(resp.read())
        log("DONE-REALESRGAN", url, "->", dst)

    if not (os.path.isdir(weights_dir) and os.listdir(weights_dir)):
        log("ERROR: Missing Real-ESRGAN weights:", weights_dir)
        sys.exit(3)

    log("DONE: Real-ESRGAN weights present ->", weights_dir)


def ensure_aesthetic_head() -> None:
    url = os.environ.get("AESTHETIC_HEAD_URL", "").strip()
    if not url:
        return
    import urllib.request

    target_dir = os.path.join(MODELS_DIR, "aesthetic")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "laion_v2_linear_L14.pt")
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        log("exists", "aesthetic_head", "->", target_path)
        return
    log("START-AESTHETIC", url, "->", target_path)
    with urllib.request.urlopen(url) as resp:
        with open(target_path, "wb") as f:
            f.write(resp.read())
    log("DONE-AESTHETIC", url, "->", target_path)


def main() -> None:
    ensure_pkg()

    done_marker = os.path.join(MODELS_DIR, "manifests", ".bootstrap_done")
    if os.path.exists(done_marker):
        log("marker-present", done_marker, "— will still verify all mandatory models")
    else:
        log("marker-missing", done_marker, "— full bootstrap will run")

    for rid, key, allow in HF_MODELS:
        snapshot(rid, key, allow)

    for url, key in GIT_REPOS:
        git_clone(url, key)

    ensure_rvc_titan()
    ensure_rife_vfi()
    ensure_realesrgan_weights()
    ensure_aesthetic_head()

    missing = [k for k in MANDATORY_DIRS if not (os.path.isdir(os.path.join(MODELS_DIR, k)) and os.listdir(os.path.join(MODELS_DIR, k)))]
    if missing:
        log("ERROR: Missing mandatory models:", missing)
        sys.exit(3)

    log("SUMMARY: ALL MANDATORY MODELS DOWNLOADED")

    try:
        os.makedirs(os.path.dirname(done_marker), exist_ok=True)
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write(str(int(time.time())))
    except Exception:
        log("WARN: failed to write done marker", done_marker)

    log("✅ bootstrap complete into", MODELS_DIR)


if __name__ == "__main__":
    main()
