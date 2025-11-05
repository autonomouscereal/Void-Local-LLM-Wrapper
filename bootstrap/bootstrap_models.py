import os
import sys
import json
import subprocess
import threading
import time
from pathlib import Path


MODELS_DIR = os.environ.get("FILM2_MODELS", "/opt/models")
HF_HOME = os.environ.get("HF_HOME", os.path.join(MODELS_DIR, ".hf"))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["TORCH_HOME"] = os.path.join(MODELS_DIR, ".torch")
os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "info")
os.environ.setdefault("HF_HUB_ENABLE_XET", "0")
HF_MAX_WORKERS = int(os.environ.get("HF_MAX_WORKERS", "4"))

# status tracking
STATUS_PATH = os.path.join(MODELS_DIR, "manifests", "bootstrap_status.json")
STATUS: dict[str, dict[str, dict[str, str]]] = {"hf": {}, "git": {}}

def write_status() -> None:
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(STATUS, f, indent=2)


YUE_REPO = os.environ.get("YUE_REPO", "").strip()

HF_MODELS = [
    ("tencent/HunyuanVideo",                  "hunyuan",        None),
    ("Lightricks/LTX-Video",                  "ltx_video",      None),
    ("InstantX/InstantID",                    "instantid",      None),
    ("h94/IP-Adapter",                        "ip_adapter",     None),
    ("lllyasviel/ControlNet-v1-1",            "controlnet",     None),
    ("facebook/sam2-hiera-large",             "sam",            ["sam2_hiera_large.pt"]),
    ("depth-anything/Depth-Anything-V2-Base", "depth_anything", None),
    ("openai/clip-vit-large-patch14-336",     "clip",           None),
    ("laion/clap-htsat-unfused",              "clap",           None),
    ("Salesforce/blip2-flan-t5-xl",           "blip2",          None),
    ("openai/whisper-large-v3",               "whisper",        None),
    # Audio composition/variation (optional pre-warm)
    ("facebook/musicgen-large",                "musicgen-large", None),
    ("facebook/musicgen-melody",               "musicgen-melody", None),
    ("facebook/audiogen-medium",               "audiogen-medium", None),
    ("cvssp/audioldm2-large",                  "audioldm2-large", None),
]

# Never append YuE dynamically in bootstrap; YuE is handled as a local artifact if used.


GIT_REPOS = [
    ("https://github.com/hpcaitech/Open-Sora.git",                   "opensora"),
    ("https://github.com/princeton-vl/RAFT.git",                     "raft"),
    ("https://github.com/ckkelvinchan/BasicVSR_PlusPlus.git",        "basicvsrpp"),
    ("https://github.com/xinntao/EDVR.git",                          "edvr"),
    ("https://github.com/TencentARC/GFPGAN.git",                     "gfpgan"),
    ("https://github.com/sczhou/CodeFormer.git",                     "codeformer"),
    ("https://github.com/MegEngine/ECCV2022-RIFE.git",               "rife"),
    ("https://github.com/CMU-Perceptual-Computing-Lab/openpose.git", "openpose"),
]

# Mandatory model directories must exist (non-empty) after bootstrap
MANDATORY_DIRS = [
    "hunyuan",
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
]
# Do not require 'yue' as mandatory; removed from bootstrap scope


def log(*a: object) -> None:
    print("[bootstrap]", *a, flush=True)


def human(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024.0:
            return f"{v:.1f}{u}"
        v /= 1024.0
    return f"{v:.1f}PB"


class SizeBeat:
    def __init__(self, path: str, label: str, interval: float = 5.0) -> None:
        self.path = path
        self.label = label
        self.interval = interval
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop:
            total = 0
            for root, _, files in os.walk(self.path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total += os.path.getsize(fp)
                    except Exception:
                        pass
            log(f"PROGRESS {self.label}", human(total))
            time.sleep(self.interval)

    def start(self) -> None:
        self.t.start()

    def stop(self) -> None:
        self._stop = True
        try:
            self.t.join(timeout=2)
        except Exception:
            pass


def ensure_pkg() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub>=0.23.0"])  # noqa: S603,S607


def snapshot(repo_id: str, local_key: str, allow_patterns: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download
    tgt = os.path.join(MODELS_DIR, local_key)
    if os.path.isdir(tgt) and os.listdir(tgt):
        log("exists", local_key, "->", tgt)
        return
    # Guard: skip invalid repo ids (paths/env-like assignments)
    if not isinstance(repo_id, str) or "=" in repo_id or repo_id.strip().startswith(("/", "./", "../")):
        log("skip-invalid-repo-id", repo_id, "->", tgt)
        return
    log("START-HF", repo_id, "->", tgt)
    os.makedirs(tgt, exist_ok=True)
    STATUS.setdefault("hf", {})[local_key] = {"repo": repo_id, "state": "downloading"}
    write_status()
    beat = SizeBeat(tgt, f"HF:{local_key}")
    beat.start()
    try:
        kw = dict(
            repo_id=repo_id,
            local_dir=tgt,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=HF_MAX_WORKERS,
        )
        if allow_patterns:
            kw["allow_patterns"] = allow_patterns
        try:
            snapshot_download(**kw)
        except RuntimeError as e:
            msg = str(e).lower()
            if ("xet" in msg) or ("cas service" in msg):
                os.environ["HF_HUB_ENABLE_XET"] = "0"
                kw["max_workers"] = 2
                snapshot_download(**kw)
            else:
                raise
    finally:
        beat.stop()
    STATUS["hf"][local_key]["state"] = "done"
    write_status()
    log("DONE-HF", repo_id, "->", tgt)


def git_clone(url: str, local_key: str) -> None:
    tgt = os.path.join(MODELS_DIR, local_key)
    if os.path.exists(os.path.join(tgt, ".git")) or (os.path.isdir(tgt) and os.listdir(tgt)):
        log("exists", local_key, "->", tgt)
        return
    log("START-GIT", url, "->", tgt)
    STATUS.setdefault("git", {})[local_key] = {"repo": url, "state": "cloning"}
    write_status()
    subprocess.check_call(["git", "clone", "--depth=1", url, tgt])  # noqa: S603,S607
    STATUS["git"][local_key]["state"] = "done"
    write_status()
    log("DONE-GIT", url, "->", tgt)


def main() -> None:
    ensure_pkg()
    # Early exit if done marker exists
    done_marker = os.path.join(MODELS_DIR, "manifests", ".bootstrap_done")
    if os.path.exists(done_marker):
        log("marker-present", done_marker, "— skipping downloads")
    else:
        for rid, key, allow in HF_MODELS:
            snapshot(rid, key, allow)
        for url, key in GIT_REPOS:
            git_clone(url, key)
    manifest = {
        "hf": {rid: key for rid, key, _ in HF_MODELS},
        "git": {url: key for url, key in GIT_REPOS},
    }
    mpath = os.path.join(MODELS_DIR, "manifests")
    os.makedirs(mpath, exist_ok=True)
    with open(os.path.join(mpath, "model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    # Summary
    pending = {k: v for k, v in STATUS.get("hf", {}).items() if v.get("state") != "done"}
    pending.update({f"git:{k}": v for k, v in STATUS.get("git", {}).items() if v.get("state") != "done"})
    if pending:
        log("SUMMARY: PENDING/MISSING", sorted(pending.keys()))
    else:
        log("SUMMARY: ALL MODELS DOWNLOADED")
    # Hard fail if any mandatory directory is missing or empty
    missing = [
        k for k in MANDATORY_DIRS
        if not (os.path.isdir(os.path.join(MODELS_DIR, k)) and os.listdir(os.path.join(MODELS_DIR, k)))
    ]
    if missing:
        log("ERROR: Missing mandatory models:", missing)
        sys.exit(3)
    log("SUMMARY: ALL MANDATORY MODELS DOWNLOADED")
    # Write done marker
    try:
        os.makedirs(os.path.dirname(done_marker), exist_ok=True)
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write(str(int(time.time())))
    except Exception:
        pass
    log("✅ bootstrap complete into", MODELS_DIR)


if __name__ == "__main__":
    main()


