import os
import sys
import json
import subprocess
import threading
import time
from pathlib import Path


MODELS_DIR = os.environ.get("FILM2_MODELS", "/opt/models")
HF_HOME = os.environ.get("HF_HOME", os.path.join(MODELS_DIR, ".hf"))
# Optional Hugging Face token (forwarded from docker-compose as HUGGINGFACE_TOKEN).
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

# status tracking
STATUS_PATH = os.path.join(MODELS_DIR, "manifests", "bootstrap_status.json")
STATUS: dict[str, dict[str, dict[str, str]]] = {"hf": {}, "git": {}}

def write_status() -> None:
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(STATUS, f, indent=2)


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

# Never append external/custom music engines dynamically in bootstrap; when used
# they are handled as local artifacts outside this manifest.


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
    "rvc_titan",
]
# Do not require external/custom music engines as mandatory; removed from bootstrap scope


def ensure_aesthetic_head() -> None:
    """
    Ensure the LAION aesthetic head weights are present under MODELS_DIR/aesthetic.

    This uses AESTHETIC_HEAD_URL if provided; otherwise it is a no-op. The orchestrator
    will look for the file at AESTHETIC_HEAD_PATH (defaulting to /opt/models/aesthetic/...),
    which should map into this directory via docker-compose volumes.
    """
    url = os.environ.get("AESTHETIC_HEAD_URL", "").strip()
    if not url:
        return
    import urllib.request  # noqa: S310

    target_dir = os.path.join(MODELS_DIR, "aesthetic")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "laion_v2_linear_L14.pt")
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        log("exists", "aesthetic_head", "->", target_path)
        return
    log("START-AESTHETIC", url, "->", target_path)
    try:
        with urllib.request.urlopen(url) as resp:  # nosec: B310 – trusted URL configured by user
            with open(target_path, "wb") as f:
                f.write(resp.read())
        log("DONE-AESTHETIC", url, "->", target_path)
    except Exception as ex:
        log("ERROR-AESTHETIC", url, "->", target_path, ":", ex)


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
        self._err_count = 0
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop:
            total = 0
            for root, _, files in os.walk(self.path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total += os.path.getsize(fp)
                    except Exception as exc:
                        # Non-fatal; progress reporting must never crash bootstrap.
                        # But it must also not be silent; rate-limit warnings.
                        self._err_count += 1
                        if self._err_count <= 3 or (self._err_count % 100) == 0:
                            log("WARN: SizeBeat getsize failed", fp, ":", exc)
            log(f"PROGRESS {self.label}", human(total))
            time.sleep(self.interval)

    def start(self) -> None:
        self.t.start()

    def stop(self) -> None:
        self._stop = True
        try:
            self.t.join()
        except Exception as exc:
            log("WARN: SizeBeat join failed", self.label, ":", exc)
            return


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
        # Forward the compose-level token when available so private/gated models
        # can be accessed without relying on global env defaults.
        if HF_TOKEN:
            kw["token"] = HF_TOKEN
        try:
            snapshot_download(**kw)
        except RuntimeError as e:
            msg = str(e).lower()
            if ("xet" in msg) or ("cas service" in msg):
                os.environ["HF_HUB_ENABLE_XET"] = "0"
                kw["max_workers"] = 2
                snapshot_download(**kw)
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


def ensure_rvc_titan() -> None:
    """
    Ensure the Titan RVC pretrained weights are present under MODELS_DIR/rvc_titan.

    We rely primarily on snapshot() to clone the full blaise-tk/TITAN repo, but
    this helper guarantees that the 48k Medium D/G checkpoints are materialized
    inside /opt/models/rvc_titan so the rvc_service startup check never sees an
    empty directory.
    """
    from huggingface_hub import hf_hub_download

    target_dir = os.path.join(MODELS_DIR, "rvc_titan")
    os.makedirs(target_dir, exist_ok=True)
    d_name = "D-f048k-TITAN-Medium.pth"
    g_name = "G-f048k-TITAN-Medium.pth"
    d_path = os.path.join(target_dir, d_name)
    g_path = os.path.join(target_dir, g_name)

    # If both files already exist and are non-empty, nothing to do.
    if os.path.isfile(d_path) and os.path.getsize(d_path) > 0 and os.path.isfile(g_path) and os.path.getsize(g_path) > 0:
        log("exists", "rvc_titan.DG48k", "->", target_dir)
        return

    log("START-RVC-TITAN-DG48k", "blaise-tk/TITAN", "->", target_dir)
    # Use the explicit 48k Medium pretrained paths for D and G.
    d_remote = "models/medium/48k/pretrained/" + d_name
    g_remote = "models/medium/48k/pretrained/" + g_name

    # Download D
    d_local = hf_hub_download(
        repo_id="blaise-tk/TITAN",
        filename=d_remote,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
    # Download G
    g_local = hf_hub_download(
        repo_id="blaise-tk/TITAN",
        filename=g_remote,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )
    log("DONE-RVC-TITAN-DG48k", d_local, g_local, "->", target_dir)


def main() -> None:
    ensure_pkg()
    done_marker = os.path.join(MODELS_DIR, "manifests", ".bootstrap_done")
    if os.path.exists(done_marker):
        # Marker means "bootstrap completed at least once", NOT "skip all work".
        # We still re-verify all mandatory models on every run so newly-added
        # directories like rvc_titan are populated even after the initial bootstrap.
        log("marker-present", done_marker, "— will still verify all mandatory models")
    else:
        log("marker-missing", done_marker, "— full bootstrap will run")
    # Idempotent per-model downloads; snapshot/git_clone short-circuit when targets exist.
    for rid, key, allow in HF_MODELS:
        snapshot(rid, key, allow)
    for url, key in GIT_REPOS:
        git_clone(url, key)
    # Ensure Titan's 48k Medium D/G checkpoints are present for rvc_service,
    # then optional aesthetic head weights for orchestrator image scoring.
    ensure_rvc_titan()
    ensure_aesthetic_head()
    manifest = {
        "hf": {rid: key for rid, key, _ in HF_MODELS},
        "git": {url: key for url, key in GIT_REPOS},
    }
    mpath = os.path.join(MODELS_DIR, "manifests")
    os.makedirs(mpath, exist_ok=True)
    with open(os.path.join(mpath, "model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    # Summary for HF/git sources (best effort; STATUS only tracks fresh downloads).
    pending = {k: v for k, v in STATUS.get("hf", {}).items() if v.get("state") != "done"}
    pending.update({f"git:{k}": v for k, v in STATUS.get("git", {}).items() if v.get("state") != "done"})
    if pending:
        log("SUMMARY: PENDING/MISSING", sorted(pending.keys()))
    # Hard fail if any mandatory directory is missing or empty (authoritative check).
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
        log("WARN: failed to write done marker", done_marker)
    log("✅ bootstrap complete into", MODELS_DIR)


if __name__ == "__main__":
    main()


