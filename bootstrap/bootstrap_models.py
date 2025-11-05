import os
import sys
import json
import subprocess


MODELS_DIR = os.environ.get("FILM2_MODELS", "/opt/models")
HF_HOME = os.environ.get("HF_HOME", os.path.join(MODELS_DIR, ".hf"))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["TORCH_HOME"] = os.path.join(MODELS_DIR, ".torch")


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
]


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


def log(*a: object) -> None:
    print("[bootstrap]", *a, flush=True)


def ensure_pkg() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub>=0.23.0"])  # noqa: S603,S607


def snapshot(repo_id: str, local_key: str, allow_patterns: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download
    tgt = os.path.join(MODELS_DIR, local_key)
    if os.path.isdir(tgt) and os.listdir(tgt):
        log("exists", tgt)
        return
    os.makedirs(tgt, exist_ok=True)
    kw = dict(repo_id=repo_id, local_dir=tgt, local_dir_use_symlinks=False, resume_download=True)
    if allow_patterns:
        kw["allow_patterns"] = allow_patterns
    snapshot_download(**kw)
    log("dl-ok", repo_id, "->", tgt)


def git_clone(url: str, local_key: str) -> None:
    tgt = os.path.join(MODELS_DIR, local_key)
    if os.path.exists(os.path.join(tgt, ".git")) or os.path.isdir(tgt):
        log("exists", tgt)
        return
    subprocess.check_call(["git", "clone", "--depth=1", url, tgt])  # noqa: S603,S607
    log("git-ok", url, "->", tgt)


def main() -> None:
    ensure_pkg()
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
    log("✅ bootstrap complete into", MODELS_DIR)


if __name__ == "__main__":
    main()


