from __future__ import annotations

import logging
import os
import subprocess
import sys
import zipfile
from typing import Optional
import time

from huggingface_hub import hf_hub_download
import requests


try:
    from logging.handlers import RotatingFileHandler

    _log_dir = os.getenv("LOG_DIR", "/workspace/logs").strip() or "/workspace/logs"
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.getenv("LOG_FILE", "").strip() or os.path.join(_log_dir, "comfyui_init.log")
    _lvl = getattr(logging, (os.getenv("LOG_LEVEL", "INFO") or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_lvl,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(process)d/%(threadName)s %(name)s %(pathname)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(_log_file, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"),
        ],
        force=True,
    )
    logging.getLogger("comfyui_init.logging").info(
        "comfyui_init logging configured file=%r level=%s", _log_file, logging.getLevelName(_lvl)
    )
except Exception as _ex:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logging.getLogger("comfyui_init.logging").warning("comfyui_init file logging disabled: %s", _ex, exc_info=True)

log = logging.getLogger("comfyui_init")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def file_exists_nonempty(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def copy_to_dir(src: str, target_dir: str, rename: Optional[str] = None) -> None:
    ensure_dir(target_dir)
    dst = os.path.join(target_dir, rename or os.path.basename(src))
    if file_exists_nonempty(dst):
        log.info("copy_to_dir exists skipping dst=%r", dst)
        return
    os.system(f'cp -f "{src}" "{dst}"')
    log.info("copy_to_dir copied src=%r dst=%r", src, dst)


def dl(repo: str, filename: str, target_dir: str, rename: Optional[str] = None, token: Optional[str] = None) -> None:
    """Download a file from HuggingFace. Falls back to raw URL if hf_hub_download fails."""
    ensure_dir(target_dir)
    dst = os.path.join(target_dir, rename or os.path.basename(filename))
    if file_exists_nonempty(dst):
        log.info("dl exists skipping dst=%r repo=%r filename=%r", dst, repo, filename)
        return
    try:
        t0 = time.perf_counter()
        p = hf_hub_download(repo_id=repo, filename=filename, token=token)
        copy_to_dir(p, target_dir, rename)
        log.info(
            "dl hf_hub_download ok repo=%r filename=%r dst=%r duration_ms=%s",
            repo,
            filename,
            dst,
            int((time.perf_counter() - t0) * 1000),
        )
        return
    except Exception as ex:
        log.warning("dl hf_hub_download failed repo=%r filename=%r err=%s", repo, filename, ex, exc_info=True)
    # Fallback: try raw URL
    try:
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        headers = {"Authorization": f"Bearer {token}"} if token else None
        t0 = time.perf_counter()
        r = requests.get(url, headers=headers)
        with open(dst, "wb") as f:
            f.write(r.content)
        log.info(
            "dl raw_url ok url=%r dst=%r status=%s bytes=%s duration_ms=%s",
            url,
            dst,
            int(getattr(r, "status_code", 0) or 0),
            int(len(getattr(r, "content", b"") or b"")),
            int((time.perf_counter() - t0) * 1000),
        )
    except Exception as ex:
        log.error("dl raw_url failed repo=%r filename=%r err=%s", repo, filename, ex, exc_info=True)


def dl_candidates(repo: str, filenames: list[str], target_dir: str, rename: Optional[str] = None, token: Optional[str] = None) -> None:
    """Try multiple candidate paths under a repo until one succeeds."""
    ensure_dir(target_dir)
    for fn in filenames:
        try:
            dst = os.path.join(target_dir, rename or os.path.basename(fn))
            log.info("dl_candidates try repo=%r fn=%r dst=%r", repo, fn, dst)
            dl(repo, fn, target_dir, rename, token)
            if file_exists_nonempty(dst):
                log.info("dl_candidates success repo=%r chosen_fn=%r dst=%r", repo, fn, dst)
                return
        except Exception as ex:
            log.warning("dl_candidates candidate failed repo=%r fn=%r err=%s", repo, fn, ex, exc_info=True)
            continue
    log.error("dl_candidates all candidates failed repo=%r filenames=%r", repo, filenames)


def download_with_retries(url: str, dst: str, max_retries: int = 3, backoff_seconds: int = 2, headers: Optional[dict] = None) -> None:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.perf_counter()
            r = requests.get(url, headers=headers)
            with open(dst, "wb") as f:
                f.write(r.content)
            log.info(
                "download ok attempt=%s/%s url=%r dst=%r status=%s bytes=%s duration_ms=%s",
                attempt,
                max_retries,
                url,
                dst,
                int(getattr(r, "status_code", 0) or 0),
                int(len(getattr(r, "content", b"") or b"")),
                int((time.perf_counter() - t0) * 1000),
            )
            return
        except Exception as ex:
            last_err = ex
            log.warning(
                "download failed attempt=%s/%s url=%r err=%s",
                attempt,
                max_retries,
                url,
                ex,
                exc_info=True,
            )
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    log.error("download failed exhausted url=%r dst=%r last_err=%s", url, dst, last_err)


def download_url(url: str, target_dir: str, filename: str) -> None:
    ensure_dir(target_dir)
    dst = os.path.join(target_dir, filename)
    if file_exists_nonempty(dst):
        log.info("download_url exists skipping dst=%r", dst)
        return
    download_with_retries(url, dst)


def main() -> None:
    token = os.environ.get("HUGGINGFACE_TOKEN")
    log.info("comfyui_init start token_present=%s", bool(token))
    # Root paths aligned with docker-compose volume mounts
    MODELS_ROOT = "/comfyui/models"
    CUSTOM_ROOT = "/comfyui/custom_nodes"

    # Base SDXL checkpoints
    base_ckpts = [
        ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", f"{MODELS_ROOT}/checkpoints", None),
        ("stabilityai/stable-diffusion-xl-refiner-1.0", "sd_xl_refiner_1.0.safetensors", f"{MODELS_ROOT}/checkpoints", None),
    ]
    for repo, fn, tgt, rename in base_ckpts:
        dl(repo, fn, tgt, rename, token)

    # ControlNet (SD1.5 variants)
    # ControlNet SD15: support fp16 filenames as fallback
    dl_candidates("lllyasviel/control_v11p_sd15_canny",
                  ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors"],
                  f"{MODELS_ROOT}/controlnet", "controlnet-canny-sd15.safetensors", token)
    dl_candidates("lllyasviel/control_v11p_sd15_openpose",
                  ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors"],
                  f"{MODELS_ROOT}/controlnet", "controlnet-openpose-sd15.safetensors", token)
    dl_candidates("lllyasviel/control_v11f1e_sd15_tile",
                  ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.fp16.safetensors"],
                  f"{MODELS_ROOT}/controlnet", "controlnet-tile-sd15.safetensors", token)

    # IP-Adapter (FaceID and encoder)
    # IP-Adapter: try multiple face SDXL filenames
    dl_candidates("h94/IP-Adapter",
                  [
                      "models/ip-adapter-plus-face_sdxl_vit-h.safetensors",
                      "models/ip-adapter-plus-face_sdxl.safetensors",
                      "models/ip-adapter-faceid-plus_sdxl.safetensors",
                  ],
                  f"{MODELS_ROOT}/ipadapter", "ip-adapter-plus-face_sdxl_vit-h.safetensors", token)
    dl("h94/IP-Adapter", "models/image_encoder/model.safetensors", f"{MODELS_ROOT}/ipadapter", "ip-adapter-image-encoder.safetensors", token)

    # InstantID (ControlNet + adapter) + antelopev2 InsightFace pack.
    # InstantID weights are optional (some paths require license/cookies). Try, but do not block init if 404.
    try:
        dl_candidates(
            "InstantX/InstantID",
            ["models/ControlNetModel/diffusion_pytorch_model.safetensors"],
            f"{MODELS_ROOT}/controlnet",
            "controlnet-instantid.safetensors",
            token,
        )
        dl(
            "InstantX/InstantID",
            "models/ip-adapter/instantid.safetensors",
            f"{MODELS_ROOT}/ipadapter",
            "instantid.safetensors",
            token,
        )
    except Exception as ex:
        log.warning("InstantID assets optional; continuing err=%s", ex, exc_info=True)

    # Canonical InsightFace antelopev2 ONNX pack for InstantID / FaceID.
    # Use LPDoctor/insightface, which exposes the full antelopev2 directory.
    try:
        antel_dir = f"{MODELS_ROOT}/insightface/models/antelopev2"
        ensure_dir(antel_dir)
        antel_base = "https://huggingface.co/LPDoctor/insightface/resolve/main/models/antelopev2"
        antel_files = [
            "1k3d68.onnx",
            "2d106det.onnx",
            "genderage.onnx",
            "glintr100.onnx",
            "scrfd_10g_bnkps.onnx",
        ]
        for name in antel_files:
            url = f"{antel_base}/{name}"
            download_url(url, antel_dir, name)
    except Exception as ex:
        # Antelopev2 is strongly recommended for InstantID; log loudly but do not abort container init.
        log.error("Failed to fetch antelopev2 ONNX pack err=%s", ex, exc_info=True)

    # AnimateDiff motion modules (public stabilized alternatives)
    try:
        download_url(
            "https://huggingface.co/manshoety/AD_Stabilized_Motion/resolve/main/mm-Stabilized_mid.pth",
            f"{MODELS_ROOT}/animatediff_models",
            "mm-Stabilized_mid.pth",
        )
        download_url(
            "https://huggingface.co/manshoety/AD_Stabilized_Motion/resolve/main/mm-Stabilized_high.pth",
            f"{MODELS_ROOT}/animatediff_models",
            "mm-Stabilized_high.pth",
        )
    except Exception as ex:
        log.warning("Failed to fetch stabilized Animatediff modules err=%s", ex, exc_info=True)

    # Also mirror all AnimateDiff motion modules (.ckpt/.pth) to the custom node path so Evolved finds them
    try:
        src_dir = f"{MODELS_ROOT}/animatediff_models"
        dst_dir = f"{CUSTOM_ROOT}/ComfyUI-AnimateDiff-Evolved/models"
        ensure_dir(dst_dir)
        if os.path.isdir(src_dir):
            for name in os.listdir(src_dir):
                if name.endswith(".ckpt") or name.endswith(".pth"):
                    p = os.path.join(src_dir, name)
                    copy_to_dir(p, dst_dir)
    except Exception as ex:
        log.warning("Failed to mirror AnimateDiff models err=%s", ex, exc_info=True)

    # Clone commonly used custom nodes (idempotent)
    nodes = [
        ("https://github.com/cubiq/ComfyUI_IPAdapter_plus", f"{CUSTOM_ROOT}/ComfyUI_IPAdapter_plus"),
        ("https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID", f"{CUSTOM_ROOT}/ComfyUI-InstantID"),
        ("https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved", f"{CUSTOM_ROOT}/ComfyUI-AnimateDiff-Evolved"),
        ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite", f"{CUSTOM_ROOT}/ComfyUI-VideoHelperSuite"),
        # RealESRGAN clone can fail without GitHub creds; treat as optional
        # ("https://github.com/KoreTeknology/ComfyUI-RealESRGAN", f"{CUSTOM_ROOT}/ComfyUI-RealESRGAN"),
    ]
    for url, path in nodes:
        try:
            if not os.path.exists(path):
                log.info("git clone start url=%r path=%r", url, path)
                subprocess.run(["git", "clone", "--depth", "1", url, path], check=False)
            else:
                log.info("git clone exists skipping path=%r", path)
        except Exception as ex:
            log.warning("git clone failed url=%r err=%s", url, ex, exc_info=True)

    # Attempt to fetch popular Real-ESRGAN model (direct GitHub release avoids HF 401)
    try:
        download_url(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            f"{MODELS_ROOT}/upscale_models",
            "realesr-general-x4v3.pth",
        )
    except Exception as ex:
        log.warning("Failed to fetch Real-ESRGAN model via GitHub err=%s", ex, exc_info=True)

    # InsightFace antelopev2 (public zip -> extract); satisfies face embeddings without gated InstantID
    try:
        dest_dir = f"{MODELS_ROOT}/insightface/models/antelopev2"
        m2 = os.path.join(dest_dir, "m2.0.onnx")
        ga = os.path.join(dest_dir, "genderage.onnx")
        if file_exists_nonempty(m2) and file_exists_nonempty(ga):
            log.info("InsightFace antelopev2 exists, skipping download dest_dir=%r", dest_dir)
        else:
            ensure_dir(dest_dir)
            tmp_zip = "/tmp/antelopev2.zip"
            headers = {"Authorization": f"Bearer {token}"} if token else None
            download_with_retries(
                "https://huggingface.co/deepinsight/insightface/resolve/main/models/antelopev2.zip",
                tmp_zip,
                max_retries=3,
                backoff_seconds=3,
                headers=headers,
            )
            with zipfile.ZipFile(tmp_zip, 'r') as z:
                z.extractall(path=f"{MODELS_ROOT}/insightface/models/")
            log.info("Extracted antelopev2.zip to %r", f"{MODELS_ROOT}/insightface/models/")
    except Exception as ex:
        log.warning("Failed to fetch/extract InsightFace antelopev2 err=%s", ex, exc_info=True)

    # Prepare directory for VideoHelperSuite RIFE models (download is optional; VHS can auto-download)
    try:
        ensure_dir(f"{MODELS_ROOT}/VideoHelperSuite/RIFE")
    except Exception as ex:
        log.warning("Failed to prepare VHS RIFE dir err=%s", ex, exc_info=True)

    log.info("comfyui_init finished")


if __name__ == "__main__":
    main()


