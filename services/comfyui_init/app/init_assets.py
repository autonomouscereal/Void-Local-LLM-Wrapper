from __future__ import annotations

import os
import subprocess
import zipfile
from typing import Optional
import time

from huggingface_hub import hf_hub_download
import requests


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
        print("Exists, skipping:", dst)
        return
    os.system(f'cp -f "{src}" "{dst}"')
    print("Copied", src, "->", dst)


def dl(repo: str, filename: str, target_dir: str, rename: Optional[str] = None, token: Optional[str] = None) -> None:
    """Download a file from HuggingFace. Falls back to raw URL if hf_hub_download fails."""
    ensure_dir(target_dir)
    dst = os.path.join(target_dir, rename or os.path.basename(filename))
    if file_exists_nonempty(dst):
        print("Exists, skipping:", dst)
        return
    try:
        p = hf_hub_download(repo_id=repo, filename=filename, token=token)
        copy_to_dir(p, target_dir, rename)
        return
    except Exception as ex:
        print("hf_hub_download failed:", repo, filename, ex)
    # Fallback: try raw URL
    try:
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        headers = {"Authorization": f"Bearer {token}"} if token else None
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        with open(dst, "wb") as f:
            f.write(r.content)
        print("Fetched via raw URL:", url, "->", dst)
    except Exception as ex:
        print("Raw URL fallback failed:", repo, filename, ex)


def dl_candidates(repo: str, filenames: list[str], target_dir: str, rename: Optional[str] = None, token: Optional[str] = None) -> None:
    """Try multiple candidate paths under a repo until one succeeds."""
    ensure_dir(target_dir)
    for fn in filenames:
        try:
            dst = os.path.join(target_dir, rename or os.path.basename(fn))
            dl(repo, fn, target_dir, rename, token)
            if file_exists_nonempty(dst):
                return
        except Exception as ex:
            print("Candidate failed:", repo, fn, ex)
            continue
    print("All candidates failed for", repo, filenames)


def download_with_retries(url: str, dst: str, timeout: int = 300, max_retries: int = 3, backoff_seconds: int = 2, headers: Optional[dict] = None) -> None:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            with open(dst, "wb") as f:
                f.write(r.content)
            print(f"Downloaded ({attempt}/{max_retries}):", url, "->", dst)
            return
        except Exception as ex:
            last_err = ex
            print(f"Download failed (attempt {attempt}/{max_retries}):", url, ex)
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
    if last_err:


def download_url(url: str, target_dir: str, filename: str) -> None:
    ensure_dir(target_dir)
    dst = os.path.join(target_dir, filename)
    if file_exists_nonempty(dst):
        print("Exists, skipping:", dst)
        return
    download_with_retries(url, dst)


def main() -> None:
    token = os.environ.get("HUGGINGFACE_TOKEN")
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
        print("InstantID assets (controlnet/ip-adapter) optional, continuing:", ex)

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
        print("Failed to fetch antelopev2 ONNX pack:", ex)

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
        print("Failed to fetch stabilized Animatediff modules:", ex)

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
        print("Failed to mirror AnimateDiff models to custom_nodes path:", ex)

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
                print("Cloning", url, "->", path)
                subprocess.run(["git", "clone", "--depth", "1", url, path], check=False)
            else:
                print("Exists, skipping clone:", path)
        except Exception as ex:
            print("Failed to clone", url, ex)

    # Attempt to fetch popular Real-ESRGAN model (direct GitHub release avoids HF 401)
    try:
        download_url(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
            f"{MODELS_ROOT}/upscale_models",
            "realesr-general-x4v3.pth",
        )
    except Exception as ex:
        print("Failed to fetch Real-ESRGAN model via GitHub:", ex)

    # InsightFace antelopev2 (public zip -> extract); satisfies face embeddings without gated InstantID
    try:
        dest_dir = f"{MODELS_ROOT}/insightface/models/antelopev2"
        m2 = os.path.join(dest_dir, "m2.0.onnx")
        ga = os.path.join(dest_dir, "genderage.onnx")
        if file_exists_nonempty(m2) and file_exists_nonempty(ga):
            print("InsightFace antelopev2 exists, skipping download")
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
            print("Extracted antelopev2.zip to", f"{MODELS_ROOT}/insightface/models/")
    except Exception as ex:
        print("Failed to fetch/extract InsightFace antelopev2:", ex)

    # Prepare directory for VideoHelperSuite RIFE models (download is optional; VHS can auto-download)
    try:
        ensure_dir(f"{MODELS_ROOT}/VideoHelperSuite/RIFE")
    except Exception as ex:
        print("Failed to prepare VHS RIFE dir:", ex)


if __name__ == "__main__":
    main()


