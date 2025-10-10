from __future__ import annotations

import os
import subprocess
from typing import Optional

from huggingface_hub import hf_hub_download
import requests


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_to_dir(src: str, target_dir: str, rename: Optional[str] = None) -> None:
    ensure_dir(target_dir)
    dst = os.path.join(target_dir, rename or os.path.basename(src))
    os.system(f'cp -f "{src}" "{dst}"')
    print("Copied", src, "->", dst)


def dl(repo: str, filename: str, target_dir: str, rename: Optional[str] = None, token: Optional[str] = None) -> None:
    """Download a file from HuggingFace. Falls back to raw URL if hf_hub_download fails."""
    try:
        p = hf_hub_download(repo_id=repo, filename=filename, token=token)
        copy_to_dir(p, target_dir, rename)
        return
    except Exception as ex:
        print("hf_hub_download failed:", repo, filename, ex)
    # Fallback: try raw URL
    try:
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        ensure_dir(target_dir)
        dst = os.path.join(target_dir, rename or os.path.basename(filename))
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(dst, "wb") as f:
            f.write(r.content)
        print("Fetched via raw URL:", url, "->", dst)
    except Exception as ex:
        print("Raw URL fallback failed:", repo, filename, ex)


def dl_candidates(repo: str, filenames: list[str], target_dir: str, rename: Optional[str] = None, token: Optional[str] = None) -> None:
    """Try multiple candidate paths under a repo until one succeeds."""
    for fn in filenames:
        try:
            dl(repo, fn, target_dir, rename, token)
            return
        except Exception:
            continue
    print("All candidates failed for", repo, filenames)


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
    controlnet = [
        ("lllyasviel/control_v11p_sd15_canny", "diffusion_pytorch_model.safetensors", f"{MODELS_ROOT}/controlnet", "controlnet-canny-sd15.safetensors"),
        ("lllyasviel/control_v11p_sd15_openpose", "diffusion_pytorch_model.safetensors", f"{MODELS_ROOT}/controlnet", "controlnet-openpose-sd15.safetensors"),
        ("lllyasviel/control_v11f1e_sd15_tile", "diffusion_pytorch_model.safetensors", f"{MODELS_ROOT}/controlnet", "controlnet-tile-sd15.safetensors"),
    ]
    for repo, fn, tgt, rename in controlnet:
        dl(repo, fn, tgt, rename, token)

    # IP-Adapter (FaceID and encoder)
    ipadapter = [
        ("h94/IP-Adapter", "models/ip-adapter-plus-face_sdxl_vit-h.safetensors", f"{MODELS_ROOT}/ipadapter", "ip-adapter-plus-face_sdxl_vit-h.safetensors"),
        ("h94/IP-Adapter", "models/image_encoder/model.safetensors", f"{MODELS_ROOT}/ipadapter", "ip-adapter-image-encoder.safetensors"),
    ]
    for repo, fn, tgt, rename in ipadapter:
        dl(repo, fn, tgt, rename, token)

    # InstantID (ControlNet + adapter + insightface)
    instantid = [
        ("InstantX/InstantID", "models/ControlNetModel/diffusion_pytorch_model.safetensors", f"{MODELS_ROOT}/controlnet", "controlnet-instantid.safetensors"),
        ("InstantX/InstantID", "models/ip-adapter/instantid.safetensors", f"{MODELS_ROOT}/ipadapter", "instantid.safetensors"),
        ("InstantX/InstantID", "models/insightface/models/antelopev2/m2.0.onnx", f"{MODELS_ROOT}/insightface", "m2.0.onnx"),
        ("InstantX/InstantID", "models/insightface/models/antelopev2/genderage.onnx", f"{MODELS_ROOT}/insightface", "genderage.onnx"),
    ]
    for repo, fn, tgt, rename in instantid:
        dl(repo, fn, tgt, rename, token)

    # AnimateDiff motion modules (place where ComfyUI expects them)
    # Animatediff candidates: try multiple known paths per file
    dl_candidates("guoyww/animatediff",
                  [
                      "animatediff_motion_module/mm_sd_v15_v2.ckpt",
                      "v3_mm/sd15/mm_sd_v15_v2.ckpt",
                      "mm_sd_v15_v2.ckpt",
                  ],
                  f"{MODELS_ROOT}/animatediff_models", "mm_sd_v15_v2.ckpt", token)
    dl_candidates("guoyww/animatediff",
                  [
                      "animatediff_motion_module/mm_sdxl_v10.ckpt",
                      "v3_mm/sdxl/mm_sdxl_v10.ckpt",
                      "mm_sdxl_v10.ckpt",
                  ],
                  f"{MODELS_ROOT}/animatediff_models", "mm_sdxl_v10.ckpt", token)

    # Also mirror AnimateDiff motion models into the custom node's expected path so ComfyUI-AnimateDiff-Evolved finds them
    try:
        src_dir = f"{MODELS_ROOT}/animatediff_models"
        dst_dir = f"{CUSTOM_ROOT}/ComfyUI-AnimateDiff-Evolved/models"
        ensure_dir(dst_dir)
        for name in ("mm_sd_v15_v2.ckpt", "mm_sdxl_v10.ckpt"):
            p = os.path.join(src_dir, name)
            if os.path.exists(p):
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

    # Attempt to fetch popular Real-ESRGAN model
    try:
        dl("xinntao/Real-ESRGAN", "weights/realesr-general-x4v3.pth", f"{MODELS_ROOT}/upscale_models", "realesr-general-x4v3.pth", token)
    except Exception as ex:
        print("Failed to fetch Real-ESRGAN model:", ex)

    # Prepare directory for VideoHelperSuite RIFE models (download is optional; VHS can auto-download)
    try:
        ensure_dir(f"{MODELS_ROOT}/VideoHelperSuite/RIFE")
    except Exception as ex:
        print("Failed to prepare VHS RIFE dir:", ex)


if __name__ == "__main__":
    main()


