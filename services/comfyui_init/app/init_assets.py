from __future__ import annotations

import os
import subprocess
from typing import Optional

from huggingface_hub import hf_hub_download


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_to_dir(src: str, target_dir: str, rename: Optional[str] = None) -> None:
    ensure_dir(target_dir)
    dst = os.path.join(target_dir, rename or os.path.basename(src))
    os.system(f'cp -f "{src}" "{dst}"')
    print("Copied", src, "->", dst)


def dl(repo: str, filename: str, target_dir: str, rename: Optional[str] = None, token: Optional[str] = None) -> None:
    try:
        p = hf_hub_download(repo_id=repo, filename=filename, token=token)
        copy_to_dir(p, target_dir, rename)
    except Exception as ex:
        print("Failed to download", repo, filename, ex)


def main() -> None:
    token = os.environ.get("HUGGINGFACE_TOKEN")

    # Base SDXL checkpoints
    base_ckpts = [
        ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", "/models/checkpoints", None),
        ("stabilityai/stable-diffusion-xl-refiner-1.0", "sd_xl_refiner_1.0.safetensors", "/models/checkpoints", None),
    ]
    for repo, fn, tgt, rename in base_ckpts:
        dl(repo, fn, tgt, rename, token)

    # ControlNet (SD1.5 variants)
    controlnet = [
        ("lllyasviel/control_v11p_sd15_canny", "diffusion_pytorch_model.safetensors", "/models/controlnet", "controlnet-canny-sd15.safetensors"),
        ("lllyasviel/control_v11p_sd15_openpose", "diffusion_pytorch_model.safetensors", "/models/controlnet", "controlnet-openpose-sd15.safetensors"),
        ("lllyasviel/control_v11f1e_sd15_tile", "diffusion_pytorch_model.safetensors", "/models/controlnet", "controlnet-tile-sd15.safetensors"),
    ]
    for repo, fn, tgt, rename in controlnet:
        dl(repo, fn, tgt, rename, token)

    # IP-Adapter (FaceID and encoder)
    ipadapter = [
        ("h94/IP-Adapter", "models/ip-adapter-plus-face_sdxl_vit-h.safetensors", "/models/ipadapter", "ip-adapter-plus-face_sdxl_vit-h.safetensors"),
        ("h94/IP-Adapter", "models/image_encoder/model.safetensors", "/models/ipadapter", "ip-adapter-image-encoder.safetensors"),
    ]
    for repo, fn, tgt, rename in ipadapter:
        dl(repo, fn, tgt, rename, token)

    # InstantID (ControlNet + adapter + insightface)
    instantid = [
        ("InstantX/InstantID", "models/ControlNetModel/diffusion_pytorch_model.safetensors", "/models/controlnet", "controlnet-instantid.safetensors"),
        ("InstantX/InstantID", "models/ip-adapter/instantid.safetensors", "/models/ipadapter", "instantid.safetensors"),
        ("InstantX/InstantID", "models/insightface/models/antelopev2/m2.0.onnx", "/models/insightface", "m2.0.onnx"),
        ("InstantX/InstantID", "models/insightface/models/antelopev2/genderage.onnx", "/models/insightface", "genderage.onnx"),
    ]
    for repo, fn, tgt, rename in instantid:
        dl(repo, fn, tgt, rename, token)

    # AnimateDiff motion modules
    animatediff = [
        ("guoyww/animatediff", "animatediff_motion_module/mm_sd_v15_v2.ckpt", "/models/animatediff", "mm_sd_v15_v2.ckpt"),
        ("guoyww/animatediff", "animatediff_motion_module/mm_sdxl_v10.ckpt", "/models/animatediff", "mm_sdxl_v10.ckpt"),
    ]
    for repo, fn, tgt, rename in animatediff:
        dl(repo, fn, tgt, rename, token)

    # Clone commonly used custom nodes (idempotent)
    nodes = [
        ("https://github.com/cubiq/ComfyUI_IPAdapter_plus", "/custom_nodes/ComfyUI_IPAdapter_plus"),
        ("https://github.com/ZHO-ZHO-ZHO/ComfyUI-InstantID", "/custom_nodes/ComfyUI-InstantID"),
        ("https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved", "/custom_nodes/ComfyUI-AnimateDiff-Evolved"),
        ("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite", "/custom_nodes/ComfyUI-VideoHelperSuite"),
        ("https://github.com/KoreTeknology/ComfyUI-RealESRGAN", "/custom_nodes/ComfyUI-RealESRGAN"),
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
        dl("xinntao/Real-ESRGAN", "weights/realesr-general-x4v3.pth", "/models/upscale_models", "realesr-general-x4v3.pth", token)
    except Exception as ex:
        print("Failed to fetch Real-ESRGAN model:", ex)

    # Prepare directory for VideoHelperSuite RIFE models (download is optional; VHS can auto-download)
    try:
        ensure_dir("/models/VideoHelperSuite/RIFE")
    except Exception as ex:
        print("Failed to prepare VHS RIFE dir:", ex)


if __name__ == "__main__":
    main()


