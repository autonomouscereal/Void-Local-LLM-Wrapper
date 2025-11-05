from __future__ import annotations

import time
import uuid
from typing import Dict, Any, List


def _default(val, d):
    return d if val is None else val


def build_full_graph(req: Dict[str, Any], uploaded: Dict[str, str]) -> Dict[str, Any]:
    # Auto-fill
    prompt = _default(req.get("prompt"), "")
    neg = _default(req.get("negative_prompt"), "extra fingers, deformed hands, lowres")
    steps = int(_default(req.get("steps"), 32))
    cfg = float(_default(req.get("cfg"), 5.5))
    sampler = _default(req.get("sampler"), "dpmpp_2m_sde_karras")
    seed = int(_default(req.get("seed"), int(time.time()) & 0x7FFFFFFF))
    w = int(_default(req.get("width"), 1024))
    h = int(_default(req.get("height"), 1024))
    fname = f"job_{uuid.uuid4().hex[:8]}"

    nodes: Dict[str, Dict[str, Any]] = {}
    # 3: SDXL
    nodes["3"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}}
    # positive / negative text encodes
    nodes["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 1]}}
    nodes["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["3", 1]}}
    # latent image
    nodes["lat"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}

    # Build adapters + controlnets from ALP-like request
    locks = req.get("locks") or {}
    model_in: List[Any] = ["3", 0]

    # FaceID adapters
    for i, f in enumerate(locks.get("faces") or []):
        face_img_b64 = f.get("ref_b64")
        face_filename = uploaded.get(f"face_{i}") or uploaded.get("face")
        if not face_filename and not face_img_b64:
            continue
        # Loader and Apply
        lid = f"fload_{i}"
        aid = f"fapply_{i}"
        # Load image to tensor first
        img_node = f"fimg_{i}"
        nodes[img_node] = {"class_type": "LoadImage", "inputs": {"image": face_filename or face_img_b64}}
        nodes[lid] = {"class_type": "IPAdapterModelLoader", "inputs": {"model_name": "ip-adapter-faceid-plusv2_sdxl.bin"}}
        nodes[aid] = {
            "class_type": "IPAdapterApply",
            "inputs": {
                "model": [lid, 0],
                "weight": float(f.get("weight", 0.9)),
                "image": [img_node, 0],
                "clip_vision": ["3", 1],
                "vae": ["3", 2],
                "model": model_in,
            },
        }
        model_in = [aid, 0]

    # Style/Color adapters
    for i, s in enumerate(locks.get("clothes_styles") or []):
        s_img_b64 = s.get("ref_b64")
        s_filename = uploaded.get(f"style_{i}") or uploaded.get("style")
        if not s_filename and not s_img_b64:
            continue
        lid = f"sload_{i}"
        aid = f"sapply_{i}"
        img_node = f"simg_{i}"
        nodes[img_node] = {"class_type": "LoadImage", "inputs": {"image": s_filename or s_img_b64}}
        nodes[lid] = {"class_type": "IPAdapterModelLoader", "inputs": {"model_name": "ip-adapter-plus_sdxl_vit-h.safetensors"}}
        nodes[aid] = {
            "class_type": "IPAdapterApply",
            "inputs": {
                "model": [lid, 0],
                "weight": float(s.get("weight", 0.7)),
                "image": [img_node, 0],
                "clip_vision": ["3", 1],
                "vae": ["3", 2],
                "model": model_in,
            },
        }
        model_in = [aid, 0]

    def add_cnet(name: str, image_key: str, weight: float, idx: int) -> List[Any]:
        l = f"cnl_{idx}_{name}"
        a = f"cna_{idx}_{name}"
        img_n = f"cnimg_{idx}_{name}"
        nodes[img_n] = {"class_type": "LoadImage", "inputs": {"image": uploaded.get(image_key)}}
        nodes[l] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": name}}
        nodes[a] = {
            "class_type": "ApplyControlNet",
            "inputs": {
                "strength": float(weight),
                "start_percent": 0.0,
                "end_percent": 1.0,
                "control_net": [l, 0],
                "image": [img_n, 0],
                "model": model_in,
            },
        }
        return [a, 0]

    # Layout
    for i, lref in enumerate(locks.get("layouts") or []):
        cnet_name = "controlnet-sd-xl-1.0-softedge-dexined.safetensors" if (lref.get("type", "softedge") == "softedge") else "controlnet-lineart-sdxl.safetensors"
        model_in = add_cnet(cnet_name, f"layout_{i}", lref.get("weight", 0.6), i)

    # Depth
    for i, d in enumerate(locks.get("depths") or []):
        model_in = add_cnet("controlnet-depth-sdxl-1.0.safetensors", f"depth_{i}", d.get("weight", 0.6), i)

    # Pose (only if present)
    for i, p in enumerate(locks.get("poses") or []):
        model_in = add_cnet("controlnet-openpose-sdxl-1.0.safetensors", f"pose_{i}", p.get("weight", 1.0), i)

    # Sampler / decode / save
    nodes["sam"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": "normal",
            "denoise": 1.0,
            "model": model_in,
            "positive": ["pos", 0],
            "negative": ["neg", 0],
            "latent_image": ["lat", 0],
        },
    }
    nodes["vae"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sam", 0], "vae": ["3", 2]}}
    nodes["save"] = {"class_type": "SaveImage", "inputs": {"images": ["vae", 0], "filename_prefix": fname}}

    # Hi-res
    hires = (req.get("hires") or {}).get("enabled", False)
    if hires:
        factor = float((req.get("hires") or {}).get("latent_upscale_factor", 2.0))
        nodes["up"] = {"class_type": "LatentUpscale", "inputs": {"samples": ["sam", 0], "upscale_method": "nearest-exact", "scale": factor}}
        nodes["vae"] = {"class_type": "VAEDecode", "inputs": {"samples": ["up", 0], "vae": ["3", 2]}}
        nodes["save"]["inputs"]["images"] = ["vae", 0]

    return {"prompt": nodes}


