import os
import sys

REQUIRED = [
    "/opt/models/hunyuan",
    "/opt/models/instantid",
    "/opt/models/ip_adapter",
    "/opt/models/controlnet",
    "/opt/models/sam/sam2_hiera_large.pt",
    "/opt/models/depth_anything",
    "/opt/models/clip",
    "/opt/models/clap",
    "/opt/models/blip2",
    "/opt/models/whisper",
]


def check(paths=REQUIRED) -> None:
    missing: list[str] = []
    for p in paths:
        if p.endswith(".pt"):
            if not os.path.isfile(p):
                missing.append(p)
        else:
            if (not os.path.isdir(p)) or (not os.listdir(p)):
                missing.append(p)
    if missing:
        print("[fatal] missing mandatory models:", missing, file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    check()


