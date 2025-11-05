import os
import sys
try:
    import torch  # type: ignore
except Exception as _e:  # pragma: no cover
    print(f"[guard] torch not importable: {_e}", file=sys.stderr)
    sys.exit(5)


def ensure_cuda_safe() -> str:
    """Return 'cpu' or 'cuda:<idx>' and enforce cu118 on Pascal (sm_61).

    - If CUDA is unavailable, force CPU and keep service healthy.
    - If device capability is (6, 1) (Pascal/P40) but torch is not a +cu118 wheel,
      print fatal and exit 5 (prevents cryptic 'no kernel image' later).
    """
    if not torch.cuda.is_available():
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        print("[guard] CUDA not available -> CPU mode")
        return "cpu"
    dev = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(dev)
    ver = torch.__version__
    if cap == (6, 1) and "+cu118" not in ver:
        print(f"[fatal] P40 (sm_61) requires torch +cu118; found {ver}", file=sys.stderr)
        sys.exit(5)
    return f"cuda:{dev}"


if __name__ == "__main__":  # manual check
    print(ensure_cuda_safe())


