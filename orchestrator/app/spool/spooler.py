from __future__ import annotations

import os
import io
import time
import hashlib
from typing import Optional, Dict, Any


def _sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


class Spooler:
    """
    Write bytes into a buffer with hard limits.
    - First N KB kept in RAM (BytesIO).
    - Beyond spill_threshold, write to tmp file (append).
    - On finalize(): fsync + atomic rename to target path.
    """

    def __init__(self, tmp_dir: str, target_path: str, ram_limit_bytes: int = 512 * 1024, spill_threshold: int = 256 * 1024):
        os.makedirs(tmp_dir, exist_ok=True)
        self.tmp_path = os.path.join(tmp_dir, f".spool_{int(time.time()*1000)}_{os.getpid()}.tmp")
        self.target_path = target_path
        self.ram_limit = int(ram_limit_bytes)
        self.spill_threshold = int(spill_threshold)
        self.ram = io.BytesIO()
        self.spilled = False
        self.bytes = 0

    def write(self, b: bytes | bytearray | str):
        if not isinstance(b, (bytes, bytearray)):
            b = str(b).encode("utf-8")
        n = len(b)
        self.bytes += n
        if (not self.spilled) and ((self.ram.tell() + n) <= self.ram_limit):
            self.ram.write(b)
            return
        # spill path
        if not self.spilled:
            with open(self.tmp_path, "ab") as f:
                f.write(self.ram.getvalue())
                f.write(b)
                f.flush(); os.fsync(f.fileno())
            self.spilled = True
            self.ram = None
        else:
            with open(self.tmp_path, "ab") as f:
                f.write(b)
                f.flush(); os.fsync(f.fileno())

    def finalize(self) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(self.target_path), exist_ok=True)
        if not self.spilled:
            tmp = self.tmp_path
            with open(tmp, "wb") as f:
                f.write(self.ram.getvalue())
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, self.target_path)
        else:
            os.replace(self.tmp_path, self.target_path)
        return {"path": self.target_path, "bytes": int(self.bytes), "sha256": _sha256_file(self.target_path)}


