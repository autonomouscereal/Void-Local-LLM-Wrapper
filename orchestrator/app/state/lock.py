from __future__ import annotations

import os
import time


class LockError(Exception):
    pass


def _lock_path(root: str, key: str) -> str:
    return os.path.join(root, f"{key}.lock")


def acquire_lock(root: str, key: str, timeout_s: int = 10) -> str:
    os.makedirs(root, exist_ok=True)
    p = _lock_path(root, key)
    t0 = time.time()
    while True:
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return p
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                raise LockError(f"timeout acquiring lock {key}")
            time.sleep(0.05)


def release_lock(root: str, key: str):
    try:
        os.unlink(_lock_path(root, key))
    except FileNotFoundError:
        pass


