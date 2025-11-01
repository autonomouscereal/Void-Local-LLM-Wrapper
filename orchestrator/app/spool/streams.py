from __future__ import annotations

from typing import Iterable, Dict, Any
from .spooler import Spooler


def stream_model_to_spool(provider_stream: Iterable, spooler: Spooler, chunk_field: str = "delta") -> Dict[str, Any]:
    for ch in provider_stream:
        if isinstance(ch, dict):
            ch = ch.get(chunk_field, "") or ""
        if not isinstance(ch, (bytes, bytearray)):
            ch = str(ch).encode("utf-8")
        spooler.write(ch)
    return spooler.finalize()


def stream_text_to_spool(lines_iterable: Iterable, spooler: Spooler) -> Dict[str, Any]:
    for line in lines_iterable:
        if not isinstance(line, (bytes, bytearray)):
            line = (str(line) + "\n").encode("utf-8")
        spooler.write(line)
    return spooler.finalize()


