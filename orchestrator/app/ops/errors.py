from __future__ import annotations


class OpErr(Exception):
    def __init__(self, kind: str, code: str | None, msg: str, where: str | None = None):
        super().__init__(msg)
        self.kind = kind
        self.code = code
        self.where = where or "unknown"


def classify_provider_exc(e: Exception) -> OpErr:
    s = str(e).lower()
    if "429" in s or "rate" in s:
        return OpErr("retryable", "429", "rate limited")
    if "timeout" in s or "408" in s:
        return OpErr("retryable", "408", "timeout")
    if any(x in s for x in ("502", "503", "504", "overload")):
        return OpErr("retryable", "503", "provider overloaded")
    return OpErr("permanent", None, str(e))


def error_envelope(e: OpErr) -> dict:
    return {"error": {"kind": e.kind, "code": e.code, "where": e.where, "message": str(e)}}


