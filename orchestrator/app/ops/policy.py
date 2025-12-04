from __future__ import annotations

import sys
import types
from typing import Optional


_BANNED_MODULES = {
    # Allow pydantic imports (FastAPI depends on it); avoid using it in our code instead
    "sqlalchemy": "BANNED: sqlalchemy/ORM is forbidden. Use asyncpg + raw SQL.",
}


class _BannedLoader:
    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        return None

    def exec_module(self, module):  # type: ignore[no-untyped-def]


class _BannedFinder:
    def find_spec(self, fullname: str, path: Optional[list[str]], target=None):  # type: ignore[no-untyped-def]
        base = fullname.split(".", 1)[0]
        if base in _BANNED_MODULES:
            try:
                from importlib.machinery import ModuleSpec  # type: ignore
                return ModuleSpec(fullname, _BannedLoader(fullname, _BANNED_MODULES[base]))
            except Exception:
                return None
        return None


def _patch_http_clients() -> None:
    try:
        import httpx  # type: ignore

        _OrigAsync = httpx.AsyncClient
        _OrigSync = httpx.Client

        class _PatchedAsyncClient(_OrigAsync):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                kwargs["timeout"] = None
                kwargs.setdefault("trust_env", False)
                super().__init__(*args, **kwargs)

        class _PatchedClient(_OrigSync):  # type: ignore[misc]
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                kwargs["timeout"] = None
                kwargs.setdefault("trust_env", False)
                super().__init__(*args, **kwargs)

        httpx.AsyncClient = _PatchedAsyncClient  # type: ignore[assignment]
        httpx.Client = _PatchedClient  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import requests  # type: ignore

        _orig_request = requests.sessions.Session.request

        def _no_timeout(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
            if "timeout" in kwargs:
                kwargs.pop("timeout", None)
            return _orig_request(self, method, url, **kwargs)

        requests.sessions.Session.request = _no_timeout  # type: ignore[assignment]
    except Exception:
        pass


def enforce_core_policy() -> None:
    # Sanitize any broken meta_path entries left by prior runs/hooks
    try:
        sanitized: list[object] = []
        for finder in list(sys.meta_path):
            try:
                if isinstance(finder, types.SimpleNamespace):
                    continue
                if not (hasattr(finder, "find_spec") or hasattr(finder, "find_module")):
                    continue
                sanitized.append(finder)
            except Exception:
                continue
        sys.meta_path[:] = sanitized
    except Exception:
        pass

    # Install import guard once
    try:
        if not any(isinstance(f, _BannedFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _BannedFinder())
    except Exception:
        pass
    _patch_http_clients()


