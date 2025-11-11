from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple, Optional


class JSONParser:
    """
    Deterministic, schema-driven JSON parser and normalizer.

    Rules:
      - Exactly two json.loads try/except blocks in this file (in _attempt_loads).
      - No other try/except anywhere in this file.
      - parse_strict(text) -> (ok: bool, obj|None): strict load with one repair pass.
      - parse(text, expected) -> coerced structure to 'expected' shape.
   
    Compatibility:
      - Keeps self.errors, self.repairs, self.last_error for callers that log them.
      - Exposes parse_best_effort as an alias to parse.
    """

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.repairs: List[str] = []
        self.last_error: Optional[str] = None

    # ---------- public API ----------

    def parse_strict(self, text: str) -> Tuple[bool, Optional[Any]]:
        s = self._prep(text)
        ok, data = self._attempt_loads(s)
        if not ok:
            return False, None
        return True, data

    def parse(self, text: str, expected: Any) -> Any:
        s = self._prep(text)
        ok, data = self._attempt_loads(s)
        if not ok:
            # On failure, default container based on expected, then coerce
            data = {} if isinstance(expected, dict) else ([] if isinstance(expected, list) else None)
        return self._ensure_structure(data, expected)

    # Backwards-compatible alias
    def parse_best_effort(self, text: str, expected: Any) -> Any:
        return self.parse(text, expected)

    # ---------- core loading (the ONLY try/except blocks) ----------

    def _attempt_loads(self, s: str) -> Tuple[bool, Optional[Any]]:
        # 1) Pristine attempt
        try:
            data = json.loads(s)
            return True, data
        except Exception as e1:
            self.last_error = str(e1)
            self.errors.append(f"loads_pristine:{self.last_error}")
        # 2) Repair then final attempt
        s2 = self._repair(s)
        try:
            data = json.loads(s2)
            return True, data
        except Exception as e2:
            self.last_error = str(e2)
            self.errors.append(f"loads_repaired:{self.last_error}")
            return False, None

    # ---------- normalization & repair (no try/except here) ----------

    def _prep(self, s: str) -> str:
        if not isinstance(s, str):
            s = ""
        # strip common wrappers
        t = s.replace("\ufeff", "").strip()
        if t.startswith("```"):
            parts = t.split("```")
            if len(parts) >= 3:
                body = parts[1]
                if body.startswith(("json", "jsonc")):
                    body = body.split("\n", 1)[-1]
                t = body
        # normalize curly quotes
        t = (t.replace("“", '"').replace("”", '"')
               .replace("‘", "'").replace("’", "'"))
        return t.strip()

    def _repair(self, s: str) -> str:
        # Prefer last balanced object/array if multiple payloads present
        seg = self._last_balanced_segment(s)
        if seg:
            s = seg
        # unify quotes for keys: abc: -> "abc":
        s = re.sub(r'(?<=\{|,)\s*([A-Za-z0-9_\-]+)\s*:', r'"\1":', s)
        # single quotes to double quotes for strings (pragmatic)
        s = re.sub(r"'", r'"', s)
        # remove trailing commas
        s = s.replace(",}", "}").replace(",]", "]")
        # join adjacent objects
        s = s.replace("}{", "},{")
        # if result is multiple concatenated JSON values, wrap into array if looks like that case
        if s.count("}{") > 0 and not (s.strip().startswith("[") and s.strip().endswith("]")):
            s = "[" + s + "]"
        # last balanced again post-repair
        seg2 = self._last_balanced_segment(s)
        return seg2 if seg2 else s

    def _last_balanced_segment(self, s: str) -> str:
        stack: List[str] = []
        start = -1
        last = ""
        in_str = False
        esc = False
        str_delim = ""
        for i, ch in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == str_delim:
                    in_str = False
                continue
            if ch in ('"', "'"):
                in_str = True
                str_delim = ch
                continue
            if ch in "{[":
                if not stack:
                    start = i
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    top = stack[-1]
                    if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                        stack.pop()
                        if not stack and start != -1:
                            last = s[start:i+1]
        return last

    # ---------- structure coercion ----------

    def _ensure_structure(self, data: Any, expected: Any) -> Any:
        # list expectation
        if isinstance(expected, list):
            item_shape = expected[0] if expected else {}
            if not isinstance(data, list):
                data = []
            out_list: List[Any] = []
            for v in data:
                if isinstance(item_shape, dict):
                    out_list.append(self._ensure_structure(v if isinstance(v, dict) else {}, item_shape))
                elif isinstance(item_shape, list):
                    out_list.append(self._ensure_structure(v if isinstance(v, list) else [], item_shape))
                else:
                    out_list.append(self._coerce_scalar(v, item_shape))
            return out_list
        # dict expectation
        if isinstance(expected, dict):
            src = data if isinstance(data, dict) else {}
            out: Dict[str, Any] = {}
            for k, shape in expected.items():
                v = src.get(k) if isinstance(src, dict) else None
                if isinstance(shape, dict):
                    out[k] = self._ensure_structure(v if isinstance(v, dict) else {}, shape)
                elif isinstance(shape, list):
                    if isinstance(v, list):
                        out[k] = self._ensure_structure(v, shape)
                    else:
                        out[k] = []
                else:
                    out[k] = self._coerce_scalar(v, shape)
            return out
        # scalar expectation
        return self._coerce_scalar(data, expected)

    def _coerce_scalar(self, v: Any, t: Any) -> Any:
        if t is int:
            return v if isinstance(v, int) else 0
        if t is float:
            return v if isinstance(v, (int, float)) else 0.0
        if t is list:
            return v if isinstance(v, list) else []
        if t is dict:
            return v if isinstance(v, dict) else {}
        return v if isinstance(v, str) else ""

    # kept for list item primitive coercion when shape is dict-like
    def _coerce_primitive(self, v: Any, item_shape: Any) -> Any:
        if isinstance(item_shape, dict):
            return self._ensure_structure({}, item_shape)
        return v

    # ---------- diagnostics (no try/except) ----------

    def _note(self, msg: str) -> None:
        self.repairs.append(msg)

    def _err(self, msg: str) -> None:
        self.last_error = msg
        self.errors.append(msg)

