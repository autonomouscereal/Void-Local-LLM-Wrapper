from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple, Optional


# Configure logging (non-intrusive by default to avoid I/O stalls)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


_JSON_MAX = 2_000_000  # soft cap to avoid pathological inputs


class JSONParser:
    """
    Robust JSON normalizer + parser.

    Modes:
      - parse_strict(text) -> (ok, obj|None)
      - parse_best_effort(text, expected_structure)

    Side effects:
      - self.errors: List[str] of human-readable issues
      - self.repairs: List[str] listing repairs applied in order
      - self.last_error: Optional[str]
    """

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.repairs: List[str] = []
        self.last_error: Optional[str] = None

    # Backwards-compatible shim (maps to best-effort)
    def parse(self, text: str, expected_structure: Any) -> Any:
        return self.parse_best_effort(text, expected_structure)

    # ---------- public API ----------

    def parse_strict(self, text: str) -> Tuple[bool, Optional[Any]]:
        s = self._normalize_input(text)
        if not s:
            self._err("empty_after_normalization")
            return False, None
        segments = self._balanced_segments(s)
        if not segments:
            self._err("no_balanced_json_segment_found")
            return False, None
        candidate = segments[-1]
        candidate2 = self._safe_repairs(candidate)
        obj, err = self._loads(candidate2)
        if err is not None:
            self._err("json_loads_failed:" + err)
            return False, None
        return True, obj

    def parse_best_effort(self, text: str, expected_structure: Any) -> Any:
        s = self._normalize_input(text)
        if not s:
            self._err("empty_after_normalization")
            return self._ensure_structure({} if isinstance(expected_structure, dict) else [], expected_structure)
        segs = self._balanced_segments(s)
        if not segs:
            self._err("no_balanced_json_segment_found")
            return self._ensure_structure({} if isinstance(expected_structure, dict) else [], expected_structure)
        if isinstance(expected_structure, list):
            merged = self._merge_segments_into_array(segs)
            obj, err = self._loads(self._safe_repairs(merged))
            if err:
                self._err("json_loads_failed_best_effort:" + err)
                return self._ensure_structure([], expected_structure)
            return self._ensure_structure(obj, expected_structure)
        candidate = self._pick_last_object_or_first_array(segs)
        obj, err = self._loads(self._safe_repairs(candidate))
        if err:
            self._err("json_loads_failed_best_effort:" + err)
            return self._ensure_structure({}, expected_structure)
        return self._ensure_structure(obj, expected_structure)

    # ---------- normalization & repair pipeline ----------

    def _normalize_input(self, text: str) -> str:
        s = text or ""
        if len(s) > _JSON_MAX:
            s = s[:_JSON_MAX]
            self._note("truncated_to_max")
        s2 = s.strip()
        if s2.startswith("```"):
            s2 = self._strip_code_fences(s)
        else:
            s2 = s
        s3 = (
            s2.replace("\ufeff", "")
              .replace("\u200b", "")
              .replace("\u200c", "")
              .replace("\u200d", "")
              .replace("\u2060", "")
        )
        s4 = (s3.replace("“", '"').replace("”", '"')
                 .replace("‘", "'").replace("’", "'"))
        return s4.strip()

    def _strip_code_fences(self, s: str) -> str:
        t = s.strip()
        if t.startswith("```"):
            parts = t.split("```")
            if len(parts) >= 3:
                return parts[1].split("\n", 1)[-1] if parts[1].startswith(("json", "jsonc")) else parts[1]
        return s

    def _balanced_segments(self, s: str) -> List[str]:
        out: List[str] = []
        stack: List[str] = []
        start = -1
        in_str = False
        str_delim = ""
        esc = False
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
                        if not stack and start >= 0:
                            out.append(s[start:i+1])
        return out

    def _safe_repairs(self, s: str) -> str:
        step = s
        step = self._convert_single_quoted_strings(step)
        step = self._quote_unquoted_keys(step)
        step = self._remove_trailing_commas(step)
        step = self._fix_bare_literals(step)
        return step

    def _convert_single_quoted_strings(self, s: str) -> str:
        out = []
        in_str = False
        delim = ""
        esc = False
        for ch in s:
            if in_str:
                if esc:
                    out.append(ch)
                    esc = False
                elif ch == "\\":
                    out.append(ch)
                    esc = True
                elif ch == delim:
                    if delim == "'":
                        out.append('"')
                    else:
                        out.append(ch)
                    in_str = False
                else:
                    if delim == "'":
                        if ch == '"':
                            out.append('\\"')
                        else:
                            out.append(ch)
                    else:
                        out.append(ch)
            else:
                if ch in ('"', "'"):
                    in_str = True
                    delim = ch
                    out.append('"' if ch == "'" else '"')
                else:
                    out.append(ch)
        return "".join(out)

    def _quote_unquoted_keys(self, s: str) -> str:
        out = []
        in_str = False
        esc = False
        expect_key = False
        stack: List[str] = []
        i = 0
        L = len(s)
        while i < L:
            ch = s[i]
            if in_str:
                out.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
                continue
            if ch in "{[":
                stack.append(ch)
                out.append(ch)
                expect_key = (ch == "{")
                i += 1
                continue
            if ch in "}]":
                if stack:
                    stack.pop()
                out.append(ch)
                expect_key = False
                i += 1
                continue
            if ch == ",":
                out.append(ch)
                expect_key = (stack and stack[-1] == "{")
                i += 1
                continue
            if expect_key:
                if ch.isspace():
                    out.append(ch)
                    i += 1
                    continue
                if ch == '"':
                    in_str = True
                    out.append(ch)
                    i += 1
                    continue
                start = i
                while i < L and s[i] not in (":", " ", "\n", "\r", "\t"):
                    if s[i] in "{}[],":
                        break
                    i += 1
                token = s[start:i]
                if token and token[0] != '"' and token[0] != "'" and ":" in s[i:i+1]:
                    out.append('"')
                    out.append(token)
                    out.append('"')
                    expect_key = False
                    continue
                j = i
                while j < L and s[j].isspace():
                    j += 1
                if j < L and s[j] == ":" and token:
                    out.append('"')
                    out.append(token)
                    out.append('"')
                    i = j
                    continue
                out.append(token)
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _remove_trailing_commas(self, s: str) -> str:
        out = []
        in_str = False
        esc = False
        stack: List[str] = []
        i = 0
        L = len(s)
        while i < L:
            ch = s[i]
            if in_str:
                out.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
                continue
            if ch in "{[":
                stack.append(ch)
                out.append(ch)
                i += 1
                continue
            if ch in "}]":
                if out and out[-1] == ",":
                    out.pop()
                    self._note("removed_trailing_comma")
                if stack:
                    stack.pop()
                out.append(ch)
                i += 1
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def _fix_bare_literals(self, s: str) -> str:
        out = []
        in_str = False
        esc = False
        token = ""
        def flush_token():
            nonlocal token
            if token in ("True", "False", "None"):
                m = {"True": "true", "False": "false", "None": "null"}[token]
                out.append(m)
                self._note("fixed_bare_literal:" + token)
            else:
                out.append(token)
            token = ""
        for ch in s:
            if in_str:
                out.append(ch)
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                flush_token()
                in_str = True
                out.append(ch)
                continue
            if ch.isalpha():
                token += ch
                continue
            if token:
                flush_token()
            out.append(ch)
        if token:
            flush_token()
        return "".join(out)

    def _merge_segments_into_array(self, segs: List[str]) -> str:
        repaired = [self._safe_repairs(seg) for seg in segs]
        return "[" + ",".join(repaired) + "]"

    def _pick_last_object_or_first_array(self, segs: List[str]) -> str:
        for seg in reversed(segs):
            t = seg.strip()
            if t.startswith("{") and t.endswith("}"):
                return seg
        for seg in segs:
            t = seg.strip()
            if t.startswith("[") and t.endswith("]"):
                return seg
        return segs[-1]

    # ---------- structure coercion ----------

    def _ensure_structure(self, parsed_json: Any, expected_structure: Any) -> Any:
        if isinstance(expected_structure, list):
            want = expected_structure[0] if expected_structure else {}
            if not isinstance(parsed_json, list):
                parsed = []
            else:
                parsed = parsed_json
            return [self._ensure_structure(item, want) if isinstance(item, dict) else self._ensure_structure({}, want)
                    for item in parsed]
        result: Dict[str, Any] = {}
        for key, expected_type in expected_structure.items():
            val = parsed_json.get(key) if isinstance(parsed_json, dict) else None
            if isinstance(expected_type, list):
                if isinstance(val, list):
                    inner = expected_type[0] if expected_type else {}
                    result[key] = [self._ensure_structure(v, inner) if isinstance(v, dict) else v for v in val]
                else:
                    result[key] = []
            elif isinstance(expected_type, dict):
                if isinstance(val, dict):
                    result[key] = self._ensure_structure(val, expected_type)
                else:
                    result[key] = {}
            else:
                if isinstance(val, expected_type):
                    result[key] = val
                else:
                    result[key] = self._default_value(expected_type)
        return result

    def _default_value(self, expected_type: Any) -> Any:
        if expected_type is int:
            return 0
        if expected_type is float:
            return 0.0
        if expected_type is list:
            return []
        if expected_type is dict:
            return {}
        return ""

    # ---------- helpers ----------

    def _loads(self, s: str) -> Tuple[Optional[Any], Optional[str]]:
        try:
            return json.loads(s), None
        except Exception as e:
            self.last_error = str(e)
            self.errors.append(self.last_error)
            return None, self.last_error

    def _note(self, msg: str) -> None:
        self.repairs.append(msg)

    def _err(self, msg: str) -> None:
        self.last_error = msg
        self.errors.append(msg)

