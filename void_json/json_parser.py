from __future__ import annotations

import base64
import json
import logging
import math
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

logger = logging.getLogger()


class JSONParser:
    """
    parse(source, expected_structure) -> coerced result

    Strict ordering (your rule):
    1) If source already dict/list: return it immediately (no fixes).
    2) If source is str: try json.loads immediately (no fixes). If ok, return it.
    3) Only if json.loads fails: then do repairs/extraction/fallback parsing.
    4) Finally: coerce to expected_structure without dropping extras, and pack misplaced keys.

    EXTRA SAFETY (added):
    - The final returned value is guaranteed JSON-serializable by converting
      bytes/bytearray/memoryview anywhere in the tree into base64 strings.
    """

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.repairs: List[str] = []
        self.last_error: Optional[str] = None

    # ---------------- public API ----------------

    def parse(self, source: Any, expected_structure: Any) -> Any:
        # 1) already json-like: return as-is immediately
        if isinstance(source, (dict, list)):
            raw = source
            return self._coerce_node_safe(raw, expected_structure)

        # 2) if string, try json.loads before ANY fixes
        if isinstance(source, str):
            obj = self._try_json_loads(source, "json.loads.raw")
            if obj is not None:
                raw = obj
                return self._coerce_node_safe(raw, expected_structure)
            # only now do the heavy stuff
            raw = self._parse_raw_from_text_safe(source, expected_structure)
            return self._coerce_node_safe(raw, expected_structure)

        # Non-string / non-dict/list types:
        # Try to preserve type while still obeying "loads before fixes" for text.
        raw = self._parse_raw_any_safe(source, expected_structure)
        return self._coerce_node_safe(raw, expected_structure)

    # ---------------- safe wrappers ----------------

    def _parse_raw_any_safe(self, source: Any, expected_structure: Any) -> Any:
        try:
            return self._parse_raw_any(source, expected_structure)
        except Exception as exc:
            self._err(f"parse_raw_any:{type(exc).__name__}:{exc}")
            return self._default_for_schema(expected_structure)

    def _parse_raw_from_text_safe(self, text: str, expected_structure: Any) -> Any:
        try:
            return self._parse_raw_from_text(text, expected_structure)
        except Exception as exc:
            self._err(f"parse_raw_from_text:{type(exc).__name__}:{exc}")
            return self._default_for_schema(expected_structure)

    def _coerce_node_safe(self, raw: Any, expected_structure: Any) -> Any:
        try:
            coerced = self._coerce_node(raw, expected_structure)
        except Exception as exc:
            self._err(f"coerce_node:{type(exc).__name__}:{exc}")
            coerced = self._default_for_schema(expected_structure)
        if coerced is None:
            coerced = self._default_for_schema(expected_structure)

        # CRITICAL: ensure the final output is JSON-serializable (fixes bytes leaks).
        coerced = self._json_sanitize_any_safe(coerced)
        return coerced

    # ---------------- JSON-serializable sanitation (bytes-safe) ----------------

    def _bytes_to_b64(self, v: Any) -> str:
        try:
            b = bytes(v)
        except Exception as exc:
            self._err(f"bytes_to_b64:{type(exc).__name__}:{exc}")
            return "b64:"
        try:
            return "b64:" + base64.b64encode(b).decode("ascii")
        except Exception as exc:
            self._err(f"bytes_b64encode:{type(exc).__name__}:{exc}")
            return "b64:"

    def _json_sanitize_any_safe(self, v: Any) -> Any:
        try:
            return self._json_sanitize_any(v)
        except Exception as exc:
            self._err(f"json_sanitize_any:{type(exc).__name__}:{exc}")
            # last-resort: stringify so response never kills the connection
            try:
                return str(v)
            except Exception:
                return ""

    def _json_sanitize_any(self, v: Any) -> Any:
        # bytes anywhere -> base64 string (lossless)
        if isinstance(v, (bytes, bytearray, memoryview)):
            return self._bytes_to_b64(v)

        # dict: sanitize keys + values
        if isinstance(v, dict):
            out: Dict[str, Any] = {}
            for k, val in v.items():
                # JSON requires string keys; ensure keys are safe
                if isinstance(k, (bytes, bytearray, memoryview)):
                    key = self._bytes_to_b64(k)
                else:
                    key = str(k)
                out[key] = self._json_sanitize_any(val)
            return out

        # list/tuple/set: sanitize elements (and normalize to list)
        if isinstance(v, list):
            return [self._json_sanitize_any(x) for x in v]
        if isinstance(v, (tuple, set, frozenset)):
            return [self._json_sanitize_any(x) for x in list(v)]

        # primitives are fine
        if v is None or isinstance(v, (str, int, float, bool)):
            return v

        # anything else: to be absolutely safe, stringify here instead of returning raw objects.
        return str(v)

    # ---------------- any-type raw parsing ----------------

    def _parse_raw_any(self, source: Any, expected_structure: Any) -> Any:
        # mapping-like
        if self._is_mapping_like(source):
            d = self._to_dict_safe(source)
            return d if d is not None else {}

        # iterable containers -> list
        if isinstance(source, (tuple, set, frozenset)):
            try:
                return list(source)
            except Exception as exc:
                self._err(f"iterable_to_list:{type(exc).__name__}:{exc}")
                return []

        # bytes -> decode, then obey "loads before fixes"
        if isinstance(source, (bytes, bytearray, memoryview)):
            try:
                s = bytes(source).decode("utf-8", errors="replace")
            except Exception as exc:
                self._err(f"bytes_decode:{type(exc).__name__}:{exc}")
                return self._default_for_schema(expected_structure)

            obj = self._try_json_loads(s, "json.loads.bytes")
            if obj is not None:
                return obj
            return self._parse_raw_from_text(s, expected_structure)

        # dataclass / __dict__ object -> dict
        d2 = self._object_to_dict_safe(source)
        if d2 is not None:
            return d2

        # fallback stringify -> obey "loads before fixes"
        try:
            s2 = str(source)
        except Exception as exc:
            self._err(f"str_source_failed:{type(exc).__name__}:{exc}")
            return self._default_for_schema(expected_structure)

        obj = self._try_json_loads(s2, "json.loads.str")
        if obj is not None:
            return obj
        return self._parse_raw_from_text(s2, expected_structure)

    # ---------------- text raw parsing ----------------

    def _parse_raw_from_text(self, text: str, expected_structure: Any) -> Any:
        s = self._normalize_text(text)

        # We already tried json.loads(text) before reaching here.
        # Do NOT re-try json.loads(s) until after we extract/repair.
        s = self._strip_fence_tokens(s)

        # Collect candidates (JSON segments embedded anywhere)
        segments = self._collect_balanced_segments(s)

        best_obj: Any = None
        best_score = -1

        i = 0
        while i < len(segments):
            seg = segments[i]
            repaired = self._repair_json_text(seg)
            obj2 = self._try_json_loads(repaired, "json.loads.segment.repaired")
            if obj2 is not None:
                score = self._score_candidate(obj2, expected_structure)
                if score > best_score:
                    best_score = score
                    best_obj = obj2
            i += 1

        if best_obj is not None:
            return best_obj

        # Repair whole string and try
        repaired_all = self._repair_json_text(s)
        obj3 = self._try_json_loads(repaired_all, "json.loads.repaired_all")
        if obj3 is not None:
            return obj3

        # Markdown table fallback
        table_obj = self._parse_markdown_table(repaired_all)
        if table_obj is not None:
            return table_obj

        # Key/value prose fallback
        kv_obj = self._parse_key_value_fallback(repaired_all)
        if kv_obj is not None:
            return kv_obj

        return {} if isinstance(expected_structure, dict) else []

    # ---------------- strict json.loads ----------------

    def _try_json_loads(self, s: str, tag: str) -> Optional[Any]:
        try:
            return json.loads(s)
        except Exception as exc:
            self._err(f"{tag}:{type(exc).__name__}:{exc}")
            return None

    # ---------------- normalization ----------------

    def _normalize_text(self, s: str) -> str:
        try:
            out = s if isinstance(s, str) else ("" if s is None else str(s))
        except Exception:
            return ""

        try:
            out = unicodedata.normalize("NFKC", out)
        except Exception as exc:
            self._err(f"normalize_nfkc:{type(exc).__name__}:{exc}")

        out = out.replace("\u201c", '"').replace("\u201d", '"')
        out = out.replace("\u2018", "'").replace("\u2019", "'")
        out = out.replace("\uFF1A", ":")
        out = out.replace("\u2192", "->")
        out = out.replace("\u21D2", "=>")
        out = out.replace("\u2013", "-").replace("\u2014", "-")
        out = out.replace("\u200b", "").replace("\u2060", "")
        out = out.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
        return out.strip()

    def _strip_fence_tokens(self, s: str) -> str:
        try:
            out = s.replace("```json", "").replace("```JSON", "").replace("```", "")
            out = out.replace("'''json", "").replace("'''JSON", "").replace("'''", "")

            lines = out.splitlines()
            kept: List[str] = []
            i = 0
            while i < len(lines):
                t = lines[i].strip().lower()
                if t == "json":
                    i += 1
                    continue
                kept.append(lines[i])
                i += 1
            return "\n".join(kept).strip()
        except Exception as exc:
            self._err(f"strip_fence_tokens:{type(exc).__name__}:{exc}")
            return s if isinstance(s, str) else ""

    # ---------------- balanced extraction ----------------

    def _collect_balanced_segments(self, s: str) -> List[str]:
        try:
            segs: List[str] = []
            segs.extend(self._collect_balanced_for_delims(s, "{", "}"))
            segs.extend(self._collect_balanced_for_delims(s, "[", "]"))

            # prefer longer first (manual sort)
            i = 0
            while i < len(segs):
                j = i + 1
                while j < len(segs):
                    if len(segs[j]) > len(segs[i]):
                        tmp = segs[i]
                        segs[i] = segs[j]
                        segs[j] = tmp
                    j += 1
                i += 1
            return segs
        except Exception as exc:
            self._err(f"collect_balanced_segments:{type(exc).__name__}:{exc}")
            return []

    def _collect_balanced_for_delims(self, s: str, start: str, end: str) -> List[str]:
        out: List[str] = []
        in_string = False
        escape = False
        quote = ""
        depth = 0
        start_idx = -1

        i = 0
        while i < len(s):
            ch = s[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                    quote = ""
                i += 1
                continue

            if ch == '"' or ch == "'":
                in_string = True
                quote = ch
                i += 1
                continue

            if ch == start:
                if depth == 0:
                    start_idx = i
                depth += 1
                i += 1
                continue

            if ch == end and depth > 0:
                depth -= 1
                if depth == 0 and start_idx != -1:
                    out.append(s[start_idx : i + 1])
                    start_idx = -1
                i += 1
                continue

            i += 1

        return out

    # ---------------- deterministic repair (no regex) ----------------

    def _repair_json_text(self, s: str) -> str:
        try:
            out = s.strip()
            out2 = self._remove_trailing_commas(out)
            if out2 != out:
                self.repairs.append("remove_trailing_commas")
                out = out2

            out2 = self._normalize_python_constants(out)
            if out2 != out:
                self.repairs.append("normalize_python_constants")
                out = out2

            out2 = self._single_quotes_to_double_quotes(out)
            if out2 != out:
                self.repairs.append("single_to_double_quotes")
                out = out2

            out2 = self._quote_unquoted_keys_and_values(out)
            if out2 != out:
                self.repairs.append("quote_unquoted_keys_and_values")
                out = out2

            out2 = self._close_unbalanced(out)
            if out2 != out:
                self.repairs.append("close_unbalanced")
                out = out2

            return out
        except Exception as exc:
            self._err(f"repair_json_text:{type(exc).__name__}:{exc}")
            return s if isinstance(s, str) else ""

    def _remove_trailing_commas(self, s: str) -> str:
        out: List[str] = []
        in_string = False
        escape = False
        quote = ""
        i = 0
        while i < len(s):
            ch = s[i]
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                    quote = ""
                i += 1
                continue

            if ch == '"' or ch == "'":
                in_string = True
                quote = ch
                out.append(ch)
                i += 1
                continue

            if ch == ",":
                j = i + 1
                while j < len(s) and s[j] in (" ", "\t", "\n"):
                    j += 1
                if j < len(s) and (s[j] == "}" or s[j] == "]"):
                    i += 1
                    continue

            out.append(ch)
            i += 1
        return "".join(out)

    def _normalize_python_constants(self, s: str) -> str:
        out: List[str] = []
        in_string = False
        escape = False
        quote = ""
        i = 0
        while i < len(s):
            ch = s[i]
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                    quote = ""
                i += 1
                continue

            if ch == '"' or ch == "'":
                in_string = True
                quote = ch
                out.append(ch)
                i += 1
                continue

            if ch.isalpha():
                start = i
                j = i
                while j < len(s) and (s[j].isalpha() or s[j].isdigit() or s[j] == "_"):
                    j += 1
                token = s[start:j]
                if token == "True":
                    out.append("true")
                elif token == "False":
                    out.append("false")
                elif token == "None":
                    out.append("null")
                else:
                    out.append(token)
                i = j
                continue

            out.append(ch)
            i += 1
        return "".join(out)

    def _single_quotes_to_double_quotes(self, s: str) -> str:
        out: List[str] = []
        in_string = False
        escape = False
        quote = ""
        i = 0
        while i < len(s):
            ch = s[i]

            if in_string:
                if escape:
                    out.append(ch)
                    escape = False
                    i += 1
                    continue
                if ch == "\\":
                    out.append(ch)
                    escape = True
                    i += 1
                    continue
                if ch == quote:
                    out.append('"')
                    in_string = False
                    quote = ""
                    i += 1
                    continue
                if quote == "'" and ch == '"':
                    out.append("\\")
                    out.append('"')
                    i += 1
                    continue
                out.append(ch)
                i += 1
                continue

            if ch == "'":
                in_string = True
                quote = "'"
                out.append('"')
                i += 1
                continue
            if ch == '"':
                in_string = True
                quote = '"'
                out.append('"')
                i += 1
                continue

            out.append(ch)
            i += 1

        if in_string and quote == "'":
            out.append('"')
        return "".join(out)

    def _quote_unquoted_keys_and_values(self, s: str) -> str:
        out: List[str] = []
        stack: List[str] = []
        in_string = False
        escape = False
        quote = ""
        expecting_key = False
        expecting_value = False

        i = 0
        while i < len(s):
            ch = s[i]

            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                    quote = ""
                i += 1
                continue

            if ch == '"' or ch == "'":
                in_string = True
                quote = '"'
                out.append('"')
                i += 1
                continue

            if ch == "{":
                stack.append("{")
                out.append("{")
                expecting_key = True
                expecting_value = False
                i += 1
                continue
            if ch == "[":
                stack.append("[")
                out.append("[")
                expecting_key = False
                expecting_value = True
                i += 1
                continue
            if ch == "}" or ch == "]":
                if stack:
                    stack.pop()
                out.append(ch)
                expecting_key = (stack and stack[-1] == "{")
                expecting_value = False
                i += 1
                continue
            if ch == ":":
                out.append(":")
                expecting_key = False
                expecting_value = True
                i += 1
                continue
            if ch == ",":
                out.append(",")
                expecting_key = (stack and stack[-1] == "{")
                expecting_value = (stack and stack[-1] == "[")
                i += 1
                continue

            if ch in (" ", "\t", "\n"):
                out.append(ch)
                i += 1
                continue

            if stack and stack[-1] == "{" and expecting_key:
                if self._is_bareword_start(ch):
                    token, j = self._read_bareword(s, i)
                    k = j
                    while k < len(s) and s[k] in (" ", "\t", "\n"):
                        k += 1
                    next_ch = s[k] if k < len(s) else ""

                    out.append('"'); out.append(token); out.append('"'); out.append(":")
                    expecting_key = False
                    expecting_value = True

                    if next_ch == ":" or next_ch == "=":
                        i = k + 1
                    else:
                        i = k
                    continue

            if expecting_value:
                if self._is_bareword_start(ch) or ch.isdigit() or ch in ("+", "-", "."):
                    token, j = self._read_value_token(s, i)
                    low = token.lower()

                    if low in ("true", "false", "null"):
                        out.append(low)
                    elif self._looks_number(token):
                        out.append(token)
                    else:
                        out.append('"'); out.append(token); out.append('"')

                    expecting_value = False
                    i = j
                    continue

            out.append(ch)
            i += 1

        return "".join(out)

    def _close_unbalanced(self, s: str) -> str:
        in_string = False
        escape = False
        quote = ""
        brace = 0
        bracket = 0
        i = 0
        while i < len(s):
            ch = s[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                    quote = ""
                i += 1
                continue
            if ch == '"' or ch == "'":
                in_string = True
                quote = ch
                i += 1
                continue
            if ch == "{":
                brace += 1
            elif ch == "}":
                brace = max(0, brace - 1)
            elif ch == "[":
                bracket += 1
            elif ch == "]":
                bracket = max(0, bracket - 1)
            i += 1
        if brace > 0:
            s = s + ("}" * brace)
        if bracket > 0:
            s = s + ("]" * bracket)
        return s

    def _is_bareword_start(self, ch: str) -> bool:
        return ch.isalpha() or ch == "_" or ch == "-"

    def _read_bareword(self, s: str, start: int) -> Tuple[str, int]:
        i = start
        out: List[str] = []
        while i < len(s):
            ch = s[i]
            if ch.isalnum() or ch in ("_", "-", "."):
                out.append(ch)
                i += 1
            else:
                break
        return "".join(out), i

    def _read_value_token(self, s: str, start: int) -> Tuple[str, int]:
        i = start
        out: List[str] = []
        while i < len(s):
            ch = s[i]
            if ch in (",", "}", "]"):
                break
            if ch in (" ", "\t", "\n"):
                break
            out.append(ch)
            i += 1
        return "".join(out), i

    def _looks_number(self, token: str) -> bool:
        if not token:
            return False
        t = token.strip()
        if not t:
            return False
        i = 0
        if t[0] in ("+", "-"):
            i = 1
        dot = 0
        digit = 0
        while i < len(t):
            ch = t[i]
            if ch.isdigit():
                digit += 1
            elif ch == ".":
                dot += 1
                if dot > 1:
                    return False
            else:
                return False
            i += 1
        return digit > 0

    # ---------------- markdown table fallback ----------------

    def _parse_markdown_table(self, s: str) -> Optional[Any]:
        try:
            lines = s.splitlines()
            table_lines: List[str] = []
            i = 0
            while i < len(lines):
                t = lines[i].strip()
                if "|" in t:
                    table_lines.append(t)
                i += 1
            if len(table_lines) < 2:
                return None

            header = self._split_md_table_row(table_lines[0])
            if not header:
                return None
            if not self._looks_like_md_separator(table_lines[1]):
                return None

            rows: List[Dict[str, Any]] = []
            r = 2
            while r < len(table_lines):
                cols = self._split_md_table_row(table_lines[r])
                if cols:
                    row: Dict[str, Any] = {}
                    c = 0
                    while c < len(header):
                        key = header[c]
                        val = cols[c] if c < len(cols) else ""
                        row[key] = val
                        c += 1
                    rows.append(row)
                r += 1

            return rows if rows else None
        except Exception as exc:
            self._err(f"parse_markdown_table:{type(exc).__name__}:{exc}")
            return None

    def _split_md_table_row(self, line: str) -> List[str]:
        t = line.strip()
        if not t:
            return []
        if t.startswith("|"):
            t = t[1:]
        if t.endswith("|"):
            t = t[:-1]
        parts = t.split("|")
        out: List[str] = []
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if part != "":
                out.append(part)
            i += 1
        return out

    def _looks_like_md_separator(self, line: str) -> bool:
        t = line.strip()
        if not t:
            return False
        chars: List[str] = []
        i = 0
        while i < len(t):
            ch = t[i]
            if ch not in ("|", " ", "\t"):
                chars.append(ch)
            i += 1
        if not chars:
            return False
        dash = 0
        i = 0
        while i < len(chars):
            if chars[i] == "-":
                dash += 1
            elif chars[i] == ":":
                # Alignment marker is allowed in markdown separators (e.g. :---:)
                dash += 0
            else:
                return False
            i += 1
        return dash >= 3

    # ---------------- prose/key-value fallback ----------------

    def _parse_key_value_fallback(self, s: str) -> Optional[Dict[str, Any]]:
        try:
            if not isinstance(s, str) or not s.strip():
                return None
            root: Dict[str, Any] = {}
            lines = s.splitlines()
            i = 0
            while i < len(lines):
                t = lines[i].strip()
                if not t:
                    i += 1
                    continue
                # key:value or key value
                key, val = self._split_kv_line(t)
                if key:
                    root[key] = self._coerce_fallback_scalar(val)
                i += 1
            return root if root else None
        except Exception as exc:
            self._err(f"parse_key_value_fallback:{type(exc).__name__}:{exc}")
            return None

    def _split_kv_line(self, t: str) -> Tuple[str, str]:
        delims = [":", "=", "->", "=>"]
        i = 0
        while i < len(delims):
            d = delims[i]
            pos = t.find(d)
            if pos != -1:
                return t[:pos].strip(), t[pos + len(d):].strip()
            i += 1
        j = 0
        while j < len(t) and t[j] not in (" ", "\t"):
            j += 1
        return t[:j].strip(), t[j:].strip()

    def _coerce_fallback_scalar(self, v: str) -> Any:
        if v is None:
            return ""
        t = v.strip()
        if not t:
            return ""
        low = t.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("null", "none"):
            return None
        if self._looks_number(t):
            try:
                if "." in t:
                    f = float(t)
                    return f if math.isfinite(f) else t
                return int(t)
            except Exception:
                return t
        if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
            return t[1:-1]
        return t

    # ---------------- coercion: preserve extras + pack keys ----------------

    def _coerce_node(self, data: Any, expected: Any) -> Any:
        origin = get_origin(expected)
        if origin is list:
            args = get_args(expected)
            item_schema = args[0] if args else Any
            return self._coerce_node(data, [item_schema])

        if origin is dict:
            args = get_args(expected)
            value_schema = args[1] if len(args) == 2 else Any
            src = self._as_dict(data)
            if src is None:
                return {} if data is None else {"_raw": data}
            out: Dict[str, Any] = dict(src)
            for k in list(out.keys()):
                out[k] = self._coerce_node(out.get(k), value_schema)
            return out

        if isinstance(expected, list):
            if not expected:
                if isinstance(data, list):
                    return data
                return [] if data is None else [data]
            item_schema = expected[0]
            if isinstance(data, list):
                out_list: List[Any] = []
                i = 0
                while i < len(data):
                    out_list.append(self._coerce_node(data[i], item_schema))
                    i += 1
                return out_list
            if isinstance(data, dict):
                return [self._coerce_node(data, item_schema)]
            return [] if data is None else [self._coerce_node(data, item_schema)]

        if isinstance(expected, dict):
            src = self._as_dict(data)
            if src is None:
                src = {} if data is None else {"_raw": data}
            if not expected:
                return src

            out: Dict[str, Any] = dict(src)  # preserve extras
            for key, schema in expected.items():
                if key in src:
                    out[key] = self._coerce_node(src.get(key), schema)
                else:
                    out[key] = self._default_for_schema(schema)

            self._pack_children_from_parent(out, expected)
            return out

        return self._coerce_scalar(data, expected)

    def _pack_children_from_parent(self, out: Dict[str, Any], expected: Dict[str, Any]) -> None:
        # Move matching keys from parent into nested dicts if expected
        try:
            for child_key, child_schema in expected.items():
                if not isinstance(child_schema, dict) or not child_schema:
                    continue
                child_val = out.get(child_key)
                if not isinstance(child_val, dict):
                    child_val = {} if child_val is None else {"_raw": child_val}
                    out[child_key] = child_val
                for ck, cschema in child_schema.items():
                    if ck in out and ck not in child_val:
                        child_val[ck] = self._coerce_node(out.get(ck), cschema)
                        del out[ck]
        except Exception as exc:
            self._err(f"pack_children_from_parent:{type(exc).__name__}:{exc}")

    def _default_for_schema(self, schema: Any) -> Any:
        try:
            if isinstance(schema, dict):
                return {}
            if isinstance(schema, list):
                return []
            origin = get_origin(schema)
            if origin is list:
                return []
            if origin is dict:
                return {}
            if schema is bool:
                return False
            if schema is int:
                return 0
            if schema is float:
                return 0.0
            if schema is str:
                return ""
            return None
        except Exception as exc:
            self._err(f"default_for_schema:{type(exc).__name__}:{exc}")
            return None

    def _coerce_scalar(self, value: Any, schema: Any) -> Any:
        try:
            if schema is Any or schema is object:
                return value
            origin = get_origin(schema)
            if origin is not None:
                args = list(get_args(schema) or [])
                if args:
                    for a in args:
                        if a is type(None):
                            continue
                        v = self._coerce_scalar(value, a)
                        if v not in (None, "", 0, 0.0, False, [], {}):
                            return v
                    for a in args:
                        if a is not type(None):
                            return self._coerce_scalar(value, a)
                return None
            if schema is bool:
                return self._to_bool(value)
            if schema is int:
                return self._to_int(value)
            if schema is float:
                return self._to_float(value)
            if schema is str:
                return "" if value is None else str(value)
            return value
        except Exception as exc:
            self._err(f"coerce_scalar:{type(exc).__name__}:{exc}")
            return self._default_for_schema(schema)

    def _to_bool(self, v: Any) -> bool:
        try:
            if isinstance(v, bool):
                return v
            if v is None:
                return False
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    return bool(int(v))
                except Exception:
                    return False
            if isinstance(v, str):
                t = v.strip().lower()
                if t in ("true", "1", "yes", "y", "on"):
                    return True
                if t in ("false", "0", "no", "n", "off", ""):
                    return False
                return True
            return bool(v)
        except Exception as exc:
            self._err(f"to_bool:{type(exc).__name__}:{exc}")
            return False

    def _to_int(self, v: Any) -> int:
        try:
            if v is None:
                return 0
            if isinstance(v, bool):
                return 1 if v else 0
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                if not math.isfinite(v):
                    return 0
                return int(v)
            if isinstance(v, str):
                t = v.strip()
                if not t:
                    return 0
                try:
                    f = float(t)
                    if not math.isfinite(f):
                        return 0
                    return int(f)
                except Exception:
                    return 0
            return int(v)
        except Exception as exc:
            self._err(f"to_int:{type(exc).__name__}:{exc}")
            return 0

    def _to_float(self, v: Any) -> float:
        try:
            if v is None:
                return 0.0
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            if isinstance(v, (int, float)):
                f = float(v)
                return f if math.isfinite(f) else 0.0
            if isinstance(v, str):
                t = v.strip()
                if not t:
                    return 0.0
                try:
                    f = float(t)
                    return f if math.isfinite(f) else 0.0
                except Exception:
                    return 0.0
            f = float(v)
            return f if math.isfinite(f) else 0.0
        except Exception as exc:
            self._err(f"to_float:{type(exc).__name__}:{exc}")
            return 0.0

    def _as_dict(self, v: Any) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(v, dict):
                return v
            if hasattr(v, "items") and callable(getattr(v, "items")):
                out: Dict[str, Any] = {}
                for k, val in v.items():  # type: ignore[attr-defined]
                    out[k] = val
                return out
        except Exception as exc:
            self._err(f"as_dict:{type(exc).__name__}:{exc}")
            return None
        return None

    # ---------------- scoring ----------------

    def _score_candidate(self, candidate: Any, expected: Any) -> int:
        try:
            if isinstance(expected, dict):
                if not isinstance(candidate, dict):
                    return 0
                if not expected:
                    return len(candidate)
                score = 0
                for k, sub in expected.items():
                    if k in candidate:
                        score += 2
                        if isinstance(sub, dict) and isinstance(candidate.get(k), dict):
                            score += 1
                score += min(len(candidate), 50)
                return score
            if isinstance(expected, list):
                if isinstance(candidate, list):
                    return len(candidate) + 1
                if isinstance(candidate, dict):
                    return 1
                return 0
            return 0 if candidate is None else 1
        except Exception as exc:
            self._err(f"score_candidate:{type(exc).__name__}:{exc}")
            return 0

    # ---------------- helpers ----------------

    def _is_mapping_like(self, obj: Any) -> bool:
        try:
            return hasattr(obj, "items") and callable(getattr(obj, "items"))
        except Exception:
            return False

    def _to_dict_safe(self, obj: Any) -> Optional[Dict[str, Any]]:
        try:
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "items"):
                out: Dict[str, Any] = {}
                for k, v in obj.items():  # type: ignore[attr-defined]
                    out[k] = v
                return out
        except Exception as exc:
            self._err(f"to_dict_safe:{type(exc).__name__}:{exc}")
            return None
        return None

    def _object_to_dict_safe(self, obj: Any) -> Optional[Dict[str, Any]]:
        try:
            # dataclass
            try:
                import dataclasses
                if dataclasses.is_dataclass(obj):
                    return dataclasses.asdict(obj)
            except Exception as exc:
                # Non-fatal; fall back to __dict__ below.
                self._err(f"object_to_dict_safe:dataclasses:{type(exc).__name__}:{exc}")
            # __dict__
            if hasattr(obj, "__dict__"):
                d = getattr(obj, "__dict__", None)
                if isinstance(d, dict):
                    return dict(d)
        except Exception as exc:
            self._err(f"object_to_dict_safe:{type(exc).__name__}:{exc}")
            return None
        return None

    # ---------------- diagnostics ----------------

    def _err(self, msg: str) -> None:
        self.last_error = msg
        self.errors.append(msg)
        logger.info("JSONParser error: %s", msg)
