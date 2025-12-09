from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple


# Use the process-wide logging configuration only; no per-module handlers or
# import-time I/O. This logger is typically configured in the orchestrator
# entrypoint so all parser diagnostics go to the same sinks.
logger = logging.getLogger()


class JSONParser:
    """
    Robust, schema-driven JSON parser and normalizer.

    - Always returns a structure shaped by the explicit expected schema.
    - Never raises to callers; all failures are contained inside this class.
    - Aggressively repairs common LLM / tool JSON formatting issues.
    """

    def __init__(self) -> None:
        # Diagnostic fields kept for compatibility; callers can inspect if desired.
        self.errors: List[str] = []
        self.repairs: List[str] = []
        self.last_error: Optional[str] = None

    # ---------- public API ----------

    def parse(self, source: Any, expected_structure: Any) -> Any:
        """
        Best-effort parsing with repairs, followed by coercion into expected_structure.
        Never raises; always returns something matching the expected shape.

        Accepts:
          - str containing JSON (with or without markdown fences)
          - dict / list objects (already-parsed JSON)
          - any other value, which will be stringified before parsing
        """
        # Fast path: already a JSON-like structure; just coerce to the schema.
        if isinstance(source, (dict, list)):
            return self.ensure_structure(source, expected_structure)

        # Normalize non-string scalars into text so the repair pipeline can run.
        if not isinstance(source, str):
            source = "" if source is None else str(source)

        json_string = self.strip_markdown(source)
        json_string = self.attempt_repair(json_string)

        # Apply a sequence of parsing strategies, keeping the "best" result.
        methods = [
            self.method_json_loads,
            self.method_replace_quotes,
            self.method_fix_braces,
            self.method_fix_commas,
            self.method_fix_all,
            self.method_extract_segment,
            self.regex_parse,
        ]

        best_result: Any = None

        for method in methods:
            try:
                parsed_json = method(json_string)
                best_result = self.select_best_result(
                    best_result, parsed_json, expected_structure
                )
            except Exception as exc:
                msg = f"method:{method.__name__} failed:{exc}"
                logger.debug(msg, exc_info=True)
                self._err(msg)

        # Final fallback: if nothing succeeded, synthesise structure from schema alone.
        if best_result is None:
            if isinstance(expected_structure, list):
                best_result = self.ensure_structure([], expected_structure)
            else:
                best_result = self.ensure_structure({}, expected_structure)
        else:
            best_result = self.ensure_structure(best_result, expected_structure)

        # Salvage pass: when the source was textual, attempt to recover missing
        # scalar fields (str/int/float) directly from the raw text by scanning
        # for "key: value" style patterns, even if the JSON structure itself
        # was badly malformed.
        if isinstance(source, str):
            best_result = self._salvage_from_text(source, expected_structure, best_result)

        return best_result

    # Backwards-compatible alias
    def parse_best_effort(self, source: Any, expected_structure: Any) -> Any:
        """
        Alias for parse(...); kept for compatibility with older call sites.
        Accepts any JSON-like source (str/dict/list/scalar).
        """
        return self.parse(source, expected_structure)

    def parse_strict(self, source: Any) -> Tuple[bool, Optional[Any]]:
        """
        Strict load without schema coercion: returns (ok, raw_obj).
        Still uses markdown stripping and basic repair, but does not enforce shape.

        Accepts:
          - dict / list → treated as already-strict JSON (returns (True, source))
          - str → parsed via json.loads with a light repair pass
          - other scalars → stringified and parsed as JSON when possible
        """
        # Already-parsed JSON objects are treated as strictly valid.
        if isinstance(source, (dict, list)):
            return True, source

        if not isinstance(source, str):
            source = "" if source is None else str(source)
        s = self.strip_markdown(source)
        try:
            obj = json.loads(s)
            return True, obj
        except Exception as e1:
            self._err(f"parse_strict:{e1}")
        repaired = self.attempt_repair(s)
        try:
            obj = json.loads(repaired)
            return True, obj
        except Exception as e2:
            self._err(f"parse_strict_repaired:{e2}")
            # Never raise to callers; strict mode simply reports failure.
            return False, None

    def parse_superset(self, source: Any, expected_structure: Any) -> Dict[str, Any]:
        """
        Return a richer result that includes:
          - coerced: structure shaped to expected_structure (best-effort, with repairs)
          - raw: the best raw JSON object we could parse (preferring strict JSON when possible)
          - extras: parts of raw not represented in coerced (keys/values outside the schema)
          - vars: numeric hints extracted from string leaves
          - repairs/errors/last_error: diagnostics

        Strategy:
          1) Attempt a strict json.loads on the response text (via parse_strict), so we
             retain the most faithful view of the model/tool output when it's valid JSON.
          2) Independently run the full repair/heuristic pipeline via parse(...), which
             searches multiple strategies and coerces into expected_structure.
          3) Use the best-effort result from parse(...) as `coerced`, but keep the strict
             JSON object (when available) as `raw`. Any information present in raw but
             not representable in the schema is surfaced under `extras`.
        """
        # Step 1: strict-ish raw JSON
        ok, raw_strict = self.parse_strict(source)
        if not ok:
            # When strict parsing fails entirely, raw_strict becomes a typed empty shell
            # so callers still see a shape that matches the schema.
            if isinstance(expected_structure, dict):
                raw_strict = {}
            elif isinstance(expected_structure, list):
                raw_strict = []
            else:
                raw_strict = None

        # Step 2: best-effort coerced structure using all repair strategies
        coerced = self.parse(source, expected_structure)

        # Step 3: choose raw baseline and compute extras/vars
        # Prefer genuine strict JSON structure (dict/list) when we have one, otherwise
        # fall back to the coerced structure so raw is never "less informative" than coerced.
        if ok and isinstance(raw_strict, (dict, list)):
            raw: Any = raw_strict
        else:
            raw = coerced

        extras = self._diff_extras(raw, coerced)
        vars_map = self._extract_vars(raw)
        return {
            "coerced": coerced,
            "raw": raw,
            "extras": extras,
            "vars": vars_map,
            "repairs": list(self.repairs),
            "errors": list(self.errors),
            "last_error": self.last_error or "",
        }

    # ---------- original "GOOD" repair helpers ----------

    def strip_markdown(self, response: str) -> str:
        """Remove common ```json fences and surrounding whitespace."""
        if not isinstance(response, str):
            response = "" if response is None else str(response)
        response = response.replace("```json", "").replace("```", "").strip()
        # Also strip BOM / stray whitespace
        response = response.replace("\ufeff", "").strip()
        return response

    def correct_common_errors(self, text: str) -> str:
        corrections = [
            ('\\"', '"'),
            ("\\'", "'"),
            ('"{', "{"),
            ('}"', "}"),
            (",}", "}"),
            (",]", "]"),
            ('"False"', "false"),
            ('"True"', "true"),
        ]
        for old, new in corrections:
            if old in text:
                text = text.replace(old, new)
        return text

    def extract_json_segment(
        self, text: str, start_delim: str = "{", end_delim: str = "}"
    ) -> List[str]:
        """Return all balanced {...} segments from text (last one is usually the answer)."""
        stack: List[str] = []
        json_segments: List[str] = []
        start_index = -1
        for i, char in enumerate(text):
            if char == start_delim:
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == end_delim and stack:
                stack.pop()
                if not stack and start_index != -1:
                    segment = text[start_index : i + 1]
                    json_segments.append(segment)
        return json_segments

    def fix_missing_commas(self, json_string: str) -> str:
        json_string = json_string.replace("}{", "},{")
        return json_string

    def attempt_repair(self, json_string: str) -> str:
        json_string = self.correct_common_errors(json_string)
        json_string = self.fix_missing_commas(json_string)
        return json_string

    def regex_parse(self, json_string: str) -> Any:
        """
        Regex-based fallback: handle common LLM quirks like single quotes and
        unquoted keys. This is intentionally permissive and only used after
        simpler strategies fail.
        """
        if not isinstance(json_string, str):
            json_string = "" if json_string is None else str(json_string)
        try:
            return json.loads(json_string)
        except Exception as exc:
            # Record the failure so callers can see that the strict branch failed
            # before we fall back to the more permissive regex-based repair.
            self._err(f"regex_parse_strict:{exc}")
        json_string = re.sub(r"'", '"', json_string)
        json_string = re.sub(
            r"(?<=\{|,)\s*([A-Za-z0-9_]+)\s*:",
            r'"\1":',
            json_string,
        )
        try:
            return json.loads(json_string)
        except Exception as exc:
            self._err(f"regex_parse:{exc}")
            return None

    # ---------- method variants, each using json.loads ----------

    def method_json_loads(self, json_string: str) -> Any:
        if not isinstance(json_string, str):
            json_string = "" if json_string is None else str(json_string)
        try:
            return json.loads(json_string)
        except Exception as exc:
            self._err(f"json_loads:{exc}")
            return None

    def method_replace_quotes(self, json_string: str) -> Any:
        if not isinstance(json_string, str):
            json_string = "" if json_string is None else str(json_string)
        json_string = json_string.replace("\\'", "'").replace('\\"', '"')
        try:
            return json.loads(json_string)
        except Exception as exc:
            self._err(f"replace_quotes:{exc}")
            return None

    def method_fix_braces(self, json_string: str) -> Any:
        if not isinstance(json_string, str):
            json_string = "" if json_string is None else str(json_string)
        try:
            json_string = json_string.replace('"{', "{").replace('}"', "}")
            return json.loads(json_string)
        except Exception as exc:
            self._err(f"fix_braces:{exc}")
            return None

    def method_fix_commas(self, json_string: str) -> Any:
        if not isinstance(json_string, str):
            json_string = "" if json_string is None else str(json_string)
        json_string = json_string.replace(",}", "}").replace(",]", "]")
        try:
            return json.loads(json_string)
        except Exception as exc:
            self._err(f"fix_commas:{exc}")
            return None

    def method_fix_all(self, json_string: str) -> Any:
        if not isinstance(json_string, str):
            json_string = "" if json_string is None else str(json_string)
        json_string = (
            json_string.replace('"{', "{")
            .replace('}"', "}")
            .replace(",}", "}")
            .replace(",]", "]")
        )
        json_string = self.fix_missing_commas(json_string)
        try:
            return json.loads(json_string)
        except Exception as exc:
            self._err(f"fix_all:{exc}")
            return None

    def method_extract_segment(self, json_string: str) -> Any:
        if not isinstance(json_string, str):
            json_string = "" if json_string is None else str(json_string)
        json_segments = self.extract_json_segment(json_string)
        if json_segments:
            try:
                return json.loads(json_segments[-1])
            except Exception as exc:
                self._err(f"extract_segment:{exc}")
                return None
        return {}

    # ---------- schema coercion (shape enforcement) ----------

    def ensure_structure(self, parsed_json: Any, expected_structure: Any) -> Any:
        """
        Public wrapper around the coercion logic. Always returns something that
        matches expected_structure (dict, list, or scalar).
        """
        return self._ensure_structure_internal(parsed_json, expected_structure)

    def _ensure_structure_internal(self, data: Any, expected: Any) -> Any:
        if isinstance(expected, list):
            item_shape = expected[0] if expected else {}
            if not isinstance(data, list):
                if isinstance(data, dict) and isinstance(item_shape, dict):
                    data = [data]
                elif data is None:
                    data = []
                else:
                    data = [data]
            out_list: List[Any] = []
            for v in data:
                if isinstance(item_shape, dict):
                    out_list.append(
                        self._ensure_structure_internal(
                            v if isinstance(v, dict) else {}, item_shape
                        )
                    )
                elif isinstance(item_shape, list):
                    out_list.append(
                        self._ensure_structure_internal(
                            v if isinstance(v, list) else [], item_shape
                        )
                    )
                else:
                    out_list.append(self._coerce_scalar(v, item_shape))
            return out_list

        if isinstance(expected, dict):
            src = data if isinstance(data, dict) else {}
            out: Dict[str, Any] = {}
            for key, expected_type in expected.items():
                value = src.get(key) if isinstance(src, dict) else None
                if isinstance(expected_type, list):
                    if not isinstance(value, list):
                        value = [] if not isinstance(value, dict) else [value]
                    out[key] = self._ensure_structure_internal(value, expected_type)
                elif isinstance(expected_type, dict):
                    if isinstance(value, list) and value:
                        v0 = value[0] if isinstance(value[0], dict) else {}
                        out[key] = self._ensure_structure_internal(v0, expected_type)
                    else:
                        out[key] = self._ensure_structure_internal(
                            value if isinstance(value, dict) else {}, expected_type
                        )
                else:
                    out[key] = self._coerce_scalar(value, expected_type)
            return out

        return self._coerce_scalar(data, expected)

    def _coerce_scalar(self, value: Any, expected_type: Any) -> Any:
        if expected_type is int:
            return value if isinstance(value, int) else 0
        if expected_type is float:
            return value if isinstance(value, (int, float)) else 0.0
        if expected_type is list:
            return value if isinstance(value, list) else []
        if expected_type is dict:
            return value if isinstance(value, dict) else {}
        return value if isinstance(value, str) else ""

    def default_value(self, expected_type: Any) -> Any:
        if expected_type == int:
            return 0
        if expected_type == float:
            return 0.0
        if expected_type == list:
            return []
        if expected_type == dict:
            return {}
        return ""

    def ensure_list_structure(
        self, parsed_json_list: List[Dict[str, Any]], expected_structure: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        result_list: List[Dict[str, Any]] = []
        for item in parsed_json_list:
            result: Dict[str, Any] = {}
            for key, expected_type in expected_structure.items():
                value = item.get(key, self.default_value(expected_type))
                if not isinstance(value, expected_type):
                    value = self.default_value(expected_type)
                result[key] = value
            result_list.append(result)
        return result_list

    def select_best_result(
        self,
        current_best: Any,
        new_result: Any,
        expected_structure: Any,
    ) -> Any:
        if isinstance(expected_structure, list):
            if isinstance(new_result, list):
                parsed_list = new_result
            elif isinstance(new_result, dict):
                parsed_list = [new_result]
            else:
                parsed_list = []
            return self.ensure_structure(parsed_list, expected_structure)

        if not isinstance(expected_structure, dict):
            return self._coerce_scalar(new_result, expected_structure)

        if not isinstance(current_best, dict):
            current_best = {}
        if isinstance(new_result, dict):
            for key, expected_type in expected_structure.items():
                new_val = new_result.get(key)
                # Nested list/dict schemas are handled via full structure coercion
                # instead of direct isinstance checks, to avoid treating schema
                # templates (e.g. [{"tool": str, "args": dict}]) as type objects.
                if isinstance(expected_type, (list, dict)):
                    if new_val is not None:
                        current_best[key] = self._ensure_structure_internal(
                            new_val,
                            expected_type,
                        )
                    elif key not in current_best:
                        current_best[key] = self.default_value(expected_type)
                    continue
                # Scalar leaf types (str, int, float, etc.)
                if isinstance(new_val, expected_type):
                    current_best[key] = new_val
                elif key not in current_best:
                    current_best[key] = self.default_value(expected_type)
        return current_best

    # ---------- extras diff & vars extraction ----------

    def _diff_extras(self, raw: Any, coerced: Any) -> Any:
        if isinstance(raw, dict) and isinstance(coerced, dict):
            out: Dict[str, Any] = {}
            for k, v in raw.items():
                if k not in coerced:
                    out[k] = v
                else:
                    sub = self._diff_extras(v, coerced.get(k))
                    if self._non_empty(sub):
                        out[k] = sub
            return out
        if isinstance(raw, list) and isinstance(coerced, list):
            extras_list: List[Any] = []
            n = max(len(raw), len(coerced))
            for i in range(n):
                rv = raw[i] if i < len(raw) else None
                cv = coerced[i] if i < len(coerced) else None
                sub = self._diff_extras(rv, cv)
                if self._non_empty(sub):
                    extras_list.append(sub)
            if len(raw) > len(coerced):
                for j in range(len(coerced), len(raw)):
                    extras_list.append(raw[j])
            return extras_list
        if raw == coerced:
            return None
        return raw

    def _non_empty(self, v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, (list, dict)):
            return len(v) > 0
        return True

    def _extract_vars(self, raw: Any) -> Dict[str, Any]:
        vars_map: Dict[str, Any] = {}

        def _scan_str(s: str) -> None:
            t = s.strip()
            for key in ("width", "height", "steps", "cfg", "seed"):
                pat = r"\b" + key + r"\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\b"
                for m in re.finditer(pat, t, flags=re.IGNORECASE):
                    val = m.group(1)
                    if "." in val and key not in ("steps", "seed"):
                        if val.replace(".", "", 1).isdigit():
                            if key in ("steps", "seed"):
                                vars_map[key] = int(float(val))
                            else:
                                vars_map[key] = float(val)
                    elif val.isdigit():
                        if key in ("steps", "seed"):
                            vars_map[key] = int(val)
                        else:
                            vars_map[key] = int(val)
            for m in re.finditer(
                r"\b([A-Za-z0-9_\-]+)\s*[:=]\s*([A-Za-z0-9_\-\.]+)\b", t
            ):
                k = m.group(1).lower()
                v = m.group(2)
                if k not in vars_map:
                    if v.replace(".", "", 1).isdigit():
                        if "." in v:
                            vars_map[k] = float(v)
                        else:
                            vars_map[k] = int(v)
                    else:
                        vars_map[k] = v

        def _walk(x: Any) -> None:
            if isinstance(x, str):
                _scan_str(x)
                return
            if isinstance(x, list):
                for it in x:
                    _walk(it)
                return
            if isinstance(x, dict):
                for it in x.values():
                    _walk(it)
                return

        _walk(raw)
        return vars_map

    def _is_default_scalar(self, value: Any, expected_type: Any) -> bool:
        """
        Return True if the current value should be considered "missing" for the
        purposes of salvage. We only overwrite such default-ish values; any
        non-default (already sensible) value is left untouched.
        """
        if expected_type is int:
            return not isinstance(value, int) or value == 0
        if expected_type is float:
            return not isinstance(value, (int, float)) or float(value) == 0.0
        if expected_type is str:
            return not isinstance(value, str) or not value.strip()
        if isinstance(value, (list, dict)):
            return len(value) == 0
        return value is None

    def _extract_scalar_from_text(self, text: str, key: str, expected_type: Any) -> Optional[Any]:
        """
        Extract a scalar value for a given key from arbitrary text. This is a
        heuristic, last-resort helper and must tolerate spaces, punctuation,
        and partial JSON or natural language fragments.
        """
        if not isinstance(text, str) or not text:
            return None
        key_pattern = re.escape(str(key))
        # Look for "key: <value>" or "key = <value>" patterns; capture everything
        # to the end of the line or until a closing delimiter, then trim.
        pattern = rf"{key_pattern}\s*[:=]\s*(?P<val>.+)"
        m = re.search(pattern, text)
        if m:
            raw_val = m.group("val")
            if not isinstance(raw_val, str):
                raw_val = str(raw_val)
            raw = raw_val.lstrip()
            # First, if another key-like token appears later on this line
            # (e.g., "color: blue color2: red"), treat the start of that token
            # as a delimiter so we only keep "blue".
            next_key = re.search(r"\s+[A-Za-z0-9_\-]+\s*:", raw)
            if next_key:
                raw = raw[: next_key.start()].rstrip()
            # Then truncate at common structural delimiters while allowing
            # punctuation and spaces within the value itself.
            for sep in [",", "\n", "\r", ";", "}", "]"]:
                idx = raw.find(sep)
                if idx != -1:
                    raw = raw[:idx]
                    break
        else:
            # Fallback: handle the pattern where the key appears on its own line
            # and the value is on the next non-empty line, e.g.:
            #   color
            #   red
            raw = ""
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if re.fullmatch(rf"\s*{key_pattern}\s*[:=]?\s*$", line):
                    # Find the next non-empty line.
                    for j in range(idx + 1, len(lines)):
                        candidate = lines[j].strip()
                        if not candidate:
                            continue
                        # If the next non-empty line looks like another key
                        # declaration (e.g., "color2:"), treat this as no value.
                        if re.match(r"[A-Za-z0-9_\-]+\s*[:=]\s*", candidate):
                            raw = ""
                            break
                        raw = candidate
                        break
                    break
        raw = raw.strip().strip("\"'")
        if not raw:
            return None
        # If the extracted chunk immediately looks like the *next* key (e.g.
        # "color: color2: blue"), treat this as "no value" for the current key.
        # i.e., a new identifier followed by ":" means we shouldn't salvage.
        if re.match(r"[A-Za-z0-9_\-]+\s*:", raw):
            return None
        # Treat common "no value" markers as intentional empties; do not try to
        # salvage a concrete value from them.
        lowered = raw.lower()
        if lowered in ("none", "null", "n/a", "na", "-", "--"):
            return None
        try:
            if expected_type is int:
                return int(float(raw))
            if expected_type is float:
                return float(raw)
            if expected_type is str:
                # Collapse excessive whitespace but preserve punctuation.
                return re.sub(r"\s+", " ", raw)
        except (ValueError, TypeError):
            return None
        return None

    def _salvage_node_from_text(self, text: str, expected: Any, current: Any) -> Any:
        """
        Walk the expected structure (dicts/lists) and, for any scalar fields
        that are still default/missing in `current`, attempt to salvage a value
        from the raw text. Never overwrites non-default values.
        """
        # Dict: recurse into nested fields.
        if isinstance(expected, dict):
            if not isinstance(current, dict):
                current = {}  # type: ignore[assignment]
            for key, expected_type in expected.items():
                if isinstance(expected_type, (dict, list)):
                    cur_val = current.get(key)
                    current[key] = self._salvage_node_from_text(text, expected_type, cur_val)
                    continue
                if expected_type not in (int, float, str):
                    continue
                cur_val = current.get(key)
                if not self._is_default_scalar(cur_val, expected_type):
                    # Do not clobber an already-meaningful value.
                    continue
                extracted = self._extract_scalar_from_text(text, key, expected_type)
                if extracted is not None:
                    current[key] = extracted
            return current

        # List: apply salvage to each element using the first element as schema.
        if isinstance(expected, list) and expected:
            proto = expected[0]
            if not isinstance(current, list):
                items: List[Any] = [current] if current is not None else []
            else:
                items = list(current)
            return [self._salvage_node_from_text(text, proto, item) for item in items]

        # Scalar or unsupported schema node: nothing to salvage here.
        return current

    def _salvage_from_text(self, text: Any, expected_structure: Any, result: Any) -> Any:
        """
        Best-effort salvage of scalar fields from raw, possibly non-JSON text.

        This is a last-resort path: it is only invoked after best-effort JSON
        parsing, and it is only allowed to fill in fields that are clearly
        default/missing (0, 0.0, empty string, empty list/dict, or None).
        It never overwrites non-default values.
        """
        if not isinstance(text, str):
            return result
        return self._salvage_node_from_text(text, expected_structure, result)

    # ---------- diagnostics ----------

    def _note(self, msg: str) -> None:
        self.repairs.append(msg)

    def _err(self, msg: str) -> None:
        self.last_error = msg
        self.errors.append(msg)
        # Surface parser diagnostics through the shared logger so they land
        # in the central orchestrator logs instead of a private file.
        logger.info(f"JSONParser error: {msg}")


