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
            self._err(f"parse_strict_pristine:{e1}")
        repaired = self.attempt_repair(s)
        try:
            obj = json.loads(repaired)
            return True, obj
        except Exception as e2:
            self._err(f"parse_strict_repaired:{e2}")
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
        try:
            return json.loads(json_string)
        except Exception:
            pass
        json_string = re.sub(r"'", '"', json_string)
        json_string = re.sub(
            r"(?<=\{|,)\s*([A-Za-z0-9_]+)\s*:",
            r'"\1":',
            json_string,
        )
        return json.loads(json_string)

    # ---------- method variants, each using json.loads ----------

    def method_json_loads(self, json_string: str) -> Any:
        return json.loads(json_string)

    def method_replace_quotes(self, json_string: str) -> Any:
        json_string = json_string.replace("\\'", "'").replace('\\"', '"')
        return json.loads(json_string)

    def method_fix_braces(self, json_string: str) -> Any:
        json_string = json_string.replace('"{', "{").replace('}"', "}")
        return json.loads(json_string)

    def method_fix_commas(self, json_string: str) -> Any:
        json_string = json_string.replace(",}", "}").replace(",]", "]")
        return json.loads(json_string)

    def method_fix_all(self, json_string: str) -> Any:
        json_string = (
            json_string.replace('"{', "{")
            .replace('}"', "}")
            .replace(",}", "}")
            .replace(",]", "]")
        )
        json_string = self.fix_missing_commas(json_string)
        return json.loads(json_string)

    def method_extract_segment(self, json_string: str) -> Any:
        json_segments = self.extract_json_segment(json_string)
        if json_segments:
            return json.loads(json_segments[-1])
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

    def _salvage_from_text(self, text: str, expected_structure: Any, result: Any) -> Any:
        """
        Best-effort salvage of scalar fields from raw, possibly non-JSON text.

        This is intended as a last resort when normal JSON parsing/repair cannot
        recover certain fields. We scan the original text for simple "key: value"
        style patterns and, when we find something that looks like a valid scalar
        for that key, we inject it into the coerced result without overwriting
        non-default values.

        Salvage is **schema-driven but fully generic**: it walks the expected
        structure recursively (dicts/lists) and tries to recover any scalar
        (int/float/str) fields it finds, regardless of depth.
        """
        if not isinstance(text, str):
            return result
        lower_text = text

        def _is_default(val: Any, typ: Any) -> bool:
            if typ is int:
                return not isinstance(val, int) or val == 0
            if typ is float:
                return not isinstance(val, (int, float)) or float(val) == 0.0
            if typ is str:
                return not isinstance(val, str) or not val.strip()
            if isinstance(val, (list, dict)):
                return not bool(val)
            return val is None

        def _salvage_node(exp: Any, res: Any) -> Any:
            # Dict: recurse into nested fields.
            if isinstance(exp, dict):
                if not isinstance(res, dict):
                    res = {}  # type: ignore[assignment]
                for key, et in exp.items():
                    if isinstance(et, (dict, list)):
                        cur = res.get(key)
                        res[key] = _salvage_node(et, cur)
                        continue
                    # Scalar salvage (int/float/str)
                    if et not in (int, float, str):
                        continue
                    cur_val = res.get(key)
                    if not _is_default(cur_val, et):
                        continue
                    try:
                        key_pattern = re.escape(str(key))
                        if et in (int, float):
                            val_pattern = r"([-+]?\d+(?:\.\d+)?)"
                        else:
                            val_pattern = r"(.+?)"
                        pattern = rf"{key_pattern}\s*[:=\-]?\s*{val_pattern}(?:[\n\r,;]|$)"
                        m = re.search(pattern, lower_text)
                        if not m:
                            continue
                        raw_val = m.group(1)
                        if not isinstance(raw_val, str):
                            raw_val = str(raw_val)
                        raw_val = raw_val.strip().strip("\"' \t")
                        if not raw_val:
                            continue
                        if et is int:
                            try:
                                res[key] = int(float(raw_val))
                            except Exception:
                                continue
                        elif et is float:
                            try:
                                res[key] = float(raw_val)
                            except Exception:
                                continue
                        else:  # str
                            cleaned = re.sub(r"\s+", " ", raw_val)
                            res[key] = cleaned
                    except Exception:
                        continue
                return res

            # List: apply salvage to each element using the first element as schema.
            if isinstance(exp, list) and exp:
                prototype = exp[0]
                if not isinstance(res, list):
                    res_list: List[Any] = []
                else:
                    res_list = res
                out_list: List[Any] = []
                for item in res_list:
                    out_list.append(_salvage_node(prototype, item))
                return out_list

            # Scalar or unsupported schema node: nothing to salvage here.
            return res

        return _salvage_node(expected_structure, result)

    # ---------- diagnostics ----------

    def _note(self, msg: str) -> None:
        self.repairs.append(msg)

    def _err(self, msg: str) -> None:
        self.last_error = msg
        self.errors.append(msg)
        # Surface parser diagnostics through the shared logger so they land
        # in the central orchestrator logs instead of a private file.
        logger.info(f"JSONParser error: {msg}")


