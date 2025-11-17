from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.FileHandler("json_parser.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


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

    def parse(self, json_string: str, expected_structure: Any) -> Any:
        """
        Best-effort parsing with repairs, followed by coercion into expected_structure.
        Never raises; always returns something matching the expected shape.
        """
        # logger.debug("Starting JSON parsing process")
        if not isinstance(json_string, str):
            json_string = "" if json_string is None else str(json_string)
        # logger.debug("Original JSON string: %s", json_string)

        # Step 1: normalize wrapper noise (markdown fences, BOM, whitespace)
        json_string = self.strip_markdown(json_string)
        # logger.debug("After strip_markdown: %s", json_string)

        # Step 2: quick repair pass for very common issues
        json_string = self.attempt_repair(json_string)
        # logger.debug("After attempt_repair: %s", json_string)

        # Step 3: apply a sequence of parsing strategies, keeping the "best" result
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
                # logger.debug("Method %s produced: %r", method.__name__, parsed_json)
                best_result = self.select_best_result(
                    best_result, parsed_json, expected_structure
                )
            except Exception as exc:
                msg = f"method:{method.__name__} failed:{exc}"
                logger.debug(msg, exc_info=True)
                self._err(msg)

        # Final fallback: if nothing succeeded, synthesise structure from schema alone
        if best_result is None:
            # logger.debug("No parsing strategy succeeded; synthesising from schema")
            if isinstance(expected_structure, list):
                best_result = self.ensure_structure([], expected_structure)
            else:
                best_result = self.ensure_structure({}, expected_structure)
        else:
            # Ensure the final result conforms exactly to the expected schema.
            best_result = self.ensure_structure(best_result, expected_structure)

        # logger.debug("Final coerced result: %r", best_result)
        return best_result

    # Backwards-compatible alias
    def parse_best_effort(self, text: str, expected_structure: Any) -> Any:
        return self.parse(text, expected_structure)

    def parse_strict(self, text: str) -> Tuple[bool, Optional[Any]]:
        """
        Strict load without schema coercion: returns (ok, raw_obj).
        Still uses markdown stripping and basic repair, but does not enforce shape.
        """
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        s = self.strip_markdown(text)
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

    def parse_superset(self, text: str, expected_structure: Any) -> Dict[str, Any]:
        """
        Return a richer result that includes:
          - coerced: structure shaped to expected_structure
          - raw: best-effort raw object from JSON
          - extras: parts of raw not represented in coerced
          - vars: numeric hints extracted from string leaves
          - repairs/errors/last_error: diagnostics
        """
        ok, raw = self.parse_strict(text)
        if not ok:
            raw = {} if isinstance(expected_structure, dict) else (
                [] if isinstance(expected_structure, list) else None
            )
        coerced = self.ensure_structure(raw, expected_structure)
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
        # logger.debug("Stripping markdown from response")
        if not isinstance(response, str):
            response = "" if response is None else str(response)
        response = response.replace("```json", "").replace("```", "").strip()
        # logger.debug("Response after stripping markdown: %s", response)
        # Also strip BOM / stray whitespace
        response = response.replace("\ufeff", "").strip()
        return response

    def correct_common_errors(self, text: str) -> str:
        # logger.debug("Correcting common JSON format errors")
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
                # logger.debug("Replaced %r with %r -> %s", old, new, text)
        return text

    def extract_json_segment(
        self, text: str, start_delim: str = "{", end_delim: str = "}"
    ) -> List[str]:
        """Return all balanced {...} segments from text (last one is usually the answer)."""
        # logger.debug("Extracting JSON segments from text")
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
                    # logger.debug("Extracted JSON segment: %s", segment)
        return json_segments

    def fix_missing_commas(self, json_string: str) -> str:
        #logger.debug("Fixing missing commas in JSON string")
        json_string = json_string.replace("}{", "},{")
        # logger.debug("JSON string after fixing missing commas: %s", json_string)
        return json_string

    def attempt_repair(self, json_string: str) -> str:
        # logger.debug("Attempting to repair JSON string")
        json_string = self.correct_common_errors(json_string)
        json_string = self.fix_missing_commas(json_string)
        # logger.debug("JSON string after repair: %s", json_string)
        return json_string

    def regex_parse(self, json_string: str) -> Any:
        """
        Regex-based fallback: handle common LLM quirks like single quotes and
        unquoted keys. This is intentionally permissive and only used after
        simpler strategies fail.
        """
        # logger.debug("Attempting regex-based JSON parsing")
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
        # logger.debug("JSON string after regex-based fixes: %s", json_string)
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
                # logger.debug("Ensured key %r with value: %r", key, result[key])
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

    # ---------- diagnostics ----------

    def _note(self, msg: str) -> None:
        self.repairs.append(msg)

    def _err(self, msg: str) -> None:
        self.last_error = msg
        self.errors.append(msg)


