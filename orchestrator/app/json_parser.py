from __future__ import annotations

import json
import logging
import re


# Configure logging (non-intrusive by default to avoid I/O stalls)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class JSONParser:
    def __init__(self):
        pass

    def strip_markdown(self, response: str) -> str:
        response = response.replace('```json', '').replace('```', '').strip()
        return response

    def correct_common_errors(self, text: str) -> str:
        corrections = [
            ('\\"', '"'),
            ("\\'", "'"),
            ('"{', '{'),
            ('}"', '}'),
            (",}", "}"),
            (",]", "]"),
            ('"False"', 'false'),
            ('"True"', 'true'),
        ]
        for old, new in corrections:
            text = text.replace(old, new)
        return text

    def extract_json_segment(self, text: str, start_delim: str = '{', end_delim: str = '}'):
        stack = []
        json_segments = []
        start_index = -1
        for i, char in enumerate(text):
            if char == start_delim:
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == end_delim and stack:
                stack.pop()
                if not stack:
                    segment = text[start_index:i+1]
                    json_segments.append(segment)
        return json_segments

    def fix_missing_commas(self, json_string: str) -> str:
        json_string = json_string.replace('}{', '},{')
        return json_string

    def attempt_repair(self, json_string: str) -> str:
        json_string = self.correct_common_errors(json_string)
        json_string = self.fix_missing_commas(json_string)
        return json_string

    def regex_parse(self, json_string: str):
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            pass
        json_string = re.sub(r"'", '"', json_string)
        json_string = re.sub(r'(?<=\{|,)\s*([A-Za-z0-9_]+)\s*:', r'"\1":', json_string)
        return json.loads(json_string)

    def ensure_structure(self, parsed_json, expected_structure):
        if isinstance(expected_structure, list):
            if not isinstance(parsed_json, list):
                parsed_json = []
            return [self.ensure_structure(item, expected_structure[0]) for item in parsed_json]

        result = {}
        for key, expected_type in expected_structure.items():
            value = parsed_json.get(key, self.default_value(expected_type))
            if isinstance(expected_type, list):
                if not isinstance(value, list):
                    value = []
                else:
                    item_structure = expected_type[0]
                    value = [self.ensure_structure(item, item_structure) if isinstance(item, dict) else item for item in value]
            elif isinstance(expected_type, dict):
                if isinstance(value, dict):
                    value = self.ensure_structure(value, expected_type)
                else:
                    value = self.default_value(dict)
            else:
                try:
                    if not isinstance(value, expected_type):
                        value = self.default_value(expected_type)
                except TypeError:
                    value = self.default_value(str)
            result[key] = value
        return result

    def default_value(self, expected_type):
        if expected_type == int:
            return 0
        elif expected_type == float:
            return 0.0
        elif expected_type == list:
            return []
        elif expected_type == dict:
            return {}
        else:
            return ""

    def parse(self, json_string: str, expected_structure):
        stripped = self.strip_markdown(json_string)
        try:
            pristine = self.method_json_loads(stripped)
            return self.ensure_structure(pristine, expected_structure)
        except Exception:
            pass
        json_string = self.attempt_repair(stripped)
        methods = [
            self.method_json_loads,
            self.method_replace_quotes,
            self.method_fix_braces,
            self.method_fix_commas,
            self.method_fix_all,
            self.method_extract_segment,
            self.regex_parse,
        ]
        best_result = {}
        for method in methods:
            try:
                parsed_json = method(json_string)
                if isinstance(parsed_json, list) and all(isinstance(item, dict) for item in parsed_json):
                    best_result = self.ensure_structure(parsed_json, expected_structure)
                else:
                    best_result = self.select_best_result(best_result, parsed_json, expected_structure)
            except Exception:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("fallback method failed", exc_info=True)
        if not best_result:
            if isinstance(expected_structure, list):
                best_result = self.ensure_structure([], expected_structure)
            else:
                best_result = self.ensure_structure({}, expected_structure)
        return best_result

    def ensure_list_structure(self, parsed_json_list, expected_structure):
        result_list = []
        for item in parsed_json_list:
            result = {}
            for key, expected_type in expected_structure.items():
                value = item.get(key, self.default_value(expected_type))
                if not isinstance(value, expected_type):
                    value = self.default_value(expected_type)
                result[key] = value
            result_list.append(result)
        return result_list

    def select_best_result(self, current_best, new_result, expected_structure):
        if isinstance(expected_structure, list):
            if isinstance(new_result, list):
                parsed_list = new_result
            elif isinstance(new_result, dict):
                parsed_list = [new_result]
            else:
                parsed_list = []
            return self.ensure_structure(parsed_list, expected_structure)

        if not isinstance(current_best, dict):
            current_best = {}

        if isinstance(new_result, dict):
            for key, expected_type in expected_structure.items():
                new_val = new_result.get(key)
                if isinstance(expected_type, list):
                    if isinstance(new_val, list):
                        current_best[key] = self.ensure_structure(new_val, expected_type)
                    else:
                        current_best[key] = []
                    continue
                if isinstance(expected_type, dict):
                    if isinstance(new_val, dict):
                        current_best[key] = self.ensure_structure(new_val, expected_type)
                    else:
                        current_best[key] = {}
                    continue
                try:
                    if isinstance(new_val, expected_type):
                        current_best[key] = new_val
                    elif key not in current_best:
                        current_best[key] = self.default_value(expected_type)
                except TypeError:
                    if key not in current_best:
                        current_best[key] = self.default_value(dict if isinstance(expected_type, dict) else list if isinstance(expected_type, list) else str)
        return current_best

    def method_json_loads(self, json_string: str):
        return json.loads(json_string)

    def method_replace_quotes(self, json_string: str):
        json_string = json_string.replace("\\'", "'").replace('\\"', '"')
        return json.loads(json_string)

    def method_fix_braces(self, json_string: str):
        json_string = json_string.replace('"{', '{').replace('}"', '}')
        return json.loads(json_string)

    def method_fix_commas(self, json_string: str):
        json_string = json_string.replace(",}", "}").replace(",]", "]")
        return json.loads(json_string)

    def method_fix_all(self, json_string: str):
        json_string = json_string.replace('"{', '{').replace('}"', '}').replace(",}", "}").replace(",]", "]")
        json_string = self.fix_missing_commas(json_string)
        return json.loads(json_string)

    def method_extract_segment(self, json_string: str):
        json_segments = self.extract_json_segment(json_string)
        if json_segments:
            return json.loads(json_segments[-1])
        return {}


