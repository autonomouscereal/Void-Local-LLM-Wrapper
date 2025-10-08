import json
import logging
import re

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
handler = logging.FileHandler('json_parser.log')
handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

class JSONParser:
    def __init__(self):
        pass

    def strip_markdown(self, response):
        logger.debug("Stripping markdown from response")
        response = response.replace('```json', '').replace('```', '').strip()
        logger.debug(f"Response after stripping markdown: {response}")
        return response

    def correct_common_errors(self, text):
        logger.debug("Correcting common JSON format errors")
        corrections = [
            ('\\"', '"'),      # Unescape double quotes
            ("\\'", "'"),      # Unescape single quotes
            ('"{', '{'),       # Remove quotes around braces
            ('}"', '}'),
            (",}", "}"),       # Trailing comma before a closing brace
            (",]", "]"),       # Trailing comma before a closing bracket
            ('"False"', 'false'),  # Incorrectly capitalized boolean
            ('"True"', 'true')
        ]
        for old, new in corrections:
            text = text.replace(old, new)
            logger.debug(f"Text after replacing {old} with {new}: {text}")
        return text

    def extract_json_segment(self, text, start_delim='{', end_delim='}'):
        logger.debug("Extracting JSON segments from text")
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
                    logger.debug(f"Extracted JSON segment: {segment}")
        return json_segments

    def fix_missing_commas(self, json_string):
        logger.debug("Fixing missing commas in JSON string")
        json_string = json_string.replace('}{', '},{')
        logger.debug(f"JSON string after fixing missing commas: {json_string}")
        return json_string

    def attempt_repair(self, json_string):
        logger.debug("Attempting to repair JSON string")
        json_string = self.correct_common_errors(json_string)
        json_string = self.fix_missing_commas(json_string)
        logger.debug(f"JSON string after repair: {json_string}")
        return json_string

    def regex_parse(self, json_string):
        logger.debug("Attempting regex-based JSON parsing")
        json_string = re.sub(r'(?<!\\)"', r'\"', json_string)  # Escape double quotes
        json_string = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_string)  # Escape backslashes not followed by a valid escape character
        json_string = re.sub(r"'", r'"', json_string)  # Replace single quotes with double quotes
        json_string = re.sub(r'(\w+):', r'"\1":', json_string)  # Add double quotes around keys
        logger.debug(f"JSON string after regex-based parsing: {json_string}")
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
            elif not isinstance(value, expected_type):
                value = self.default_value(expected_type)
            result[key] = value
            logger.debug(f"Ensured key '{key}' with value: {result[key]}")
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

    def parse(self, json_string, expected_structure):
        logger.debug("Starting JSON parsing process")
        logger.debug(f"Original JSON string: {json_string}")
        json_string = self.strip_markdown(json_string)
        json_string = self.attempt_repair(json_string)

        # Attempt multiple parsing methods
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
            except Exception as e:
                logger.exception(f"Error in method {method.__name__}: {e}")

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
                logger.debug(f"Ensured key '{key}' with value: {result[key]}")
            result_list.append(result)
        return result_list

    def select_best_result(self, current_best, new_result, expected_structure):
        for key, expected_type in expected_structure.items():
            if key not in current_best or not isinstance(current_best[key], expected_type):
                current_best[key] = new_result.get(key, self.default_value(expected_type)) if isinstance(new_result, dict) else self.default_value(expected_type)
        return current_best

    def method_json_loads(self, json_string):
        return json.loads(json_string)

    def method_replace_quotes(self, json_string):
        json_string = json_string.replace("\\'", "'").replace('\\"', '"')
        return json.loads(json_string)

    def method_fix_braces(self, json_string):
        json_string = json_string.replace('"{', '{').replace('}"', '}')
        return json.loads(json_string)

    def method_fix_commas(self, json_string):
        json_string = json_string.replace(",}", "}").replace(",]", "]")
        return json.loads(json_string)

    def method_fix_all(self, json_string):
        json_string = json_string.replace('"{', '{').replace('}"', '}').replace(",}", "}").replace(",]", "]")
        json_string = self.fix_missing_commas(json_string)
        return json.loads(json_string)

    def method_extract_segment(self, json_string):
        json_segments = self.extract_json_segment(json_string)
        if json_segments:
            return json.loads(json_segments[0])
        return {}

    def regex_parse(self, json_string):
        return json.loads(json_string)


