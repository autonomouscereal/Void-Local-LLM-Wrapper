'''
JSONParser.py
'''


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
        # First try a direct json.loads – many times the string is already
        # valid JSON and we can avoid any risky regex transformations.
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            pass  # proceed with more permissive recovery steps

        # ------------------------------------------------------------------
        # Loose, best-effort fixes – *without* destroying already correct
        # syntax.  We no longer escape every double quote which previously
        # produced invalid JSON like {\"foo\": 1}.
        # ------------------------------------------------------------------
        # 1) Replace single quotes with double quotes (common LLM output)
        json_string = re.sub(r"'", '"', json_string)
        # 2) Add double quotes around unquoted keys (e.g. {foo: 1})
        json_string = re.sub(r'(?<=\{|,)\s*([A-Za-z0-9_]+)\s*:', r'"\1":', json_string)

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
                # Downgrade to DEBUG to avoid expected fallback failures spamming ERROR logs
                logger.debug(f"Method %s failed during JSON parse fallbacks: %s", method.__name__, e)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Stack trace:", exc_info=True)

        # ------------------------------------------------------------------
        # NEW – Fallback: ensure we always return a structure matching
        # *expected_structure* even when none of the parsing strategies
        # succeeded.  This guarantees callers never receive ``{}``/``[]``
        # placeholders which simplifies downstream error handling.
        # ------------------------------------------------------------------
        if not best_result:
            # When *expected_structure* is a list, return an empty list with
            # the correct item schema.  Otherwise create a dict with default
            # values for each expected key.
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
                logger.debug(f"Ensured key '{key}' with value: {result[key]}")
            result_list.append(result)
        return result_list

    def select_best_result(self, current_best, new_result, expected_structure):
        """Merge *new_result* into *current_best* while ensuring it matches *expected_structure*.

        The logic originally assumed *expected_structure* is always a mapping (dict) which
        is not true – it can also be a list specifying the schema of list items (see
        the example at the bottom of this file).  Trying to iterate over
        ``expected_structure.items()`` in that situation raises **AttributeError**.
        This revised implementation handles both cases gracefully:
        • When *expected_structure* is a list, we normalise *new_result* to a list and
          delegate to ``ensure_structure`` which already knows how to build the list
          output.
        • When it is a dict, we keep the previous merging behaviour.
        """
        # ------------------------------------------------------------------
        # Case 1 – list-based expected structure (e.g. ``[{"foo": str}]``)
        # ------------------------------------------------------------------
        if isinstance(expected_structure, list):
            # Normalise *new_result* into a list so that ``ensure_structure`` can
            # process it uniformly.
            if isinstance(new_result, list):
                parsed_list = new_result
            elif isinstance(new_result, dict):
                parsed_list = [new_result]
            else:
                parsed_list = []  # Unable to coerce – fall back to empty list

            return self.ensure_structure(parsed_list, expected_structure)

        # ------------------------------------------------------------------
        # Case 2 – mapping/dict-based expected structure (original behaviour)
        # ------------------------------------------------------------------
        if not isinstance(current_best, dict):
            current_best = {}

        if isinstance(new_result, dict):
            for key, expected_type in expected_structure.items():
                new_val = new_result.get(key)

                # Accept immediately if the new value is of the right type.
                if isinstance(new_val, expected_type):
                    current_best[key] = new_val
                elif key not in current_best:
                    # Fallback – keep at least a default value.
                    current_best[key] = self.default_value(expected_type)

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
            # The LLM may include example JSON objects before the final answer.
            # Heuristic: the *last* JSON segment appearing in the text is most
            # likely the actual answer, so parse that one instead of the first.
            return json.loads(json_segments[-1])
        return {}

# ---------------------------------------------------------------------------
# Example usage   – executed only when running this module directly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Only run the demo when the file is executed as a script, not on import.
    parser = JSONParser()

    json_string = '''
    [
      {
        "time_passed": "1 second",
        "description": "Luna, the elf with a bow and arrow, takes a careful step forward, raising her bow slightly as she scans the shadows with focused eyes, positioned at coordinates (X: 10, Y: 5, Z: 0). Thorne, the dwarf with a large hammer, moves closer to Luna, his rugged features tense with anticipation, standing at (X: 9, Y: 4, Z: 0). Elyse, the mage with a glowing staff, hovers slightly behind, her staff emitting a soft, mystical light, located at (X: 10, Y: 3, Z: 0). The forest remains dense and eerie, with faint moonlight peeking through the thick canopy."
      },
      {
        "time_passed": "2 seconds",
        "description": "Luna's keen elven senses detect a slight rustling to the left. She adjusts her position swiftly, aiming her bow in the direction of the noise, now at (X: 11, Y: 6, Z: 0). Thorne stands his ground, ready to protect his companions, gripping his hammer firmly at (X: 9, Y: 4, Z: 0). Elyse maintains her magical stance, observing the surroundings closely with her staff glowing brighter, located at (X: 10, Y: 3, Z: 0). The shadows in the forest deepen as the adventurers remain vigilant."
      },
      {
        "time_passed": "3 seconds",
        "description": "A pair of glowing eyes peeks out from behind a tree, revealing a mystical creature watching the adventurers intently. Luna's arrow is drawn, her bow pointing directly at the creature, now positioned at (X: 12, Y: 7, Z: 0). Thorne readies his hammer, muscles tensed, prepared for any sudden movements, standing firm at (X: 9, Y: 4, Z: 0). Elyse's staff pulsates with magical energy, casting a soft light on the creature, situated at (X: 10, Y: 3, Z: 0). The forest comes alive with unseen dangers lurking in the shadows."
      }
    ]
    '''

    expected_structure = [
        {
            "time_passed": str,
            "description": str
        }
    ]

    result = parser.parse(json_string, expected_structure)
    print(json.dumps(result, indent=2))