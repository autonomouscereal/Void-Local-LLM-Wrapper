from __future__ import annotations

import json
import re


class JSONParser:
    def strip_markdown(self, response: str) -> str:
        return response.replace('```json', '').replace('```', '').strip()

    def fix_missing_commas(self, s: str) -> str:
        return s.replace('}{', '},{').replace(',}', '}').replace(',]', ']')

    def parse(self, s: str, expected_structure):
        txt = self.strip_markdown(s or '')
        # try pristine
        try:
            obj = json.loads(txt)
            return self.ensure_structure(obj, expected_structure)
        except Exception:
            pass
        # simple repairs
        try:
            obj = json.loads(self.fix_missing_commas(txt))
            return self.ensure_structure(obj, expected_structure)
        except Exception:
            pass
        # regex-based relaxed parse: quote keys and single quotes
        try:
            tmp = re.sub(r"'", '"', txt)
            tmp = re.sub(r'(?<=\{|,)\s*([A-Za-z0-9_]+)\s*:', r'"\1":', tmp)
            obj = json.loads(tmp)
            return self.ensure_structure(obj, expected_structure)
        except Exception:
            return self.ensure_structure({}, expected_structure)

    def ensure_structure(self, parsed, expected):
        if isinstance(expected, list):
            item = expected[0]
            if not isinstance(parsed, list):
                return []
            return [self.ensure_structure(p, item) if isinstance(p, dict) else p for p in parsed]
        out = {}
        for k, t in (expected or {}).items():
            v = parsed.get(k) if isinstance(parsed, dict) else None
            if isinstance(t, list):
                if isinstance(v, list):
                    out[k] = v
                else:
                    out[k] = []
            elif isinstance(t, dict):
                if isinstance(v, dict):
                    out[k] = self.ensure_structure(v, t)
                else:
                    out[k] = {}
            else:
                out[k] = v if isinstance(v, t) else (0 if t == int else 0.0 if t == float else "" if t == str else None)
        return out


