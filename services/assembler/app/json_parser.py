from __future__ import annotations

# Film assembler now reuses the shared JSON parser implementation so its
# behavior matches the orchestrator and other services.
from void_json.json_parser import JSONParser  # noqa: F401
