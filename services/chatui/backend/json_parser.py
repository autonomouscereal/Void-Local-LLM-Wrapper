from __future__ import annotations

# ChatUI backend now delegates to the shared JSON parser used by the
# orchestrator and other services. This keeps behavior consistent while
# avoiding code duplication.
from void_json.json_parser import JSONParser  # noqa: F401
