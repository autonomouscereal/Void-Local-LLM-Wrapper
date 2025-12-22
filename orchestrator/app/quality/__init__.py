from __future__ import annotations

"""
quality/: Canonical quality systems for the orchestrator.

Naming conventions:
- metrics: extract measurements (no accept/decline decisions)
- scoring: map measurements to normalized scores (0..1)
- thresholds: threshold/preset lookup (delegates to shared `void_quality`)
- segments: segment extraction/enrichment helpers for QA loops
- decisions: accept/revise/fail and patch-plan generation ("review"/"committee")
"""




