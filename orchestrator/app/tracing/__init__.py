"""
Tracing sinks (runtime vs training) with explicit APIs.

This package exists to prevent accidental mixing of:
  - runtime/logging traces -> <uploads>/state/traces/<trace_id>/trace.jsonl
  - training sample traces -> <uploads>/datasets/trace/<modality>.jsonl
"""





