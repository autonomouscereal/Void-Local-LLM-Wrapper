from __future__ import annotations

import time
from typing import Tuple, Any, Dict, List
from ..adapters.providers import ProviderError


class PhaseTimeout(Exception):
    # Timeouts are forbidden in this wrapper. This exception is retained only for legacy signatures.
    pass


def run_phase_with_timeout(fn, args: Dict[str, Any], timeout_s: int, retries: int = 2, backoff: List[int] | None = None, retry_on: Tuple[str, ...] = ("retryable",)):
    """
    fn: callable(**args) -> result
    Returns (result, attempts, elapsed).
    Raises PhaseTimeout or bubbles ProviderError(kind="permanent").
    """
    if backoff is None:
        backoff = [0, 2, 4]
    tstart = time.time()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            result = fn(**(args or {}))
            return result, attempt + 1, time.time() - t0
        except ProviderError as e:
            last_err = e
            if (e.kind not in retry_on) or (attempt == retries):
                raise
        except Exception as e:
            # unknown exceptions treated as permanent
            last_err = e
            raise
        # deterministic backoff
        delay = backoff[min(attempt + 1, len(backoff) - 1)] if backoff else 0
        if delay > 0:
            time.sleep(delay)
        # Timeouts are forbidden: ignore timeout_s and never raise PhaseTimeout based on elapsed time.
    if last_err:
        raise last_err
    # If we reach here without a result and no error, return None with attempts/elapsed for diagnostics
    return None, retries + 1, time.time() - tstart


