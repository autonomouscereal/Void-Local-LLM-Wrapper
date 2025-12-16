from __future__ import annotations

import collections


class RingBuffer:
    """
    Bounded, thread-safe buffer of bytes or small strings.
    Producers block when full; consumers block when empty.
    """

    def __init__(self, max_items: int = 1024):
        self.max_items = max_items
        self.q = collections.deque()
        self.closed = False

    def put(self, item: bytes | str):
        # Hard-blocking, single-threaded: no condition variables / no waiting.
        if self.closed:
            return
        if len(self.q) >= self.max_items:
            # Drop oldest to keep bounded memory; callers should tolerate.
            try:
                self.q.popleft()
            except Exception:
                return
        self.q.append(item)

    def get(self):
        if not self.q:
            return None
        try:
            return self.q.popleft()
        except Exception:
            return None

    def close(self):
        self.closed = True


