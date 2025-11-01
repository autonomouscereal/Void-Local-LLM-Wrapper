from __future__ import annotations

import threading
import collections


class RingBuffer:
    """
    Bounded, thread-safe buffer of bytes or small strings.
    Producers block when full; consumers block when empty.
    """

    def __init__(self, max_items: int = 1024):
        self.max_items = max_items
        self.q = collections.deque()
        self.cv = threading.Condition()
        self.closed = False

    def put(self, item: bytes | str):
        with self.cv:
            while len(self.q) >= self.max_items and not self.closed:
                self.cv.wait()
            if self.closed:
                return
            self.q.append(item)
            self.cv.notify_all()

    def get(self):
        with self.cv:
            while not self.q and not self.closed:
                self.cv.wait()
            if not self.q:
                return None
            item = self.q.popleft()
            self.cv.notify_all()
            return item

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()


