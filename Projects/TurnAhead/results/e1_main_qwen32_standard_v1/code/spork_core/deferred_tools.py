"""Allow speculative model computation while holding all child tool execution."""
from __future__ import annotations

import threading
import time


class DiscardedChild(RuntimeError):
    pass


class DeferredTools:
    def __init__(self, executor):
        self.executor = executor
        self.event = threading.Event()
        self.allowed = False
        self.wait_s = 0.0
        self.blocked_calls = 0
        self.executed_calls = 0
        self._lock = threading.Lock()

    def __getattr__(self, key):
        return getattr(self.executor, key)

    def allow(self):
        self.allowed = True
        self.event.set()

    def discard(self):
        self.allowed = False
        self.event.set()

    def _wait(self):
        start = time.perf_counter()
        with self._lock:
            self.blocked_calls += int(not self.event.is_set())
        self.event.wait()
        with self._lock:
            self.wait_s += time.perf_counter() - start
        if not self.allowed:
            raise DiscardedChild("Parent rejected this isolated child")
        with self._lock:
            self.executed_calls += 1

    def execute(self, call):
        self._wait()
        return self.executor.execute(call)

    def execute_speculative(self, call):
        self._wait()
        return self.executor.execute_speculative(call)
