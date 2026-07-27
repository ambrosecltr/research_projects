"""Thread-safe token buckets for bounded external API workloads."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock


@dataclass(slots=True)
class DualTokenBucket:
    """Atomically reserve request and model-token capacity."""

    requests_per_minute: int
    tokens_per_minute: int
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _request_tokens: float = field(init=False, repr=False)
    _model_tokens: float = field(init=False, repr=False)
    _updated_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("requests_per_minute", self.requests_per_minute),
            ("tokens_per_minute", self.tokens_per_minute),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._request_tokens = float(self.requests_per_minute)
        self._model_tokens = float(self.tokens_per_minute)
        self._updated_at = self.clock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated_at)
        self._request_tokens = min(
            float(self.requests_per_minute),
            self._request_tokens + elapsed * self.requests_per_minute / 60.0,
        )
        self._model_tokens = min(
            float(self.tokens_per_minute),
            self._model_tokens + elapsed * self.tokens_per_minute / 60.0,
        )
        self._updated_at = now

    def acquire(self, model_tokens: int) -> None:
        """Block until one request and the stated model-token estimate are reserved."""
        if isinstance(model_tokens, bool) or not isinstance(model_tokens, int) or model_tokens < 1:
            raise ValueError("model_tokens must be a positive integer")
        if model_tokens > self.tokens_per_minute:
            raise ValueError("one request cannot exceed the token bucket capacity")

        while True:
            with self._lock:
                self._refill(self.clock())
                if self._request_tokens >= 1.0 and self._model_tokens >= model_tokens:
                    self._request_tokens -= 1.0
                    self._model_tokens -= model_tokens
                    return
                request_wait = max(0.0, 1.0 - self._request_tokens) * 60.0
                request_wait /= self.requests_per_minute
                token_wait = max(0.0, model_tokens - self._model_tokens) * 60.0
                token_wait /= self.tokens_per_minute
                wait_seconds = max(request_wait, token_wait)
            self.sleeper(wait_seconds)

    def refund_model_tokens(self, reserved: int, used: int) -> None:
        """Return an overestimate after the provider reports actual total usage."""
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (reserved, used)
        ):
            raise ValueError("reserved and used token counts must be non-negative integers")
        refund = max(0, reserved - used)
        with self._lock:
            self._refill(self.clock())
            self._model_tokens = min(
                float(self.tokens_per_minute),
                self._model_tokens + refund,
            )
