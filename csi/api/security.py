"""API protection: optional API-key check and a per-client rate limit.

Both are FastAPI dependencies attached to every /v1 route.
  * CSI_API_KEY      if set, requests must send the same value in the `X-API-Key` header.
  * CSI_RATE_LIMIT   requests per minute per client (API key, or IP when no key is used). 0 disables it.
"""
from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from csi.config import settings


class RateLimiter:
    """Sliding-window limiter kept in memory (enough for a single API process)."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window_s = limit, window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def retry_after(self, client: str, now: float | None = None) -> float:
        """Record a request. Returns 0 if it is allowed, otherwise the seconds to wait."""
        if self.limit <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[client]
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return round(hits[0] + self.window_s - now, 1)
            hits.append(now)
            return 0.0


limiter = RateLimiter(settings.rate_limit_per_minute)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and not hmac.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(401, "Missing or invalid X-API-Key header")


def rate_limit(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    client = x_api_key or (request.client.host if request.client else "unknown")
    wait = limiter.retry_after(client)
    if wait:
        raise HTTPException(429, f"Rate limit of {limiter.limit} requests/minute exceeded",
                            headers={"Retry-After": str(max(1, int(wait + 0.5)))})
