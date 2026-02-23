from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict

@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    burst: int = 30

class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.last = time.time()

    def allow(self, cost: float = 1.0) -> bool:
        now = time.time()
        elapsed = max(0.0, now - self.last)
        self.last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

class RateLimiter:
    def __init__(self, cfg: RateLimitConfig) -> None:
        self.cfg = cfg
        self.buckets: Dict[str, TokenBucket] = {}
        self.rate = cfg.requests_per_minute / 60.0

    def allow(self, client_key: str) -> bool:
        if client_key not in self.buckets:
            self.buckets[client_key] = TokenBucket(rate_per_sec=self.rate, capacity=float(self.cfg.burst))
        return self.buckets[client_key].allow(1.0)
