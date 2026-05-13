import threading
import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, Dict


class MetricsStore:
    MAX_LATENCY_SAMPLES = 2000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._latency: Dict[str, list] = defaultdict(list)
        self._start_time = time.time()

    def inc(self, key: str, value: int = 1) -> None:
        with self._lock:
            self._counters[key] += value

    def observe_latency(self, route: str, latency_ms: float) -> None:
        with self._lock:
            bucket = self._latency[route]
            bucket.append(latency_ms)
            if len(bucket) > self.MAX_LATENCY_SAMPLES:
                del bucket[: len(bucket) - self.MAX_LATENCY_SAMPLES]

    @contextmanager
    def track_latency(self, route: str, subkey: str = ""):
        full_key = f"{route}:{subkey}" if subkey else route
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.observe_latency(full_key, elapsed_ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            latency_stats = {}
            for route, values in self._latency.items():
                latency_stats[route] = {
                    "avg_ms": round(sum(values) / len(values), 2) if values else 0,
                    "max_ms": round(max(values), 2) if values else 0,
                    "p95_ms": round(
                        sorted(values)[int(len(values) * 0.95)] if values else 0, 2
                    ),
                }

            return {
                "counters": dict(self._counters),
                "latency_stats": latency_stats,
                "uptime_seconds": int(time.time() - self._start_time),
                "route_count": len(self._latency),
            }


metrics_store = MetricsStore()