import time
from typing import Any, Dict, Optional
from functools import wraps

from app.utils.logging import get_logger
from app.config import settings

logger = get_logger(__name__)


class SimpleCache:
    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        if not settings.cache_enabled:
            self._misses += 1
            return None
        if key in self._cache:
            if time.time() - self._timestamps[key] < self._ttl:
                self._hits += 1
                return self._cache[key]
            del self._cache[key]
            del self._timestamps[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if not settings.cache_enabled:
            return
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def _evict_oldest(self) -> None:
        if not self._timestamps:
            return
        oldest_key = min(self._timestamps, key=self._timestamps.get)
        del self._cache[oldest_key]
        del self._timestamps[oldest_key]
        logger.debug(f"Cache evicted oldest entry: {oldest_key}")

    def clear(self) -> None:
        self._cache.clear()
        self._timestamps.clear()

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total * 100, 2) if total > 0 else 0,
            "entries": len(self._cache),
        }


_cache_instance: Optional[SimpleCache] = None


def get_cache() -> SimpleCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SimpleCache(
            ttl_seconds=settings.cache_ttl,
            max_size=100
        )
    return _cache_instance


def cached(ttl: int = 300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cached_val = get_cache().get(key)
            if cached_val is not None:
                logger.debug(f"Cache hit: {key}")
                return cached_val
            result = func(*args, **kwargs)
            get_cache().set(key, result)
            logger.debug(f"Cache miss, stored: {key}")
            return result
        return wrapper
    return decorator