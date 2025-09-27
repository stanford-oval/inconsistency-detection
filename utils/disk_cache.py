import asyncio
import functools
import pickle
from typing import cast

from diskcache import Cache


cache = Cache(directory="./.diskcache")


def diskcache_cache(func):
    """Decorator that caches the results of the decorated function."""
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = pickle.dumps((func.__module__, func.__qualname__, args, tuple(kwargs.items())))
            if key in cache:
                try:
                    cached_bytes = cast(bytes, cache[key])
                    return pickle.loads(cached_bytes)
                except ModuleNotFoundError:
                    # Cached artefacts may reference modules that no longer exist after refactors.
                    # If deserialisation fails, drop the stale cache entry and recompute.
                    del cache[key]
            result = await func(*args, **kwargs)
            cache[key] = pickle.dumps(result)
            return result

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = pickle.dumps((func.__module__, func.__qualname__, args, tuple(kwargs.items())))
            if key in cache:
                try:
                    cached_bytes = cast(bytes, cache[key])
                    return pickle.loads(cached_bytes)
                except ModuleNotFoundError:
                    del cache[key]
            result = func(*args, **kwargs)
            cache[key] = pickle.dumps(result)
            return result

        return sync_wrapper
