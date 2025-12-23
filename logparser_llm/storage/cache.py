"""
Caching system for template lookups.
Supports memory, file, and Redis backends.
"""
import json
import hashlib
from typing import Optional, Dict, Any
from pathlib import Path
from abc import ABC, abstractmethod
import time


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str):
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class MemoryCache(CacheBackend):
    """In-memory LRU cache."""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of items
            ttl: Default time-to-live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value with TTL check."""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if expired
            if entry['expires_at'] and time.time() > entry['expires_at']:
                del self.cache[key]
                return None
            
            # Update access time for LRU
            entry['accessed_at'] = time.time()
            return entry['value']
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value with TTL."""
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        expires_at = None
        if ttl or self.default_ttl:
            expires_at = time.time() + (ttl or self.default_ttl)
        
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'accessed_at': time.time()
        }
    
    def delete(self, key: str):
        """Delete key."""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
    
    def exists(self, key: str) -> bool:
        """Check existence."""
        return self.get(key) is not None
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]['accessed_at']
        )
        del self.cache[lru_key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0
        }


class FileCache(CacheBackend):
    """File-based persistent cache."""
    
    def __init__(self, cache_dir: str = "cache/", ttl: int = 86400):
        """
        Initialize file cache.
        
        Args:
            cache_dir: Directory for cache files
            ttl: Default time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = ttl
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file."""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check TTL
            if data['expires_at'] and time.time() > data['expires_at']:
                file_path.unlink()
                return None
            
            return data['value']
        except Exception:
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Save value to file."""
        file_path = self._get_file_path(key)
        
        expires_at = None
        if ttl or self.default_ttl:
            expires_at = time.time() + (ttl or self.default_ttl)
        
        data = {
            'key': key,
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Failed to write cache: {e}")
    
    def delete(self, key: str):
        """Delete cache file."""
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
    
    def clear(self):
        """Clear all cache files."""
        for file_path in self.cache_dir.glob("*.json"):
            file_path.unlink()
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None


class RedisCache(CacheBackend):
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = 3600):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            ttl: Default time-to-live in seconds
        """
        try:
            import redis
            self.redis = redis.from_url(redis_url, decode_responses=True)
            self.default_ttl = ttl
            # Test connection
            self.redis.ping()
        except ImportError:
            raise ImportError("redis package required. Install with: pip install redis")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            value = self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception:
            pass
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis."""
        try:
            serialized = json.dumps(value)
            if ttl or self.default_ttl:
                self.redis.setex(key, ttl or self.default_ttl, serialized)
            else:
                self.redis.set(key, serialized)
        except Exception as e:
            print(f"Failed to set Redis key: {e}")
    
    def delete(self, key: str):
        """Delete key from Redis."""
        self.redis.delete(key)
    
    def clear(self):
        """Clear all keys (use with caution!)."""
        self.redis.flushdb()
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.redis.exists(key) > 0


class CacheManager:
    """
    High-level cache manager with multiple backend support.
    """
    
    def __init__(
        self,
        backend: str = "memory",
        ttl: int = 3600,
        **kwargs
    ):
        """
        Initialize cache manager.
        
        Args:
            backend: Cache backend type (memory, file, redis)
            ttl: Default time-to-live
            **kwargs: Backend-specific arguments
        """
        self.backend_type = backend
        self.ttl = ttl
        
        if backend == "memory":
            self.backend = MemoryCache(
                max_size=kwargs.get('max_size', 10000),
                ttl=ttl
            )
        elif backend == "file":
            self.backend = FileCache(
                cache_dir=kwargs.get('cache_dir', 'cache/'),
                ttl=ttl
            )
        elif backend == "redis":
            self.backend = RedisCache(
                redis_url=kwargs.get('redis_url', 'redis://localhost:6379/0'),
                ttl=ttl
            )
        else:
            raise ValueError(f"Unknown cache backend: {backend}")
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value with statistics tracking."""
        value = self.backend.get(key)
        if value is not None:
            self.hits += 1
        else:
            self.misses += 1
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value."""
        self.backend.set(key, value, ttl or self.ttl)
    
    def delete(self, key: str):
        """Delete key."""
        self.backend.delete(key)
    
    def clear(self):
        """Clear all cache."""
        self.backend.clear()
        self.hits = 0
        self.misses = 0
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.backend.exists(key)
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'backend': self.backend_type,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'ttl': self.ttl
        }
        
        # Add backend-specific stats
        if hasattr(self.backend, 'get_statistics'):
            stats.update(self.backend.get_statistics())
        
        return stats


# Example usage
if __name__ == "__main__":
    # Memory cache example
    cache = CacheManager(backend="memory", max_size=100)
    
    # Set values
    cache.set("key1", "value1")
    cache.set("key2", {"data": "value2"})
    
    # Get values
    print(cache.get("key1"))  # "value1"
    print(cache.get("key2"))  # {"data": "value2"}
    print(cache.get("key3"))  # None
    
    # Statistics
    print(cache.get_statistics())
    
    # File cache example
    file_cache = CacheManager(backend="file", cache_dir="test_cache/")
    file_cache.set("persistent", "data")
    print(file_cache.get("persistent"))