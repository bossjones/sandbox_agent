"""Redis client for the sandbox agent."""

from __future__ import annotations

import redis

from sandbox_agent.aio_settings import aiosettings


class RedisClient:
    def __init__(self):
        self.client = redis.Redis(
            host=aiosettings.redis_host,
            port=aiosettings.redis_port,
            db=aiosettings.redis_base,
            username=aiosettings.redis_user,
            password=aiosettings.redis_pass.get_secret_value() if aiosettings.redis_pass else None,
        )

    def set(self, key, value):
        return self.client.set(key, value)

    def get(self, key):
        return self.client.get(key)
