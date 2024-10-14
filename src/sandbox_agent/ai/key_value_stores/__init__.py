"""Key-value stores for the sandbox agent."""

from __future__ import annotations

from langchain.storage import InMemoryStore, LocalFileStore, RedisStore


class KeyValueStoreFactory:
    @staticmethod
    def create(store_type: str):
        if store_type == "redis":
            return RedisStore()
        elif store_type == "in_memory":
            return InMemoryStore()
        elif store_type == "local_file":
            return LocalFileStore()
        # Add more key-value stores as needed
        else:
            raise ValueError(f"Unsupported key-value store type: {store_type}")
