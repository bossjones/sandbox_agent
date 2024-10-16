"""Key-value stores for the sandbox agent."""

from __future__ import annotations

from langchain.storage import LocalFileStore
from langchain_community.storage import RedisStore
from langchain_core.stores import InMemoryStore

from sandbox_agent.aio_settings import aiosettings


class KeyValueStoreFactory:
    @staticmethod
    def create(store_type: str = aiosettings.llm_key_value_stores_type):
        """
        Create a key-value store instance based on the given store type.

        Args:
            store_type (str): The type of key-value store to create.
                Defaults to the value of `aiosettings.llm_key_value_stores_type`.

        Returns:
            The created key-value store instance.

        Raises:
            ValueError: If an unsupported key-value store type is provided.
        """
        if store_type == "redis":
            return RedisStore(redis_url=aiosettings.redis_url)
        elif store_type == "in_memory":
            return InMemoryStore()
        elif store_type == "local_file":
            return LocalFileStore(root_path=str(aiosettings.localfilestore_root_path))
        # Add more key-value stores as needed
        else:
            raise ValueError(f"Unsupported key-value store type: {store_type}")
