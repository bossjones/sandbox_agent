"""Tests for the KeyValueStoreFactory class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.storage.file_system import LocalFileStore
from langchain.storage.redis import RedisStore
from langchain_core.stores import InMemoryStore

import pytest

from sandbox_agent.ai.key_value_stores import KeyValueStoreFactory
from sandbox_agent.aio_settings import aiosettings


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestKeyValueStoreFactory:
    """Tests for the KeyValueStoreFactory class."""

    @pytest.mark.parametrize(
        "store_type, expected_store",
        [
            ("redis", RedisStore),
            ("in_memory", InMemoryStore),
            ("local_file", LocalFileStore),
        ],
    )
    def test_create_supported_stores(self, store_type: str, expected_store: type, mocker: MockerFixture) -> None:
        """Test that the factory creates the correct store for supported types."""
        mocker.patch.object(aiosettings, "redis_url", "redis://localhost:6379")
        mocker.patch.object(aiosettings, "localfilestore_root_path", "/test/root/path")
        store = KeyValueStoreFactory.create(store_type)
        assert isinstance(store, expected_store)

    def test_create_unsupported_store(self) -> None:
        """Test that the factory raises an error for unsupported store types."""
        unsupported_store_type = "unsupported"
        with pytest.raises(ValueError) as excinfo:
            KeyValueStoreFactory.create(unsupported_store_type)
        assert str(excinfo.value) == f"Unsupported key-value store type: {unsupported_store_type}"

    def test_create_local_file_store(self, mocker: MockerFixture) -> None:
        """Test that the factory creates a LocalFileStore with the correct root path."""
        root_path = "/test/root/path"
        mocker.patch.object(aiosettings, "localfilestore_root_path", root_path)
        store = KeyValueStoreFactory.create("local_file")
        assert isinstance(store, LocalFileStore)
        assert str(store.root_path) == root_path
