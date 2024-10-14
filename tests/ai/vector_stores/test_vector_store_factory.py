"""Tests for the VectorStoreFactory class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.vectorstores import FAISS, Chroma, PGVector
from langchain.vectorstores.redis import Redis as RedisVectorStore

import pytest

from sandbox_agent.ai.vector_stores import VectorStoreFactory


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestVectorStoreFactory:
    """Tests for the VectorStoreFactory class."""

    @pytest.mark.parametrize(
        "store_type, expected_store",
        [
            ("chroma", Chroma),
            ("faiss", FAISS),
            ("pgvector", PGVector),
            ("redis", RedisVectorStore),
        ],
    )
    def test_create_supported_stores(self, store_type: str, expected_store: type) -> None:
        """Test that the factory creates the correct vector store for supported types."""
        store = VectorStoreFactory.create(store_type)
        assert store == expected_store

    def test_create_unsupported_store(self) -> None:
        """Test that the factory raises an error for unsupported vector store types."""
        unsupported_store_type = "unsupported"
        with pytest.raises(ValueError) as excinfo:
            VectorStoreFactory.create(unsupported_store_type)
        assert str(excinfo.value) == f"Unsupported vector store type: {unsupported_store_type}"

    def test_create_custom_store(self, mocker: MockerFixture) -> None:
        """Test that the factory can create a custom vector store."""
        custom_store_type = "custom"
        custom_store_class = mocker.Mock()
        mocker.patch.object(
            VectorStoreFactory,
            "create",
            side_effect=lambda x: custom_store_class if x == custom_store_type else None,
        )
        store = VectorStoreFactory.create(custom_store_type)
        assert store == custom_store_class
