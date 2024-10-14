"""Tests for the RetrieverFactory class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever

import pytest

from sandbox_agent.ai.retrievers import RetrieverFactory


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestRetrieverFactory:
    """Tests for the RetrieverFactory class."""

    def test_create_vector_store_retriever(self, mocker: MockerFixture) -> None:
        """Test that the factory creates a VectorStoreRetriever for the 'vector_store' retriever type."""
        vector_store_mock = mocker.Mock(spec=VectorStore)
        retriever = RetrieverFactory.create("vector_store", vector_store_mock)
        assert isinstance(retriever, VectorStoreRetriever)
        assert retriever.vectorstore == vector_store_mock

    def test_create_unsupported_retriever(self, mocker: MockerFixture) -> None:
        """Test that the factory raises an error for unsupported retriever types."""
        unsupported_retriever_type = "unsupported"
        with pytest.raises(ValueError) as excinfo:
            RetrieverFactory.create(unsupported_retriever_type, mocker.Mock())
        assert str(excinfo.value) == f"Unsupported retriever type: {unsupported_retriever_type}"

    def test_create_custom_retriever(self, mocker: MockerFixture) -> None:
        """Test that the factory can create a custom retriever."""
        custom_retriever_type = "custom"
        custom_retriever_class = mocker.Mock()
        vector_store_mock = mocker.Mock(spec=VectorStore)
        mocker.patch.object(
            RetrieverFactory,
            "create",
            side_effect=lambda x, y: custom_retriever_class if x == custom_retriever_type else None,
        )
        retriever = RetrieverFactory.create(custom_retriever_type, vector_store_mock)
        assert retriever == custom_retriever_class
        custom_retriever_class.assert_called_once_with(vectorstore=vector_store_mock)
