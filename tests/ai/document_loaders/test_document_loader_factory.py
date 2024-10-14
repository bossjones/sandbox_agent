"""Tests for the DocumentLoaderFactory class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader, PyPDFLoader

import pytest

from sandbox_agent.ai.document_loaders import DocumentLoaderFactory


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestDocumentLoaderFactory:
    """Tests for the DocumentLoaderFactory class."""

    @pytest.mark.parametrize(
        "loader_type, expected_loader",
        [
            ("pdf", PyPDFLoader),
            ("pymupdf", PyMuPDFLoader),
            ("directory", DirectoryLoader),
        ],
    )
    def test_create_supported_loaders(self, loader_type: str, expected_loader: type) -> None:
        """Test that the factory creates the correct loader for supported types."""
        loader = DocumentLoaderFactory.create(loader_type)
        assert loader == expected_loader

    def test_create_unsupported_loader(self, mocker: MockerFixture) -> None:
        """Test that the factory raises an error for unsupported loader types."""
        unsupported_loader_type = "unsupported"
        with pytest.raises(ValueError) as excinfo:
            DocumentLoaderFactory.create(unsupported_loader_type)
        assert str(excinfo.value) == f"Unsupported loader type: {unsupported_loader_type}"

    def test_create_custom_loader(self, mocker: MockerFixture) -> None:
        """Test that the factory can create a custom loader."""
        custom_loader_type = "custom"
        custom_loader_class = mocker.Mock()
        mocker.patch.object(
            DocumentLoaderFactory,
            "create",
            side_effect=lambda x: custom_loader_class if x == custom_loader_type else None,
        )
        loader = DocumentLoaderFactory.create(custom_loader_type)
        assert loader == custom_loader_class
