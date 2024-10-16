"""Document loaders for the sandbox agent."""

from __future__ import annotations

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, PyPDFLoader

from sandbox_agent.aio_settings import aiosettings


class DocumentLoaderFactory:
    @staticmethod
    def create(loader_type: str = aiosettings.llm_document_loader_type):
        """
        Create a document loader instance based on the given loader type.

        Args:
            loader_type (str): The type of document loader to create.
                Defaults to the value of `aiosettings.llm_document_loader_type`.

        Returns:
            The created document loader class.

        Raises:
            ValueError: If an unsupported loader type is provided.
        """
        if loader_type == "pdf":
            return PyPDFLoader
        elif loader_type == "pymupdf":
            return PyMuPDFLoader
        elif loader_type == "directory":
            return DirectoryLoader
        # Add more loaders as needed
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")
