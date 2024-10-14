"""Document loaders for the sandbox agent."""

from __future__ import annotations

from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader, PyPDFLoader


class DocumentLoaderFactory:
    @staticmethod
    def create(loader_type: str):
        if loader_type == "pdf":
            return PyPDFLoader
        elif loader_type == "pymupdf":
            return PyMuPDFLoader
        elif loader_type == "directory":
            return DirectoryLoader
        # Add more loaders as needed
        else:
            raise ValueError(f"Unsupported loader type: {loader_type}")
