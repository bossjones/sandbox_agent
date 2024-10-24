"""Retrievers for the sandbox agent."""

from __future__ import annotations

from langchain.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever

from sandbox_agent.aio_settings import aiosettings


class RetrieverFactory:
    """Factory class for creating retriever instances."""

    @staticmethod
    def create(
        retriever_type: str = aiosettings.llm_retriever_type, vector_store: VectorStore = None
    ) -> VectorStoreRetriever:
        """
        Create a retriever instance based on the given retriever type and vector store.

        Args:
            retriever_type (str): The type of retriever to create. Defaults to the value of `aiosettings.llm_retriever_type`.
            vector_store (VectorStore): The vector store to use for the retriever.

        Returns:
            VectorStoreRetriever: The created retriever instance.

        Raises:
            ValueError: If an unsupported retriever type is provided.
        """
        if retriever_type == "vector_store":
            return VectorStoreRetriever(vectorstore=vector_store)
        # Add more retrievers as needed
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")
