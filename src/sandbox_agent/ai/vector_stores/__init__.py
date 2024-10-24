"""Vector stores for the sandbox agent."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS, Chroma, PGVector, SKLearnVectorStore
from langchain_community.vectorstores import Redis as RedisVectorStore

from sandbox_agent.aio_settings import aiosettings


class VectorStoreFactory:
    @staticmethod
    def create(
        store_type: str = aiosettings.llm_vectorstore_type,
    ) -> type[FAISS | Chroma | PGVector | RedisVectorStore | SKLearnVectorStore]:
        """
        Create a vector store instance based on the given store type.

        Args:
            store_type (str): The type of vector store to create.
                Defaults to the value of `aiosettings.llm_vectorstore_type`.

        Returns:
            type[FAISS | Chroma | PGVector | RedisVectorStore | SKLearnVectorStore]: The created vector store class.

        Raises:
            ValueError: If an unsupported vector store type is provided.
        """
        if store_type == "chroma":
            return Chroma
        elif store_type == "faiss":
            return FAISS
        elif store_type == "pgvector":
            return PGVector
        elif store_type == "redis":
            return RedisVectorStore
        elif store_type == "sklearn":
            return SKLearnVectorStore
        # Add more vector stores as needed
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
