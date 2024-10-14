"""Vector stores for the sandbox agent."""

from __future__ import annotations

from langchain_community.vectorstores import FAISS, Chroma, PGVector, Redis as RedisVectorStore


class VectorStoreFactory:
    @staticmethod
    def create(store_type: str):
        if store_type == "chroma":
            return Chroma
        elif store_type == "faiss":
            return FAISS
        elif store_type == "pgvector":
            return PGVector
        elif store_type == "redis":
            return RedisVectorStore
        # Add more vector stores as needed
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
