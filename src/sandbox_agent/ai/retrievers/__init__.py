"""Retrievers for the sandbox agent."""

from __future__ import annotations

from langchain.retrievers import VectorStoreRetriever


class RetrieverFactory:
    @staticmethod
    def create(retriever_type: str, vector_store):
        if retriever_type == "vector_store":
            return VectorStoreRetriever(vectorstore=vector_store)
        # Add more retrievers as needed
        else:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")
