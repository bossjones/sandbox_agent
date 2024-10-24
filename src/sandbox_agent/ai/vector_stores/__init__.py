# SOURCE: https://github.com/langchain-ai/langchain/issues/15944
# Making a generalized method to ingest documents in any vector database.
"""Vector stores for the sandbox agent."""

from __future__ import annotations

import os
import pathlib

from abc import ABC, abstractmethod
from typing import List, Optional, Type

import chromadb

from chromadb.api import ClientAPI, ServerAPI
from chromadb.config import Settings as ChromaSettings
from langchain.docstore.document import Document
from langchain.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS, Chroma, PGVector, SKLearnVectorStore
from langchain_community.vectorstores import Redis as RedisVectorStore
from loguru import logger

from sandbox_agent.ai.embedding_models import EmbeddingModelFactory
from sandbox_agent.aio_settings import aiosettings


HERE = os.path.dirname(__file__)

CHROMA_DATA_PATH = os.path.join(HERE, "..", "data", "chroma", "documents")
CHROMA_PATH = os.path.join(HERE, "..", "data", "chroma", "vectorstorage")
CHROMA_PATH_API = pathlib.Path(CHROMA_PATH)


class VectorStoreAdapter(ABC):
    """
    An abstract base class defining the common interface for all vector store adapters.
    This allows the client code to work with different vector stores interchangeably,
    without needing to know the specific details of each store.
    """

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """
        Add a list of documents to the vector store.

        Args:
        ----
            documents: A list of Document objects to be added to the vector store.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 4) -> list[Document]:
        """
        Search for the top k most similar documents to the query.

        Args:
        ----
            query: The search query string.
            k: The number of documents to return. Defaults to 4.

        Returns:
        -------
            List[Document]: A list of the most similar documents.
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """
        Persist the vector store to disk if supported, otherwise do nothing.
        """
        pass

    @abstractmethod
    def retrieve(self, ids: list[str]) -> list[Document]:
        """
        Retrieve documents by their IDs.

        Args:
        ----
            ids: A list of document IDs to retrieve.

        Returns:
        -------
            List[Document]: A list of retrieved documents.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load the vector store from disk if supported, otherwise do nothing.
        """
        pass

    @abstractmethod
    def delete(self) -> None:
        """
        Delete the vector store from disk if supported, otherwise do nothing.
        """
        pass


class SKLearnVectorStoreAdapter(VectorStoreAdapter):
    """
    Adapter for SKLearnVectorStore.

    This adapter implements the VectorStoreAdapter interface for the SKLearnVectorStore.
    It handles initialization, document addition, searching, persistence, loading, and deletion.
    """

    def __init__(self, vectorstore: type[SKLearnVectorStore], documents: Optional[list[Document]] = None) -> None:
        """
        Initialize the SKLearnVectorStoreAdapter.

        Args:
        ----
            vectorstore: The SKLearnVectorStore class.
            documents: Optional list of documents to initialize the vector store with.

        """
        self.embeddings = EmbeddingModelFactory.create()
        if pathlib.Path(aiosettings.sklearn_persist_path).exists() and not documents:
            logger.debug(
                f"No documents provided, loading SKLearn vector store from {aiosettings.sklearn_persist_path} ...."
            )
            logger.info(
                f"SKLearnVectorStore(persist_path={aiosettings.sklearn_persist_path}, embedding={repr(self.embeddings)}, serializer={aiosettings.sklearn_serializer}, metric={aiosettings.sklearn_metric})"
            )
            self.vectorstore: SKLearnVectorStore = vectorstore(
                persist_path=aiosettings.sklearn_persist_path,
                embedding=self.embeddings,
                serializer=aiosettings.sklearn_serializer,
                metric=aiosettings.sklearn_metric,
            )
        else:
            logger.info("Documents argument provided, creating SKLearnVectorStore from documents ...")
            logger.info(
                f"SKLearnVectorStore.from_documents(documents={repr(documents)}, persist_path={aiosettings.sklearn_persist_path}, embedding={repr(self.embeddings)}, serializer={aiosettings.sklearn_serializer}, metric={aiosettings.sklearn_metric})"
            )
            self.vectorstore: SKLearnVectorStore = vectorstore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                serializer=aiosettings.sklearn_serializer,
                metric=aiosettings.sklearn_metric,
            )
            logger.info(f"Persisting SKLearn vector store to {aiosettings.sklearn_persist_path} ....")
            self.vectorstore.persist()

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add a list of documents to the vector store.

        Args:
        ----
            documents: List of documents to add.

        """
        self.vectorstore.add_documents(documents)

    def search(self, query: str, k: int = 4) -> list[Document]:
        """
        Search for the top k most similar documents to the query.

        Args:
        ----
            query: The search query string.
            k: The number of documents to return. Defaults to 4.

        Returns:
        -------
            List of the most similar documents.

        """
        return self.vectorstore.similarity_search(query, k=k)

    def persist(self) -> None:
        """
        Persist the vector store to disk.

        """
        logger.info(f"Persisting SKLearn vector store to {aiosettings.sklearn_persist_path} ....")
        self.vectorstore.persist()

    def retrieve(self, ids: list[str]) -> list[Document]:
        """
        Retrieve documents by their IDs.

        Args:
        ----
            ids: List of document IDs to retrieve.

        Returns:
        -------
            List of retrieved documents.

        """
        # Note: This method needs to be implemented. SKLearnVectorStore doesn't have a built-in retrieve method.
        raise NotImplementedError("Retrieve method not implemented for SKLearnVectorStore")

    def load(self) -> None:
        """
        Load the vector store from disk.

        """
        logger.info(f"Loading SKLearn vector store from {aiosettings.sklearn_persist_path} ....")
        logger.info(
            f"SKLearnVectorStore(persist_path={aiosettings.sklearn_persist_path}, embedding={repr(self.embeddings)}, serializer={aiosettings.sklearn_serializer}, metric={aiosettings.sklearn_metric})"
        )
        self.vectorstore = SKLearnVectorStore(
            persist_path=aiosettings.sklearn_persist_path,
            embedding=self.embeddings,
            serializer=aiosettings.sklearn_serializer,
            metric=aiosettings.sklearn_metric,
        )

    def delete(self) -> None:
        """
        Delete the vector store from disk.

        Note: This method is currently not implemented.

        """
        logger.info(f"Deleting SKLearn vector store from {aiosettings.sklearn_persist_path} ....")
        # NOTE: would run this but need to check if it works
        # os.remove(aiosettings.sklearn_persist_path)
        raise NotImplementedError("Delete method not implemented for SKLearnVectorStore")


def get_vector_store_adapter(store_type: str) -> VectorStoreAdapter:
    """
    A factory function that creates the appropriate vector store adapter based on the store_type.
    """
    logger.info(f"Getting vector store adapter for {store_type} ....")
    if store_type == "sklearn":
        return SKLearnVectorStoreAdapter()
    else:
        raise ValueError(f"Unsupported vector store adaptor type: {store_type}")


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
