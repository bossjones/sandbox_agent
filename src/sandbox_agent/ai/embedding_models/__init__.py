"""Embedding models for the sandbox agent."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from sandbox_agent.aio_settings import aiosettings


class EmbeddingModelFactory:
    @staticmethod
    def create(model_name: str = aiosettings.llm_embedding_model_name):
        if model_name == "text-embedding-3-large":
            return OpenAIEmbeddings(model=model_name)
        # Add more models as needed
        else:
            raise ValueError(f"Unsupported model: {model_name}")
