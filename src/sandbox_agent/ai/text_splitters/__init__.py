"""Chat models for the sandbox agent."""

from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from loguru import logger

from sandbox_agent.aio_settings import aiosettings


class TextSplitterFactory:
    """Factory class for creating text splitter instances."""

    @staticmethod
    def create(splitter_type: str = aiosettings.llm_text_splitter_type) -> RecursiveCharacterTextSplitter:
        """
        Create a text splitter instance based on the given model name.

        Args:
            splitter_type (str): The type of text splitter to create.
                Defaults to the value of `aiosettings.llm_text_splitter_type`.

        Returns:
            RecursiveCharacterTextSplitter: The created text splitter instance.

        Raises:
            ValueError: If an unsupported text splitter type is provided.
        """
        if splitter_type == "recursive_text":
            logger.info(
                f"Creating recursive_text splitter with chunk_size: {aiosettings.text_chunk_size} and chunk_overlap: {aiosettings.text_chunk_overlap}"
            )
            return RecursiveCharacterTextSplitter(
                chunk_size=aiosettings.text_chunk_size,
                chunk_overlap=aiosettings.text_chunk_overlap,
            )

        # Add more models as needed
        else:
            raise ValueError(f"Unsupported text splitter type: {splitter_type}")
