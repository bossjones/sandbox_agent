"""Chat models for the sandbox agent."""

from __future__ import annotations

from langchain_openai.chat_models import ChatOpenAI
from loguru import logger

from sandbox_agent.aio_settings import aiosettings


class ChatModelFactory:
    """Factory class for creating chat model instances."""

    @staticmethod
    def create(model_name: str = aiosettings.llm_model_name) -> ChatOpenAI:
        """
        Create a chat model instance based on the given model name.

        Args:
            model_name (str): The name of the model to create.
                Defaults to the value of `aiosettings.llm_model_name`.

        Returns:
            ChatOpenAI: The created chat model instance.

        Raises:
            ValueError: If an unsupported model name is provided.
        """
        if model_name == "gpt-4o":
            logger.info(
                f"Creating gpt-4o model with max_tokens: {aiosettings.max_tokens} and max_retries: {aiosettings.llm_max_retries}"
            )
            return ChatOpenAI(
                model_name="gpt-4o",
                temperature=aiosettings.llm_temperature,
                max_tokens=aiosettings.max_tokens,
                max_retries=aiosettings.llm_max_retries,
            )
        elif model_name == "gpt-4":
            logger.info(
                f"Creating gpt-4 model with max_tokens: {aiosettings.max_tokens} and max_retries: {aiosettings.llm_max_retries}"
            )
            return ChatOpenAI(
                model_name="gpt-4",
                temperature=aiosettings.llm_temperature,
                max_tokens=aiosettings.max_tokens,
                max_retries=aiosettings.llm_max_retries,
            )
        elif model_name == "gpt-4o-mini":
            logger.info(
                f"Creating gpt-4o-mini model with max_tokens: {aiosettings.max_tokens} and max_retries: {aiosettings.llm_max_retries}"
            )
            return ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=aiosettings.llm_temperature,
                max_tokens=aiosettings.max_tokens,
                max_retries=aiosettings.llm_max_retries,
            )
        elif model_name == "gpt-4-0314":
            logger.info(
                f"Creating gpt-4-0314 model with max_tokens: {aiosettings.max_tokens} and max_retries: {aiosettings.llm_max_retries}"
            )
            return ChatOpenAI(
                model_name="gpt-4-0314",
                temperature=aiosettings.llm_temperature,
                max_tokens=aiosettings.max_tokens,
                max_retries=aiosettings.llm_max_retries,
            )
        elif model_name == "gpt-4-32k":
            logger.info(
                f"Creating gpt-4-32k model with max_tokens: {aiosettings.max_tokens} and max_retries: {aiosettings.llm_max_retries}"
            )
            return ChatOpenAI(
                model_name="gpt-4-32k",
                temperature=aiosettings.llm_temperature,
                max_tokens=aiosettings.max_tokens,
                max_retries=aiosettings.llm_max_retries,
            )
        elif model_name == "gpt-4-32k-0314":
            logger.info(
                f"Creating gpt-4-32k-0314 model with max_tokens: {aiosettings.max_tokens} and max_retries: {aiosettings.llm_max_retries}"
            )
            return ChatOpenAI(
                model_name="gpt-4-32k-0314",
                temperature=aiosettings.llm_temperature,
                max_tokens=aiosettings.max_tokens,
                max_retries=aiosettings.llm_max_retries,
            )
        elif model_name == "gpt-3.5-turbo":
            logger.info(
                f"Creating gpt-3.5-turbo model with max_tokens: {aiosettings.max_tokens} and max_retries: {aiosettings.llm_max_retries}"
            )
            return ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=aiosettings.llm_temperature,
                max_tokens=aiosettings.max_tokens,
                max_retries=aiosettings.llm_max_retries,
            )
        elif model_name == "gpt-3.5-turbo-0301":
            logger.info(
                f"Creating gpt-3.5-turbo-0301 model with max_tokens: {aiosettings.max_tokens} and max_retries: {aiosettings.llm_max_retries}"
            )
            return ChatOpenAI(
                model_name="gpt-3.5-turbo-0301",
                temperature=aiosettings.llm_temperature,
                max_tokens=aiosettings.max_tokens,
                max_retries=aiosettings.llm_max_retries,
            )
        elif model_name == "o1-preview":
            logger.info(
                f"Creating o1-preview model with max_tokens: {aiosettings.max_tokens} and max_retries: {aiosettings.llm_max_retries}"
            )
            return ChatOpenAI(
                model_name="o1-preview",
                temperature=aiosettings.llm_temperature,
                max_tokens=aiosettings.max_tokens,
                max_retries=aiosettings.llm_max_retries,
            )
        elif model_name == "o1-mini":
            logger.info(
                f"Creating o1-mini model with max_tokens: {aiosettings.max_tokens} and max_retries: {aiosettings.llm_max_retries}"
            )
            return ChatOpenAI(
                model_name="o1-mini",
                temperature=aiosettings.llm_temperature,
                max_tokens=aiosettings.max_tokens,
                max_retries=aiosettings.llm_max_retries,
            )
        # Add more models as needed
        else:
            raise ValueError(f"Unsupported model: {model_name}")
