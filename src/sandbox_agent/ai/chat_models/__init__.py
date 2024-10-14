"""Chat models for the sandbox agent."""

from __future__ import annotations

from langchain_openai.chat_models import ChatOpenAI

from sandbox_agent.aio_settings import aiosettings


class ChatModelFactory:
    """Factory class for creating chat model instances."""

    @staticmethod
    def create(model_name: str = aiosettings.chat_model) -> ChatOpenAI:
        """Create a chat model instance based on the given model name.

        Args:
            model_name (str): The name of the model to create.
                Defaults to the value of `aiosettings.chat_model`.

        Returns:
            ChatOpenAI: The created chat model instance.

        Raises:
            ValueError: If an unsupported model name is provided.
        """
        if model_name == "gpt-4":
            return ChatOpenAI(model_name="gpt-4", temperature=aiosettings.llm_temperature)
        elif model_name == "gpt-4-0314":
            return ChatOpenAI(model_name="gpt-4-0314", temperature=aiosettings.llm_temperature)
        elif model_name == "gpt-4-32k":
            return ChatOpenAI(model_name="gpt-4-32k", temperature=aiosettings.llm_temperature)
        elif model_name == "gpt-4-32k-0314":
            return ChatOpenAI(model_name="gpt-4-32k-0314", temperature=aiosettings.llm_temperature)
        elif model_name == "gpt-3.5-turbo":
            return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=aiosettings.llm_temperature)
        elif model_name == "gpt-3.5-turbo-0301":
            return ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=aiosettings.llm_temperature)
        elif model_name == "o1-preview":
            return ChatOpenAI(model_name="o1-preview", temperature=aiosettings.llm_temperature)
        elif model_name == "o1-mini":
            return ChatOpenAI(model_name="o1-mini", temperature=aiosettings.llm_temperature)
        # Add more models as needed
        else:
            raise ValueError(f"Unsupported model: {model_name}")
