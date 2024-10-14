"""Chat models for the sandbox agent."""

from __future__ import annotations

from langchain.chat_models import ChatOpenAI

from sandbox_agent.aio_settings import aiosettings


class ChatModelFactory:
    @staticmethod
    def create(model_name: str = aiosettings.chat_model):
        if model_name == "gpt-4o":
            return ChatOpenAI(model_name="gpt-4", temperature=aiosettings.llm_temperature)
        elif model_name == "gpt-3.5-turbo":
            return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=aiosettings.llm_temperature)
        # Add more models as needed
        else:
            raise ValueError(f"Unsupported model: {model_name}")
