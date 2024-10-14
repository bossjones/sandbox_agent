# pylint: disable=no-member
# mypy: disable-error-code="return"
# mypy: disable-error-code="str-byte-safe"
# mypy: disable-error-code="misc"

"""Configuration for the sandbox agent."""

from __future__ import annotations

from sandbox_agent.aio_settings import aiosettings


# Use aiosettings here
# For example:
CHAT_MODEL = aiosettings.chat_model
EMBEDDING_MODEL = aiosettings.llm_embedding_model_name
DISCORD_TOKEN = aiosettings.discord_token.get_secret_value()
