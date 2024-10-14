"""Tests for the chat models module."""

from __future__ import annotations

from langchain_openai.chat_models import ChatOpenAI

import pytest

from sandbox_agent.ai.chat_models import ChatModelFactory
from sandbox_agent.aio_settings import aiosettings


# if __name__ == "__main__":
#     pytest.main([__file__])


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o",
        "gpt-4-0314",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "o1-preview",
        "o1-mini",
    ],
)
def test_chat_model_factory_create_valid_models(model_name: str) -> None:
    """
    Test ChatModelFactory.create method with valid model names.

    Args:
        model_name: The name of the model to test.

    This test covers:
    1. Creating chat models with all supported model names.
    2. Verifying that the created model is an instance of ChatOpenAI.
    3. Checking that the model name and temperature are set correctly.
    """
    chat_model = ChatModelFactory.create(model_name)

    assert isinstance(chat_model, ChatOpenAI)
    assert chat_model.model_name == model_name
    assert chat_model.temperature == aiosettings.llm_temperature


def test_chat_model_factory_create_default_model() -> None:
    """
    Test ChatModelFactory.create method with the default model.

    This test covers:
    1. Creating a chat model without specifying a model name.
    2. Verifying that the default model from aiosettings is used.
    """
    chat_model = ChatModelFactory.create()

    assert isinstance(chat_model, ChatOpenAI)
    assert chat_model.model_name == aiosettings.chat_model
    assert chat_model.temperature == aiosettings.llm_temperature


def test_chat_model_factory_create_invalid_model() -> None:
    """
    Test ChatModelFactory.create method with an invalid model name.

    This test covers:
    1. Attempting to create a chat model with an unsupported model name.
    2. Verifying that a ValueError is raised with an appropriate error message.
    """
    invalid_model_name = "invalid-model"

    with pytest.raises(ValueError) as excinfo:
        ChatModelFactory.create(invalid_model_name)

    assert str(excinfo.value) == f"Unsupported model: {invalid_model_name}"


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4",
        "gpt-3.5-turbo",
        "o1-preview",
    ],
)
def test_chat_model_factory_create_with_custom_temperature(model_name: str) -> None:
    """
    Test ChatModelFactory.create method with a custom temperature setting.

    Args:
        model_name: The name of the model to test.

    This test covers:
    1. Creating chat models with a custom temperature setting.
    2. Verifying that the temperature is set correctly on the created model.
    """
    custom_temperature = 0.8
    original_temperature = aiosettings.llm_temperature

    try:
        aiosettings.llm_temperature = custom_temperature
        chat_model = ChatModelFactory.create(model_name)

        assert isinstance(chat_model, ChatOpenAI)
        assert chat_model.model_name == model_name
        assert chat_model.temperature == custom_temperature
    finally:
        # Restore the original temperature setting
        aiosettings.llm_temperature = original_temperature


def test_chat_model_factory_create_model_consistency() -> None:
    """
    Test consistency of ChatModelFactory.create method across multiple calls.

    This test covers:
    1. Creating multiple instances of the same model.
    2. Verifying that all instances have consistent properties.
    """
    model_name = "gpt-3.5-turbo"
    num_instances = 5

    chat_models = [ChatModelFactory.create(model_name) for _ in range(num_instances)]

    for chat_model in chat_models:
        assert isinstance(chat_model, ChatOpenAI)
        assert chat_model.model_name == model_name
        assert chat_model.temperature == aiosettings.llm_temperature

    # Verify that all instances have the same properties
    assert all(chat_model.model_name == chat_models[0].model_name for chat_model in chat_models)
    assert all(chat_model.temperature == chat_models[0].temperature for chat_model in chat_models)
