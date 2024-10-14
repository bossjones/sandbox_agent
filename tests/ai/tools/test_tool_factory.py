"""Tests for the ToolFactory class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun

import pytest

from sandbox_agent.ai.tools import ToolFactory
from sandbox_agent.aio_settings import aiosettings


if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestToolFactory:
    """Tests for the ToolFactory class."""

    @pytest.mark.parametrize(
        "tool_name, expected_tool",
        [
            ("duckduckgo", DuckDuckGoSearchRun),
            ("brave", BraveSearch),
        ],
    )
    def test_create_supported_tools(self, tool_name: str, expected_tool: type, mocker: MockerFixture) -> None:
        """Test that the factory creates the correct tool for supported tool names."""
        mocker.patch.object(aiosettings, "brave_search_api_key", SecretStr("dummy_api_key"))
        tool = ToolFactory.create(tool_name)
        assert isinstance(tool, expected_tool)

    def test_create_unsupported_tool(self) -> None:
        """Test that the factory raises an error for unsupported tool names."""
        unsupported_tool_name = "unsupported"
        with pytest.raises(ValueError) as excinfo:
            ToolFactory.create(unsupported_tool_name)
        assert str(excinfo.value) == f"Unsupported tool: {unsupported_tool_name}"

    def test_create_custom_tool(self, mocker: MockerFixture) -> None:
        """Test that the factory can create a custom tool."""
        custom_tool_name = "custom"
        custom_tool_class = mocker.Mock()
        mocker.patch.object(
            ToolFactory,
            "create",
            side_effect=lambda x: custom_tool_class() if x == custom_tool_name else None,
        )
        tool = ToolFactory.create(custom_tool_name)
        assert isinstance(tool, custom_tool_class)
        custom_tool_class.assert_called_once()
