"""Tests for the ToolFactory class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.tools import Tool
from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun, TavilySearchResults

import pytest

from sandbox_agent.ai.tools.tool_factory import ToolFactory
from sandbox_agent.aio_settings import aiosettings


if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def all_tools_allowed(monkeypatch: MonkeyPatch) -> None:
    """
    Fixture to set up the environment with all tools allowed.

    Args:
        monkeypatch (MonkeyPatch): The monkeypatch fixture for modifying aio_settings.

    """
    monkeypatch.setattr(aiosettings, "tool_allowlist", ["tavily_search", "duckduckgo_search", "brave_search"])
    monkeypatch.setattr(aiosettings, "tavily_search_max_results", 5)
    monkeypatch.setattr(aiosettings, "brave_search_api_key", "test_api_key")


@pytest.fixture
def no_tools_allowed(monkeypatch: MonkeyPatch) -> None:
    """
    Fixture to set up the environment with no tools allowed.

    Args:
        monkeypatch (MonkeyPatch): The monkeypatch fixture for modifying aio_settings.

    """
    monkeypatch.setattr(aiosettings, "tool_allowlist", [])


@pytest.fixture
def partial_tools_allowed(monkeypatch: MonkeyPatch) -> None:
    """
    Fixture to set up the environment with partial tools allowed.

    Args:
        monkeypatch (MonkeyPatch): The monkeypatch fixture for modifying aio_settings.

    """
    monkeypatch.setattr(aiosettings, "tool_allowlist", ["tavily_search"])
    monkeypatch.setattr(aiosettings, "tavily_search_max_results", 3)


@pytest.mark.aidergenerated
def test_create_tools_all_allowed(all_tools_allowed: None) -> None:
    """
    Test the create_tools method with all tools allowed.

    This test verifies that the create_tools method correctly creates and returns
    a list of Tool instances when all tools are allowed in the aio_settings.

    Args:
        all_tools_allowed (None): The fixture to set up the environment with all tools allowed.

    """
    tools = ToolFactory.create_tools()

    assert len(tools) == 3
    assert isinstance(tools[0], TavilySearchResults)
    assert tools[0].max_results == 5
    assert isinstance(tools[1], DuckDuckGoSearchRun)
    assert isinstance(tools[2], BraveSearch)
    assert tools[2].search_wrapper.api_key == "test_api_key"


@pytest.mark.aidergenerated
def test_create_tools_no_allowed(no_tools_allowed: None) -> None:
    """
    Test the create_tools method with no tools allowed.

    This test verifies that the create_tools method returns an empty list
    when no tools are allowed in the aio_settings.

    Args:
        no_tools_allowed (None): The fixture to set up the environment with no tools allowed.

    """
    tools = ToolFactory.create_tools()

    assert len(tools) == 0


@pytest.mark.aidergenerated
def test_create_tools_partial_allowed(partial_tools_allowed: None) -> None:
    """
    Test the create_tools method with partial tools allowed.

    This test verifies that the create_tools method correctly creates and returns
    a list of Tool instances when only a subset of tools is allowed in the aio_settings.

    Args:
        partial_tools_allowed (None): The fixture to set up the environment with partial tools allowed.

    """
    tools = ToolFactory.create_tools()

    assert len(tools) == 1
    assert isinstance(tools[0], TavilySearchResults)
    assert tools[0].max_results == 3
