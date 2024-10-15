"""Tool factory for the sandbox agent."""

from __future__ import annotations

from typing import List

from langchain.tools import Tool
from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun, TavilySearchResults

from sandbox_agent.aio_settings import aiosettings


class ToolFactory:
    @staticmethod
    def create_tools() -> list[Tool]:
        """
        Create a list of tool instances based on the allowlist defined in aio_settings.

        Returns:
            List[Tool]: A list of created tool instances.
        """
        tools = []

        if "tavily_search" in aiosettings.tool_allowlist:
            tools.append(TavilySearchResults(max_results=aiosettings.tavily_search_max_results))

        if "duckduckgo_search" in aiosettings.tool_allowlist:
            tools.append(DuckDuckGoSearchRun())

        if "brave_search" in aiosettings.tool_allowlist:
            tools.append(BraveSearch(api_key=aiosettings.brave_search_api_key))

        # Add more tools based on the allowlist and their respective configurations

        return tools
