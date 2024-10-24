"""Tool factory for the sandbox agent."""

from __future__ import annotations

from typing import List

from langchain.tools import Tool
from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun, TavilySearchResults
from loguru import logger as LOGGER

from sandbox_agent.ai.tools.tool_utils import magic_function
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
            LOGGER.info(f"Added TavilySearchResults tool with max_results={aiosettings.tavily_search_max_results}")

        if "magic_function" in aiosettings.tool_allowlist:
            tools.append(magic_function)
            LOGGER.info("Added magic_function tool")

        if "duckduckgo_search" in aiosettings.tool_allowlist:
            tools.append(DuckDuckGoSearchRun())
            LOGGER.info("Added DuckDuckGoSearchRun tool")

        if "brave_search" in aiosettings.tool_allowlist:
            tools.append(
                BraveSearch.from_api_key(api_key=aiosettings.brave_search_api_key, search_kwargs={"num_results": 3})
            )
            LOGGER.info("Added BraveSearch tool to agent executor with search_kwargs='num_results': 3")
        # Add more tools based on the allowlist and their respective configurations

        return tools
