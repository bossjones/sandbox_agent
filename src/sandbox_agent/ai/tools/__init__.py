"""Tools for the sandbox agent."""

from __future__ import annotations

from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun


class ToolFactory:
    @staticmethod
    def create(tool_name: str):
        if tool_name == "duckduckgo":
            return DuckDuckGoSearchRun()
        elif tool_name == "brave":
            return BraveSearch()
        # Add more tools as needed
        else:
            raise ValueError(f"Unsupported tool: {tool_name}")
