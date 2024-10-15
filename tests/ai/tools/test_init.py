"""Tests for the __init__ module of the tools package."""

from __future__ import annotations

import pytest

from sandbox_agent.ai.tools import ToolFactory


@pytest.mark.aidergenerated
def test_tool_factory_import() -> None:
    """
    Test the import of the ToolFactory class from the __init__ module.

    This test verifies that the ToolFactory class can be imported correctly
    from the __init__ module of the tools package.

    """
    assert ToolFactory is not None
    assert hasattr(ToolFactory, "create_tools")
