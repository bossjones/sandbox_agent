"""Tests for the advanced workflows module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

import pytest

from sandbox_agent.ai.workflows.advanced import AgentState, call_model, graph, should_continue, tool_node, workflow


if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.asyncio
async def test_call_model(monkeypatch: MonkeyPatch) -> None:
    """
    Test the call_model function.

    Args:
        monkeypatch (MonkeyPatch): The pytest monkeypatch fixture for patching.
    """

    # Mock the model.invoke method
    async def mock_invoke(messages: list[BaseMessage], config: RunnableConfig) -> BaseMessage:
        return BaseMessage(content="Mock response")

    monkeypatch.setattr("sandbox_agent.ai.workflows.advanced.model.invoke", mock_invoke)

    state: AgentState = {"messages": [SystemMessage(content="Test message")]}
    config = RunnableConfig()

    result = await call_model(state, config)

    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], BaseMessage)
    assert result["messages"][0].content == "Mock response"


def test_tool_node() -> None:
    """Test the tool_node function."""
    state: AgentState = {
        "messages": [
            ToolMessage(
                content='{"result": "Test tool result"}',
                name="test_tool",
                tool_call_id="test_id",
            )
        ]
    }

    result = tool_node(state)

    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert result["messages"][0].content == '{"result": "Test tool result"}'
    assert result["messages"][0].name == "test_tool"
    assert result["messages"][0].tool_call_id == "test_id"


def test_should_continue_with_tool_calls() -> None:
    """Test the should_continue function when there are tool calls."""
    state: AgentState = {"messages": [ToolMessage(content="Test", name="test", tool_call_id="id")]}

    result = should_continue(state)

    assert result == "continue"


def test_should_continue_without_tool_calls() -> None:
    """Test the should_continue function when there are no tool calls."""
    state: AgentState = {"messages": [BaseMessage(content="Test")]}

    result = should_continue(state)

    assert result == "end"


def test_workflow_instance() -> None:
    """Test the workflow instance."""
    assert isinstance(workflow, StateGraph)
    assert "agent" in workflow.nodes
    assert "tools" in workflow.nodes
    assert workflow.entry_point == "agent"


def test_graph_instance() -> None:
    """Test the graph instance."""
    assert callable(graph)
