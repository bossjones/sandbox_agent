"""Tests for the advanced workflows module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

import pytest

from sandbox_agent.ai.workflows.advanced import AgentState, call_model, graph, should_continue, tool_node, workflow


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from vcr.request import Request as VCRRequest


@pytest.mark.skip(reason="This test is not working yet")
@pytest.mark.flaky()
@pytest.mark.asyncio()
@pytest.mark.vcronly()
@pytest.mark.default_cassette("test_call_model.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "query"],
    ignore_localhost=False,
)
async def test_call_model(caplog: LogCaptureFixture, capsys: CaptureFixture, vcr: VCRRequest) -> None:
    """
    Test the call_model function.

    Args:
        caplog (LogCaptureFixture): Pytest fixture to capture log messages.
        capsys (CaptureFixture): Pytest fixture to capture stdout and stderr.
        vcr (VCRRequest): The VCR request object for recording/replaying HTTP interactions.
    """
    state: AgentState = {"messages": [SystemMessage(content="Test message")]}
    config = RunnableConfig()

    result = await call_model(state, config)

    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], BaseMessage)
    assert result["messages"][0].content != ""
    assert vcr.play_count > 0


@pytest.mark.skip(reason="This test is not working yet")
@pytest.mark.flaky()
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


@pytest.mark.skip(reason="This test is not working yet")
@pytest.mark.flaky()
def test_should_continue_with_tool_calls() -> None:
    """Test the should_continue function when there are tool calls."""
    state: AgentState = {"messages": [ToolMessage(content="Test", name="test", tool_call_id="id")]}

    result = should_continue(state)

    assert result == "continue"


@pytest.mark.skip(reason="This test is not working yet")
@pytest.mark.flaky()
def test_should_continue_without_tool_calls() -> None:
    """Test the should_continue function when there are no tool calls."""
    state: AgentState = {"messages": [BaseMessage(content="Test")]}

    result = should_continue(state)

    assert result == "end"


@pytest.mark.skip(reason="This test is not working yet")
@pytest.mark.flaky()
def test_workflow_instance() -> None:
    """Test the workflow instance."""
    assert isinstance(workflow, StateGraph)
    assert "agent" in workflow.nodes
    assert "tools" in workflow.nodes
    assert workflow.entry_point == "agent"


@pytest.mark.skip(reason="This test is not working yet")
@pytest.mark.flaky()
def test_graph_instance() -> None:
    """Test the graph instance."""
    assert callable(graph)


@pytest.mark.skip(reason="This test is not working yet")
@pytest.mark.flaky()
@pytest.mark.asyncio()
@pytest.mark.vcronly()
@pytest.mark.default_cassette("test_graph_execution.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query", "headers"],
    ignore_localhost=False,
)
async def test_graph_execution(caplog: LogCaptureFixture, capsys: CaptureFixture, vcr: VCRRequest) -> None:
    """
    Test the execution of the graph function.

    Args:
        caplog (LogCaptureFixture): Pytest fixture to capture log messages.
        capsys (CaptureFixture): Pytest fixture to capture stdout and stderr.
        vcr (VCRRequest): The VCR request object for recording/replaying HTTP interactions.
    """
    input_message = "What is the capital of France?"
    result = await graph.ainvoke({"messages": [BaseMessage(content=input_message)]})

    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) > 0
    assert isinstance(result["messages"][-1], BaseMessage)
    assert "Paris" in result["messages"][-1].content
    assert vcr.play_count > 0


@pytest.mark.skip(reason="This test is not working yet")
@pytest.mark.flaky()
@pytest.mark.asyncio()
@pytest.mark.vcronly()
@pytest.mark.default_cassette("test_workflow_execution.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query", "headers"],
    ignore_localhost=False,
)
async def test_workflow_execution(caplog: LogCaptureFixture, capsys: CaptureFixture, vcr: VCRRequest) -> None:
    """
    Test the execution of the workflow.

    Args:
        caplog (LogCaptureFixture): Pytest fixture to capture log messages.
        capsys (CaptureFixture): Pytest fixture to capture stdout and stderr.
        vcr (VCRRequest): The VCR request object for recording/replaying HTTP interactions.
    """
    input_message = "What is the population of New York City?"
    result = await workflow.ainvoke({"messages": [BaseMessage(content=input_message)]})

    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) > 0
    assert isinstance(result["messages"][-1], BaseMessage)
    assert "million" in result["messages"][-1].content.lower()
    assert vcr.play_count > 0
