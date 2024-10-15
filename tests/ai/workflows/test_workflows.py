# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.schema import HumanMessage
from langgraph.graph import END, StateGraph

import pytest

from sandbox_agent.ai.workflows import (
    Act,
    Plan,
    PlanExecute,
    Response,
    WorkflowFactory,
    build_workflow,
    execute_step,
    get_graph,
    plan_step,
    replan_step,
    should_end,
)


if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock import MockerFixture


@pytest.mark.cursorgenerated
@pytest.mark.asyncio
async def test_execute_step(mocker: MockerFixture) -> None:
    """
    Test the execute_step function.

    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_agent_executor = mocker.patch("sandbox_agent.ai.workflows.agent_executor")
    mock_agent_executor.ainvoke.return_value = {"messages": [HumanMessage(content="Mocked response")]}

    state = PlanExecute(input="Test input", plan=["Step 1", "Step 2"])
    result = await execute_step(state)

    assert result == {"past_steps": [("Step 1", "Mocked response")]}
    mock_agent_executor.ainvoke.assert_called_once()


@pytest.mark.cursorgenerated
@pytest.mark.asyncio
async def test_plan_step(mocker: MockerFixture) -> None:
    """
    Test the plan_step function.

    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_planner = mocker.patch("sandbox_agent.ai.workflows.PLANNER")
    mock_planner.ainvoke.return_value = {"plan": ["Step 1", "Step 2"]}

    state = PlanExecute(input="Test input")
    result = await plan_step(state)

    assert result == {"plan": ["Step 1", "Step 2"]}
    mock_planner.ainvoke.assert_called_once_with({"messages": [("user", "Test input")]})


@pytest.mark.cursorgenerated
@pytest.mark.asyncio
async def test_replan_step_response(mocker: MockerFixture) -> None:
    """
    Test the replan_step function when it returns a response.

    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_replanner = mocker.patch("sandbox_agent.ai.workflows.REPLANNER")
    mock_replanner.ainvoke.return_value = {"response": "Mocked response"}

    state = PlanExecute(input="Test input", plan=["Step 1", "Step 2"], past_steps=[("Step 1", "Executed Step 1")])
    result = await replan_step(state)

    assert result == {"response": "Mocked response"}
    mock_replanner.ainvoke.assert_called_once_with(state)


@pytest.mark.cursorgenerated
@pytest.mark.asyncio
async def test_replan_step_plan(mocker: MockerFixture) -> None:
    """
    Test the replan_step function when it returns an updated plan.

    Args:
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_replanner = mocker.patch("sandbox_agent.ai.workflows.REPLANNER")
    mock_replanner.ainvoke.return_value = {"plan": ["Step 2", "Step 3"]}

    state = PlanExecute(input="Test input", plan=["Step 1", "Step 2"], past_steps=[("Step 1", "Executed Step 1")])
    result = await replan_step(state)

    assert result == {"plan": ["Step 2", "Step 3"]}
    mock_replanner.ainvoke.assert_called_once_with(state)


@pytest.mark.cursorgenerated
def test_should_end_agent() -> None:
    """Test the should_end function when it returns "agent"."""
    state = PlanExecute(input="Test input", plan=["Step 1", "Step 2"])
    result = should_end(state)
    assert result == "agent"


@pytest.mark.cursorgenerated
def test_should_end_end() -> None:
    """Test the should_end function when it returns END."""
    state = PlanExecute(input="Test input", response="Mocked response")
    result = should_end(state)
    assert result == END


@pytest.mark.cursorgenerated
def test_build_workflow() -> None:
    """Test the build_workflow function."""
    workflow = build_workflow()
    assert isinstance(workflow, StateGraph)
    assert "planner" in workflow.nodes
    assert "agent" in workflow.nodes
    assert "replan" in workflow.nodes


@pytest.mark.cursorgenerated
def test_get_graph() -> None:
    """Test the get_graph function."""
    graph = get_graph()
    assert callable(graph)


@pytest.mark.cursorgenerated
def test_workflow_factory_create_plan_and_execute(monkeypatch: MonkeyPatch) -> None:
    """
    Test the WorkflowFactory.create method for the "plan_and_execute" agent type.

    Args:
        monkeypatch (MonkeyPatch): The pytest monkeypatch fixture for patching.
    """
    monkeypatch.setattr("sandbox_agent.aio_settings.aiosettings.agent_type", "plan_and_execute")
    workflow = WorkflowFactory.create()
    assert callable(workflow)


@pytest.mark.cursorgenerated
def test_workflow_factory_create_unsupported_agent_type(monkeypatch: MonkeyPatch) -> None:
    """
    Test the WorkflowFactory.create method for an unsupported agent type.

    Args:
        monkeypatch (MonkeyPatch): The pytest monkeypatch fixture for patching.
    """
    monkeypatch.setattr("sandbox_agent.aio_settings.aiosettings.agent_type", "unsupported_agent_type")
    with pytest.raises(ValueError, match="Unsupported agent type: unsupported_agent_type"):
        WorkflowFactory.create()
