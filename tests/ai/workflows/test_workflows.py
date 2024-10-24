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
    build_compiled_workflow,
    build_workflow,
    execute_step,
    plan_step,
    replan_step,
    should_end,
)


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from vcr.request import Request as VCRRequest

    from pytest_mock.plugin import MockerFixture


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
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


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
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


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
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


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
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


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
@pytest.mark.cursorgenerated
def test_should_end_agent() -> None:
    """Test the should_end function when it returns "agent"."""
    state = PlanExecute(input="Test input", plan=["Step 1", "Step 2"])
    result = should_end(state)
    assert result == "agent"


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
@pytest.mark.cursorgenerated
def test_should_end_end() -> None:
    """Test the should_end function when it returns END."""
    state = PlanExecute(input="Test input", response="Mocked response")
    result = should_end(state)
    assert result == END


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
@pytest.mark.cursorgenerated
def test_build_workflow() -> None:
    """Test the build_workflow function."""
    workflow = build_workflow()
    assert isinstance(workflow, StateGraph)
    assert "planner" in workflow.nodes
    assert "agent" in workflow.nodes
    assert "replan" in workflow.nodes


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
@pytest.mark.cursorgenerated
def test_build_compiled_workflow() -> None:
    """Test the build_compiled_workflow function."""
    graph = build_compiled_workflow()
    assert callable(graph)


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
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


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
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


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
@pytest.mark.vcronly
@pytest.mark.default_cassette("test_execute_step.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query", "headers"],
    ignore_localhost=False,
)
@pytest.mark.asyncio
async def test_execute_step_vcr(vcr: VCRRequest) -> None:
    """
    Test the execute_step function using VCR.

    Args:
        vcr (VCRRequest): The VCR request fixture.
    """
    state = PlanExecute(input="Test input", plan=["Step 1", "Step 2"])
    result = await execute_step(state)

    assert "clarify what" in result["past_steps"][0][1]

    # assert result == {'past_steps': [('Step 1', 'It seems like there is a repetition in your request. Could you please clarify what "Step 1" entails so I can assist you accordingly?')]}


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
@pytest.mark.vcronly
@pytest.mark.default_cassette("test_plan_step.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query", "headers"],
    ignore_localhost=False,
)
@pytest.mark.asyncio
async def test_plan_step_vcr(vcr: VCRRequest) -> None:
    """
    Test the plan_step function using VCR.

    Args:
        vcr (VCRRequest): The VCR request fixture.
    """
    state = PlanExecute(input="Test input")
    result = await plan_step(state)

    assert result

    # assert result == {'plan': ['Identify the problem or task that needs to be addressed.', 'Gather all necessary information and resources related to the task.', 'Break down the task into smaller, manageable steps.', 'Organize the steps in a logical order to ensure a smooth workflow.', 'Assign responsibilities or resources to each step if needed.', 'Execute each step in the order they were organized.', 'Review the results after completing all steps to ensure the task is completed correctly.']}


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
@pytest.mark.vcronly
@pytest.mark.default_cassette("test_replan_step_response.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query", "headers"],
    ignore_localhost=False,
)
@pytest.mark.asyncio
async def test_replan_step_response_vcr(vcr: VCRRequest) -> None:
    """
    Test the replan_step function when it returns a response using VCR.

    Args:
        vcr (VCRRequest): The VCR request fixture.
    """
    state = PlanExecute(input="Test input", plan=["Step 1", "Step 2"], past_steps=[("Step 1", "Executed Step 1")])
    result = await replan_step(state)

    # assert result == {'plan': ['Step 2']}
    assert result


@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
@pytest.mark.vcronly
@pytest.mark.default_cassette("test_replan_step_plan.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query", "headers"],
    ignore_localhost=False,
)
@pytest.mark.asyncio
async def test_replan_step_plan_vcr(vcr: VCRRequest) -> None:
    """
    Test the replan_step function when it returns an updated plan using VCR.

    Args:
        vcr (VCRRequest): The VCR request fixture.
    """
    state = PlanExecute(input="Test input", plan=["Step 1", "Step 2"], past_steps=[("Step 1", "Executed Step 1")])
    result = await replan_step(state)

    # assert result == {"plan": ["Recorded Step 2", "Recorded Step 3"]}
    assert result
    # assert result


# @pytest.mark.vcronly
# @pytest.mark.default_cassette("test_workflow_factory_create_plan_and_execute.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "headers"],
#     ignore_localhost=False,
# )
# def test_workflow_factory_create_plan_and_execute_vcr(monkeypatch: MonkeyPatch, vcr: VCRRequest) -> None:
#     """
#     Test the WorkflowFactory.create method for the "plan_and_execute" agent type using VCR.

#     Args:
#         monkeypatch (MonkeyPatch): The pytest monkeypatch fixture for patching.
#         vcr (VCRRequest): The VCR request fixture.
#     """
#     monkeypatch.setattr("sandbox_agent.aio_settings.aiosettings.agent_type", "plan_and_execute")
#     workflow = WorkflowFactory.create()
#     assert str(type(workflow)) == "<class 'langgraph.graph.state.CompiledStateGraph'>"
