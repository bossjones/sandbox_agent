# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
"""Tests for the AgentExecutorFactory class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain import hub
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent
from vcr.request import Request as VCRRequest

import pytest

from sandbox_agent.agents.agent_executor import AgentExecutorFactory
from sandbox_agent.ai.chat_models import ChatModelFactory
from sandbox_agent.ai.tools import ToolFactory
from sandbox_agent.factories import ChatModelFactory


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture


@pytest.mark.integration()
@pytest.mark.vcronly()
@pytest.mark.default_cassette("test_create_plan_and_execute_agent.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query", "headers"],
    ignore_localhost=False,
)
def test_create_plan_and_execute_agent(
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
    vcr: VCRRequest,
):
    """Test the create_plan_and_execute_agent method."""
    # Call the method under test
    agent_executor = AgentExecutorFactory.create_plan_and_execute_agent(verbose=True)

    # Assert that the agent executor was created successfully
    assert agent_executor is not None
    # Add more specific assertions based on the expected behavior

    # # Assert that the hub.pull method was called with the correct arguments
    # assert hub.pull.call_args[0][0] == "ih/ih-react-agent-executor"

    # # Assert that the ToolFactory.create_tools method was called
    # assert ToolFactory.create_tools.called

    # # Assert that the create_react_agent function was called
    # assert create_react_agent.called
    # # Add more specific assertions for the create_react_agent call if needed


@pytest.mark.integration()
@pytest.mark.vcronly()
@pytest.mark.default_cassette("test_create_plan_and_execute_agent_with_custom_llm.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query", "headers"],
    ignore_localhost=False,
)
def test_create_plan_and_execute_agent_with_custom_llm(
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
    vcr: VCRRequest,
):
    """
    Test the create_plan_and_execute_agent method with a custom LLM.

    Args:
        caplog (LogCaptureFixture): Fixture for capturing log messages.
        capsys (CaptureFixture): Fixture for capturing stdout and stderr output.
        vcr (VCRRequest): Fixture for recording and replaying HTTP interactions.
    """
    # Create a custom LLM instance using ChatModelFactory
    custom_llm = ChatModelFactory.create("gpt-3.5-turbo")

    # Call the method under test with the custom LLM
    agent_executor = AgentExecutorFactory.create_plan_and_execute_agent(llm=custom_llm)

    # Assert that the agent executor was created successfully
    assert agent_executor is not None
    # Add more specific assertions based on the expected behavior

    # # Assert that the hub.pull method was called with the correct arguments
    # assert hub.pull.call_args[0][0] == "ih/ih-react-agent-executor"

    # # Assert that the ToolFactory.create_tools method was called
    # assert ToolFactory.create_tools.called

    # # Assert that the create_react_agent function was called
    # assert create_react_agent.called
    # # Add more specific assertions for the create_react_agent call if needed
