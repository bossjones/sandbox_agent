"""Eval tests for Tool Selection Evaluation. See more: https://docs.smith.langchain.com/old/cookbook/testing-examples/tool-selection"""

# pylint: disable=too-many-function-args
# mypy: disable-error-code="arg-type, var-annotated, list-item, no-redef, truthy-bool, return-value"
# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
# mypy: disable-error-code="arg-type, var-annotated, list-item, no-redef"
from __future__ import annotations

import time

from importlib.metadata import version
from typing import TYPE_CHECKING

import langsmith

from sandbox_agent.llm_manager import LlmManager
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run, TracerSessionResult

import pytest


if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


runs = []
examples = []


# @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.integration()
@pytest.mark.evals()
@pytest.mark.slow()
@pytest.mark.flaky()
def test_evals_sandbox_agent_tool_selection(caplog: LogCaptureFixture):
    ls_client = langsmith.Client()
    llm = LlmManager().llm
    CHAT_CLS = str(llm.__repr__)
    DATASET_NAME = "goob-ai-llm-custom-tool-selection-dataset"
    DATASET_VERSION = "latest"
    CONCURRENCY = 1
    EXPERIMENT_PREFIX = "Tool Calling"
    EXPECTED_AVG = 1.0

    MODEL = ""
    DEPLOYMENT_NAME = ""
    FULL_MODEL_NAME = ""
    if "ChatOpenAI" in CHAT_CLS:
        MODEL = llm.model_name
        FULL_MODEL_NAME = f"{MODEL}"

    def invoke_llm_tool_selection(inputs: dict):
        # Convert the messages into LangChain format
        messages = [
            (payload["type"], payload["data"]["content"]) for payload in inputs["input"]
        ]
        # Extract the available tools from the inputs
        tools = inputs["functions"]
        # Bind the tools to the LLM
        llm_with_tools = llm.bind_tools(tools)

        return llm_with_tools.invoke(messages)

    # The custom evaluator created to check to see if correct tool was being called
    def correct_tools(root_run: Run, example: Example) -> dict:
        tool_calls = [
            tool_call["name"] for tool_call in root_run.outputs["output"].tool_calls
        ]
        expected_tool_calls = [
            tool_call["name"]
            for tool_call in example.outputs["output"]["data"]["tool_calls"]
        ]
        return {
            "score": int(set(tool_calls) == set(expected_tool_calls)),
            "key": "correct_tools",
        }

    results = evaluate(
        invoke_llm_tool_selection,
        data=ls_client.list_examples(dataset_name=DATASET_NAME),
        evaluators=[correct_tools],
        experiment_prefix=EXPERIMENT_PREFIX,
        metadata={
            "model": FULL_MODEL_NAME,
            "langchain_version": version("langchain"),
            "langchain_community_version": version("langchain_community"),
            "langchain_core_version": version("langchain_core"),
            "langchain_openai_version": version("langchain_openai"),
            "langchain_text_splitters_version": version("langchain_text_splitters"),
            "langsmith_version": version("langsmith"),
            "pydantic_version": version("pydantic"),
            "pydantic_settings_version": version("pydantic_settings"),
        },
    )

    # Use langsmith client to pull back the project results
    project_results: TracerSessionResult = ls_client.read_project(
        project_name=results.experiment_name
    )

    # sometimes we try to grab the feedback stats before everything is ready, so we need to keep trying until we get them
    if project_results.feedback_stats == {}:
        time.sleep(5)
        project_results: TracerSessionResult = ls_client.read_project(
            project_name=results.experiment_name
        )

    assert project_results.feedback_stats, "Feedback stats are empty"
    # get the average of the project you just ran
    avg: float = project_results.feedback_stats["correct_tools"]["avg"]

    # For tool selection, you want this value to always be as close to 1 as possible
    assert avg == EXPECTED_AVG

    caplog.clear()
