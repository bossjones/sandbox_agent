from __future__ import annotations

from typing import List, Optional, Tuple, Union

from langchain.llms import BaseLLM
from pydantic import BaseModel


class Plan(BaseModel):
    steps: list[str]


def plan_step(state: PlanExecuteState, planner_prompt: str, llm: BaseLLM) -> Plan:
    """
    Generate a plan based on the current state using the planner prompt and LLM.

    Args:
        state (PlanExecuteState): The current state of the plan-and-execute agent.
        planner_prompt (str): The prompt to use for generating the plan.
        llm (BaseLLM): The language model to use for generating the plan.

    Returns:
        Plan: The generated plan.
    """
    # Use the LLM to generate the plan based on the state and planner prompt
    plan_output = llm.generate([state.dict(), planner_prompt])
    plan_steps = plan_output.generations[0][0].text.strip().split("\n")
    return Plan(steps=plan_steps)


class Response(BaseModel):
    text: str


class Act(BaseModel):
    act: Union[Plan, Response]


def replan_step(state: PlanExecuteState) -> Act:
    """
    Generate a new plan or response based on the current state and previous steps.

    Args:
        state (PlanExecuteState): The current state of the plan-and-execute agent.

    Returns:
        Act: The generated action (either a new plan or a response).
    """
    # Implement the logic to generate a new plan or response based on the state
    # This may involve using the LLM to analyze the previous steps and results
    # and determine the next course of action
    # For simplicity, this example always generates a response
    response = Response(text="I have completed the task.")
    return Act(act=response)


class PlanExecuteState(BaseModel):
    input: str = ""
    plan: Plan = Plan(steps=[])
    past_steps: list[tuple[str, str]] = []
    act: Optional[Act] = None
