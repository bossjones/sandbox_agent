# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import operator

from typing import Annotated, List, Literal, Tuple, Union

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class PlanExecute(TypedDict):
    """
    Represents the state to track for the plan-and-execute agent.

    Attributes:
        input (str): The original input or objective.
        plan (list[str]): The current plan represented as a list of strings.
        past_steps (Annotated[list[tuple], operator.add]): Previously executed steps represented as a list of tuples.
        response (str): The final response to the user.
    """

    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple], operator.add]
    response: str


# Planning Step
# This step creates a plan using function calling.
class Plan(BaseModel):
    """
    Represents a plan to follow in the future.

    Attributes:
        steps (list[str]): Different steps to follow, should be in sorted order.
    """

    steps: list[str] = Field(description="Different steps to follow, should be in sorted order.")


class Response(BaseModel):
    """
    Represents a response to the user.

    Attributes:
        response (str): The response to the user.
    """

    response: str


# Re-Plan Step
# This step re-does the plan based on the result of the previous step.
class Act(BaseModel):
    """
    Represents an action to perform.

    Attributes:
        action (Union[Response, Plan]): The action to perform. If you want to respond to the user, use Response.
                                         If you need to further use tools to get the answer, use Plan.
    """

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )
