# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import Any, List, Literal, Union

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from sandbox_agent.agents.agent_executor import AgentExecutorFactory
from sandbox_agent.ai.graph import Act, Plan, PlanExecute, Response
from sandbox_agent.ai.tools import ToolFactory
from sandbox_agent.aio_settings import aiosettings
from sandbox_agent.factories import ChatModelFactory


# Create the agent executor
agent_executor = AgentExecutorFactory.create_plan_and_execute_agent()

# Define the planner prompt
PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# Create the planner using the prompt and a ChatOpenAI model
PLANNER = PLANNER_PROMPT | ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Plan)

# Define the replanner prompt
REPLANNER_PROMPT = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

# Create the replanner using the prompt and a ChatOpenAI model
REPLANNER = REPLANNER_PROMPT | ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Act)


async def execute_step(state: PlanExecute) -> dict[str, Any]:
    """
    Execute a step in the plan.

    Args:
        state (PlanExecute): The current state of the plan execution.

    Returns:
        dict[str, Any]: The updated state with the executed step added to past_steps.
    """
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke({"messages": [("user", task_formatted)]})
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute) -> dict[str, list[str]]:
    """
    Create a plan based on the input state.

    Args:
        state (PlanExecute): The current state of the plan execution.

    Returns:
        dict[str, list[str]]: The updated state with the generated plan.
    """
    plan = await PLANNER.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute) -> dict[str, Union[str, list[str]]]:
    """
    Replan based on the current state.

    Args:
        state (PlanExecute): The current state of the plan execution.

    Returns:
        dict[str, Union[str, list[str]]]: The updated state with either a response or an updated plan.
    """
    output = await REPLANNER.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute) -> Literal["agent", END]:
    """
    Determine if the plan execution should end.

    Args:
        state (PlanExecute): The current state of the plan execution.

    Returns:
        Literal["agent", END]: "agent" if the execution should continue, END if it should end.
    """
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


def build_workflow() -> CompiledStateGraph:
    """
    Build the workflow graph.

    Returns:
        CompiledStateGraph: The compiled state graph.
    """
    # Create the workflow graph
    workflow = StateGraph(PlanExecute)

    # Add the plan node
    workflow.add_node("planner", plan_step)

    # Add the execution step
    workflow.add_node("agent", execute_step)

    # Add a replan node
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")

    # From plan we go to agent
    workflow.add_edge("planner", "agent")

    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
        ["agent", END],
    )

    return workflow


def build_compiled_workflow() -> CompiledStateGraph:
    """
    Get the compiled state graph for the workflow.

    Returns:
        CompiledStateGraph: The compiled state graph.
    """
    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    workflow = build_workflow()
    return workflow.compile()


class WorkflowFactory:
    """Factory class for creating workflow instances."""

    @staticmethod
    def create() -> CompiledStateGraph:
        """
        Create a workflow instance based on the agent type defined in aio_settings.

        Returns:
            CompiledStateGraph: The created workflow instance.

        Raises:
            ValueError: If an unsupported agent type is provided.
        """
        agent_type = aiosettings.agent_type

        if agent_type == "plan_and_execute":
            return build_compiled_workflow()
        # Add more agent types and their corresponding workflows as needed
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
