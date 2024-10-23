"""ReAct agent from scratch

A ReAct agent, or reasoning and action agent, is a framework that combines the reasoning capabilities of large language models (LLMs) with the ability to take actions. This allows ReAct agents to understand and process information, assess situations, and take appropriate actions.
ReAct agents are designed to interact with the real world through actions, such as searching the web, accessing databases, or controlling physical devices. They can flexibly adjust their strategy based on continuous feedback from the environment, allowing them to solve complex or dynamic tasks.
ReAct agents use a step-by-step problem-solving approach, alternating between generating thoughts and task-specific actions. They maintain a list of previous messages, which serves as input to the LLM model to process context and provide relevant responses.
ReAct agents can be used in a variety of applications, such as customer support systems, where they can think about a customer's issue, retrieve relevant data, and refine their approach to resolve the issue.
"""

# SOURCE: https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/
# SOURCE: https://langchain-ai.github.io/langgraph/tutorials/introduction/#part-2-enhancing-the-chatbot-with-tools
# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import json

from collections.abc import Sequence
from typing import Annotated, Any, Dict, List, Literal, TypedDict, Union

import pysnooper

from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from loguru import logger

from sandbox_agent.agents.agent_executor import AgentExecutorFactory
from sandbox_agent.ai.graph import Act, Plan, PlanExecute, Response
from sandbox_agent.aio_settings import aiosettings
from sandbox_agent.factories import ChatModelFactory, MemoryFactory, ToolFactory


class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]


class State(TypedDict):
    """
    Represents the state of the conversation.

    Attributes:
    ----------
    messages : Annotated[List[dict], add_messages]
        A list of message dictionaries. The `add_messages` function
        defines how this state key should be updated
        (in this case, it appends messages to the list, rather than overwriting them).
    """

    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)

    messages: Annotated[list[dict], add_messages]


model = ChatModelFactory.create()
tools = ToolFactory.create_tools()
model = model.bind_tools(tools)
memory = MemoryFactory.create()


"""
Define nodes and edges¶
Next let's define our nodes and edges. In our basic ReAct agent there are only two nodes, one for calling the model and one for using tools, however you can modify this basic structure to work better for your use case. The tool node we define here is a simplified version of the prebuilt ToolNode, which has some additional features.

Perhaps you want to add a node for adding structured output or a node for executing some external action (sending an email, adding a calendar event, etc.). Maybe you just want to change the way the call_model node works and how should_continue decides whether to call tools - the possibilities are endless and LangGraph makes it easy to customize this basic structure for your specific use case.
"""


tools_by_name = {tool.name: tool for tool in tools}


# Define our tool node
def tool_node(state: AgentState) -> dict[str, list[ToolMessage]]:
    """
    Execute the tools specified in the agent's last message.

    This function processes the tool calls from the last message in the agent's state,
    invokes the corresponding tools, and generates ToolMessage responses.

    Args:
    ----
        state (AgentState): The current state of the agent, containing messages and tool calls.

    Returns:
    -------
        Dict[str, List[ToolMessage]]: A dictionary with a 'messages' key containing a list of ToolMessages
        representing the results of the tool executions.
    """
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])  # type: ignore
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],  # type: ignore
                tool_call_id=tool_call["id"],  # type: ignore
            )
        )
    return {"messages": outputs}


# Define the node that calls the model
def call_model(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, list[BaseMessage]]:
    """
    Generate a response using the language model based on the current agent state.

    This function adds a system prompt to the existing messages and invokes the language model
    to generate a response.

    Args:
    ----
        state (AgentState): The current state of the agent, containing previous messages.
        config (RunnableConfig): Configuration for the model invocation.

    Returns:
    -------
        Dict[str, List[BaseMessage]]: A dictionary with a 'messages' key containing a list
        with the model's response message.
    """
    system_prompt = SystemMessage(
        "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["end", "continue"]:
    """
    Determine whether the agent should continue processing or end the conversation.

    This function checks the last message in the agent's state for tool calls.
    If there are tool calls, it indicates that the conversation should continue.
    Otherwise, it signals that the conversation should end.

    Args:
    ----
        state (AgentState): The current state of the agent, containing messages.

    Returns:
    -------
        Literal["end", "continue"]: "continue" if there are tool calls in the last message,
        "end" otherwise.
    """
    # messages = state["messages"]
    # last_message = messages[-1]
    # return "continue" if last_message.tool_calls else "end"

    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")


# Define a function to build the compile arguments
def build_compile_args() -> dict[str, Any]:
    """
    Build the compile arguments for the workflow graph.

    This function constructs a dictionary of arguments to be passed to the `workflow.compile()`
    method. The arguments include:

    - `checkpointer`: An instance of `MemorySaver` if `aiosettings.llm_memory_enabled` is True.
    - `interrupt_before`: A list of node names before which the execution should be interrupted
      to allow for human input. In this case, it is set to `["tools"]` if
      `aiosettings.llm_human_loop_enabled` is True.
    - `interrupt_after`: A list of node names after which the execution should be interrupted
      to allow for human input. This is currently commented out, but you can uncomment the line
      if you want to interrupt after the "tools" node as well.

    Returns:
        Dict[str, Any]: A dictionary containing the compile arguments.
    """
    compile_args = {}

    if aiosettings.llm_memory_enabled:
        logger.info("Adding checkpointer to compile args")
        compile_args["checkpointer"] = memory

    if aiosettings.llm_human_loop_enabled:
        logger.info("Adding interrupt_before to compile args")
        compile_args["interrupt_before"] = ["tools"]
        # Uncomment the following line if you want to interrupt after tools as well
        # compile_args["interrupt_after"] = ["tools"]

    logger.info(f"Compile args: {compile_args}")
    return compile_args


# Now we can compile the graph with conditional arguments
graph = workflow.compile(**build_compile_args())
