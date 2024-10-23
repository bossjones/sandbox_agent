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
from typing import Annotated, Any, List, Literal, TypedDict, Union

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
from sandbox_agent.factories import ChatModelFactory, ToolFactory


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


"""
Define nodes and edges¶
Next let's define our nodes and edges. In our basic ReAct agent there are only two nodes, one for calling the model and one for using tools, however you can modify this basic structure to work better for your use case. The tool node we define here is a simplified version of the prebuilt ToolNode, which has some additional features.

Perhaps you want to add a node for adding structured output or a node for executing some external action (sending an email, adding a calendar event, etc.). Maybe you just want to change the way the call_model node works and how should_continue decides whether to call tools - the possibilities are endless and LangGraph makes it easy to customize this basic structure for your specific use case.
"""


tools_by_name = {tool.name: tool for tool in tools}


# Define our tool node
def tool_node(state: AgentState):
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
):
    # this is similar to customizing the create_react_agent with state_modifier, but is a lot more flexible
    system_prompt = SystemMessage(
        "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
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

# Now we can compile and visualize our graph
graph = workflow.compile()


# # Initialize the StateGraph
# # The first thing you do when you define a graph is define the State of the graph. The State consists of the schema of the graph as well as reducer functions which specify how to apply updates to the state. In our example State is a TypedDict with a single key: messages. The messages key is annotated with the add_messages reducer function, which tells LangGraph to append new messages to the existing list, rather than overwriting it. State keys without an annotation will be overwritten by each update, storing the most recent value. Check out this conceptual guide to learn more about state, reducers and other low-level concepts.
# graph_builder: StateGraph = StateGraph(State)

# # Create the agent executor
# llm = AgentExecutorFactory.create_plan_and_execute_agent()

# def chatbot(state: State) -> dict:
#     """
#     Process the current state and generate a response using the LLM.

#     Args:
#     ----
#     state : State
#         The current state of the conversation.

#     Returns:
#     -------
#     dict
#         A dictionary containing the LLM's response message.
#     """
#     return {"messages": [llm.invoke(state["messages"])]}

# # Add the chatbot node to the graph
# # The first argument is the unique node name
# # The second argument is the function or object that will be called whenever
# # the node is used.
# graph_builder.add_node("chatbot", chatbot)
