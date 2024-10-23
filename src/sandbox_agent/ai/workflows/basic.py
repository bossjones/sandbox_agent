from __future__ import annotations

from typing import Annotated, List

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict


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


def chatbot(state: State) -> dict:
    """
    Process the current state and generate a response using the LLM.

    Args:
    ----
    state : State
        The current state of the conversation.

    Returns:
    -------
    dict
        A dictionary containing the LLM's response message.
    """
    return {"messages": [llm.invoke(state["messages"])]}


# Initialize the StateGraph
# The first thing you do when you define a graph is define the State of the graph. The State consists of the schema of the graph as well as reducer functions which specify how to apply updates to the state. In our example State is a TypedDict with a single key: messages. The messages key is annotated with the add_messages reducer function, which tells LangGraph to append new messages to the existing list, rather than overwriting it. State keys without an annotation will be overwritten by each update, storing the most recent value. Check out this conceptual guide to learn more about state, reducers and other low-level concepts.
graph_builder: StateGraph = StateGraph(State)

# Initialize the LLM
llm: ChatAnthropic = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Add the chatbot node to the graph
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

# Set the entry and finish points
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# Compile the graph
graph = graph_builder.compile()
