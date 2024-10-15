"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, List, Literal, Optional, Sequence, TYPE_CHECKING, Union
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import Input
from langchain_core.tools import BaseTool, InjectedToolArg
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel

if TYPE_CHECKING:
    ...
INVALID_TOOL_NAME_ERROR_TEMPLATE = ...
TOOL_CALL_ERROR_TEMPLATE = ...
def msg_content_output(output: Any) -> str | List[dict]:
    ...

class ToolNode(RunnableCallable):
    """A node that runs the tools called in the last AIMessage.

    It can be used either in StateGraph with a "messages" key or in MessageGraph. If
    multiple tool calls are requested, they will be run in parallel. The output will be
    a list of ToolMessages, one for each tool call.

    The `ToolNode` is roughly analogous to:

    ```python
    tools_by_name = {tool.name: tool for tool in tools}
    def tool_node(state: dict):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        return {"messages": result}
    ```

    Important:
        - The state MUST contain a list of messages.
        - The last message MUST be an `AIMessage`.
        - The `AIMessage` MUST have `tool_calls` populated.
    """
    name: str = ...
    def __init__(self, tools: Sequence[Union[BaseTool, Callable]], *, name: str = ..., tags: Optional[list[str]] = ..., handle_tool_errors: Optional[bool] = ...) -> None:
        ...

    def invoke(self, input: Input, config: Optional[RunnableConfig] = ..., **kwargs: Any) -> Any:
        ...

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig] = ..., **kwargs: Any) -> Any:
        ...



def tools_condition(state: Union[list[AnyMessage], dict[str, Any], BaseModel]) -> Literal["tools", "__end__"]:
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (Union[list[AnyMessage], dict[str, Any], BaseModel]): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.


    Examples:
        Create a custom ReAct-style agent with tools.

        ```pycon
        >>> from langchain_anthropic import ChatAnthropic
        >>> from langchain_core.tools import tool
        ...
        >>> from langgraph.graph import StateGraph
        >>> from langgraph.prebuilt import ToolNode, tools_condition
        >>> from langgraph.graph.message import add_messages
        ...
        >>> from typing import TypedDict, Annotated
        ...
        >>> @tool
        >>> def divide(a: float, b: float) -> int:
        ...     \"\"\"Return a / b.\"\"\"
        ...     return a / b
        ...
        >>> llm = ChatAnthropic(model="claude-3-haiku-20240307")
        >>> tools = [divide]
        ...
        >>> class State(TypedDict):
        ...     messages: Annotated[list, add_messages]
        >>>
        >>> graph_builder = StateGraph(State)
        >>> graph_builder.add_node("tools", ToolNode(tools))
        >>> graph_builder.add_node("chatbot", lambda state: {"messages":llm.bind_tools(tools).invoke(state['messages'])})
        >>> graph_builder.add_edge("tools", "chatbot")
        >>> graph_builder.add_conditional_edges(
        ...     "chatbot", tools_condition
        ... )
        >>> graph_builder.set_entry_point("chatbot")
        >>> graph = graph_builder.compile()
        >>> graph.invoke({"messages": {"role": "user", "content": "What's 329993 divided by 13662?"}})
        ```
    """
    ...

class InjectedState(InjectedToolArg):
    """Annotation for a Tool arg that is meant to be populated with the graph state.

    Any Tool argument annotated with InjectedState will be hidden from a tool-calling
    model, so that the model doesn't attempt to generate the argument. If using
    ToolNode, the appropriate graph state field will be automatically injected into
    the model-generated tool args.

    Args:
        field: The key from state to insert. If None, the entire state is expected to
            be passed in.

    Example:
        ```python
        from typing import List
        from typing_extensions import Annotated, TypedDict

        from langchain_core.messages import BaseMessage, AIMessage
        from langchain_core.tools import tool

        from langgraph.prebuilt import InjectedState, ToolNode


        class AgentState(TypedDict):
            messages: List[BaseMessage]
            foo: str

        @tool
        def state_tool(x: int, state: Annotated[dict, InjectedState]) -> str:
            '''Do something with state.'''
            if len(state["messages"]) > 2:
                return state["foo"] + str(x)
            else:
                return "not enough messages"

        @tool
        def foo_tool(x: int, foo: Annotated[str, InjectedState("foo")]) -> str:
            '''Do something else with state.'''
            return foo + str(x + 1)

        node = ToolNode([state_tool, foo_tool])

        tool_call1 = {"name": "state_tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}
        tool_call2 = {"name": "foo_tool", "args": {"x": 1}, "id": "2", "type": "tool_call"}
        state = {
            "messages": [AIMessage("", tool_calls=[tool_call1, tool_call2])],
            "foo": "bar",
        }
        node.invoke(state)
        ```

        ```pycon
        [
            ToolMessage(content='not enough messages', name='state_tool', tool_call_id='1'),
            ToolMessage(content='bar2', name='foo_tool', tool_call_id='2')
        ]
        ```
    """
    def __init__(self, field: Optional[str] = ...) -> None:
        ...



class InjectedStore(InjectedToolArg):
    """Annotation for a Tool arg that is meant to be populated with LangGraph store.

    Any Tool argument annotated with InjectedStore will be hidden from a tool-calling
    model, so that the model doesn't attempt to generate the argument. If using
    ToolNode, the appropriate store field will be automatically injected into
    the model-generated tool args. Note: if a graph is compiled with a store object,
    the store will be automatically propagated to the tools with InjectedStore args
    when using ToolNode.

    !!! Warning
        `InjectedStore` annotation requires `langchain-core >= 0.3.8`

    Example:
        ```python
        from typing import Any
        from typing_extensions import Annotated

        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool

        from langgraph.store.memory import InMemoryStore
        from langgraph.prebuilt import InjectedStore, ToolNode

        store = InMemoryStore()
        store.put(("values",), "foo", {"bar": 2})

        @tool
        def store_tool(x: int, my_store: Annotated[Any, InjectedStore()]) -> str:
            '''Do something with store.'''
            stored_value = my_store.get(("values",), "foo").value["bar"]
            return stored_value + x

        node = ToolNode([store_tool])

        tool_call = {"name": "store_tool", "args": {"x": 1}, "id": "1", "type": "tool_call"}
        state = {
            "messages": [AIMessage("", tool_calls=[tool_call])],
        }

        node.invoke(state, store=store)
        ```

        ```pycon
        {
            "messages": [
                ToolMessage(content='3', name='store_tool', tool_call_id='1'),
            ]
        }
        ```
    """
    ...