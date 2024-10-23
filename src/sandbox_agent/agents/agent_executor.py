"""Agent executor for the sandbox agent."""

from __future__ import annotations

import inspect

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Union
from uuid import UUID

from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler, StdOutCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools.simple import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger as LOGGER
from tenacity import RetryCallState

from sandbox_agent.ai.graph import Act, Plan
from sandbox_agent.ai.tools import ToolFactory
from sandbox_agent.aio_settings import aiosettings
from sandbox_agent.factories import ChatModelFactory, MemoryFactory


if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult


class AgentExecutorFactory:
    """Factory class for creating agent executors."""

    @staticmethod
    def create_plan_and_execute_agent(
        llm: ChatOpenAI | None = None,
        verbose: bool = False,
        memory_enabled: bool | None = None,
        **kwargs: Any,
    ) -> CompiledStateGraph:
        """
        Create a plan-and-execute agent executor.

        Args:
            llm (ChatOpenAI, optional): Language model to use for the agent. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            memory_enabled (bool, optional): Whether to enable memory for the agent.
                Defaults to None, which will use the value from aiosettings.llm_memory_enabled.
            **kwargs (CompiledStateGraph): Additional keyword arguments for the agent.

        Returns:
            AgentExecutor: The created plan-and-execute agent executor.
        """
        if llm is None:
            llm = ChatModelFactory.create(aiosettings.chat_model)
            LOGGER.info(f"Creating LLM from {aiosettings.chat_model}. llm: {llm}")

        # Get the prompt to use - you can modify this!
        prompt = hub.pull("ih/ih-react-agent-executor")
        prompt.pretty_print()

        tools: list[Tool] = ToolFactory.create_tools()

        # Determine if memory should be enabled
        if memory_enabled is None:
            memory_enabled = aiosettings.llm_memory_enabled

        # Choose the LLM that will drive the agent
        if memory_enabled:
            memory = MemoryFactory.create()
            agent_executor = create_react_agent(llm, tools, state_modifier=prompt, memory=memory, debug=True)
            LOGGER.info(f"Created agent executor with memory: {agent_executor}")
        else:
            agent_executor = create_react_agent(llm, tools, state_modifier=prompt, debug=True)
            LOGGER.info(f"Created agent executor without memory: {agent_executor}")

        return agent_executor


# SOURCE: https://python.langchain.com/docs/how_to/callbacks_runtime/
class SandboxLoggingHandler(BaseCallbackHandler):
    """
    A callback handler for logging events during agent execution.

    This class inherits from BaseCallbackHandler and provides methods to log
    events related to chat model start, LLM end, chain start, and chain end.
    """

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """
        Log when a chat model starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chat model.
            messages (List[List[BaseMessage]]): The messages to be processed by the chat model.
            **kwargs (Any): Additional keyword arguments.
        """
        print("Chat model started")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Log when an LLM (Language Model) finishes running.

        Args:
            response (LLMResult): The result from the LLM.
            **kwargs (Any): Additional keyword arguments.
        """
        print(f"Chat model ended, response: {response}")

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        Log when a chain starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Dict[str, Any]): The inputs to the chain.
            **kwargs (Any): Additional keyword arguments.
        """
        print(f"Chain {serialized.get('name')} started")

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """
        Log when a chain finishes running.

        Args:
            outputs (Dict[str, Any]): The outputs from the chain.
            **kwargs (Any): Additional keyword arguments.
        """
        print(f"Chain ended, outputs: {outputs}")


class AsyncLoggingCallbackHandler(AsyncCallbackHandler):
    """Async Tracer that logs via the input Logger."""

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running.

        **ATTENTION**: This method is called for non-chat models (regular LLMs). If
            you're implementing a handler for a chat model,
            you should use on_chat_model_start instead.

        Args:
            serialized (Dict[str, Any]): The serialized LLM.
            prompts (List[str]): The prompts.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_llm_start | serialized: {serialized} | prompts: {prompts} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | metadata: {metadata} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_llm_start | inspect: {inspect.signature(self.on_llm_start)}")
        await LOGGER.complete()

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running.

        **ATTENTION**: This method is called for chat models. If you're implementing
            a handler for a non-chat model, you should use on_llm_start instead.

        Args:
            serialized (Dict[str, Any]): The serialized chat model.
            messages (List[List[BaseMessage]]): The messages.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_chat_model_start | serialized: {serialized} | messages: {messages} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | metadata: {metadata} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_chat_model_start | inspect: {inspect.signature(self.on_chat_model_start)}")
        await LOGGER.complete()

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled.

        Args:
            token (str): The new token.
            chunk (GenerationChunk | ChatGenerationChunk): The new generated chunk,
              containing content and other information.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_llm_new_token | token: {token} | chunk: {chunk} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_llm_new_token | inspect: {inspect.signature(self.on_llm_new_token)}")
        await LOGGER.complete()

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The response which was generated.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_llm_end | response: {response} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_llm_end | inspect: {inspect.signature(self.on_llm_end)}")
        await LOGGER.complete()

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors.

        Args:
            error: The error that occurred.
            run_id: The run ID. This is the ID of the current run.
            parent_run_id: The parent run ID. This is the ID of the parent run.
            tags: The tags.
            kwargs (Any): Additional keyword arguments.
                - response (LLMResult): The response which was generated before
                    the error occurred.
        """
        LOGGER.info(
            f"on_llm_error | error: {error} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_llm_error | inspect: {inspect.signature(self.on_llm_error)}")
        await LOGGER.complete()

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when a chain starts running.

        Args:
            serialized (Dict[str, Any]): The serialized chain.
            inputs (Dict[str, Any]): The inputs.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_chain_start | serialized: {serialized} | inputs: {inputs} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | metadata: {metadata} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_chain_start | inspect: {inspect.signature(self.on_chain_start)}")
        await LOGGER.complete()

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when a chain ends running.

        Args:
            outputs (Dict[str, Any]): The outputs of the chain.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_chain_end | outputs: {outputs} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_chain_end | inspect: {inspect.signature(self.on_chain_end)}")
        await LOGGER.complete()

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_chain_error | error: {error} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_chain_error | inspect: {inspect.signature(self.on_chain_error)}")
        await LOGGER.complete()

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when the tool starts running.

        Args:
            serialized (Dict[str, Any]): The serialized tool.
            input_str (str): The input string.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            inputs (Optional[Dict[str, Any]]): The inputs.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_tool_start | serialized: {serialized} | input_str: {input_str} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | metadata: {metadata} | inputs: {inputs} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_tool_start | inspect: {inspect.signature(self.on_tool_start)}")
        await LOGGER.complete()

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when the tool ends running.

        Args:
            output (Any): The output of the tool.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_tool_end | output: {output} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_tool_end | inspect: {inspect.signature(self.on_tool_end)}")
        await LOGGER.complete()

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_tool_error | error: {error} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_tool_error | inspect: {inspect.signature(self.on_tool_error)}")
        await LOGGER.complete()

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on an arbitrary text.

        Args:
            text (str): The text.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_text | text: {text} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_text | inspect: {inspect.signature(self.on_text)}")
        await LOGGER.complete()

    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on a retry event.

        Args:
            retry_state (RetryCallState): The retry state.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_retry | retry_state: {retry_state} | run_id: {run_id} | parent_run_id: {parent_run_id} | kwargs: {kwargs}"
        )
        await LOGGER.complete()

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action.

        Args:
            action (AgentAction): The agent action.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_agent_action | action: {action} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_agent_action | inspect: {inspect.signature(self.on_agent_action)}")
        await LOGGER.complete()

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on the agent end.

        Args:
            finish (AgentFinish): The agent finish.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_agent_finish | finish: {finish} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_agent_finish | inspect: {inspect.signature(self.on_agent_finish)}")
        await LOGGER.complete()

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on the retriever start.

        Args:
            serialized (Dict[str, Any]): The serialized retriever.
            query (str): The query.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            metadata (Optional[Dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_retriever_start | serialized: {serialized} | query: {query} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | metadata: {metadata} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_retriever_start | inspect: {inspect.signature(self.on_retriever_start)}")
        await LOGGER.complete()

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on the retriever end.

        Args:
            documents (Sequence[Document]): The documents retrieved.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments."""
        LOGGER.info(
            f"on_retriever_end | documents: {documents} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_retriever_end | inspect: {inspect.signature(self.on_retriever_end)}")
        await LOGGER.complete()

    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on retriever error.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[List[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """
        LOGGER.info(
            f"on_retriever_error | error: {error} | run_id: {run_id} | parent_run_id: {parent_run_id} | tags: {tags} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_retriever_error | inspect: {inspect.signature(self.on_retriever_error)}")
        await LOGGER.complete()

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Override to define a handler for a custom event.

        Args:
            name: The name of the custom event.
            data: The data for the custom event. Format will match
                  the format specified by the user.
            run_id: The ID of the run.
            tags: The tags associated with the custom event
                (includes inherited tags).
            metadata: The metadata associated with the custom event
                (includes inherited metadata).

        .. versionadded:: 0.2.15
        """
        LOGGER.info(
            f"on_custom_event | name: {name} | data: {data} | run_id: {run_id} | tags: {tags} | metadata: {metadata} | kwargs: {kwargs}"
        )
        LOGGER.info(f"on_custom_event | inspect: {inspect.signature(self.on_custom_event)}")
        await LOGGER.complete()


# NOTE: Sequence[MessageLikeRepresentation]
# Sequence[MessageLikeRepresentation]
def format_user_message(message: str) -> dict[str, list[tuple[str, str]]]:
    """
    Format a user message into a dictionary for use with agent_executor.invoke().

    Args:
        message (str): The user message to format.

    Returns:
        dict[str, list[tuple[str, str]]]: A dictionary containing the formatted message.
    """
    return {"messages": [("user", message)]}


def format_user_input(message: str) -> dict[str, str]:
    """
    Format a user message into a dictionary for use with agent_executor.invoke().

    Args:
        message (str): The user message to format.

    Returns:
        dict[str, str]: A dictionary containing the formatted message as the input.
    """
    return {"input": message}
