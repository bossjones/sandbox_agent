"""Agent executor for the sandbox agent."""

from __future__ import annotations

from typing import Any, List, Union

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from sandbox_agent.ai.graph import Act, Plan
from sandbox_agent.ai.tools import ToolFactory
from sandbox_agent.aio_settings import aiosettings
from sandbox_agent.factories import ChatModelFactory


class AgentExecutorFactory:
    """Factory class for creating agent executors."""

    @staticmethod
    def create_plan_and_execute_agent(
        llm: ChatOpenAI = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Create a plan-and-execute agent executor.

        Args:
            llm (ChatOpenAI, optional): Language model to use for the agent. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs (Any): Additional keyword arguments for the agent.

        Returns:
            AgentExecutor: The created plan-and-execute agent executor.
        """
        if llm is None:
            llm = ChatModelFactory.create(aiosettings.chat_model)

        # Get the prompt to use - you can modify this!
        prompt = hub.pull("ih/ih-react-agent-executor")
        prompt.pretty_print()

        tools = ToolFactory.create_tools()

        # Choose the LLM that will drive the agent
        agent_executor = create_react_agent(llm, tools, state_modifier=prompt)

        return agent_executor
