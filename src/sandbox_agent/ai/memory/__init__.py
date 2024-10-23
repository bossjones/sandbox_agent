"""Chat models for the sandbox agent."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from sandbox_agent.aio_settings import aiosettings


class MemoryFactory:
    """Factory class for creating memory instances."""

    @staticmethod
    def create(memory_type: str = aiosettings.llm_memory_type) -> MemorySaver:
        """
        Create a memory instance based on the given memory type.

        Args:
            memory_type (str): The type of memory to create.
                Defaults to the value of `aiosettings.llm_memory_type`.

        Returns:
            MemorySaver: The created memory instance.

        Raises:
            ValueError: If an unsupported memory type is provided.
        """
        if memory_type == "memorysaver":
            return MemorySaver()
        # elif memory_type == "redis":
        #     return RedisChatMessageHistory()

        # Add more models as needed
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")
