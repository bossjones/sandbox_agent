Using the design patterns discussed in @tutorials help me come up the proper creational, structural, and behavioral patterns needed for a llm chatbot project that leverages langchain and langgraph. I want to have an abstration layer that allows me to interchange things like llm models, embedding models, vector stores and retrivals on the fly

expand up the folder structure, and give me an example of what should be in factories.llm_factory

Final Context in Folders
Here's an expanded folder structure with an example of what should be in factories/llm_factory.py:

```
src/
├── sandbox_agent/
│   ├── __init__.py
│   ├── aio_settings.py
│   ├── llm_manager.py
│   ├── factories/
│   │   ├── __init__.py
│   │   ├── llm_factory.py
│   │   ├── embedding_factory.py
│   │   ├── vector_store_factory.py
│   │   └── retrieval_factory.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── embedding.py
│   │   ├── vector_store.py
│   │   └── retrieval.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── retrieval_strategy.py
│   │   ├── semantic_retrieval.py
│   │   └── keyword_retrieval.py
│   └── facades/
│       ├── __init__.py
│       └── chatbot_facade.py
└── tests/
    ├── __init__.py
    ├── test_llm_factory.py
    ├── test_embedding_factory.py
    ├── test_vector_store_factory.py
    ├── test_retrieval_factory.py
    ├── test_retrieval_strategy.py
    └── test_chatbot_facade.py
```

Here's an example of what factories/llm_factory.py might contain:

```python
from abc import ABC, abstractmethod
from typing import Type

from sandbox_agent.models.llm import LLM


class LLMFactory(ABC):
    """
    Abstract factory for creating LLM instances.
    """

    @abstractmethod
    def create_llm(self) -> LLM:
        """
        Create an LLM instance.

        Returns:
            LLM: The created LLM instance.
        """
        pass


class OpenAILLMFactory(LLMFactory):
    """
    Concrete factory for creating OpenAI LLM instances.
    """

    def create_llm(self) -> LLM:
        """
        Create an OpenAI LLM instance.

        Returns:
            LLM: The created OpenAI LLM instance.
        """
        # Implementation for creating an OpenAI LLM instance
        # ...


class AnthropicLLMFactory(LLMFactory):
    """
    Concrete factory for creating Anthropic LLM instances.
    """

    def create_llm(self) -> LLM:
        """
        Create an Anthropic LLM instance.

        Returns:
            LLM: The created Anthropic LLM instance.
        """
        # Implementation for creating an Anthropic LLM instance
        # ...


def get_llm_factory(factory_type: str) -> LLMFactory:
    """
    Get the appropriate LLM factory based on the factory type.

    Args:
        factory_type (str): The type of LLM factory to get.

    Returns:
        LLMFactory: The LLM factory instance.

    Raises:
        ValueError: If an invalid factory type is provided.
    """
    factories = {
        "openai": OpenAILLMFactory,
        "anthropic": AnthropicLLMFactory,
    }

    if factory_type not in factories:
        raise ValueError(f"Invalid LLM factory type: {factory_type}")

    return factories[factory_type]()
```

In this example, the LLMFactory is an abstract base class that defines the interface for creating LLM instances. The OpenAILLMFactory and AnthropicLLMFactory are concrete implementations of the LLMFactory that create LLM instances specific to OpenAI and Anthropic, respectively.
The get_llm_factory function is a utility function that returns the appropriate LLM factory based on the provided factory_type. It raises a ValueError if an invalid factory type is provided.
Note that the actual implementation details for creating the LLM instances are not shown here and would need to be filled in based on the specific requirements and libraries used in your project.
