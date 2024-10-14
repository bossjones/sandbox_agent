# Decision Registry

## 2024-03-18: Initial Project Structure and Implementation

______________________________________________________________________

### Decision Summary

Established the initial project structure and implemented core components for a Discord-based LLM chatbot using
Langchain and Langgraph.

### Context

The project aims to create a flexible and extensible Discord chatbot that leverages large language models (LLMs) and
various AI components. The structure needs to support interchangeability of components like LLM models, embedding
models, vector stores, and more.

### Details

1. **Project Structure**

    - Created a modular structure with separate directories for different components (AI, clients, factories, services,
        etc.).
    - Implemented factory patterns for easy creation and swapping of components.

1. **AI Components**

    - Implemented factories for chat models, embedding models, document loaders, vector stores, retrievers, tools, and
        key-value stores.
    - Each factory supports multiple implementations and can be easily extended.

1. **Clients**

    - Created client classes for Discord, Redis, PostgreSQL, and SQLite.
    - Clients are configured using settings from `aio_settings.py`.

1. **Configuration**

    - Utilized `aio_settings.py` for centralized configuration management.
    - Created a `config` module to expose commonly used settings.

1. **Constants and Exceptions**

    - Defined project-wide constants in `constants.py`.
    - Created custom exceptions for better error handling.

1. **Main Bot Implementation**

    - Implemented `SandboxAgent` class extending `DiscordClient`.
    - Set up basic command structure and message handling.

1. **Factories and Services**

    - Established a factory pattern for creating various components.
    - Prepared a structure for implementing service classes.

1. **Asynchronous Design**

    - Implemented asynchronous patterns where appropriate, especially in the Discord client and main bot implementation.

### Consequences

- The modular structure allows for easy extension and modification of individual components.
- Factory patterns enable simple swapping of implementations (e.g., changing LLM models or vector stores).
- Centralized configuration through `aio_settings.py` simplifies management of environment-specific settings.
- The structure supports future implementation of more complex AI workflows using Langchain and Langgraph.

### Open Questions

- Specific implementation details for AI workflows and chat functionalities need to be developed.
- Integration with Langgraph for more complex conversational flows is yet to be implemented.
- Testing strategy and continuous integration setup need to be defined.

### Next Steps

1. Implement core chat functionality using the established structure.
1. Develop more advanced AI workflows using Langchain and Langgraph.
1. Set up a comprehensive testing suite.
1. Implement logging and monitoring solutions.
1. Develop documentation for the project structure and usage.
