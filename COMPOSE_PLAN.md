Using the design patterns discussed in @tutorials help me come up the proper creational, structural, and behavioral
patterns needed for a discord based LLM chatbot project that leverages langchain and langgraph. I want to have an
abstration layer that allows me to interchange things like llm models, embedding models, vector stores, document
loaders, embedding models, and key-value stores on the fly. Some of the functions/methods being using need async
versions that allow better concurrency. As a result you might also need to consider concurrency patterns and event
driven design patterns as well. Here are some example topics for both:

Concurrency Patterns:

- Producer-Consumer Pattern
- Future-Promise Pattern
- Readers-Writers Pattern
- Leader-Follower Pattern
- Thread Pool Pattern
- Half-Sync/Half-Async Pattern
- Monitor Object Pattern
- Scheduler Pattern
- Active Object Pattern
- Balking Pattern
- Double-Checked Locking Pattern
- Guarded Suspension Pattern
- Thread-Specific Storage Pattern

Event driven and Network design patterns:

Event-Driven Patterns:

- Reactor Pattern
- Event Queue Pattern
- Event Aggregator Pattern
- Event Sourcing Pattern
- Event-driven pattern comparison

Network Design Patterns:

- Client-Server Pattern
- Peer-to-Peer (P2P) Pattern
- Broker Pattern
- Request-Reply Pattern
- Publisher-Subscriber Pattern
- Layered (Tiered) Pattern
- Retry Pattern
- Circuit Breaker Pattern
- Throttling Pattern
- Load Balancer Pattern

Here are some more specifications:

Chat Framework:

- I am using Rapptz/discord.py as my discord client

LLM Architecture:

- Must use langchain along with langgraph. Leverage LCEL when creating chains.

LLM Components:

- Chat Models: Openai or Anthropic models for chat. grab the model name from aiosettings.vision_model and
    aiosettings.chat_model.

- Embedding Models: The bot should be able to use Fake or OpenAIEmbeddings. grab the model name from
    aiosettings.llm_embedding_model_name.

- Document Loaders: The app should support the following document loaders

    - PyPDF
    - PyMuPDF
    - DirectoryLoader

- Vector Stores: The app should support the following vector stores

    - Chroma
    - Faiss
    - InMemoryVectorStore
    - PGVector
    - Redis

- Retrievers: The app should be able to use something like an adaptor to change between vector store and retriever.

- Tools: Should be able to use the following tools

    - DuckDuckgoSearch
    - Brave

- Key-Value Stores:

    - RedisStore
    - InMemoryByteStore
    - LocalFileStore

- Message History:

    - Redis Chat Message History

Common libaries and clients: The app should also support common client usage for the following technologies

- Redis
- Postgres
- SQLite
- Discord

Custom Tools To Implement:

- 'get_current_time': Returns the current time.
- 'get_random_number': Returns a random number between 1 and 100.
- 'open_browser': Opens a browser tab with the best-fitting URL based on the user's prompt.
- 'generate_diagram': Generates mermaid diagrams based on the user's prompt.
- 'runnable_code_check': Checks if the code in the specified file is runnable and provides necessary changes if not.
- 'run_python': Executes a Python script from the 'scratch_pad_dir' based on the user's prompt and returns the output.
- 'create_file': Generates content for a new file based on the user's prompt and file name.
- 'update_file': Updates a file based on the user's prompt.
- 'delete_file': Deletes a file based on the user's prompt.
- 'discuss_file': Discusses a file's content based on the user's prompt, considering the current memory content.
- 'read_file_into_memory': Reads a file from the scratch_pad_dir and saves its content into memory based on the user's
    prompt.
- 'read_dir_into_memory': Reads all files from the scratch_pad_dir and saves their content into memory.
- 'clipboard_to_memory': Copies the content from the clipboard to memory.
- 'add_to_memory': Adds a key-value pair to memory.
- 'remove_variable_from_memory': Removes a variable from memory based on the user's prompt.
- 'reset_active_memory': Resets the active memory to an empty dictionary.
- 'scrap_to_file_from_clipboard': Gets a URL from the clipboard, scrapes its content, and saves it to a file in the
    scratch_pad_dir.
