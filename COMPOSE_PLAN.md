Using the design patterns discussed in @tutorials help me come up the proper creational, structural, and behavioral patterns needed for a discord based LLM chatbot project that leverages langchain and langgraph. I want to have an abstration layer that allows me to interchange things like llm models, embedding models, vector stores, document loaders, embedding models, and key-value stores on the fly. Some of the functions/methods being using need async versions that allow better concurrency. As a result you might also need to consider concurrency patterns and event driven design patterns as well. Here are some example topics for both:

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



Here are some more specifications.

Chat Framework:
- I am using Rapptz/discord.py as my discord client

LLM Architecture:
- Must use langchain along with langgraph. Leverage LCEL when creating chains.

LLM Components:
- Chat Models: Openai or Anthropic models for chat. grab the model name from aiosettings.vision_model and aiosettings.chat_model.
- Embedding Models: The bot should be able to use Fake or OpenAIEmbeddings. grab the model name from aiosettings.llm_embedding_model_name.
- Document Loaders: The app should support the following document loaders
  - Web
  - Sitemap
  - RecursiveURL
  - PyPDF
  - PyMuPDF
  - DirectoryLoader
  - CSVLoader
  - JsonLoader

- Vector Stores: The app should support the following vector stores
  - Chroma
  - Faiss
  - InMemoryVectorStore
  - PGVector
  - Pinecone
  - PineconeVectorStore
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
