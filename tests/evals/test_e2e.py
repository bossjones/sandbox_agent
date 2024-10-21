# #!/usr/bin/env python3
# FIXME: Enable this and break it up into different components
# # NOTE: THIS IS THE END ALL FILE FOR ALL OF THIS STUFF, DONT WORRY ABOUT THE NOTEBOOKS ETC, WORRY ABOUT THIS
# # NOTE: https://python.langchain.com/v0.2/docs/how_to/qa_sources/
# # pyright: ignore[reportAttributeAccessIssue]
# from __future__ import annotations

# import os
# import sys
# import traceback
# import typing

# from typing import List

# import bpdb
# import dotenv
# import rich


# os.environ["USER_AGENT"] =  (
#     f"boss-rag-techniques/0.1.0 | Python/" f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
# )

# from langchain_core.documents import Document
# from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
# from loguru import logger as LOGGER


# # Reload the variables in your '.env' file (override the existing variables)
# dotenv.load_dotenv("../.env", override=True)


# import logging
# import os
# import sys

# from dotenv import load_dotenv
# from rich.console import Console
# from rich.markdown import Markdown


# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path since we work with notebooks
# import logging
# import os
# import sys
# import traceback
# import typing

# from importlib.metadata import version
# from typing import Any, Callable, Dict, List, Optional, Union

# import bpdb
# import bs4
# import dotenv
# import langsmith
# import openai
# import pysnooper
# import rich

# from langchain import hub
# from langchain.agents import AgentExecutor
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.question_answering import load_qa_chain
# from pydantic import BaseModel, Field
# from langchain_anthropic import ChatAnthropic
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.documents import Document
# from langchain_core.messages import AIMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# from pydantic import BaseModel, Field
# from langchain_core.runnables import (
#     ConfigurableField,
#     Runnable,
#     RunnableBranch,
#     RunnableLambda,
#     RunnableMap,
#     RunnablePassthrough,
# )
# from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langsmith import traceable
# from langsmith.evaluation import evaluate_comparative
# from langsmith.run_trees import RunTree
# from langsmith.schemas import Example, Run
# from langsmith.wrappers import wrap_openai
# from loguru import logger
# from loguru import logger as LOGGER
# from loguru._defaults import LOGURU_FORMAT
# from openai import OpenAI
# from openai.types.chat.chat_completion import ChatCompletion
# from rich.panel import Panel

# from evaluation.evalute_rag import *
# from helper_functions import *
# from logging_utils import get_logger, global_log_config


# global_log_config(
#     log_level=logging.getLevelName("DEBUG"),
#     json=False,
# )
# LOGGER.disable("ipykernel.")
# LOGGER.disable("ipykernel.kernelbase")
# LOGGER.disable("openai._base_client")
# LOGGER.disable("httpcore._trace")
# LOGGER.disable("langsmith.client:_serialize_json")

# import warnings


# # SOURCE: https://github.com/taikinman/langrila/blob/main/src/langrila/openai/model_config.py
# # SOURCE: https://github.com/taikinman/langrila/blob/main/src/langrila/openai/model_config.py
# # TODO: Look at this https://github.com/h2oai/h2ogpt/blob/542543dc23aa9eb7d4ce7fe6b9af1204a047b50f/src/enums.py#L386 and see if we can add some more models
# _TOKENS_PER_TILE = 170
# _TILE_SIZE = 512

# _OLDER_MODEL_CONFIG = {
#     "gpt-4-0613": {
#         "max_tokens": 8192,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.00003,
#         "completion_cost_per_token": 0.00006,
#     },
#     "gpt-4-32k-0314": {
#         "max_tokens": 32768,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.00006,
#         "completion_cost_per_token": 0.00012,
#     },
#     "gpt-4-32k-0613": {
#         "max_tokens": 32768,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.00006,
#         "completion_cost_per_token": 0.00012,
#     },
#     "gpt-3.5-turbo-0301": {
#         "max_tokens": 4096,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.0000015,
#         "completion_cost_per_token": 0.000002,
#     },
#     "gpt-3.5-turbo-0613": {
#         "max_tokens": 4096,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.0000015,
#         "completion_cost_per_token": 0.000002,
#     },
#     "gpt-3.5-turbo-16k-0613": {
#         "max_tokens": 16384,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.000003,
#         "completion_cost_per_token": 0.000004,
#     },
#     "gpt-3.5-turbo-instruct": {
#         "max_tokens": 4096,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.0000015,
#         "completion_cost_per_token": 0.000002,
#     },
# }

# # "claude-instant-1.2": 100000,
# # "claude-3-opus-20240229": 200000,
# # "claude-3-sonnet-20240229": 200000,
# # "claude-3-5-sonnet-20240620": 200000,
# # "claude-3-haiku-20240307": 200000,

# _NEWER_MODEL_CONFIG = {
#     "claude-3-5-sonnet-20240620": {
#         "max_tokens": 2048,
#         "max_output_tokens": 16384,
#         "prompt_cost_per_token": 0.0000025,
#         "completion_cost_per_token": 0.00001,
#     },
#     "claude-3-opus-20240229": {
#         "max_tokens": 2048,
#         "max_output_tokens": 16384,
#         "prompt_cost_per_token": 0.0000025,
#         "completion_cost_per_token": 0.00001,
#     },
#     "claude-3-sonnet-20240229": {
#         "max_tokens": 2048,
#         "max_output_tokens": 16384,
#         "prompt_cost_per_token": 0.0000025,
#         "completion_cost_per_token": 0.00001,
#     },
#     "claude-3-haiku-20240307": {
#         "max_tokens": 2048,
#         "max_output_tokens": 16384,
#         "prompt_cost_per_token": 0.0000025,
#         "completion_cost_per_token": 0.00001,
#     },
#     "gpt-4o-2024-08-06": {
#         "max_tokens": 128000,
#         "max_output_tokens": 16384,
#         "prompt_cost_per_token": 0.0000025,
#         "completion_cost_per_token": 0.00001,
#     },
#     "gpt-4o-mini-2024-07-18": {
#         # "max_tokens": 128000,
#         "max_tokens": 900,
#         "max_output_tokens": 16384,
#         "prompt_cost_per_token": 0.000000150,
#         "completion_cost_per_token": 0.00000060,
#     },
#     "gpt-4o-2024-05-13": {
#         "max_tokens": 128000,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.000005,
#         "completion_cost_per_token": 0.000015,
#     },
#     "gpt-4-turbo-2024-04-09": {
#         "max_tokens": 128000,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.00001,
#         "completion_cost_per_token": 0.00003,
#     },
#     "gpt-4-0125-preview": {
#         "max_tokens": 128000,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.00001,
#         "completion_cost_per_token": 0.00003,
#     },
#     "gpt-4-1106-preview": {
#         "max_tokens": 128000,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.00001,
#         "completion_cost_per_token": 0.00003,
#     },
#     "gpt-4-vision-preview": {
#         "max_tokens": 128000,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.00001,
#         "completion_cost_per_token": 0.00003,
#     },
#     "gpt-3.5-turbo-0125": {
#         "max_tokens": 16384,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.0000005,
#         "completion_cost_per_token": 0.0000015,
#     },
#     "gpt-3.5-turbo-1106": {
#         "max_tokens": 16384,
#         "max_output_tokens": 4096,
#         "prompt_cost_per_token": 0.000001,
#         "completion_cost_per_token": 0.000002,
#     },
# }

# _NEWER_EMBEDDING_CONFIG = {
#     "text-embedding-3-small": {
#         "max_tokens": 8192,
#         "prompt_cost_per_token": 0.00000002,
#     },
#     "text-embedding-3-large": {
#         "max_tokens": 8192,
#         "prompt_cost_per_token": 0.00000013,
#     },
# }

# _OLDER_EMBEDDING_CONFIG = {
#     "text-embedding-ada-002": {
#         "max_tokens": 8192,
#         "prompt_cost_per_token": 0.0000001,
#     },
# }


# EMBEDDING_CONFIG = {}
# EMBEDDING_CONFIG.update(_OLDER_EMBEDDING_CONFIG)
# EMBEDDING_CONFIG.update(_NEWER_EMBEDDING_CONFIG)

# MODEL_CONFIG = {}
# MODEL_CONFIG.update(_OLDER_MODEL_CONFIG)
# MODEL_CONFIG.update(_NEWER_MODEL_CONFIG)

# MODEL_POINT = {
#     "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
#     "gpt-4o": "gpt-4o-2024-08-06",
#     "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
#     "gpt-4": "gpt-4-0613",
#     "gpt-4-32k": "gpt-4-32k-0613",
#     "gpt-4-vision": "gpt-4-vision-preview",
#     "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
#     "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k-0613",
#     "claude-3-opus": "claude-3-opus-20240229",
#     "claude-3-sonnet": "claude-3-sonnet-20240229",
#     "claude-3-haiku": "claude-3-haiku-20240307",
#     "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
# }

# _MODEL_POINT_CONFIG = {
#     "gpt-4o-mini": MODEL_CONFIG[MODEL_POINT["gpt-4o-mini"]],
#     "gpt-4o": MODEL_CONFIG[MODEL_POINT["gpt-4o"]],
#     "gpt-4-turbo": MODEL_CONFIG[MODEL_POINT["gpt-4-turbo"]],
#     "gpt-4": MODEL_CONFIG[MODEL_POINT["gpt-4"]],
#     "gpt-4-32k": MODEL_CONFIG[MODEL_POINT["gpt-4-32k"]],
#     "gpt-4-vision": MODEL_CONFIG[MODEL_POINT["gpt-4-vision"]],
#     "gpt-3.5-turbo": MODEL_CONFIG[MODEL_POINT["gpt-3.5-turbo"]],
#     "gpt-3.5-turbo-16k": MODEL_CONFIG[MODEL_POINT["gpt-3.5-turbo-16k"]],
#     "claude-3-opus": MODEL_CONFIG[MODEL_POINT["claude-3-opus"]],
#     "claude-3-5-sonnet": MODEL_CONFIG[MODEL_POINT["claude-3-5-sonnet"]],
#     "claude-3-sonnet": MODEL_CONFIG[MODEL_POINT["claude-3-sonnet"]],
#     "claude-3-haiku": MODEL_CONFIG[MODEL_POINT["claude-3-haiku"]],
# }

# # contains all the models and embeddings info
# MODEL_CONFIG.update(_MODEL_POINT_CONFIG)

# # produces a list of all models and embeddings available
# MODEL_ZOO = set(MODEL_CONFIG.keys()) | set(EMBEDDING_CONFIG.keys())

# # SOURCE: https://github.com/JuliusHenke/autopentest/blob/ca822f723a356ec974d2dff332c2d92389a4c5e3/src/text_embeddings.py#L19
# # https://platform.openai.com/docs/guides/embeddings/embedding-models
# EMBEDDING_MODEL_DIMENSIONS_DATA = {
#     "text-embedding-ada-002": 1536,
#     "text-embedding-3-small": 512,
#     "text-embedding-3-large": 1024,
# }


# EVAL_MAX_CONCURRENCY = 4
# LLM_MODEL_NAME = "gpt-4o-mini"
# # LLM_MODEL_NAME = "claude-3-5-sonnet"
# # PROVIDER = "anthropic" # "openai" or "anthropic"
# PROVIDER = "openai" # "openai" or "anthropic"
# CHUNK_SIZE: int=1000
# CHUNK_OVERLAP: int = 200
# ADD_START_INDEX: bool = False
# # LLM_EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"
# LLM_EMBEDDING_MODEL_NAME: str = "text-embedding-3-large"
# DEFAULT_SEARCH_KWARGS: dict[str, int]={"k": 2}
# QUESTION_TO_ASK = "What is the main cause of climate change?"
# DATASET_NAME = "Climate Change Q&A"


# MAX_TOKENS: int = MODEL_CONFIG[LLM_MODEL_NAME]["max_tokens"]
# MAX_OUTPUT_TOKENS: int = MODEL_CONFIG[LLM_MODEL_NAME]["max_output_tokens"]
# PROMPT_COST_PER_TOKEN: float = MODEL_CONFIG[LLM_MODEL_NAME]["prompt_cost_per_token"]
# COMPLETION_COST_PER_TOKEN: float = MODEL_CONFIG[LLM_MODEL_NAME]["completion_cost_per_token"]
# MAX_RETRIES: int = 9

# rich.print(MAX_TOKENS)
# rich.print(MAX_OUTPUT_TOKENS)
# rich.print(PROMPT_COST_PER_TOKEN)
# rich.print(COMPLETION_COST_PER_TOKEN)

# EMBEDDING_MAX_TOKENS: int = EMBEDDING_MODEL_DIMENSIONS_DATA[LLM_EMBEDDING_MODEL_NAME]
# EMBEDDING_MODEL_DIMENSIONS: int = EMBEDDING_MODEL_DIMENSIONS_DATA[LLM_EMBEDDING_MODEL_NAME]

# # USE THIS TO ENABLE OR DISABLE CERTAIN EVALUATIONS
# COMPARE_MODELS_FEATURE_FLAG = False
# RAG_ANSWER_V_REFERENCE_FEATURE_FLAG = False
# HELPFULNESS_FEATURE_FLAG = False
# RAG_ANSWER_HALLUCINATION_FEATURE_FLAG = False
# RAG_DOC_RELEVANCE_FEATURE_FLAG = False
# RAG_DOC_RELEVANCE_AND_HALLUCINATION_FEATURE_FLAG = False
# RAG_ANSWER_ACCURACY_FEATURE_FLAG = True
# HELPFULNESS_TESTING_FEATURE_FLAG = False
# RAG_STRING_EMBEDDING_DISTANCE_METRICS_FEATURE_FLAG = False

# def get_model_config(model_name: str = LLM_MODEL_NAME, embedding_model_name: str = LLM_EMBEDDING_MODEL_NAME) -> dict[str, Any]:
#     """
#     Get the configuration for a specific model and embedding model.

#     Args:
#         model_name: The name of the model.
#         embedding_model_name: The name of the embedding model.

#     Returns:
#         A dictionary containing the model configuration.
#     """
#     return {
#         "model_name": model_name,
#         "embedding_model_name": embedding_model_name,
#         "max_tokens": MODEL_CONFIG[model_name]["max_tokens"],
#         "max_output_tokens": MODEL_CONFIG[model_name]["max_output_tokens"],
#         "prompt_cost_per_token": MODEL_CONFIG[model_name]["prompt_cost_per_token"],
#         "completion_cost_per_token": MODEL_CONFIG[model_name]["completion_cost_per_token"],
#         "embedding_max_tokens": EMBEDDING_CONFIG[embedding_model_name]["max_tokens"],
#         "embedding_model_dimensions": EMBEDDING_MODEL_DIMENSIONS_DATA[embedding_model_name],
#     }

# def print_model_announcment(model: str, provider: str, retriever: Any, llm: Any) -> None:
#     """
#     Print an announcement about the initialized RagBot.

#     Args:
#         model: The name of the model.
#         provider: The provider of the model.
#         retriever: The retriever object.
#         llm: The language model object.
#     """
#     update_text = f"\n\n[green]RagBot initialized with model:[/green] [yellow]{model}[/yellow]\n[green]provider:[/green] [yellow]{provider}[/yellow]\n[green]retriever:[/green] [yellow]{retriever}[/yellow]\n[green]llm:[/green] [yellow]{llm}[/yellow]\n\n"
#     console = Console()
#     panel_contents = Panel(update_text)
#     console.print()
#     console.print(panel_contents)
#     console.print()

# LLM_RUN_CONFIG = get_model_config()
# rich.print(LLM_RUN_CONFIG)

# path = "../data/Understanding_Climate_Change.pdf"


# # ### Encode document
# from langsmith.evaluation import EvaluationResults, LangChainStringEvaluator, evaluate  # noqa: I001

# def encode_pdf(path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, add_start_index: bool = ADD_START_INDEX, llm_embedding_model_name: str = LLM_EMBEDDING_MODEL_NAME) -> FAISS:
#     """
#     Encodes a PDF book into a vector store using OpenAI embeddings.

#     Args:
#         path: The path to the PDF file.
#         chunk_size: The desired size of each text chunk.
#         chunk_overlap: The amount of overlap between consecutive chunks.
#         add_start_index: Whether to add start index to the chunks.
#         llm_embedding_model_name: The name of the embedding model to use.

#     Returns:
#         A FAISS vector store containing the encoded book content.
#     """

#     # Load PDF documents
#     loader = PyPDFLoader(path)
#     documents = loader.load()

#     LOGGER.debug(f"Length of documents: {len(documents)}")

#     # TODO: Check if we should enable add_start_index
#     # In this case we'll split our documents into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it. We use the RecursiveCharacterTextSplitter, which will recursively split the document using common separators like new lines until each chunk is the appropriate size. This is the recommended text splitter for generic text use cases.

#     # We set add_start_index=True so that the character index at which each split Document starts within the initial Document is preserved as metadata attribute "start_index".
#     # FIXME: See https://python.langchain.com/v0.2/docs/tutorials/rag/


#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=add_start_index
#     )
#     texts = text_splitter.split_documents(documents)
#     LOGGER.debug(f"Num of chunks: {len(texts)}")
#     cleaned_texts = replace_t_with_space(texts)

#     # Create embeddings and vector store
#     embeddings = OpenAIEmbeddings(model=llm_embedding_model_name, dimensions=EMBEDDING_MODEL_DIMENSIONS)
#     vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

#     return vectorstore


# chunks_vector_store = encode_pdf(path, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP, add_start_index = ADD_START_INDEX, llm_embedding_model_name = LLM_EMBEDDING_MODEL_NAME)


# # ### Create retriever
# chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs=DEFAULT_SEARCH_KWARGS)


# # ### Test retriever
# test_query = QUESTION_TO_ASK
# context = retrieve_context_per_question(test_query, chunks_query_retriever)
# show_context(context)

# rag_prompt = hub.pull("rlm/rag-prompt")

# # alias retriever
# retriever = chunks_query_retriever


# class RAGResponse(BaseModel):
#     """
#     A class representing the response from a RAG (Retrieval-Augmented Generation) system.

#     Attributes:
#         question: The input question.
#         context: A list of relevant documents.
#         answer: The generated answer.
#     """
#     question: str
#     context: list[Document]
#     answer: str

# def format_docs(docs: list[Document]) -> str:
#     """
#     Format a list of documents into a single string.

#     Args:
#         docs: A list of Document objects.

#     Returns:
#         A string containing the formatted content of all documents.
#     """
#     return "\n\n".join(doc.page_content for doc in docs)

# def get_all_prompts_in_chain(chain: Any) -> None:
#     """
#     Print all the prompts in the given chain.

#     Args:
#         chain: The chain object containing prompts.
#     """
#     # SOURCE: https://python.langchain.com/v0.2/docs/how_to/lcel_cheatsheet/
#     for i, prompt in enumerate(chain.get_prompts()):
#         rich.print(f"**prompt {i=}**\n")
#         rich.print(prompt.pretty_repr())
#         rich.print("\n" * 3)


# class RagBot:
#     """
#     A class representing a RAG (Retrieval-Augmented Generation) bot.

#     Attributes:
#         _provider: The provider of the language model.
#         _retriever: The retriever object for fetching relevant documents.
#         _client: The wrapped OpenAI client.
#         _model: The name of the language model.
#         _llm: The language model object.
#     """
#     def __init__(self, retriever: Any, model: str = LLM_MODEL_NAME, provider: str = PROVIDER):
#         """
#         Initialize the RagBot.

#         Args:
#             retriever: The retriever object for fetching relevant documents.
#             model: The name of the language model to use.
#             provider: The provider of the language model.
#         """
#         self._provider = provider
#         self._retriever = retriever
#         # Wrapping the client instruments the LLM
#         self._client = wrap_openai(openai.Client())
#         self._model = model
#         if self._provider == "openai":
#             self._llm = ChatOpenAI(
#                 name="ChatOpenAI",
#                 streaming=True,
#                 model=self._model,
#                 max_retries=9,
#                 # max_tokens=900,
#                 # max_tokens=MAX_TOKENS,
#                 temperature=0.0,
#             )
#         elif self._provider == "anthropic":
#             self._llm = ChatAnthropic(
#                 name="ChatAnthropic",
#                 # streaming=True,
#                 model=self._model,
#                 max_retries=9,
#                 temperature=0.0,
#             )
#         LOGGER.info(f"\n\nRagBot initialized with model: {self._model}, provider: {self._provider}, retriever: {self._retriever} and llm: {self._llm}\n\n")
#         print_model_announcment(self._model, self._provider, self._retriever, self._llm)

#     @traceable()
#     def execute_chain(self, question: str) -> dict[str, Any]:
#         """
#         Execute the RAG chain for a given question.

#         Args:
#             question: The input question.

#         Returns:
#             A dictionary containing the response from the chain.
#         """
#         chain = self.build_chain(question)
#         return chain.invoke({"question": question})

#     @traceable()
#     def display_chain(self, question: str) -> None:
#         """
#         Display the ASCII representation of the chain graph.

#         Args:
#             question: The input question.
#         """
#         chain = self.build_chain(question)
#         chain.get_graph().print_ascii()

#     @traceable()
#     async def aexecute_chain(self, question: str) -> dict[str, Any]:
#         """
#         Asynchronously execute the RAG chain for a given question.

#         Args:
#             question: The input question.

#         Returns:
#             A dictionary containing the response from the chain.
#         """
#         chain = self.build_chain(question)
#         return await chain.ainvoke({"question": question})

#     @traceable()
#     def stream_chain(self, question: str) -> str:
#         """
#         Stream the output of the RAG chain for a given question.

#         Args:
#             question: The input question.

#         Returns:
#             The concatenated output of the chain.
#         """
#         chain = self.build_chain(question)
#         chunks = []
#         for chunk in chain.stream({"question": question}):
#             rich.print(chunk.content, flush=True)
#             chunks.append(chunk.content)
#         return "".join(chunks)

#     @traceable()
#     async def astream_chain(self, question: str) -> str:
#         """
#         Asynchronously stream the output of the RAG chain for a given question.

#         Args:
#             question: The input question.

#         Returns:
#             The concatenated output of the chain.
#         """
#         chain = self.build_chain(question)
#         chunks = []
#         async for chunk in chain.astream({"question": question}):
#             chunks.append(chunk)
#             rich.print(chunk.content, flush=True)
#         return "".join(chunks)


#     # @pysnooper.snoop()
#     @traceable()
#     def build_chain(self, question: str):
#         # NOTE: https://python.langchain.com/v0.2/docs/tutorials/rag/
#         # NOTE: Look at this for inspiration
#         """
#         Build the RAG chain for a given question.

#         Args:
#             question: The input question.

#         Returns:
#             The built chain object.
#         """
#         if self._provider == "openai":
#             LOGGER.info("OpenAI selected")
#             system_template = (
#                 "You are a helpful AI code assistant with expertise in climate change. "
#                 "Use the following docs to produce a concise code solution to the user question.\n\n"
#                 "## Docs\n\n{context}"
#             )
#             # READ THIS: https://python.langchain.com/v0.2/docs/how_to/qa_sources/

#             # FIXME: I think this should be moved to the system template and human should only have a single variable eg {question}.
#             base_human_template = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nQuestion: {question} \n\nContext: {context} \n\nAnswer:"

#             base_prompt = ChatPromptTemplate.from_messages([
#             ("system", system_template),
#             ("human", base_human_template),
#             ])

#             LOGGER.debug(base_prompt)

#             rag_chain_from_docs = (
#                 {
#                     "question": lambda x: x["question"], # input query. RunnablePassthrough() just passes the input through, ie the input query/question.
#                     "context": lambda x: format_docs(x["context"]), # context
#                 }
#                 | base_prompt   # format query and context into prompt
#                 | self._llm # generate response
#                 | StrOutputParser()  # coerce to string
#             )
#             # Pass input query to retriever
#             retrieve_docs = (lambda x: x["question"]) | self._retriever

#             # Below, we chain `.assign` calls. This takes a dict and successively
#             # adds keys-- "context" and "answer"-- where the value for each key
#             # is determined by a Runnable. The Runnable operates on all existing
#             # keys in the dict.
#             chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
#                 answer=rag_chain_from_docs
#             )


#         elif self._provider == "anthropic":
#             LOGGER.info("Anthropic selected")
#             system_template = (
#                 "You are a helpful AI code assistant with expertise in climate change. Use the following docs to produce a concise code solution to the user question.\n\n## Docs\n\n{context}"
#             )

#             base_human_template = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nQuestion: {question} \n\nContext: {context} \n\nAnswer:"


#             base_prompt = ChatPromptTemplate.from_messages(
#                 [
#                     (
#                         "system",
#                         system_template,
#                     ),
#                     ("human", base_human_template),
#                 ]
#             )

#             LOGGER.debug(base_prompt)


#             rag_chain_from_docs = (
#                 {
#                     "question": lambda x: x["question"], # input query. RunnablePassthrough() just passes the input through, ie the input query/question.
#                     "context": lambda x: format_docs(x["context"]), # context
#                 }
#                 | base_prompt   # format query and context into prompt
#                 | self._llm # generate response
#                 | StrOutputParser()  # coerce to string
#             )
#             # Pass input query to retriever
#             retrieve_docs = (lambda x: x["question"]) | self._retriever

#             # Below, we chain `.assign` calls. This takes a dict and successively
#             # adds keys-- "context" and "answer"-- where the value for each key
#             # is determined by a Runnable. The Runnable operates on all existing
#             # keys in the dict.
#             chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
#                 answer=rag_chain_from_docs
#             )

#             # chain.get_graph().print_ascii()

#         return chain

#     # @traceable()
#     # def get_answer(self, question: str):
#     #     docs = self.retrieve_docs(question)
#     #     return self.invoke_llm(question, docs)

#     @traceable()
#     def get_answer(self, question: str):
#         return self.execute_chain(question)

# """
# # Example response from get_answer()
# {
#     'question': 'What is the main cause of climate change?',
#     'context': [
#         Document(
#             metadata={'source': '../data/Understanding_Climate_Change.pdf', 'page': 0},
#             page_content='driven by human activities, particularly the emission of greenhou se gases.  \nChapter 2: Causes of Climate Change  \nGreenhouse Gases  \nThe primary cause of recent climate change is the increase in greenhouse gases in the \natmosphere.
# Greenhouse gases, such as carbon dioxide (CO2), methane (CH4), and nitrous \noxide (N2O), trap heat from the sun, creating a "greenhouse effect." This effect is  essential \nfor life on Earth, as it keeps the planet warm enough to support life. However, human \nactivities
# have intensified this natural process, leading to a warmer climate.  \nFossil Fuels  \nBurning fossil fuels for energy releases large amounts of CO2. This includes coal, oil, and \nnatural gas used for electricity, heating, and transportation. The industrial revolution
# marked \nthe beginning of a significant increase in fossil fuel consumption, which continues to rise \ntoday.  \nCoal'
#         ),
#         Document(
#             metadata={'source': '../data/Understanding_Climate_Change.pdf', 'page': 0},
#             page_content="Most of these climate changes are attributed to very small variations in Earth's orbit that \nchange the amount of solar energy our planet receives. During the Holocene epoch, which \nbegan at the end of the last ice age, human societies f
# lourished, but the industrial era has seen \nunprecedented changes.  \nModern Observations  \nModern scientific observations indicate a rapid increase in global temperatures, sea levels, \nand extreme weather events. The Intergovernmental Panel on Climate Change (IPCC)
# has \ndocumented these changes extensively. Ice core samples, tree rings, and ocean sediments \nprovide a historical record that scientists use to understand past climate conditions and \npredict future trends. The evidence overwhelmingly shows that recent changes are
# primarily \ndriven by human activities, particularly the emission of greenhou se gases.  \nChapter 2: Causes of Climate Change  \nGreenhouse Gases"
#         )
#     ],
#     'answer': 'The main cause of climate change is the increase in greenhouse gases in the atmosphere, primarily driven by human activities. These gases, such as carbon dioxide (CO2), trap heat from the sun, intensifying the natural greenhouse effect. The burning of
# fossil fuels for energy significantly contributes to this increase.'
# }
# """

# rag_bot = RagBot(retriever)


# # bot smoke test
# response = rag_bot.execute_chain(test_query)
# # rich.print(response)


# # # Let's explore the chain a little bit
# bot_chain = rag_bot.build_chain(test_query)


# # # Define a function that will:
# #
# # 1. Take a dataset example
# # 2. Extract the relevant key (e.g., question) from the example
# # 3. Pass it to the RAG chain
# # 4. Return the relevant output values from the RAG chain


# def predict_rag_answer(example: dict[str, Any]) -> dict[str, str]:
#     """
#     Predict the RAG answer for a given example.

#     Args:
#         example: A dictionary containing the input question.

#     Returns:
#         A dictionary containing the predicted answer.
#     """
#     # bpdb.set_trace()
#     response = rag_bot.get_answer(example["input_question"])
#     # DISABLED: return {"answer": response["answer"]}
#     # FIXME: THIS IS WHERE WE ARE GETTING 'TypeError: string indices must be integers'  because response is a string!!!
#     return {"answer": response["answer"]}

# def predict_rag_answer_with_context(example: dict[str, Any]) -> dict[str, Union[str, list[Document]]]:
#     """
#     Predict the RAG answer with context for a given example.

#     Args:
#         example: A dictionary containing the input question.

#     Returns:
#         A dictionary containing the predicted answer and context.
#     """
#     # bpdb.set_trace()
#     response = rag_bot.get_answer(example["input_question"])
#     return {"answer": response["answer"], "context": response["context"]}


# langsmith_client = langsmith.Client()
# dataset_name = DATASET_NAME

# examples = []
# runs = []

# # Grade prompt
# grade_prompt_answer_vs_reference = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")

# def answer_evaluator(run: Run, example: Example) -> dict[str, Union[str, float]]:
#     """
#     Evaluate the answer accuracy of a RAG response.

#     Args:
#         run: The Run object containing the RAG output.
#         example: The Example object containing the input and reference output.

#     Returns:
#         A dictionary containing the evaluation key and score.
#     """
#     # bpdb.set_trace()
#     examples.append(example)
#     runs.append(run)

#     # Get question, ground truth answer, RAG chain answer
#     input_question = example.inputs["input_question"]
#     reference = example.outputs["output_answer"]
#     prediction = run.outputs["answer"]

#     # LLM grader
#     llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-answer-v-reference", max_retries=MAX_RETRIES)

#     # Structured prompt
#     answer_grader = grade_prompt_answer_vs_reference | llm

#     # Run evaluator
#     score = answer_grader.invoke({"question": input_question,
#                                   "correct_answer": reference,
#                                   "student_answer": prediction})
#     score = score["Score"]

#     return {"key": "answer_v_reference_score", "score": score}

# def answer_evaluator_improved(root_run: Run, example: Example) -> dict:
#     """
#     A simple evaluator for RAG answer generation
#     """
#     # Custom evaluators are simple functions that take a `Run` and an `Example` and return a dictionary with the evaluation results. For example:
#     # SOURCE: https://github.com/langchain-ai/langsmith-cookbook/blob/515f4140cb2576ea93051ea5bb4adec652e31893/introduction/langsmith_introduction.ipynb
#     # FIXME: Try this out

#     grade_rag_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-accuracy")

#     # Get question, answer, and reference answer
#     rag_pipeline_run = next(
#         run for run in root_run.child_runs if run.name == "get_answer"
#     )
#     context = "\n\n".join(doc.page_content for doc in rag_pipeline_run.outputs["context"])
#     # rag_pipeline_run = next(run for run in root_run.child_runs if run.name == "get_answer")
#     # retrieve_run = next(run for run in rag_pipeline_run.child_runs if run.name == "retrieve_docs")
#     input_question = example.inputs["input_question"]
#     reference = example.outputs["output_answer"]
#     prediction = rag_pipeline_run.outputs["answer"]

#     # Data model for grade
#     class GradeAnswerScore(BaseModel):
#         """A numerical score for answer accuracy."""
#         score: int = Field(description="Answer matches the ground truth, score from 1 to 10")

#     # LLM with function call, use highest capacity model
#     llm = ChatOpenAI(model="gpt-4-turbo", temperature=0,  max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-document-relevance", max_retries=MAX_RETRIES)
#     structured_llm_grader = llm.with_structured_output(GradeAnswerScore)

#     # Prompt
#     system = """Is the Assistant's Answer grounded in and similar to the Ground Truth answer. Note that we do not expect all of the text
#             in code solution examples to be identical. We expect (1) code imports to be identical if the same import is used. (2) But, it is
#             ok if there are differences in the implementation itself. The main point is that the same concept is employed. A score of 1 means
#             that the Assistant answer is not at all conceptically grounded in and similar to the Ground Truth answer. A score of 5 means  that the Assistant
#             answer contains some information that is conceptically grounded in and similar to the Ground Truth answer. A score of 10 means that the
#             Assistant answer is fully conceptically grounded in and similar to the Ground Truth answer."""
#     # system = grade_rag_answer_accuracy

#     grade_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             ("human", "Ground Truth answer: \n\n {reference} \n\n Assistant's Answer: {prediction}"),
#         ]
#     )

#     answer_grader = grade_prompt | structured_llm_grader
#     score = answer_grader.invoke({"reference": reference, "prediction": prediction})

#     return {"key": "answer_accuracy", "score": int(score.score) / 10}

# # Now, we kick off evaluation:
# #
# # - `predict_rag_answer`: Takes an `example` from our eval set, extracts the question, passes to our RAG chain
# # - `answer_evaluator`: Passes RAG chain answer, question, and ground truth answer to an evaluator
# # from rich.console import Console

# def print_markdown(text: str) -> None:
#     """
#     Print the given text as formatted Markdown.

#     This function takes a string of Markdown text and prints it to the console
#     with proper formatting.

#     Args:
#         text (str): The Markdown text to be printed.

#     Returns:
#         None
#     """
#     console = Console()
#     md = Markdown(text)
#     console.print()
#     console.print(md)

# def print_panel(text: str) -> None:
#     """
#     Print the given text in a styled panel.

#     This function takes a string and prints it to the console inside a
#     green-colored panel for emphasis.

#     Args:
#         text (str): The text to be printed in the panel.

#     Returns:
#         None
#     """
#     update_text = f"[green]{text}"
#     console = Console()
#     panel_contents = Panel(update_text)
#     console.print()
#     console.print(panel_contents)
#     console.print()


# if RAG_ANSWER_V_REFERENCE_FEATURE_FLAG:
#     try:
#         with warnings.catch_warnings():
#             print_panel("rag-answer-v-reference")
#             warnings.simplefilter("ignore")
#             # bpdb.set_trace()
#             experiment_results = evaluate(
#                 predict_rag_answer,
#                 data=dataset_name,
#                 evaluators=[answer_evaluator],
#                 experiment_prefix="rag-answer-v-reference",
#                 max_concurrency=EVAL_MAX_CONCURRENCY,
#                 metadata={
#                     "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                     "langchain_version": version("langchain"),
#                     "langchain_community_version": version("langchain_community"),
#                     "langchain_core_version": version("langchain_core"),
#                     "langchain_openai_version": version("langchain_openai"),
#                     "langchain_text_splitters_version": version("langchain_text_splitters"),
#                     "langsmith_version": version("langsmith"),
#                     "pydantic_version": version("pydantic"),
#                     "pydantic_settings_version": version("pydantic_settings"),
#                 },
#             )

#             # experiment_results
#     except Exception as ex:
#         rich.print(f"{ex}")
#         exc_type, exc_value, exc_traceback = sys.exc_info()
#         rich.print(f"Error Class: {ex.__class__}")
#         output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#         rich.print(output)
#         rich.print(f"exc_type: {exc_type}")
#         rich.print(f"exc_value: {exc_value}")
#         traceback.print_tb(exc_traceback)
#         bpdb.pm()


# # try:
# #     with warnings.catch_warnings():
# #         print_panel("rag-answer-accuracy")
# #         warnings.simplefilter("ignore")
# #         # bpdb.set_trace()
# #         experiment_results = evaluate(
# #             predict_rag_answer,
# #             data=dataset_name,
# #             evaluators=[answer_evaluator_improved],
# #             experiment_prefix="rag-answer-accuracy",
# #             max_concurrency=EVAL_MAX_CONCURRENCY,
# #             metadata={
# #                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
# #                 "langchain_version": version("langchain"),
# #                 "langchain_community_version": version("langchain_community"),
# #                 "langchain_core_version": version("langchain_core"),
# #                 "langchain_openai_version": version("langchain_openai"),
# #                 "langchain_text_splitters_version": version("langchain_text_splitters"),
# #                 "langsmith_version": version("langsmith"),
# #                 "pydantic_version": version("pydantic"),
# #                 "pydantic_settings_version": version("pydantic_settings"),
# #             },
# #         )

# #         # experiment_results
# # except Exception as ex:
# #     rich.print(f"{ex}")
# #     exc_type, exc_value, exc_traceback = sys.exc_info()
# #     rich.print(f"Error Class: {ex.__class__}")
# #     output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
# #     rich.print(output)
# #     rich.print(f"exc_type: {exc_type}")
# #     rich.print(f"exc_value: {exc_value}")
# #     traceback.print_tb(exc_traceback)
# #     bpdb.pm()


# # # Response vs input
# # Here is an example prompt that we can use:
# #
# # https://smith.langchain.com/hub/langchain-ai/rag-answer-helpfulness
# #
# # The information flow is similar to above, but we simply look at the run answer versus the example question.


# # Grade prompt
# grade_prompt_answer_helpfulness = prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

# def answer_helpfulness_evaluator(run: Run, example: Example) -> dict[str, Union[str, float]]:
#     """
#     Evaluate the helpfulness of a RAG answer.

#     Args:
#         run: The Run object containing the RAG output.
#         example: The Example object containing the input.

#     Returns:
#         A dictionary containing the evaluation key and score.
#     """

#     # Get question, ground truth answer, RAG chain answer
#     input_question = example.inputs["input_question"]
#     prediction = run.outputs["answer"]

#     # LLM grader
#     llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-answer-helpfulness", max_retries=MAX_RETRIES)


#     # Structured prompt
#     answer_grader = grade_prompt_answer_helpfulness | llm

#     # Run evaluator
#     score = answer_grader.invoke({"question": input_question,
#                                 "student_answer": prediction})
#     score = score["Score"]

#     return {"key": "answer_helpfulness_score", "score": score}


# if HELPFULNESS_FEATURE_FLAG:
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         print_panel("rag-answer-helpfulness")
#         experiment_results = evaluate(
#             predict_rag_answer,
#             data=dataset_name,
#             evaluators=[answer_helpfulness_evaluator],
#             experiment_prefix="rag-answer-helpfulness",
#             max_concurrency=EVAL_MAX_CONCURRENCY,
#             metadata={
#                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                 "langchain_version": version("langchain"),
#                 "langchain_community_version": version("langchain_community"),
#                 "langchain_core_version": version("langchain_core"),
#                 "langchain_openai_version": version("langchain_openai"),
#                 "langchain_text_splitters_version": version("langchain_text_splitters"),
#                 "langsmith_version": version("langsmith"),
#                 "pydantic_version": version("pydantic"),
#                 "pydantic_settings_version": version("pydantic_settings"),
#                 "llm_run_config": LLM_RUN_CONFIG,
#             },
#         )


# # ### **Response vs retrieved docs**[](https://docs.smith.langchain.com/tutorials/Developers/rag#response-vs-retrieved-docs)
# #
# # Here is an example prompt that we can use:
# #
# # https://smith.langchain.com/hub/langchain-ai/rag-answer-hallucination
# #
# # Here is the a video from our LangSmith evaluation series for reference:
# #
# # https://youtu.be/IlNglM9bKLw?feature=shared


# # Prompt
# grade_prompt_hallucinations = prompt = hub.pull("langchain-ai/rag-answer-hallucination")

# def answer_hallucination_evaluator(run: Run, example: Example) -> dict[str, Union[str, float]]:
#     """
#     Evaluate if the RAG answer is grounded in the retrieved documents.

#     This function checks whether the answer generated by the RAG system is based on
#     the information present in the retrieved documents, or if it's a hallucination.

#     Args:
#         run (Run): The Run object containing the RAG output.
#         example (Example): The Example object containing the input question and reference answer.

#     Returns:
#         Dict[str, Union[str, float]]: A dictionary containing the evaluation key "answer_hallucination"
#                                       and a score between 0 and 1, where 0 indicates complete hallucination
#                                       and 1 indicates the answer is fully grounded in the documents.

#     Note:
#         This function uses a language model to grade the answer based on its relevance
#         to the retrieved documents.
#     """

#     # RAG inputs
#     input_question = example.inputs["input_question"]
#     context = run.outputs["context"]

#     # RAG answer
#     prediction = run.outputs["answer"]

#     # LLM grader
#     llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-answer-hallucination", max_retries=MAX_RETRIES)

#     # Structured prompt
#     answer_grader = grade_prompt_hallucinations | llm

#     # Get score
#     score = answer_grader.invoke({"documents": context,
#                                 "student_answer": prediction})
#     score = score["Score"]

#     return {"key": "answer_hallucination", "score": score}


# if RAG_ANSWER_HALLUCINATION_FEATURE_FLAG:
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         print_panel("rag-answer-hallucination")
#         experiment_results = evaluate(
#             predict_rag_answer_with_context,
#             data=dataset_name,
#             evaluators=[answer_hallucination_evaluator],
#             experiment_prefix="rag-answer-hallucination",
#             max_concurrency=EVAL_MAX_CONCURRENCY,
#             metadata={
#                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                 "langchain_version": version("langchain"),
#                 "langchain_community_version": version("langchain_community"),
#                 "langchain_core_version": version("langchain_core"),
#                 "langchain_openai_version": version("langchain_openai"),
#                 "langchain_text_splitters_version": version("langchain_text_splitters"),
#                 "langsmith_version": version("langsmith"),
#                 "pydantic_version": version("pydantic"),
#                 "pydantic_settings_version": version("pydantic_settings"),
#                 "llm_run_config": LLM_RUN_CONFIG,
#             },
#         )


# # ### **Retrieved docs vs input**[](https://docs.smith.langchain.com/tutorials/Developers/rag#retrieved-docs-vs-input)
# #
# # Here is an example prompt that we can use:
# #
# # https://smith.langchain.com/hub/langchain-ai/rag-document-relevance
# #
# # Here is the a video from our LangSmith evaluation series for reference:
# #
# # https://youtu.be/Fr_7HtHjcf0?feature=shared


# # Grade prompt
# grade_prompt_doc_relevance = hub.pull("langchain-ai/rag-document-relevance")

# def docs_relevance_evaluator(run: Run, example: Example) -> dict[str, Union[str, float]]:
#     """
#     Evaluate the relevance of retrieved documents to the input question. Answers the question: "Are the documents relevant to the question?"

#     Args:
#         run: The Run object containing the RAG output.
#         example: The Example object containing the input.

#     Returns:
#         A dictionary containing the evaluation key and score.
#     """

#     # RAG inputs
#     input_question = example.inputs["input_question"]
#     context = run.outputs["context"]

#     # LLM grader
#     llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-document-relevance", max_retries=MAX_RETRIES)

#     # Structured prompt
#     answer_grader = grade_prompt_doc_relevance | llm

#     # Get score
#     score = answer_grader.invoke({"question":input_question,
#                                   "documents":context})
#     score = score["Score"]

#     return {"key": "document_relevance", "score": score}


# if RAG_DOC_RELEVANCE_FEATURE_FLAG:
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         print_panel("rag-doc-relevance")
#         experiment_results = evaluate(
#             predict_rag_answer_with_context,
#             data=dataset_name,
#             evaluators=[docs_relevance_evaluator],
#             experiment_prefix="rag-doc-relevance",
#             max_concurrency=EVAL_MAX_CONCURRENCY,
#             metadata={
#                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                 "langchain_version": version("langchain"),
#                 "langchain_community_version": version("langchain_community"),
#                 "langchain_core_version": version("langchain_core"),
#                 "langchain_openai_version": version("langchain_openai"),
#                 "langchain_text_splitters_version": version("langchain_text_splitters"),
#                 "langsmith_version": version("langsmith"),
#                 "pydantic_version": version("pydantic"),
#                 "pydantic_settings_version": version("pydantic_settings"),
#                 "llm_run_config": LLM_RUN_CONFIG,
#             },
#         )


# # ## Evaluating intermediate steps[](https://docs.smith.langchain.com/tutorials/Developers/rag#evaluating-intermediate-steps)
# #
# # Above, we returned the retrieved documents as part of the final answer.
# #
# # However, we will show that this is not required.
# #
# # We can isolate them as intermediate chain steps.
# #
# # See detail on isolating intermediate chain steps [here](https://docs.smith.langchain.com/how_to_guides/evaluation/evaluate_on_intermediate_steps).
# #
# # Here is the a video from our LangSmith evaluation series for reference:
# #
# # https://youtu.be/yx3JMAaNggQ?feature=shared


# from langsmith.evaluation import evaluate
# from langsmith.schemas import Example, Run


# def document_relevance_grader(root_run: Run, example: Example) -> dict:
#     """
#     A simple evaluator that checks to see if retrieved documents are relevant to the question
#     """
#     # bpdb.set_trace()
#     # Get specific steps in our RAG pipeline, which are noted with @traceable decorator
#     rag_pipeline_run = next(
#         run for run in root_run.child_runs if run.name == "get_answer"
#     )
#     # DISABLED: # retrieve_run = next(
#     # DISABLED: #     run for run in rag_pipeline_run.child_runs if run.name == "retrieve_docs"
#     # DISABLED: # )
#     # retrieve_run = next(
#     #     run for run in rag_pipeline_run.child_runs if run.name == "Retriever"
#     # )
#     context = "\n\n".join(doc.page_content for doc in rag_pipeline_run.outputs["context"])
#     input_question = example.inputs["input_question"]

#     # LLM grader
#     llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-document-relevance", max_retries=MAX_RETRIES)

#     # Structured prompt
#     answer_grader = grade_prompt_doc_relevance | llm

#     # Get score
#     score = answer_grader.invoke({"question":input_question,
#                                   "documents":context})
#     score = score["Score"]

#     return {"key": "document_relevance", "score": score}

# def answer_hallucination_grader(root_run: Run, example: Example) -> dict[str, float]:
#     """
#     Evaluate if the RAG answer is grounded in the retrieved documents.

#     This function checks whether the answer generated by the RAG system is based on
#     the information present in the retrieved documents, or if it's a hallucination.

#     Args:
#         root_run (Run): The root Run object containing the RAG pipeline execution details.
#         example (Example): The Example object containing the input question and reference answer.

#     Returns:
#         Dict[str, float]: A dictionary containing the evaluation key "answer_hallucination"
#                           and a score between 0 and 1, where 0 indicates complete hallucination
#                           and 1 indicates the answer is fully grounded in the documents.

#     Note:
#         This function uses a language model to grade the answer based on its relevance
#         to the retrieved documents.
#     """
#     # bpdb.set_trace()

#     # RAG input
#     rag_pipeline_run = next(
#         run for run in root_run.child_runs if run.name == "get_answer"
#     )
#     # DISABLED: # retrieve_run = next(
#     # DISABLED: #     run for run in rag_pipeline_run.child_runs if run.name == "retrieve_docs"
#     # DISABLED: # )
#     # retrieve_run = next(
#     #     run for run in rag_pipeline_run.child_runs if run.name == "Retriever"
#     # )
#     # context = "\n\n".join(doc.page_content for doc in retrieve_run.outputs["output"])
#     context = "\n\n".join(doc.page_content for doc in rag_pipeline_run.outputs["context"])

#     # RAG output
#     prediction = rag_pipeline_run.outputs["answer"]

#     # LLM grader
#     llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, max_tokens=MAX_TOKENS, name="ChatOpenAI-rag-answer-hallucination", max_retries=MAX_RETRIES)

#     # Structured prompt
#     answer_grader = grade_prompt_hallucinations | llm

#     # Get score
#     score = answer_grader.invoke({"documents": context,
#                                   "student_answer": prediction})
#     score = score["Score"]

#     return {"key": "answer_hallucination", "score": score}

# if RAG_DOC_RELEVANCE_AND_HALLUCINATION_FEATURE_FLAG:
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         print_panel("rag-doc-relevance-and-hallucination-grader")
#         experiment_results = evaluate(
#             predict_rag_answer,
#             data=dataset_name,
#             evaluators=[document_relevance_grader, answer_hallucination_grader],
#             experiment_prefix="rag-doc-relevance-and-hallucination-grader",
#             max_concurrency=EVAL_MAX_CONCURRENCY,
#             metadata={
#                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                 "langchain_version": version("langchain"),
#                 "langchain_community_version": version("langchain_community"),
#                 "langchain_core_version": version("langchain_core"),
#                 "langchain_openai_version": version("langchain_openai"),
#                 "langchain_text_splitters_version": version("langchain_text_splitters"),
#                 "langsmith_version": version("langsmith"),
#                 "pydantic_version": version("pydantic"),
#                 "pydantic_settings_version": version("pydantic_settings"),
#                 "llm_run_config": LLM_RUN_CONFIG,
#             },
#         )


# # -------------------------
# # Regression Testing
# # SOURCE: https://docs.smith.langchain.com/old/evaluation/faq/regression-testing
# # -------------------------
# def predict_rag_answer_openai_gpt4o_mini(example: dict):
#     """Use this for answer evaluation"""
#     gpt4o_mini_rag_bot = RagBot(retriever,provider="openai",model="gpt-4o-mini")
#     response = gpt4o_mini_rag_bot.get_answer(example["input_question"])
#     return {"answer": response["answer"]}

# def predict_rag_answer_openai_gpt4_turbo(example: dict):
#     """Use this for answer evaluation"""
#     gpt4_turbo_rag_bot = RagBot(retriever,provider="openai",model="gpt-4-turbo")
#     response = gpt4_turbo_rag_bot.get_answer(example["input_question"])
#     return {"answer": response["answer"]}

# def predict_rag_answer_openai_gpt4o(example: dict):
#     """Use this for answer evaluation"""
#     gpt4o_rag_bot = RagBot(retriever,provider="openai",model="gpt-4o")
#     response = gpt4o_rag_bot.get_answer(example["input_question"])
#     return {"answer": response["answer"]}

# def predict_rag_answer_openai_gpt35_turbo(example: dict):
#     """Use this for answer evaluation"""
#     gpt35_rag_bot = RagBot(retriever,provider="openai",model="gpt-3.5-turbo")
#     response = gpt35_rag_bot.get_answer(example["input_question"])
#     return {"answer": response["answer"]}

# def predict_rag_answer_claude_3_5_sonnet(example: dict):
#     """Use this for answer evaluation"""
#     rag_bot = RagBot(retriever, provider="anthropic", model="claude-3-5-sonnet-20240620")
#     response = rag_bot.get_answer(example["input_question"])
#     return {"answer": response["answer"]}

# def predict_rag_answer_claude_3_opus(example: dict):
#     """Use this for answer evaluation"""
#     rag_bot = RagBot(retriever, provider="anthropic", model="claude-3-opus-20240229")
#     response = rag_bot.get_answer(example["input_question"])
#     return {"answer": response["answer"]}

# def predict_rag_answer_claude_3_haiku(example: dict):
#     """Use this for answer evaluation"""
#     rag_bot = RagBot(retriever, provider="anthropic", model="claude-3-haiku-20240307")
#     response = rag_bot.get_answer(example["input_question"])
#     return {"answer": response["answer"]}

# if COMPARE_MODELS_FEATURE_FLAG:

#     # define evaluator

#     # The "labeled_score_string" evaluator instructs an LLM to assess the prediction against a reference label on a specified scale
#     # SOURCE: https://docs.smith.langchain.com/how_to_guides/evaluation/use_langchain_off_the_shelf_evaluators
#     regression_accuracy_criteria_evaluator = LangChainStringEvaluator(
#         "labeled_score_string",
#         config={
#             "criteria": {
#                 "accuracy": "Is the Assistant's Answer grounded in and similar to the Ground Truth answer? A score of [[1]] means that the Assistant answer is not at all grounded in and similar to the Ground Truth answer. A score of [[5]] means  that the Assistant  answer contains some information that is grounded in and similar to the Ground Truth answer. A score of [[10]] means that the Assistant answer is fully grounded in and similar to the Ground Truth answer.",
#                 # NOTE: This is an alternative below
#                 # "accuracy": "How accurate is this prediction compared to the reference on a scale of 1-10?"
#             },
#             # If you want the score to be saved on a scale from 0 to 1
#             "normalize_by": 10,
#         }
#     )


#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         print_panel("rag-regression-testing-gpt4o-mini")
#         experiment_results = evaluate(
#             predict_rag_answer_openai_gpt4o_mini,
#             data=dataset_name,
#             evaluators=[regression_accuracy_criteria_evaluator],
#             experiment_prefix="rag-regression-testing-gpt4o-mini",
#             max_concurrency=EVAL_MAX_CONCURRENCY,
#             metadata={
#                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                 "langchain_version": version("langchain"),
#                 "langchain_community_version": version("langchain_community"),
#                 "langchain_core_version": version("langchain_core"),
#                 "langchain_openai_version": version("langchain_openai"),
#                 "langchain_text_splitters_version": version("langchain_text_splitters"),
#                 "langsmith_version": version("langsmith"),
#                 "pydantic_version": version("pydantic"),
#                 "pydantic_settings_version": version("pydantic_settings"),
#                 "llm_run_config": LLM_RUN_CONFIG,
#             },
#         )

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         print_panel("rag-regression-testing-claude-3-5-sonnet")
#         experiment_results = evaluate(
#             predict_rag_answer_claude_3_5_sonnet,
#             data=dataset_name,
#             evaluators=[regression_accuracy_criteria_evaluator],
#             experiment_prefix="rag-regression-testing-claude-3-5-sonnet",
#             max_concurrency=EVAL_MAX_CONCURRENCY,
#             metadata={
#                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                 "langchain_version": version("langchain"),
#                 "langchain_community_version": version("langchain_community"),
#                 "langchain_core_version": version("langchain_core"),
#                 "langchain_openai_version": version("langchain_openai"),
#                 "langchain_text_splitters_version": version("langchain_text_splitters"),
#                 "langsmith_version": version("langsmith"),
#                 "pydantic_version": version("pydantic"),
#                 "pydantic_settings_version": version("pydantic_settings"),
#                 "llm_run_config": LLM_RUN_CONFIG,
#             },
#         )

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         print_panel("rag-regression-testing-claude-3-opus")
#         experiment_results = evaluate(
#             predict_rag_answer_claude_3_opus,
#             data=dataset_name,
#             evaluators=[regression_accuracy_criteria_evaluator],
#             experiment_prefix="rag-regression-testing-claude-3-5-sonnet",
#             max_concurrency=EVAL_MAX_CONCURRENCY,
#             metadata={
#                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                 "langchain_version": version("langchain"),
#                 "langchain_community_version": version("langchain_community"),
#                 "langchain_core_version": version("langchain_core"),
#                 "langchain_openai_version": version("langchain_openai"),
#                 "langchain_text_splitters_version": version("langchain_text_splitters"),
#                 "langsmith_version": version("langsmith"),
#                 "pydantic_version": version("pydantic"),
#                 "pydantic_settings_version": version("pydantic_settings"),
#                 "llm_run_config": LLM_RUN_CONFIG,
#             },
#         )

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         print_panel("rag-regression-testing-claude-3-haiku")
#         experiment_results = evaluate(
#             predict_rag_answer_claude_3_haiku,
#             data=dataset_name,
#             evaluators=[regression_accuracy_criteria_evaluator],
#             experiment_prefix="rag-regression-testing-claude-3-5-sonnet",
#             max_concurrency=EVAL_MAX_CONCURRENCY,
#             metadata={
#                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                 "langchain_version": version("langchain"),
#                 "langchain_community_version": version("langchain_community"),
#                 "langchain_core_version": version("langchain_core"),
#                 "langchain_openai_version": version("langchain_openai"),
#                 "langchain_text_splitters_version": version("langchain_text_splitters"),
#                 "langsmith_version": version("langsmith"),
#                 "pydantic_version": version("pydantic"),
#                 "pydantic_settings_version": version("pydantic_settings"),
#                 "llm_run_config": LLM_RUN_CONFIG,
#             },
#         )

#     # DISABLED: # def predict_rag_answer_phi3(example: dict):
#     # DISABLED: #     """Use this for answer evaluation"""
#     # DISABLED: #     rag_bot = RagBot(retriever,provider="ollama",model="phi3")
#     # DISABLED: #     response = rag_bot.get_answer(example["input_question"])
#     # DISABLED: #     return {"answer": response["answer"]}

#     # -------------------------
#     # Pairwise Testing
#     # -------------------------

#     # TODO: https://docs.smith.langchain.com/how_to_guides/evaluation/evaluate_pairwise
#     # Update the values below and uncomment the code to run
#     # pairwise_evaluation_prompt = hub.pull("langchain-ai/pairwise-evaluation-2")

#     # How do multiple answer versions compare?
#     # def evaluate_pairwise(runs: list[Run], example: Example):
#     #     scores = {}

#     #     # Create the model to run your evaluator
#     #     model = ChatOpenAI(model_name="gpt-4")

#     #     runnable = pairwise_evaluation_prompt | model
#     #     response = runnable.invoke({
#     #         "question": example.inputs["question"],
#     #         "answer_a": runs[0].outputs["output"] if runs[0].outputs is not None else "N/A",
#     #         "answer_b": runs[1].outputs["output"] if runs[1].outputs is not None else "N/A",
#     #     })
#     #     score = response["Preference"]
#     #     if score == 1:
#     #         scores[runs[0].id] = 1
#     #         scores[runs[1].id] = 0
#     #     elif score == 2:
#     #         scores[runs[0].id] = 0
#     #         scores[runs[1].id] = 1
#     #     else:
#     #         scores[runs[0].id] = 0
#     #         scores[runs[1].id] = 0
#     #     return {"key": "ranked_preference", "scores": scores}


#     # evaluate_comparative(
#     #     # Replace the following array with the names or IDs of your experiments
#     #     ["my-experiment-name-1", "my-experiment-name-2"],
#     #     evaluators=[evaluate_pairwise],
#     # )


# # ---------------------------------------------------------
# # compare accuracy of different models
# # ---------------------------------------------------------
# if RAG_ANSWER_ACCURACY_FEATURE_FLAG:
#     try:
#         with warnings.catch_warnings():
#             print_panel("rag-answer-accuracy-gpt4o-mini")
#             warnings.simplefilter("ignore")
#             # bpdb.set_trace()
#             experiment_results_gpt4o_mini = evaluate(
#                 predict_rag_answer_openai_gpt4o_mini,
#                 data=dataset_name,
#                 evaluators=[answer_evaluator_improved],
#                 experiment_prefix="rag-answer-accuracy-gpt4o-mini",
#                 max_concurrency=EVAL_MAX_CONCURRENCY,
#                 metadata={
#                     "version": f"{DATASET_NAME}, gpt-4o-mini, {LLM_MODEL_NAME}",
#                     "langchain_version": version("langchain"),
#                     "langchain_community_version": version("langchain_community"),
#                     "langchain_core_version": version("langchain_core"),
#                     "langchain_openai_version": version("langchain_openai"),
#                     "langchain_text_splitters_version": version("langchain_text_splitters"),
#                     "langsmith_version": version("langsmith"),
#                     "pydantic_version": version("pydantic"),
#                     "pydantic_settings_version": version("pydantic_settings"),
#                 },
#             )
#         with warnings.catch_warnings():
#             print_panel("rag-answer-accuracy-gpt4o")
#             warnings.simplefilter("ignore")
#             # bpdb.set_trace()
#             experiment_results_gpt4o = evaluate(
#                 predict_rag_answer_openai_gpt4o,
#                 data=dataset_name,
#                 evaluators=[answer_evaluator_improved],
#                 experiment_prefix="rag-answer-accuracy-gpt4o",
#                 max_concurrency=EVAL_MAX_CONCURRENCY,
#                 metadata={
#                     "version": f"{DATASET_NAME}, gpt-4o, {LLM_MODEL_NAME}",
#                     "langchain_version": version("langchain"),
#                     "langchain_community_version": version("langchain_community"),
#                     "langchain_core_version": version("langchain_core"),
#                     "langchain_openai_version": version("langchain_openai"),
#                     "langchain_text_splitters_version": version("langchain_text_splitters"),
#                     "langsmith_version": version("langsmith"),
#                     "pydantic_version": version("pydantic"),
#                     "pydantic_settings_version": version("pydantic_settings"),
#                 },
#             )
#         with warnings.catch_warnings():
#             print_panel("rag-answer-accuracy-gpt4-turbo")
#             warnings.simplefilter("ignore")
#             # bpdb.set_trace()
#             experiment_results_gpt4_turbo = evaluate(
#                 predict_rag_answer_openai_gpt4_turbo,
#                 data=dataset_name,
#                 evaluators=[answer_evaluator_improved],
#                 experiment_prefix="rag-answer-accuracy-gpt4-turbo",
#                 max_concurrency=EVAL_MAX_CONCURRENCY,
#                 metadata={
#                     "version": f"{DATASET_NAME}, gpt-4-turbo, {LLM_MODEL_NAME}",
#                     "langchain_version": version("langchain"),
#                     "langchain_community_version": version("langchain_community"),
#                     "langchain_core_version": version("langchain_core"),
#                     "langchain_openai_version": version("langchain_openai"),
#                     "langchain_text_splitters_version": version("langchain_text_splitters"),
#                     "langsmith_version": version("langsmith"),
#                     "pydantic_version": version("pydantic"),
#                     "pydantic_settings_version": version("pydantic_settings"),
#                 },
#             )
#         with warnings.catch_warnings():
#             print_panel("rag-answer-accuracy-gpt35-turbo")
#             warnings.simplefilter("ignore")
#             # bpdb.set_trace()
#             experiment_results_gpt35_turbo = evaluate(
#                 predict_rag_answer_openai_gpt35_turbo,
#                 data=dataset_name,
#                 evaluators=[answer_evaluator_improved],
#                 experiment_prefix="rag-answer-accuracy-gpt35-turbo",
#                 max_concurrency=EVAL_MAX_CONCURRENCY,
#                 metadata={
#                     "version": f"{DATASET_NAME}, gpt-3.5-turbo, {LLM_MODEL_NAME}",
#                     "langchain_version": version("langchain"),
#                     "langchain_community_version": version("langchain_community"),
#                     "langchain_core_version": version("langchain_core"),
#                     "langchain_openai_version": version("langchain_openai"),
#                     "langchain_text_splitters_version": version("langchain_text_splitters"),
#                     "langsmith_version": version("langsmith"),
#                     "pydantic_version": version("pydantic"),
#                     "pydantic_settings_version": version("pydantic_settings"),
#                 },
#             )


#             # experiment_results
#     except Exception as ex:
#         rich.print(f"{ex}")
#         exc_type, exc_value, exc_traceback = sys.exc_info()
#         rich.print(f"Error Class: {ex.__class__}")
#         output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#         rich.print(output)
#         rich.print(f"exc_type: {exc_type}")
#         rich.print(f"exc_value: {exc_value}")
#         traceback.print_tb(exc_traceback)
#         bpdb.pm()

# # -------------------------
# # Helpfulness Testing
# # -------------------------
# # The "labeled_criteria" evaluator instructs an LLM to assess if a prediction satisfies the criteria, taking into account the reference label
# # SOURCE: https://docs.smith.langchain.com/how_to_guides/evaluation/use_langchain_off_the_shelf_evaluators
# helpfulness_labeled_criteria_evaluator = LangChainStringEvaluator(
#     "labeled_criteria",
#     config={
#         "criteria": {
#             "helpfulness": (
#                 "Is this submission helpful to the user,"
#                 " taking into account the correct reference answer?"
#             )
#         }
#     }
# )

# if HELPFULNESS_FEATURE_FLAG:
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         print_panel("rag-helpfulness-testing")
#         experiment_results = evaluate(
#             predict_rag_answer,
#             data=dataset_name,
#             evaluators=[helpfulness_labeled_criteria_evaluator],
#             experiment_prefix="rag-helpfulness-testing",
#             max_concurrency=EVAL_MAX_CONCURRENCY,
#             metadata={
#                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                 "langchain_version": version("langchain"),
#                 "langchain_community_version": version("langchain_community"),
#                 "langchain_core_version": version("langchain_core"),
#                 "langchain_openai_version": version("langchain_openai"),
#                 "langchain_text_splitters_version": version("langchain_text_splitters"),
#                 "langsmith_version": version("langsmith"),
#                 "pydantic_version": version("pydantic"),
#                 "pydantic_settings_version": version("pydantic_settings"),
#                 "llm_run_config": LLM_RUN_CONFIG,
#             },
#         )

# # -------------------------
# # Use string or embedding distance metrics
# # -------------------------

# string_distance_evaluator = LangChainStringEvaluator(
#     "string_distance",
#     config={"distance": "levenshtein", "normalize_score": True}
# )

# embedding_distance_evaluator = LangChainStringEvaluator(
#     "embedding_distance",
#     config={
#       # Defaults to OpenAI, but you can customize which embedding provider to use:
#       # "embeddings": HuggingFaceEmbeddings(model="distilbert-base-uncased"),
#       # Can also choose "euclidean", "chebyshev", "hamming", and "manhattan"
#         "distance_metric": "cosine",
#       }
# )

# if RAG_STRING_EMBEDDING_DISTANCE_METRICS_FEATURE_FLAG:
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         print_panel("rag-string-embedding-distance-metrics-testing")
#         experiment_results = evaluate(
#             predict_rag_answer,
#             data=dataset_name,
#             evaluators=[string_distance_evaluator, embedding_distance_evaluator],
#             experiment_prefix="rag-string-embedding-distance-metrics-testing",
#             max_concurrency=EVAL_MAX_CONCURRENCY,
#             metadata={
#                 "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
#                 "langchain_version": version("langchain"),
#                 "langchain_community_version": version("langchain_community"),
#                 "langchain_core_version": version("langchain_core"),
#                 "langchain_openai_version": version("langchain_openai"),
#                 "langchain_text_splitters_version": version("langchain_text_splitters"),
#                 "langsmith_version": version("langsmith"),
#                 "pydantic_version": version("pydantic"),
#                 "pydantic_settings_version": version("pydantic_settings"),
#                 "llm_run_config": LLM_RUN_CONFIG,
#             },
#         )
# # To measure the similarity between a predicted string and a reference, you can use string distance metrics:

# # The "string_distance" evaluator computes a normalized string edit distance between the prediction and reference
# # The "embedding_distance" evaluator computes the distance between the text embeddings of the prediction and reference

# # FIXME: use console export text instead of console.rule
# def write_run_tree_to_file(fname: str, run: Run, example: Example) -> None:
#     """
#     Write the run tree to a file in the current working directory.

#     Args:
#         fname: The base name for the output files.
#         run: The Run object to be written.
#         example: The Example object to be written.
#     """
#     run_name = f"../cached_evals/run-{fname}.txt"
#     example_name = f"../cached_evals/example-{fname}.txt"
#     console_recorder = Console(record=True)
#     console_recorder.print("Run: \n\n")
#     console_recorder.print(rich.inspect(run, methods=True, value=True))
#     text_run = console_recorder.export_text()
#     with open(run_name, "w") as file:
#         file.write(text_run)

#     console_recorder.print("Example: \n\n")
#     console_recorder.print(rich.inspect(example, methods=True, value=True))
#     text_example = console_recorder.export_text()
#     with open(example_name, "w") as file:
#         file.write(text_example)


# # -------------------------
# # Summary Evaluation Testing
# # -------------------------
# # Row-level evaluator
# def correct_answer(run: Run, example: Example) -> dict[str, int]:
#     """
#     Evaluate if the RAG answer matches the reference answer.

#     Args:
#         run: The Run object containing the RAG output.
#         example: The Example object containing the reference output.

#     Returns:
#         A dictionary containing the evaluation score.
#     """
#     write_run_tree_to_file("correct_answer", run, example)
#     # SOURCE: https://docs.smith.langchain.com/old/cookbook/testing-examples/movie-demo#our-pipeline
#     # SOURCE: https://docs.smith.langchain.com/how_to_guides/evaluation/evaluate_llm_application#use-a-summary-evaluator
#     # DISABLED: # score = run.outputs.get("output") == example.outputs.get("output_answer")
#     score = run.outputs["answer"] == example.outputs["output_answer"]

#     # Get question, ground truth answer, RAG chain answer
#     # input_question = example.inputs["input_question"]
#     # reference = example.outputs["output_answer"]
#     # prediction = run.outputs["answer"]
#     return {"score": int(score)}

# def summary_eval(runs: list[Run], examples: list[Example]) -> dict[str, Union[str, bool]]:
#     """
#     Perform a summary evaluation on a list of runs and examples.

#     Args:
#         runs: A list of Run objects.
#         examples: A list of Example objects.

#     Returns:
#         A dictionary containing the evaluation key and pass/fail score.
#     """
#     # write_run_tree_to_file("summary-eval", runs, examples)
#     correct = 0
#     for i, run in enumerate(runs):
#         if run.outputs["output"] == examples[i].outputs["output_answer"]:
#             correct += 1
#     if correct / len(runs) > 0.5:
#         return {"key": "pass", "score": True}
#     else:
#         return {"key": "pass", "score": False}
# # -------------------------
# # Retrival grading for summary evaluation
# # -------------------------

# ### Retrieval Grader

# # FIXME: You need to read this and implement it. https://github.com/langchain-ai/langsmith-docs/blob/bed25769ecbf29809b8c227d4e37b5467d024921/docs/evaluation/faq/custom-evaluators.mdx#L339

# # SOURCE: https://www.youtube.com/watch?v=zMgrHzs_cAg&list=PLfaIDFEXuae0um8Fj0V4dHG37fGFU8Q5S&index=12
# # Data model
# class GradeDocuments(BaseModel):
#     """
#     A class representing the binary score for relevance check on retrieved documents.

#     Attributes:
#         binary_score: A string indicating whether documents are relevant ('yes' or 'no').
#     """

#     binary_score: str = Field(
#         description="Documents are relevant to the question, 'yes' or 'no'"
#     )

# def retrieval_grader_with_structured_output(question: str) -> dict[str, str]:
#     """
#     Grade the relevance of retrieved documents to a given question.

#     Args:
#         question: The input question.

#     Returns:
#         A dictionary containing the binary relevance score.
#     """
#     # https://github.com/langchain-ai/langgraph/blob/a93775413281df9ddf6ba29cc388b2460d94b9af/examples/rag/langgraph_crag.ipynb#L144
#     # LLM with function call
#     llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
#     structured_llm_grader = llm.with_structured_output(GradeDocuments)

#     # Prompt
#     system = """You are a grader assessing relevance of a retrieved document to a user question. \n
#         If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
#         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
#     grade_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             ("human", "Retrieved document: \n\n {context} \n\n User question: {question}"),
#         ]
#     )

#     retrieval_grader = grade_prompt | structured_llm_grader
#     question = "agent memory"
#     docs = retriever.get_relevant_documents(question)
#     doc_txt = docs[1].page_content
#     rich.print(retrieval_grader.invoke({"question": question, "document": doc_txt}))
#     # returns: binary_score='yes'

# def question_rewriter(question: str) -> str:
#     """
#     Rewrite a given question to optimize it for web search.

#     Args:
#         question: The input question.

#     Returns:
#         The rewritten question.
#     """
#     # SOURCE: https://github.com/langchain-ai/langgraph/blob/a93775413281df9ddf6ba29cc388b2460d94b9af/examples/rag/langgraph_crag.ipynb#L144
#     ### Question Re-writer

#     # LLM
#     llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

#     # Prompt
#     system = """You a question re-writer that converts an input question to a better version that is optimized \n
#          for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
#     re_write_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             (
#                 "human",
#                 "Here is the initial question: \n\n {question} \n Formulate an improved question.",
#             ),
#         ]
#     )

#     question_rewriter = re_write_prompt | llm | StrOutputParser()
#     question_rewriter.invoke({"question": question})


# # # disabled for now
# # with warnings.catch_warnings():
# #     warnings.simplefilter("ignore")
# #     print_panel("rag-summary-evaluation-testing")
# #     experiment_results = evaluate(
# #         predict_rag_answer,
# #         data=dataset_name,
# #         # evaluators=[answer_evaluator],
# #         evaluators=[correct_answer],
# #         summary_evaluators=[summary_eval],
# #         experiment_prefix="rag-summary-evaluation-testing",
# #         max_concurrency=EVAL_MAX_CONCURRENCY,
# #         metadata={
# #             "version": f"{DATASET_NAME}, {LLM_MODEL_NAME}",
# #             "langchain_version": version("langchain"),
# #             "langchain_community_version": version("langchain_community"),
# #             "langchain_core_version": version("langchain_core"),
# #             "langchain_openai_version": version("langchain_openai"),
# #             "langchain_text_splitters_version": version("langchain_text_splitters"),
# #             "langsmith_version": version("langsmith"),
# #             "pydantic_version": version("pydantic"),
# #             "pydantic_settings_version": version("pydantic_settings"),
# #             "llm_run_config": LLM_RUN_CONFIG,
# #         },
# #     )


# # -------------------------
# # Back Testing
# # SOURCE: https://docs.smith.langchain.com/tutorials/Developers/backtesting
# # -------------------------
