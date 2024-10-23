"""aio_settings"""

# pylint: disable=no-name-in-module
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false
from __future__ import annotations

import enum
import os
import pathlib

from pathlib import Path
from tempfile import gettempdir
from typing import Annotated, Any, Callable, Dict, List, Optional, Set, Union, cast

from loguru import logger
from pydantic import (
    AliasChoices,
    AmqpDsn,
    BaseModel,
    Field,
    ImportString,
    Json,
    PostgresDsn,
    RedisDsn,
    SecretBytes,
    SecretStr,
    field_serializer,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.table import Table
from typing_extensions import Self, TypedDict
from yarl import URL

from sandbox_agent import __version__


#     AliasChoices,
#     AmqpDsn,
#     BaseModel,
#     Field,
#     ImportString,
#     PostgresDsn,
#     RedisDsn,
#     SecretBytes,
#     SecretStr,
#     ValidationError,
#     field_serializer,
#     model_validator,
# "claude-instant-1.2": 100000,
# "claude-3-opus-20240229": 200000,
# "claude-3-sonnet-20240229": 200000,
# "claude-3-5-sonnet-20240620": 200000,
# "claude-3-haiku-20240307": 200000,


def goob_user_agent() -> str:
    """Get a common user agent"""
    return f"goob-ai/{__version__}"


# Get rid of warning
# USER_AGENT environment variable not set, consider setting it to identify your requests.
os.environ["USER_AGENT"] = goob_user_agent()


TEMP_DIR = Path(gettempdir())
# SOURCE: https://github.com/taikinman/langrila/blob/main/src/langrila/openai/model_config.py
# SOURCE: https://github.com/taikinman/langrila/blob/main/src/langrila/openai/model_config.py
# TODO: Look at this https://github.com/h2oai/h2ogpt/blob/542543dc23aa9eb7d4ce7fe6b9af1204a047b50f/src/enums.py#L386 and see if we can add some more models
_TOKENS_PER_TILE = 170
_TILE_SIZE = 512

_OLDER_MODEL_CONFIG = {
    "gpt-4-0613": {
        "max_tokens": 8192,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00003,
        "completion_cost_per_token": 0.00006,
    },
    "gpt-4-32k-0314": {
        "max_tokens": 32768,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00006,
        "completion_cost_per_token": 0.00012,
    },
    "gpt-4-32k-0613": {
        "max_tokens": 32768,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00006,
        "completion_cost_per_token": 0.00012,
    },
    "gpt-3.5-turbo-0301": {
        "max_tokens": 4096,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-0613": {
        "max_tokens": 4096,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-16k-0613": {
        "max_tokens": 16384,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000003,
        "completion_cost_per_token": 0.000004,
    },
    "gpt-3.5-turbo-instruct": {
        "max_tokens": 4096,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
}


_NEWER_MODEL_CONFIG = {
    "claude-3-5-sonnet-20240620": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "claude-3-opus-20240229": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "claude-3-sonnet-20240229": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "claude-3-haiku-20240307": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "gpt-4o-2024-08-06": {
        "max_tokens": 128000,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "gpt-4o-mini-2024-07-18": {
        # "max_tokens": 128000,
        "max_tokens": 900,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.000000150,
        "completion_cost_per_token": 0.00000060,
    },
    "gpt-4o-2024-05-13": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000005,
        "completion_cost_per_token": 0.000015,
    },
    "gpt-4-turbo-2024-04-09": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-4-0125-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-4-1106-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-4-vision-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-3.5-turbo-0125": {
        "max_tokens": 16384,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000005,
        "completion_cost_per_token": 0.0000015,
    },
    "gpt-3.5-turbo-1106": {
        "max_tokens": 16384,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000001,
        "completion_cost_per_token": 0.000002,
    },
}

_NEWER_EMBEDDING_CONFIG = {
    "text-embedding-3-small": {
        "max_tokens": 8192,
        "prompt_cost_per_token": 0.00000002,
    },
    "text-embedding-3-large": {
        "max_tokens": 8192,
        "prompt_cost_per_token": 0.00000013,
    },
}

_OLDER_EMBEDDING_CONFIG = {
    "text-embedding-ada-002": {
        "max_tokens": 8192,
        "prompt_cost_per_token": 0.0000001,
    },
}


EMBEDDING_CONFIG = {}
EMBEDDING_CONFIG.update(_OLDER_EMBEDDING_CONFIG)
EMBEDDING_CONFIG.update(_NEWER_EMBEDDING_CONFIG)

MODEL_CONFIG = {}
MODEL_CONFIG.update(_OLDER_MODEL_CONFIG)
MODEL_CONFIG.update(_NEWER_MODEL_CONFIG)

MODEL_POINT = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "gpt-4": "gpt-4-0613",
    "gpt-4-32k": "gpt-4-32k-0613",
    "gpt-4-vision": "gpt-4-vision-preview",
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k-0613",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
}

_MODEL_POINT_CONFIG = {
    "gpt-4o-mini": MODEL_CONFIG[MODEL_POINT["gpt-4o-mini"]],
    "gpt-4o": MODEL_CONFIG[MODEL_POINT["gpt-4o"]],
    "gpt-4-turbo": MODEL_CONFIG[MODEL_POINT["gpt-4-turbo"]],
    "gpt-4": MODEL_CONFIG[MODEL_POINT["gpt-4"]],
    "gpt-4-32k": MODEL_CONFIG[MODEL_POINT["gpt-4-32k"]],
    "gpt-4-vision": MODEL_CONFIG[MODEL_POINT["gpt-4-vision"]],
    "gpt-3.5-turbo": MODEL_CONFIG[MODEL_POINT["gpt-3.5-turbo"]],
    "gpt-3.5-turbo-16k": MODEL_CONFIG[MODEL_POINT["gpt-3.5-turbo-16k"]],
    "claude-3-opus": MODEL_CONFIG[MODEL_POINT["claude-3-opus"]],
    "claude-3-5-sonnet": MODEL_CONFIG[MODEL_POINT["claude-3-5-sonnet"]],
    "claude-3-sonnet": MODEL_CONFIG[MODEL_POINT["claude-3-sonnet"]],
    "claude-3-haiku": MODEL_CONFIG[MODEL_POINT["claude-3-haiku"]],
}

# contains all the models and embeddings info
MODEL_CONFIG.update(_MODEL_POINT_CONFIG)

# produces a list of all models and embeddings available
MODEL_ZOO = set(MODEL_CONFIG.keys()) | set(EMBEDDING_CONFIG.keys())

# SOURCE: https://github.com/JuliusHenke/autopentest/blob/ca822f723a356ec974d2dff332c2d92389a4c5e3/src/text_embeddings.py#L19
# https://platform.openai.com/docs/guides/embeddings/embedding-models
EMBEDDING_MODEL_DIMENSIONS_DATA = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 512,
    "text-embedding-3-large": 1024,
}


# NOTE: DIRTY HACK TO GET AROUND CIRCULAR IMPORTS
# NOTE: There is a bug in pydantic that prevents us from using the `tilda` package and dealing with circular imports
def tilda(obj):
    """
    Wrapper for linux ~/ shell notation

    Args:
    ----
        obj (_type_): _description_

    Returns:
    -------
        _type_: _description_

    """
    if isinstance(obj, list):
        return [str(pathlib.Path(o).expanduser()) if isinstance(o, str) else o for o in obj]
    elif isinstance(obj, str):
        return str(pathlib.Path(obj).expanduser())
    else:
        return obj


def normalize_settings_path(file_path: str) -> str:
    """
    field_validator used to detect shell tilda notation and expand field automatically

    Args:
    ----
        file_path (str): _description_

    Returns:
    -------
        pathlib.PosixPath | str: _description_

    """
    # prevent circular import
    # from sandbox_agent.utils import file_functions

    return tilda(file_path) if file_path.startswith("~") else file_path


def get_rich_console() -> Console:
    """
    _summary_

    Returns
    -------
        Console: _description_

    """
    return Console()


class LogLevel(str, enum.Enum):  # noqa: WPS600
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class AioSettings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    # By default, the environment variable name is the same as the field name.

    # You can change the prefix for all environment variables by setting the env_prefix config setting, or via the _env_prefix keyword argument on instantiation:

    # add a comment to each line in model_config explaining what it does
    model_config = SettingsConfigDict(
        env_prefix="SANDBOX_AGENT_CONFIG_",
        env_file=(".env", ".envrc"),
        env_file_encoding="utf-8",
        extra="allow",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "properties": {
                "llm_retriever_type": {
                    "type": "string",
                    "default": "vector_store",
                    "description": "Type of retriever to use",
                }
            }
        },
    )

    monitor_host: str = "localhost"
    monitor_port: int = 50102

    debug_langchain: bool | None = False

    # tweetpik_background_image = "510"  # a image that you want to use as background. you need to use this as a valid url like https://mysite.com/image.png and it should not be protected by cors
    audit_log_send_channel: str = ""

    # ***************************************************
    # NOTE: these are grouped together
    # ***************************************************
    # token: str = ""
    prefix: str = "?"

    discord_admin_user_id: int | None = None

    discord_general_channel: int = 908894727779258390

    discord_server_id: int = 0
    discord_client_id: int | str = 0

    discord_token: SecretStr = ""

    vector_store_type: str = "pgvector"

    # openai_token: str = ""
    openai_api_key: SecretStr = ""

    discord_admin_user_invited: bool = False

    debug: bool = True
    log_pii: bool = True

    personalization_file: str = Field(
        env="PERSONALIZATION_FILE",
        description="Path to the personalization JSON file",
        default="./personalization.json",
    )
    scratch_pad_dir: str = Field(
        env="SCRATCH_PAD_DIR",
        description="Directory for scratch pad files",
        default="./scratchpad",
    )
    active_memory_file: str = Field(
        env="ACTIVE_MEMORY_FILE",
        description="Path to the active memory JSON file",
        default="./active_memory.json",
    )

    changelogs_github_api_token: SecretStr = Field(
        env="CHANGELOGS_GITHUB_API_TOKEN", description="GitHub API token for Changelogs", default=""
    )
    firecrawl_api_key: SecretStr = Field(env="FIRECRAWL_API_KEY", description="Firecrawl API key", default="")

    # pylint: disable=redundant-keyword-arg
    better_exceptions: bool = Field(env="BETTER_EXCEPTIONS", description="Enable better exceptions", default=1)
    pythonasynciodebug: bool = Field(
        env="PYTHONASYNCIODEBUG", description="enable or disable asyncio debugging", default=0
    )
    pythondevmode: bool = Field(
        env="PYTHONDEVMODE",
        description="The Python Development Mode introduces additional runtime checks that are too expensive to be enabled by default. It should not be more verbose than the default if the code is correct; new warnings are only emitted when an issue is detected.",
        default=0,
    )
    langchain_debug_logs: bool = Field(
        env="LANGCHAIN_DEBUG_LOGS", description="enable or disable langchain debug logs", default=0
    )

    enable_ai: bool = False
    http_client_debug_enabled: bool = False

    localfilestore_root_path: str = Field(
        env="LOCALFILESTORE_ROOT_PATH", description="root path for local file store", default="./local_file_store"
    )

    # Try loading patchmatch
    globals_try_patchmatch: bool = True

    # Use CPU even if GPU is available (main use case is for debugging MPS issues)
    globals_always_use_cpu: bool = False

    # Whether the internet is reachable for dynamic downloads
    # The CLI will test connectivity at startup time.
    globals_internet_available: bool = True

    # whether we are forcing full precision
    globals_full_precision: bool = False

    # whether we should convert ckpt files into diffusers models on the fly
    globals_ckpt_convert: bool = False

    # logging tokenization everywhere
    globals_log_tokenization: bool = False

    bot_name: str = "SandboxAgentAI"

    # Variables for Redis
    redis_host: str = "localhost"
    redis_port: int = 7600
    redis_user: Optional[str] = None
    redis_pass: SecretStr | None = None
    redis_base: Optional[int] = None
    enable_redis: bool = False
    redis_url: URL | str | None = None

    sentry_dsn: SecretStr = ""
    enable_sentry: bool = False

    # Variables for ChromaDB

    # client = chromadb.HttpClient(host="localhost", port="8010", settings=Settings(allow_reset=True))
    chroma_host: str = "localhost"
    chroma_port: str = "8010"
    enable_chroma: bool = True

    dev_mode: bool = Field(env="DEV_MODE", description="enable dev mode", default=False)
    # azure_openai_api_key: str
    # openai_api_type: str
    # openai_api_version: str
    # azure_deployment: str
    # azure_openai_endpoint: str
    llm_temperature: float = 0.0
    # vision_model: str = "gpt-4-turbo"
    vision_model: str = "gpt-4o"
    chat_model: str = "gpt-4o-mini"
    # DISABLED: # vision_model: str = "gpt-4-vision-preview"
    # DISABLED: # chat_model: str = "gpt-4o-2024-05-13"
    # chat_model: str = "gpt-3.5-turbo-0125"
    # chat_model: str = "gpt-3.5-turbo-16k" # note another option
    chat_history_buffer: int = 10

    retry_stop_after_attempt: int = 3
    retry_wait_exponential_multiplier: Union[int, float] = 2
    retry_wait_exponential_max: Union[int, float] = 5
    retry_wait_exponential_min: Union[int, float] = 1
    retry_wait_fixed: Union[int, float] = 15

    pinecone_api_key: SecretStr = Field(env="PINECONE_API_KEY", description="pinecone api key", default="")
    pinecone_env: str = Field(env="PINECONE_ENV", description="pinecone env", default="local")
    pinecone_index: str = Field(env="PINECONE_INDEX", description="pinecone index", default="")

    unstructured_api_key: SecretStr = Field(env="UNSTRUCTURED_API_KEY", description="unstructured api key", default="")
    unstructured_api_url: str = Field(
        env="UNSTRUCTURED_API_URL",
        description="unstructured api url",
        default="https://api.unstructured.io/general/v0/general",
    )
    brave_search_api_key: SecretStr = Field(env="BRAVE_SEARCH_API_KEY", description="Brave Search API key", default="")

    anthropic_api_key: SecretStr = Field(env="ANTHROPIC_API_KEY", description="claude api key", default="")
    groq_api_key: SecretStr = Field(env="GROQ_API_KEY", description="groq api key", default="")
    cohere_api_key: SecretStr = Field(env="COHERE_API_KEY", description="cohere api key", default="")
    tavily_api_key: SecretStr = Field(env="TAVILY_API_KEY", description="tavily api key", default="")
    brave_search_api_key: SecretStr = Field(env="BRAVE_SEARCH_API_KEY", description="Brave Search API key", default="")

    langchain_endpoint: str = Field(
        env="LANGCHAIN_ENDPOINT", description="langchain endpoint", default="https://api.smith.langchain.com"
    )
    langchain_tracing_v2: bool = Field(
        env="LANGCHAIN_TRACING_V2", description="langchain tracing version", default=False
    )
    langchain_api_key: SecretStr = Field(
        env="LANGCHAIN_API_KEY", description="langchain api key for langsmith", default=""
    )
    langchain_hub_api_url: str = Field(
        env="LANGCHAIN_HUB_API_URL",
        description="langchain hub api url for langsmith",
        default="https://api.hub.langchain.com",
    )
    langchain_hub_api_key: SecretStr = Field(
        env="LANGCHAIN_HUB_API_KEY", description="langchain hub api key for langsmith", default=""
    )
    langchain_project: str = Field(
        env="LANGCHAIN_PROJECT", description="langsmith project name", default="sandbox_agent"
    )
    debug_aider: bool = Field(env="DEBUG_AIDER", description="debug tests stuff written by aider", default=False)

    local_test_debug: bool = Field(env="LOCAL_TEST_DEBUG", description="enable local debug testing", default=False)
    local_test_enable_evals: bool = Field(
        env="LOCAL_TEST_ENABLE_EVALS", description="enable local debug testing with evals", default=False
    )
    python_debug: bool = Field(env="PYTHON_DEBUG", description="enable bpdb on cli", default=False)
    experimental_redis_memory: bool = Field(
        env="EXPERIMENTAL_REDIS_MEMORY", description="enable experimental redis memory", default=False
    )

    oco_openai_api_key: SecretStr = Field(env="OCO_OPENAI_API_KEY", description="opencommit api key", default="")
    oco_tokens_max_input: int = Field(env="OCO_TOKENS_MAX_INPUT", description="OCO_TOKENS_MAX_INPUT", default=4096)
    oco_tokens_max_output: int = Field(env="OCO_TOKENS_MAX_OUTPUT", description="OCO_TOKENS_MAX_OUTPUT", default=500)
    oco_model: str = Field(env="OCO_MODEL", description="OCO_MODEL", default="gpt-4o")
    oco_language: str = Field(env="OCO_LANGUAGE", description="OCO_LANGUAGE", default="en")
    oco_prompt_module: str = Field(
        env="OCO_PROMPT_MODULE", description="OCO_PROMPT_MODULE", default="conventional-commit"
    )
    oco_ai_provider: str = Field(env="OCO_AI_PROVIDER", description="OCO_AI_PROVIDER", default="openai")

    openai_embeddings_model: str = Field(
        env="OPENAI_EMBEDDINGS_MODEL", description="openai embeddings model", default="text-embedding-3-large"
    )

    editor: str = Field(env="EDITOR", description="EDITOR", default="vim")
    visual: str = Field(env="VISUAL", description="VISUAL", default="vim")
    git_editor: str = Field(env="GIT_EDITOR", description="GIT_EDITOR", default="vim")

    llm_streaming: bool = Field(env="LLM_STREAMING", description="Enable streaming for LLM", default=False)
    llm_provider: str = Field(
        env="LLM_PROVIDER", description="LLM provider (e.g., openai, anthropic)", default="openai"
    )
    llm_max_retries: int = Field(
        env="LLM_MAX_RETRIES", description="Maximum number of retries for LLM API calls", default=3
    )
    llm_recursion_limit: int = Field(env="LLM_RECURSION_LIMIT", description="Recursion limit for LLM", default=50)
    llm_document_loader_type: str = Field(
        env="LLM_DOCUMENT_LOADER_TYPE", description="Document loader type", default="pymupdf"
    )
    llm_vectorstore_type: str = Field(env="LLM_VECTORSTORE_TYPE", description="Vector store type", default="pgvector")
    llm_embedding_model_type: str = Field(
        env="LLM_EMBEDDING_MODEL_TYPE", description="Embedding model type", default="text-embedding-3-large"
    )
    llm_key_value_stores_type: str = Field(
        env="LLM_KEY_VALUE_STORES_TYPE", description="Key-value stores type", default="redis"
    )
    # Variables for Postgres/pgvector
    pgvector_driver: str = Field(
        env="PGVECTOR_DRIVER",
        description="The database driver to use for pgvector (e.g., psycopg)",
        default="psycopg",
    )
    pgvector_host: str = Field(
        env="PGVECTOR_HOST",
        description="The hostname or IP address of the pgvector database server",
        default="localhost",
    )
    pgvector_port: int = Field(
        env="PGVECTOR_PORT",
        description="The port number of the pgvector database server",
        default=6432,
    )
    pgvector_database: str = Field(
        env="PGVECTOR_DATABASE",
        description="The name of the pgvector database",
        default="langchain",
    )
    pgvector_user: str = Field(
        env="PGVECTOR_USER",
        description="The username to connect to the pgvector database",
        default="langchain",
    )
    pgvector_password: SecretStr = Field(
        env="PGVECTOR_PASSWORD",
        description="The password to connect to the pgvector database",
        default="langchain",
    )
    pgvector_pool_size: int = Field(
        env="PGVECTOR_POOL_SIZE",
        description="The size of the connection pool for the pgvector database",
        default=10,
    )
    pgvector_dsn_uri: str = Field(
        env="PGVECTOR_DSN_URI",
        description="optional DSN URI, if set other pgvector_* settings are ignored",
        default="",
    )

    # Index - text splitter settings
    text_chunk_size: int = 2000
    text_chunk_overlap: int = 200
    text_splitter: Json[dict[str, Any]] = "{}"  # custom splitter settings

    # Variables for Postgres/pgvector
    # CONNECTION_STRING = PGVector.connection_string_from_db_params(
    #     driver=os.environ.get("PGVECTOR_DRIVER", "psycopg"),
    #     host=os.environ.get("PGVECTOR_HOST", "localhost"),
    #     port=int(os.environ.get("PGVECTOR_PORT", "6432")),
    #     database=os.environ.get("PGVECTOR_DATABASE", "langchain"),
    #     user=os.environ.get("PGVECTOR_USER", "langchain"),
    #     password=os.environ.get("PGVECTOR_PASSWORD", "langchain"),
    # )
    postgres_host: str = "localhost"
    postgres_port: int = 7432
    postgres_password: Optional[str] = "langchain"
    postgres_driver: Optional[str] = "psycopg"
    postgres_database: Optional[str] = "langchain"
    postgres_collection_name: Optional[str] = "langchain"
    postgres_user: Optional[str] = "langchain"
    enable_postgres: bool = True

    # # OpenAI model settings
    # openai_model_zoo: set[str] = Field(
    #     env="OPENAI_MODEL_ZOO",
    #     description="Set of all available models and embeddings",
    #     default_factory=lambda: MODEL_ZOO,
    # )
    # openai_model_config: dict[str, dict[str, Union[int, float]]] = Field(
    #     env="OPENAI_MODEL_CONFIG", description="Configuration for all models", default_factory=lambda: MODEL_CONFIG
    # )
    # openai_model_point: dict[str, str] = Field(
    #     env="OPENAI_MODEL_POINT",
    #     description="Mapping of model names to their latest version",
    #     default_factory=lambda: MODEL_POINT,
    # )
    # openai_model_point_config: dict[str, dict[str, Union[int, float]]] = Field(
    #     env="OPENAI_MODEL_POINT_CONFIG",
    #     description="Configuration for the latest version of each model",
    #     default_factory=lambda: _MODEL_POINT_CONFIG,
    # )
    # openai_embedding_model_dimensions_data: dict[str, int] = Field(
    #     env="OPENAI_EMBEDDING_MODEL_DIMENSIONS_DATA",
    #     description="Dimensions of each embedding model",
    #     default_factory=lambda: EMBEDDING_MODEL_DIMENSIONS_DATA,
    # )

    # Evaluation settings
    eval_max_concurrency: int = Field(
        env="EVAL_MAX_CONCURRENCY", description="Maximum number of concurrent evaluations", default=4
    )
    llm_model_name: str = Field(
        env="LLM_MODEL_NAME", description="Name of the LLM model to use", default="gpt-4o-mini", init=True
    )
    provider: str = Field(env="PROVIDER", description="AI provider (openai or anthropic)", default="openai")
    chunk_size: int = Field(env="CHUNK_SIZE", description="Size of each text chunk", default=1000)
    chunk_overlap: int = Field(env="CHUNK_OVERLAP", description="Overlap between text chunks", default=200)
    add_start_index: bool = Field(
        env="ADD_START_INDEX", description="Whether to add start index to text chunks", default=False
    )
    llm_embedding_model_name: str = Field(
        env="LLM_EMBEDDING_MODEL_NAME",
        description="Name of the embedding model to use",
        default="text-embedding-3-large",
    )
    llm_retriever_type: str = Field(
        env="LLM_RETRIEVER_TYPE",
        description="Type of retriever to use",
        default="vector_store",
    )
    default_search_kwargs: dict[str, int] = Field(
        env="DEFAULT_SEARCH_KWARGS",
        description="Default arguments for similarity search",
        default_factory=lambda: {"k": 2},
    )
    question_to_ask: str = Field(
        env="QUESTION_TO_ASK",
        description="Question to ask for evaluation",
        default="What is the main cause of climate change?",
    )
    dataset_name: str = Field(
        env="DATASET_NAME", description="Name of the dataset to use for evaluation", default="Climate Change Q&A"
    )

    # Model-specific settings
    max_tokens: int = Field(env="MAX_TOKENS", description="Maximum number of tokens for the model", default=900)
    max_retries: int = Field(env="MAX_RETRIES", description="Maximum number of retries for API calls", default=9)

    # # Evaluation feature flags
    compare_models_feature_flag: bool = Field(
        env="COMPARE_MODELS_FEATURE_FLAG", description="Enable comparing different models", default=False
    )
    rag_answer_v_reference_feature_flag: bool = Field(
        env="RAG_ANSWER_V_REFERENCE_FEATURE_FLAG", description="Enable comparing RAG answer to reference", default=False
    )
    helpfulness_feature_flag: bool = Field(
        env="HELPFULNESS_FEATURE_FLAG", description="Enable helpfulness evaluation", default=False
    )
    rag_answer_hallucination_feature_flag: bool = Field(
        env="RAG_ANSWER_HALLUCINATION_FEATURE_FLAG",
        description="Enable evaluating RAG answer hallucination",
        default=False,
    )
    rag_doc_relevance_feature_flag: bool = Field(
        env="RAG_DOC_RELEVANCE_FEATURE_FLAG", description="Enable evaluating RAG document relevance", default=False
    )
    rag_doc_relevance_and_hallucination_feature_flag: bool = Field(
        env="RAG_DOC_RELEVANCE_AND_HALLUCINATION_FEATURE_FLAG",
        description="Enable evaluating RAG document relevance and hallucination",
        default=False,
    )
    rag_answer_accuracy_feature_flag: bool = Field(
        env="RAG_ANSWER_ACCURACY_FEATURE_FLAG", description="Enable evaluating RAG answer accuracy", default=True
    )
    helpfulness_testing_feature_flag: bool = Field(
        env="HELPFULNESS_TESTING_FEATURE_FLAG", description="Enable helpfulness testing", default=False
    )
    rag_string_embedding_distance_metrics_feature_flag: bool = Field(
        env="RAG_STRING_EMBEDDING_DISTANCE_METRICS_FEATURE_FLAG",
        description="Enable evaluating RAG string embedding distance metrics",
        default=False,
    )

    llm_memory_type: str = Field(env="LLM_MEMORY_TYPE", description="Type of memory to use", default="memorysaver")
    llm_memory_enabled: bool = Field(env="LLM_MEMORY_ENABLED", description="Enable memory", default=True)
    llm_human_loop_enabled: bool = Field(env="LLM_HUMAN_LOOP_ENABLED", description="Enable human loop", default=False)
    # Tool allowlist
    tool_allowlist: list[str] = ["tavily_search", "magic_function"]

    # Tool-specific configuration
    tavily_search_max_results: int = 3

    agent_type: str = Field(env="AGENT_TYPE", description="Type of agent to use", default="basic")

    @model_validator(mode="before")
    @classmethod
    def pre_update(cls, values: dict[str, Any]) -> dict[str, Any]:
        llm_model_name = values.get("llm_model_name")
        llm_embedding_model_name = values.get("llm_embedding_model_name")
        logger.info(f"llm_model_name: {llm_model_name}")
        logger.info(f"llm_embedding_model_name: {llm_embedding_model_name}")
        if llm_model_name:
            values["max_tokens"] = MODEL_CONFIG[llm_model_name]["max_tokens"]
            values["max_output_tokens"] = MODEL_CONFIG[llm_model_name]["max_output_tokens"]
            values["prompt_cost_per_token"] = MODEL_CONFIG[llm_model_name]["prompt_cost_per_token"]
            values["completion_cost_per_token"] = MODEL_CONFIG[llm_model_name]["completion_cost_per_token"]
            if llm_embedding_model_name:
                values["embedding_max_tokens"] = EMBEDDING_MODEL_DIMENSIONS_DATA[llm_embedding_model_name]
                values["embedding_model_dimensions"] = EMBEDDING_MODEL_DIMENSIONS_DATA[llm_embedding_model_name]
        else:
            llm_model_name = "gpt-4o-mini"
            llm_embedding_model_name = "text-embedding-3-large"
            logger.info(f"setting default llm_model_name: {llm_model_name}")
            logger.info(f"setting default llm_embedding_model_name: {llm_embedding_model_name}")
            values["max_tokens"] = MODEL_CONFIG[llm_model_name]["max_tokens"]
            values["max_output_tokens"] = MODEL_CONFIG[llm_model_name]["max_output_tokens"]
            values["prompt_cost_per_token"] = MODEL_CONFIG[llm_model_name]["prompt_cost_per_token"]
            values["completion_cost_per_token"] = MODEL_CONFIG[llm_model_name]["completion_cost_per_token"]
            values["embedding_max_tokens"] = EMBEDDING_MODEL_DIMENSIONS_DATA[llm_embedding_model_name]
            values["embedding_model_dimensions"] = EMBEDDING_MODEL_DIMENSIONS_DATA[llm_embedding_model_name]

        return values

    @model_validator(mode="after")
    def post_root(self) -> Self:
        redis_path = f"/{self.redis_base}" if self.redis_base is not None else ""
        redis_pass = self.redis_pass if self.redis_pass is not None else None
        redis_user = self.redis_user if self.redis_user is not None else None
        logger.info(f"before redis_path: {redis_path}")
        logger.info(f"before redis_pass: {redis_pass}")
        logger.info(f"before redis_user: {redis_user}")
        if redis_pass is None and redis_user is None:
            self.redis_url = URL.build(
                scheme="redis",
                host=self.redis_host,
                port=self.redis_port,
                path=redis_path,
            )
        else:
            self.redis_url = URL.build(
                scheme="redis",
                host=self.redis_host,
                port=self.redis_port,
                path=redis_path,
                user=redis_user,
                password=redis_pass.get_secret_value(),
            )

        return self

    @property
    def postgres_url(self) -> URL:
        """
        Assemble postgres URL from settings.

        :return: postgres URL.
        """
        return f"postgresql+{self.postgres_driver}://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    # @property
    # def redis_url(self) -> URL:
    #     """
    #     Assemble REDIS URL from settings.

    #     :return: redis URL.
    #     """
    #     path = f"/{self.redis_base}" if self.redis_base is not None else ""
    #     return URL.build(
    #         scheme="redis",
    #         host=self.redis_host,
    #         port=self.redis_port,
    #         user=self.redis_user,
    #         password=self.redis_pass.get_secret_value(),
    #         path=path,
    #     )

    @field_serializer(
        "discord_token",
        "openai_api_key",
        "redis_pass",
        "pinecone_api_key",
        "langchain_api_key",
        "langchain_hub_api_key",
        when_used="json",
    )
    def dump_secret(self, v):
        return v.get_secret_value()


aiosettings = AioSettings()  # sourcery skip: docstrings-for-classes, avoid-global-variables
