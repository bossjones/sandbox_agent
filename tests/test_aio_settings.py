# disable nsupported-membership-test
# pylint: disable=unsupported-membership-test
# pylint: disable=unsubscriptable-object
"""test_settings"""

from __future__ import annotations

import asyncio
import os

from collections.abc import Iterable, Iterator
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING

import pytest_asyncio

from pydantic import (
    AliasChoices,
    AmqpDsn,
    BaseModel,
    Field,
    ImportString,
    PostgresDsn,
    RedisDsn,
    SecretBytes,
    SecretStr,
    field_serializer,
    model_validator,
)

import pytest

from sandbox_agent import aio_settings
from sandbox_agent.utils.file_functions import tilda


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


# TODO: Make sure os,environ unsets values while running tests
@pytest.mark.unittest
class TestSettings:
    def test_defaults(
        self,
    ) -> None:  # sourcery skip: extract-method
        test_settings: aio_settings.AioSettings = aio_settings.AioSettings()
        assert test_settings.monitor_host == "localhost"
        assert test_settings.monitor_port == 50102
        assert test_settings.prefix == "?"
        assert test_settings.discord_admin_user_id == 3282
        assert test_settings.discord_general_channel == 908894727779258390
        assert test_settings.discord_admin_user_invited == False
        assert test_settings.better_exceptions == 1
        assert test_settings.pythonasynciodebug == 1
        assert test_settings.globals_try_patchmatch == True
        assert test_settings.globals_always_use_cpu == False
        assert test_settings.globals_internet_available == True
        assert test_settings.globals_full_precision == False
        assert test_settings.globals_ckpt_convert == False
        assert test_settings.globals_log_tokenization == False
        assert test_settings.redis_host == "localhost"
        assert test_settings.redis_port == 7600
        assert test_settings.redis_user is None
        assert test_settings.redis_pass is None
        assert test_settings.redis_base is None
        if test_settings.enable_ai:
            assert str(test_settings.discord_token) == "**********"
            assert str(test_settings.discord_token) == "**********"
            assert str(test_settings.openai_api_key) == "**********"
            assert str(test_settings.pinecone_api_key) == "**********"
            assert str(test_settings.langchain_api_key) == "**********"
            assert str(test_settings.langchain_hub_api_key) == "**********"
        assert str(test_settings.redis_url) == "redis://localhost:7600"

    @pytest_asyncio.fixture
    async def test_integration_with_deleted_envs(self, monkeypatch: MonkeyPatch) -> None:
        # import bpdb
        # bpdb.set_trace()
        # paranoid about weird libraries trying to read env vars during testing
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_DISCORD_TOKEN", "fake_discord_token")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_DISCORD_TOKEN", "fake_discord_token")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_DISCORD_ADMIN_USER_ID", 1337)
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_DISCORD_SERVER_ID", 1337)
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_DISCORD_CLIENT_ID", 8008)
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_OPENAI_API_KEY", "fake_openai_key")
        monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_key")
        monkeypatch.setenv("PINECONE_API_KEY", "fake_pinecone_key")
        monkeypatch.setenv("PINECONE_INDEX", "fake_test_index")
        await asyncio.sleep(0.05)

        test_settings: aio_settings.AioSettings = aio_settings.AioSettings()
        assert test_settings.discord_admin_user_id == 1337
        assert test_settings.discord_client_id == 8008
        assert test_settings.discord_server_id == 1337
        assert test_settings.discord_token == "fake_discord_token"
        assert test_settings.openai_api_key == "fake_openai_key"
        assert test_settings.pinecone_api_key == "fake_pinecone_key"
        assert test_settings.pinecone_index == "fake_test_index"

    def test_postgres_defaults(self):
        test_settings = aio_settings.AioSettings()
        assert test_settings.postgres_host == "localhost"
        assert test_settings.postgres_port == 7432
        assert test_settings.postgres_password == "langchain"
        assert test_settings.postgres_driver == "psycopg"
        assert test_settings.postgres_database == "langchain"
        assert test_settings.postgres_collection_name == "langchain"
        assert test_settings.postgres_user == "langchain"
        assert test_settings.enable_postgres == True

    def test_postgres_url(self):
        test_settings = aio_settings.AioSettings()
        expected_url = "postgresql+psycopg://langchain:langchain@localhost:7432/langchain"
        assert test_settings.postgres_url == expected_url

    @pytest.mark.parametrize(
        "host,port,user,password,driver,database,expected",
        [
            (
                "testhost",
                5432,
                "testuser",
                "testpass",
                "postgresql",
                "testdb",
                "postgresql+postgresql://testuser:testpass@testhost:5432/testdb",
            ),
            (
                "127.0.0.1",
                5433,
                "admin",
                "securepass",
                "psycopg2",
                "production",
                "postgresql+psycopg2://admin:securepass@127.0.0.1:5433/production",
            ),
        ],
    )
    def test_custom_postgres_url(self, host, port, user, password, driver, database, expected):
        custom_settings = aio_settings.AioSettings(
            postgres_host=host,
            postgres_port=port,
            postgres_user=user,
            postgres_password=password,
            postgres_driver=driver,
            postgres_database=database,
        )
        assert custom_settings.postgres_url == expected

    @pytest.mark.asyncio
    async def test_postgres_env_variables(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_POSTGRES_HOST", "envhost")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_POSTGRES_PORT", "5555")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_POSTGRES_USER", "envuser")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_POSTGRES_PASSWORD", "envpass")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_POSTGRES_DRIVER", "envdriver")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_POSTGRES_DATABASE", "envdb")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_ENABLE_POSTGRES", "false")

        test_settings = aio_settings.AioSettings()
        assert test_settings.postgres_host == "envhost"
        assert test_settings.postgres_port == 5555
        assert test_settings.postgres_user == "envuser"
        assert test_settings.postgres_password == "envpass"
        assert test_settings.postgres_driver == "envdriver"
        assert test_settings.postgres_database == "envdb"
        assert test_settings.enable_postgres == False

        expected_url = "postgresql+envdriver://envuser:envpass@envhost:5555/envdb"
        assert test_settings.postgres_url == expected_url

    # -------

    def test_redis_defaults(self):
        test_settings = aio_settings.AioSettings()
        assert test_settings.redis_host == "localhost"
        assert test_settings.redis_port == 7600
        assert test_settings.redis_user is None
        assert test_settings.redis_pass is None
        assert test_settings.redis_base is None
        assert isinstance(test_settings.enable_redis, bool)

    def test_redis_url(self):
        test_settings = aio_settings.AioSettings()
        expected_url = "redis://localhost:7600"
        assert str(test_settings.redis_url) == expected_url

    @pytest.mark.parametrize(
        "host,port,user,password,base,expected",
        [
            (
                "testhost",
                6379,
                "testuser",
                "testpass",
                0,
                "redis://testuser:testpass@testhost:6379/0",
            ),
            (
                "127.0.0.1",
                6380,
                None,
                None,
                1,
                "redis://127.0.0.1:6380/1",
            ),
        ],
    )
    def test_custom_redis_url(self, host, port, user, password, base, expected):
        custom_settings = aio_settings.AioSettings(
            redis_host=host,
            redis_port=port,
            redis_user=user,
            redis_pass=password,
            redis_base=base,
        )
        assert str(custom_settings.redis_url) == expected

    @pytest.mark.asyncio
    async def test_redis_env_variables(self, monkeypatch):
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_REDIS_HOST", "envhost")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_REDIS_PORT", "7777")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_REDIS_USER", "envuser")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_REDIS_PASS", "envpass")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_REDIS_BASE", "2")
        monkeypatch.setenv("SANDBOX_AGENT_CONFIG_ENABLE_REDIS", "true")

        test_settings = aio_settings.AioSettings()
        assert test_settings.redis_host == "envhost"
        assert test_settings.redis_port == 7777
        assert test_settings.redis_user == "envuser"
        assert test_settings.redis_pass.get_secret_value() == "envpass"
        assert test_settings.redis_base == 2
        assert test_settings.enable_redis == True

        expected_url = "redis://envuser:envpass@envhost:7777/2"
        assert str(test_settings.redis_url) == expected_url

    def test_model_settings(self):
        test_settings = aio_settings.AioSettings()
        assert isinstance(test_settings.discord_admin_user_id, int)
        assert isinstance(test_settings.discord_admin_user_invited, bool)
        assert isinstance(test_settings.discord_client_id, str)
        assert isinstance(test_settings.discord_general_channel, int)
        assert isinstance(test_settings.discord_server_id, int)
        assert isinstance(test_settings.discord_token, SecretStr)
        assert isinstance(test_settings.enable_ai, bool)
        assert isinstance(test_settings.enable_chroma, bool)
        assert isinstance(test_settings.enable_postgres, bool)
        assert isinstance(test_settings.enable_redis, bool)
        assert isinstance(test_settings.enable_sentry, bool)
        assert isinstance(test_settings.experimental_redis_memory, bool)
        assert isinstance(test_settings.oco_openai_api_key, SecretStr)
        assert isinstance(test_settings.openai_api_key, SecretStr)
        assert isinstance(test_settings.pinecone_api_key, SecretStr)
        assert isinstance(test_settings.rag_answer_accuracy_feature_flag, bool)
        assert isinstance(test_settings.rag_answer_hallucination_feature_flag, bool)
        assert isinstance(test_settings.rag_answer_v_reference_feature_flag, bool)
        assert isinstance(test_settings.rag_doc_relevance_and_hallucination_feature_flag, bool)
        assert isinstance(test_settings.rag_doc_relevance_feature_flag, bool)
        assert isinstance(test_settings.rag_string_embedding_distance_metrics_feature_flag, bool)
        assert test_settings.chat_history_buffer == 10
        assert test_settings.chat_model == "gpt-4o-mini"
        assert test_settings.editor in ["lvim", "vim", "nvim"]
        assert test_settings.eval_max_concurrency == 4
        assert test_settings.git_editor in ["lvim", "vim", "nvim"]
        assert test_settings.globals_always_use_cpu == False
        assert test_settings.globals_ckpt_convert == False
        assert test_settings.globals_full_precision == False
        assert test_settings.globals_internet_available == True
        assert test_settings.globals_log_tokenization == False
        assert test_settings.globals_try_patchmatch == True
        assert str(test_settings.groq_api_key) in [
            "**********",
            "",
            None,
            SecretStr(""),
            SecretStr("**********"),
            "None",
        ]
        assert test_settings.helpfulness_feature_flag == False
        assert test_settings.helpfulness_testing_feature_flag == False
        assert test_settings.http_client_debug_enabled == False
        assert str(test_settings.langchain_api_key) in [
            "**********",
            "",
            None,
            SecretStr(""),
            SecretStr("**********"),
            "None",
        ]
        assert test_settings.langchain_debug_logs == False
        assert test_settings.langchain_endpoint == "https://api.smith.langchain.com"
        assert str(test_settings.langchain_hub_api_key) in [
            "**********",
            "",
            None,
            SecretStr(""),
            SecretStr("**********"),
        ]
        assert test_settings.langchain_hub_api_url == "https://api.hub.langchain.com"
        assert test_settings.langchain_project == "sandbox_agent"
        assert isinstance(test_settings.langchain_tracing_v2, bool)
        assert test_settings.llm_embedding_model_name == "text-embedding-3-large"
        assert test_settings.llm_model_name == "gpt-4o-mini"
        assert test_settings.llm_temperature == 0.0
        assert isinstance(test_settings.local_test_debug, bool)
        assert isinstance(test_settings.local_test_enable_evals, bool)
        assert isinstance(test_settings.log_pii, bool)
        assert test_settings.max_retries == 9
        assert test_settings.max_tokens == 900
        assert test_settings.monitor_host == "localhost"
        assert test_settings.monitor_port == 50102
        assert test_settings.oco_ai_provider == "openai"
        assert test_settings.oco_language == "en"
        assert test_settings.oco_model == "gpt-4o"
        assert test_settings.oco_prompt_module == "conventional-commit"
        assert test_settings.oco_tokens_max_input == 4096
        assert test_settings.oco_tokens_max_output == 500
        assert test_settings.openai_embeddings_model == "text-embedding-3-large"
        assert test_settings.pinecone_env in ["us-east-1", "local"]
        assert test_settings.postgres_collection_name == "langchain"
        assert test_settings.postgres_database == "langchain"
        assert test_settings.postgres_driver == "psycopg"
        assert test_settings.postgres_host == "localhost"
        assert test_settings.postgres_password == "langchain"
        assert test_settings.postgres_port == 7432
        assert test_settings.postgres_url == "postgresql+psycopg://langchain:langchain@localhost:7432/langchain"
        assert test_settings.postgres_user == "langchain"
        assert test_settings.prefix == "?"
        assert test_settings.provider == "openai"
        assert isinstance(test_settings.python_debug, bool)
        assert isinstance(test_settings.pythonasynciodebug, bool)
        assert isinstance(test_settings.pythondevmode, bool)
        assert test_settings.question_to_ask == "What is the main cause of climate change?"
        assert test_settings.redis_host == "localhost"
        assert str(test_settings.redis_pass) in ["**********", "", None, SecretStr(""), SecretStr("**********"), "None"]
        assert test_settings.redis_port == 7600
        assert str(test_settings.redis_url) == "redis://localhost:7600"
        assert test_settings.redis_user is None
        assert test_settings.retry_stop_after_attempt == 3
        assert test_settings.retry_wait_exponential_max == 5
        assert test_settings.retry_wait_exponential_min == 1
        assert test_settings.retry_wait_exponential_multiplier == 2
        assert test_settings.retry_wait_fixed == 15
        assert str(test_settings.sentry_dsn) in ["**********", "", None, SecretStr(""), SecretStr("**********"), "None"]
        assert str(test_settings.tavily_api_key) in [
            "**********",
            "",
            None,
            SecretStr(""),
            SecretStr("**********"),
            "None",
        ]
        assert str(test_settings.unstructured_api_key) in [
            "**********",
            "",
            None,
            SecretStr(""),
            SecretStr("**********"),
        ]
        assert test_settings.unstructured_api_url == "https://api.unstructured.io/general/v0/general"
        assert test_settings.vision_model == "gpt-4o"

    # def test_model_zoo(self):
    #     test_settings = aio_settings.AioSettings()
    #     assert isinstance(test_settings.openai_model_zoo, set)
    #     assert "gpt-4o" in test_settings.openai_model_zoo
    #     assert "text-embedding-3-large" in test_settings.openai_model_zoo

    # def test_model_config(self):
    #     test_settings = aio_settings.AioSettings()
    #     assert isinstance(test_settings.openai_model_config, dict)
    #     assert "gpt-4o" in test_settings.openai_model_config
    #     assert "max_tokens" in test_settings.openai_model_config["gpt-4o"]
    #     assert "max_output_tokens" in test_settings.openai_model_config["gpt-4o"]
    #     assert "prompt_cost_per_token" in test_settings.openai_model_config["gpt-4o"]
    #     assert "completion_cost_per_token" in test_settings.openai_model_config["gpt-4o"]

    # def test_model_point(self):
    #     test_settings = aio_settings.AioSettings()
    #     assert isinstance(test_settings.openai_model_point, dict)
    #     assert "gpt-4o" in test_settings.openai_model_point
    #     assert test_settings.openai_model_point["gpt-4o"] == "gpt-4o-2024-08-06"

    # def test_model_point_config(self):
    #     test_settings = aio_settings.AioSettings()
    #     assert isinstance(test_settings.openai_model_point_config, dict)
    #     assert "gpt-4o" in test_settings.openai_model_point_config
    #     assert "max_tokens" in test_settings.openai_model_point_config["gpt-4o"]
    #     assert "max_output_tokens" in test_settings.openai_model_point_config["gpt-4o"]
    #     assert "prompt_cost_per_token" in test_settings.openai_model_point_config["gpt-4o"]
    #     assert "completion_cost_per_token" in test_settings.openai_model_point_config["gpt-4o"]

    # def test_embedding_model_dimensions(self):
    #     test_settings = aio_settings.AioSettings()
    #     assert isinstance(test_settings.openai_embedding_model_dimensions_data, dict)
    #     assert "text-embedding-3-large" in test_settings.openai_embedding_model_dimensions_data
    #     assert test_settings.openai_embedding_model_dimensions_data["text-embedding-3-large"] == 1024

    @pytest.mark.skip(reason="generated by cursor but not yet ready")
    @pytest.mark.flaky
    @pytest.mark.parametrize(
        "env_vars, expected_values",
        [
            (
                {
                    "LLM_STREAMING": "True",
                    "LLM_PROVIDER": "anthropic",
                    "LLM_MAX_RETRIES": "5",
                    "LLM_DOCUMENT_LOADER_TYPE": "unstructured",
                    "LLM_VECTORSTORE_TYPE": "pinecone",
                    "LLM_EMBEDDING_MODEL_TYPE": "text-embedding-ada-002",
                    "LLM_KEY_VALUE_STORES_TYPE": "dynamodb",
                },
                {
                    "llm_streaming": True,
                    "llm_provider": "anthropic",
                    "llm_max_retries": 5,
                    "llm_document_loader_type": "unstructured",
                    "llm_vectorstore_type": "pinecone",
                    "llm_embedding_model_type": "text-embedding-ada-002",
                    "llm_key_value_stores_type": "dynamodb",
                },
            ),
            (
                {},
                {
                    "llm_streaming": False,
                    "llm_provider": "openai",
                    "llm_max_retries": 3,
                    "llm_document_loader_type": "pymupdf",
                    "llm_vectorstore_type": "pgvector",
                    "llm_embedding_model_type": "text-embedding-3-large",
                    "llm_key_value_stores_type": "redis",
                },
            ),
        ],
    )
    def test_aio_settings_llm_configs(
        self, monkeypatch: MonkeyPatch, env_vars: dict[str, str], expected_values: dict[str, object]
    ) -> None:
        """
        Test the new llm_* configs in AioSettings.

        This test verifies that the `llm_*` configs are correctly loaded from environment variables
        and fallback to their default values when not provided.

        Args:
        ----
            monkeypatch (MonkeyPatch): Pytest monkeypatch fixture for setting environment variables.
            env_vars (dict[str, str]): Dictionary of environment variables to set.
            expected_values (dict[str, object]): Dictionary of expected config values.

        """
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

        settings = aio_settings.AioSettings()

        for key, expected_value in expected_values.items():
            assert getattr(settings, key) == expected_value

    def test_aio_settings_llm_configs_default_values(self) -> None:
        """
        Test the default values of the new llm_* configs in AioSettings.

        This test verifies that the `llm_*` configs have the correct default values
        when no environment variables are set.

        """
        settings = aio_settings.AioSettings()

        assert settings.llm_streaming is False
        assert settings.llm_provider == "openai"
        assert settings.llm_max_retries == 3
        assert settings.llm_document_loader_type == "pymupdf"
        assert settings.llm_vectorstore_type == "pgvector"
        assert settings.llm_embedding_model_type == "text-embedding-3-large"
        assert settings.llm_key_value_stores_type == "redis"

    def test_aio_settings_llm_configs_default_values(self) -> None:
        """
        Test the default values of the new llm_* configs in AioSettings.

        This test verifies that the `llm_*` configs have the correct default values
        when no environment variables are set.

        """
        settings = aio_settings.AioSettings()

        assert settings.llm_streaming is False
        assert settings.llm_provider == "openai"
        assert settings.llm_max_retries == 3
        assert settings.llm_document_loader_type == "pymupdf"
        assert settings.llm_vectorstore_type == "pgvector"
        assert settings.llm_embedding_model_type == "text-embedding-3-large"
        assert settings.llm_key_value_stores_type == "redis"
