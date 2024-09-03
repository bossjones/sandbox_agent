"""test_settings"""

from __future__ import annotations

import asyncio
import os

from collections.abc import Iterable, Iterator
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING

import pytest_asyncio

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
        monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
        monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_TOKEN", "fake_discord_token")
        monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_ADMIN_USER_ID", 1337)
        monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_SERVER_ID", 1337)
        monkeypatch.setenv("GOOB_AI_CONFIG_DISCORD_CLIENT_ID", 8008)
        monkeypatch.setenv("GOOB_AI_CONFIG_OPENAI_API_KEY", "fake_openai_key")
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
        monkeypatch.setenv("GOOB_AI_CONFIG_POSTGRES_HOST", "envhost")
        monkeypatch.setenv("GOOB_AI_CONFIG_POSTGRES_PORT", "5555")
        monkeypatch.setenv("GOOB_AI_CONFIG_POSTGRES_USER", "envuser")
        monkeypatch.setenv("GOOB_AI_CONFIG_POSTGRES_PASSWORD", "envpass")
        monkeypatch.setenv("GOOB_AI_CONFIG_POSTGRES_DRIVER", "envdriver")
        monkeypatch.setenv("GOOB_AI_CONFIG_POSTGRES_DATABASE", "envdb")
        monkeypatch.setenv("GOOB_AI_CONFIG_ENABLE_POSTGRES", "false")

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