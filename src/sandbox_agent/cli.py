"""sandbox_agent.cli"""

# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# SOURCE: https://github.com/tiangolo/typer/issues/88#issuecomment-1732469681
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import traceback
import typing

from collections.abc import Awaitable, Iterable, Sequence
from enum import Enum
from functools import partial, wraps
from importlib import import_module, metadata
from importlib.metadata import version as importlib_metadata_version
from pathlib import Path
from re import Pattern
from typing import Annotated, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import anyio
import asyncer
import bpdb
import discord
import rich
import sentry_sdk
import typer

from langchain.globals import set_debug, set_verbose
from langchain_chroma import Chroma as ChromaVectorStore
from loguru import logger as LOGGER
from pinecone import Pinecone, ServerlessSpec  # pyright: ignore[reportAttributeAccessIssue]
from pinecone.core.openapi.data.model.describe_index_stats_response import DescribeIndexStatsResponse
from pinecone.core.openapi.data.model.query_response import QueryResponse
from pinecone.core.openapi.data.model.upsert_response import UpsertResponse
from pinecone.data.index import Index
from redis.asyncio import ConnectionPool, Redis
from rich import print, print_json
from rich.console import Console
from rich.pretty import pprint
from rich.prompt import Prompt
from rich.table import Table
from typer import Typer

import sandbox_agent

from sandbox_agent.aio_settings import aiosettings, get_rich_console
from sandbox_agent.asynctyper import AsyncTyper
from sandbox_agent.bot_logger import get_logger, global_log_config
from sandbox_agent.utils import repo_typing
from sandbox_agent.utils.base import print_line_seperator
from sandbox_agent.utils.file_functions import fix_path


# from sandbox_agent.services.chroma_service import ChromaService
# from sandbox_agent.services.screencrop_service import ImageService
# from sandbox_agent import db
# from sandbox_agent.monitoring.sentry import sentry_init
# from sandbox_agent.bot import AsyncGoobBot
# from sentry_sdk.integrations.argv import ArgvIntegration
# from sentry_sdk.integrations.atexit import AtexitIntegration
# from sentry_sdk.integrations.dedupe import DedupeIntegration
# from sentry_sdk.integrations.excepthook import ExcepthookIntegration
# from sentry_sdk.integrations.modules import ModulesIntegration
# from sentry_sdk.integrations.stdlib import StdlibIntegration
# from sentry_sdk.integrations.threading import ThreadingIntegration

# SOURCE: https://python.langchain.com/v0.2/docs/how_to/debugging/
if aiosettings.debug_langchain:
    # Setting the global debug flag will cause all LangChain components with callback support (chains, models, agents, tools, retrievers) to print the inputs they receive and outputs they generate. This is the most verbose setting and will fully log raw inputs and outputs.
    set_debug(True)
    # Setting the verbose flag will print out inputs and outputs in a slightly more readable format and will skip logging certain raw outputs (like the token usage stats for an LLM call) so that you can focus on application logic.
    set_verbose(True)


# if aiosettings.enable_sentry:
#     sentry_init(
#         # Set traces_sample_rate to 1.0 to capture 100%
#         # of transactions for performance monitoring.
#         traces_sample_rate=1.0,
#         # Set profiles_sample_rate to 1.0 to profile 100%
#         # of sampled transactions.
#         # We recommend adjusting this value in production.
#         profiles_sample_rate=1.0,
#     )
#     logging.getLogger("sentry_sdk").setLevel(logging.WARNING)


global_log_config(
    log_level=logging.getLevelName("DEBUG"),
    json=False,
)


class ChromaChoices(str, Enum):
    load = "load"
    generate = "generate"
    get_response = "get_response"


APP = AsyncTyper()
console = Console()
cprint = console.print


# Load existing subcommands
def load_commands(directory: str = "subcommands"):
    script_dir = Path(__file__).parent
    subcommands_dir = script_dir / directory

    LOGGER.info(f"Loading subcommands from {subcommands_dir}")

    for filename in os.listdir(subcommands_dir):
        if filename.endswith("_cmd.py"):
            module_name = f'{__name__.split(".")[0]}.{directory}.{filename[:-3]}'
            module = import_module(module_name)
            if hasattr(module, "app"):
                APP.add_typer(module.app, name=filename[:-7])


def version_callback(version: bool) -> None:
    """Print the version of sandbox_agent."""
    if version:
        rich.print(f"sandbox_agent version: {sandbox_agent.__version__}")
        raise typer.Exit()


@APP.command()
def version() -> None:
    """Version command"""
    rich.print(f"sandbox_agent version: {sandbox_agent.__version__}")


@APP.command()
def deps() -> None:
    """Deps command"""
    rich.print(f"sandbox_agent version: {sandbox_agent.__version__}")
    rich.print(f"langchain_version: {importlib_metadata_version('langchain')}")
    rich.print(f"langchain_community_version: {importlib_metadata_version('langchain_community')}")
    rich.print(f"langchain_core_version: {importlib_metadata_version('langchain_core')}")
    rich.print(f"langchain_openai_version: {importlib_metadata_version('langchain_openai')}")
    rich.print(f"langchain_text_splitters_version: {importlib_metadata_version('langchain_text_splitters')}")
    rich.print(f"langchain_chroma_version: {importlib_metadata_version('langchain_chroma')}")
    rich.print(f"chromadb_version: {importlib_metadata_version('chromadb')}")
    rich.print(f"langsmith_version: {importlib_metadata_version('langsmith')}")
    rich.print(f"pydantic_version: {importlib_metadata_version('pydantic')}")
    rich.print(f"pydantic_settings_version: {importlib_metadata_version('pydantic_settings')}")
    rich.print(f"ruff_version: {importlib_metadata_version('ruff')}")


@APP.command()
def about() -> None:
    """About command"""
    typer.echo("This is GoobBot CLI")


@APP.command()
def show() -> None:
    """Show command"""
    cprint("\nShow sandbox_agent", style="yellow")


def main():
    APP()
    load_commands()


def entry():
    """Required entry point to enable hydra to work as a console_script."""
    main()  # pylint: disable=no-value-for-parameter


async def run_bot():
    """Run the bot"""
    LOGGER.info("Running bot")
    pass


# async def run_bot():
#     try:
#         pool: ConnectionPool = db.get_redis_conn_pool()
#     except Exception as ex:
#         print(f"{ex}")
#         exc_type, exc_value, exc_traceback = sys.exc_info()
#         print(f"Error Class: {ex.__class__}")
#         output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
#         print(output)
#         print(f"exc_type: {exc_type}")
#         print(f"exc_value: {exc_value}")
#         traceback.print_tb(exc_traceback)
#         if aiosettings.dev_mode:
#             bpdb.pm()
#     async with AsyncGoobBot() as bot:
#         if aiosettings.enable_redis:
#             bot.pool = pool
#         await bot.start()

#     await LOGGER.complete()


@APP.command()
def run_pyright() -> None:
    """Generate typestubs GoobAI"""
    typer.echo("Generating type stubs for GoobAI")
    repo_typing.run_pyright()


@APP.command()
def go() -> None:
    """Main entry point for GoobAI"""
    typer.echo("Starting up GoobAI Bot")
    asyncio.run(run_bot())


def handle_sigterm(signo, frame):
    sys.exit(128 + signo)  # this will raise SystemExit and cause atexit to be called


signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == "__main__":
    APP()
