"""Discord client for the sandbox agent."""

from __future__ import annotations

import asyncio
import base64
import datetime
import io
import json
import logging
import os
import pathlib
import re
import sys
import tempfile
import time
import traceback
import typing
import uuid

from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Coroutine, Iterable
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NoReturn, Optional, Tuple, TypeVar, Union, cast

import discord

from codetiming import Timer
from discord.ext import commands
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS, Chroma, PGVector
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.state import CompiledStateGraph
from logging_tree import printout
from loguru import logger as LOGGER
from PIL import Image
from redis.asyncio import ConnectionPool as RedisConnectionPool
from starlette.responses import JSONResponse, StreamingResponse

from sandbox_agent.aio_settings import aiosettings


DESCRIPTION = """An example bot to showcase the discord.ext.commands extension
module.

There are a number of utility commands being showcased here."""


class DiscordClient(commands.Bot):
    user: discord.ClientUser
    command_stats: Counter[str]
    socket_stats: Counter[str]
    command_types_used: Counter[bool]
    logging_handler: Any
    bot_app_info: discord.AppInfo
    old_tree_error = Callable[[discord.Interaction, discord.app_commands.AppCommandError], Coroutine[Any, Any, None]]

    chat_model: ChatOpenAI | None
    embedding_model: OpenAIEmbeddings | None
    vector_store: type(Chroma) | type[FAISS] | None
    agent: CompiledStateGraph | None
    graph: CompiledStateGraph | None

    def __init__(self):
        allowed_mentions = discord.AllowedMentions(roles=False, everyone=False, users=True)
        intents: discord.flags.Intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        intents.bans = True
        intents.emojis = True
        intents.voice_states = True
        intents.messages = True
        intents.reactions = True
        # super().__init__(command_prefix=aiosettings.prefix, intents=intents)

        super().__init__(
            command_prefix=commands.when_mentioned_or(aiosettings.prefix),
            description=DESCRIPTION,
            pm_help=None,
            help_attrs=dict(hidden=True),
            chunk_guilds_at_startup=False,
            heartbeat_timeout=150.0,
            allowed_mentions=allowed_mentions,
            intents=intents,
            enable_debug_events=True,
        )

        self.pool: RedisConnectionPool | None = None
        # ------------------------------------------------
        # from bot
        # ------------------------------------------------
        # Create a queue that we will use to store our "workload".
        self.queue: asyncio.Queue = asyncio.Queue()

        self.tasks: list[Any] = []

        self.num_workers = 3

        self.total_sleep_time = 0

        self.start_time = datetime.datetime.now()

        self.typerCtx: dict | None = None

        #### For upscaler

        self.job_queue: dict[Any, Any] = {}

        # self.db: RedisConnectionPool | None = None

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if torch.cuda.is_available():
        #     torch.set_default_tensor_type(torch.cuda.HalfTensor)

        # self.current_task = None

        self.client_id: int | str = aiosettings.discord_client_id

        # shard_id: List[datetime.datetime]
        # shows the last attempted IDENTIFYs and RESUMEs
        self.resumes: defaultdict[int, list[datetime.datetime]] = defaultdict(list)
        self.identifies: defaultdict[int, list[datetime.datetime]] = defaultdict(list)

        # in case of even further spam, add a cooldown mapping
        # for people who excessively spam commands
        self.spam_control = commands.CooldownMapping.from_cooldown(10, 12.0, commands.BucketType.user)

        # A counter to auto-ban frequent spammers
        # Triggering the rate limit 5 times in a row will auto-ban the user from the bot.
        self._auto_spam_count = Counter()

        self.channel_list = [int(x) for x in CHANNEL_ID.split(",")]

    async def on_ready(self):
        print(f"Logged in as {self.user}")
