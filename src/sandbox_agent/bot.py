# pylint: disable=no-member
# pylint: disable=possibly-used-before-assignment
# pyright: reportImportCycles=false
# pyright: reportAttributeAccessIssue=false
# mypy: disable-error-code="index"
# mypy: disable-error-code="no-redef"

"""Main bot implementation for the sandbox agent."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import traceback

from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import discord
import rich

from codetiming import Timer
from discord.ext import commands
from loguru import logger as LOGGER
from PIL import Image

from sandbox_agent import shell, utils
from sandbox_agent.aio_settings import aiosettings
from sandbox_agent.clients.discord import DiscordClient
from sandbox_agent.factories import ChatModelFactory, EmbeddingModelFactory, VectorStoreFactory
from sandbox_agent.utils import file_operations
from sandbox_agent.utils.context import Context
from sandbox_agent.utils.misc import CURRENTFUNCNAME


LOGGER.add(sys.stderr, level="DEBUG")

DESCRIPTION = """An example bot to showcase the discord.ext.commands extension
module.

There are a number of utility commands being showcased here."""

HERE = os.path.dirname(__file__)

INVITE_LINK = "https://discordapp.com/api/oauth2/authorize?client_id={}&scope=bot&permissions=0"

HOME_PATH = os.environ.get("HOME")

COMMAND_RUNNER = {"dl_thumb": shell.run_coroutine_subprocess}
THREAD_DATA = defaultdict()


class ProxyObject(discord.Object):
    def __init__(self, guild: Optional[discord.abc.Snowflake]):
        super().__init__(id=0)
        self.guild: Optional[discord.abc.Snowflake] = guild


class SandboxAgent(DiscordClient):
    def __init__(self):
        super().__init__()
        self.chat_model = ChatModelFactory.create()
        self.embedding_model = EmbeddingModelFactory.create()
        self.vector_store = VectorStoreFactory.create(aiosettings.vector_store_type)

    async def process_attachments(self, message: discord.Message) -> None:
        """Process attachments in a Discord message."""
        if len(message.attachments) <= 0:
            return

        root_temp_dir = file_operations.create_temp_directory()
        uploaded_file_paths = []
        for attachment in message.attachments:
            LOGGER.debug(f"Downloading file from {attachment.url}")
            file_path = os.path.join(root_temp_dir, attachment.filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status != 200:
                        raise aiohttp.ClientException(f"Error downloading file from {attachment.url}")
                    data = await resp.read()

                    with open(file_path, "wb") as f:
                        f.write(data)

            uploaded_file_paths.append(file_path)

    async def check_for_attachments(self, message: discord.Message) -> str:
        """Check a Discord message for attachments and process image URLs."""
        message_content: str = message.content
        url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

        if "https://tenor.com/view/" in message_content:
            # Process Tenor GIF URL
            start_index = message_content.index("https://tenor.com/view/")
            end_index = message_content.find(" ", start_index)
            tenor_url = message_content[start_index:] if end_index == -1 else message_content[start_index:end_index]
            words = tenor_url.split("/")[-1].split("-")[:-1]
            sentence = " ".join(words)
            message_content = f"{message_content} [{message.author.display_name} posts an animated {sentence} ]"
            return message_content.replace(tenor_url, "")
        elif url_pattern.match(message_content):
            # Process image URL
            response = await file_operations.download_image(message_content)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Process attached image
            image_url = message.attachments[0].url
            response = await file_operations.download_image(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")

        await LOGGER.complete()
        return message_content

    def get_attachments(
        self, message: discord.Message
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[str]]:
        """Retrieve attachment data from a Discord message."""
        attachment_data_list_dicts = []
        local_attachment_file_list = []
        local_attachment_data_list_dicts = []
        media_filepaths = []

        for attm in message.attachments:
            data = utils.attachment_to_dict(attm)
            attachment_data_list_dicts.append(data)

        return attachment_data_list_dicts, local_attachment_file_list, local_attachment_data_list_dicts, media_filepaths

    async def write_attachments_to_disk(self, message: discord.Message) -> None:
        """Save attachments from a Discord message to disk."""
        ctx = await self.get_context(message)
        attachment_data_list_dicts, local_attachment_file_list, local_attachment_data_list_dicts, media_filepaths = (
            self.get_attachments(message)
        )

        tmpdirname = file_operations.create_temp_directory()

        with Timer(text="\nTotal elapsed time: {:.1f}"):
            for an_attachment_dict in attachment_data_list_dicts:
                local_attachment_path = await file_operations.handle_save_attachment_locally(
                    an_attachment_dict, tmpdirname
                )
                local_attachment_file_list.append(local_attachment_path)

            for some_file in local_attachment_file_list:
                local_data_dict = file_operations.file_to_local_data_dict(some_file, tmpdirname)
                local_attachment_data_list_dicts.append(local_data_dict)
                path_to_image = utils.file_functions.fix_path(local_data_dict["filename"])
                media_filepaths.append(path_to_image)

            try:
                for media_fpaths in media_filepaths:
                    (
                        full_path_input_file,
                        full_path_output_file,
                        get_timestamp,
                    ) = await file_operations.details_from_file(media_fpaths, cwd=f"{tmpdirname}")
            except Exception as ex:
                await ctx.send(embed=discord.Embed(description="Could not download story...."))
                print(ex)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                LOGGER.error(f"Error Class: {str(ex.__class__)}")
                output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
                LOGGER.warning(output)
                await ctx.send(embed=discord.Embed(description=output))
                LOGGER.error(f"exc_type: {exc_type}")
                LOGGER.error(f"exc_value: {exc_value}")
                traceback.print_tb(exc_traceback)

            tree_list = file_operations.get_file_tree(tmpdirname)
            rich.print("tree_list ->")
            rich.print(tree_list)

            file_to_upload_list = [f"{p}" for p in tree_list]
            LOGGER.debug(f"{type(self).__name__} -> file_to_upload_list = {file_to_upload_list}")
            rich.print(file_to_upload_list)
            await LOGGER.complete()

    async def on_message(self, message: discord.Message) -> None:
        """Handle incoming messages and process commands."""
        LOGGER.info(f"message = {message}")
        LOGGER.info("ITS THIS ONE BOSS")

        LOGGER.info(f"You are in function: {CURRENTFUNCNAME()}")
        LOGGER.info(f"This function's caller was: {CURRENTFUNCNAME(1)}")

        LOGGER.info(f"Thread message to process - {message.author}: {message.content[:50]}")
        if message.author.bot:
            LOGGER.info(f"Skipping message from bot itself, message.author.bot = {message.author.bot}")
            return

        if message.content.startswith("%"):
            LOGGER.info("Skipping message that starts with %")
            return

        await self.process_attachments(message)
        if message.content.strip() != "":
            await self.process_commands(message)
        await LOGGER.complete()

    async def close(self) -> None:
        """Close the bot and its associated resources."""
        await super().close()

    async def start(self) -> None:
        """Start the bot and connect to Discord."""
        await super().start(aiosettings.discord_token.get_secret_value(), reconnect=True)

    @commands.command()
    async def chat(self, ctx, *, message):
        """Implement chat functionality."""
        response = await self.chat_model.agenerate([message])
        await ctx.send(response.generations[0][0].text)

    # Add more commands as needed
