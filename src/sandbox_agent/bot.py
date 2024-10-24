# pylint: disable=no-member
# pylint: disable=possibly-used-before-assignment
# pyright: reportImportCycles=false
# pyright: reportUnusedFunction=false
# pyright: reportInvalidTypeForm=false
# mypy: disable-error-code="index"
# mypy: disable-error-code="no-redef"

"""Main bot implementation for the sandbox agent."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import pathlib
import re
import sys
import traceback
import uuid

from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Coroutine, Iterable
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NoReturn, Optional, Tuple, TypeVar, Union, cast

import aiohttp
import bpdb
import discord
import rich

from codetiming import Timer
from discord.ext import commands
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS, PGVector
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler, StdOutCallbackHandler
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.state import CompiledStateGraph
from langsmith import traceable
from logging_tree import printout
from loguru import logger as LOGGER
from PIL import Image
from starlette.responses import JSONResponse

import sandbox_agent

from sandbox_agent import shell, utils
from sandbox_agent.agents.agent_executor import (
    AgentExecutorFactory,
    AsyncLoggingCallbackHandler,
    format_user_input,
    format_user_message,
)
from sandbox_agent.ai.evaluators import Evaluator
from sandbox_agent.ai.workflows import WorkflowFactory
from sandbox_agent.aio_settings import aiosettings
from sandbox_agent.bot_logger import REQUEST_ID_CONTEXTVAR
from sandbox_agent.clients.discord import DiscordClient
from sandbox_agent.clients.discord_client.utils import (
    GoobThreadConfig,
    SurfaceInfo,
    SurfaceType,
    attachment_to_dict,
    close_thread,
    details_from_file,
    download_image,
    extensions,
    file_to_local_data_dict,
    get_logger_tree_printout,
    handle_save_attachment_locally,
    preload_guild_data,
)
from sandbox_agent.constants import (
    ACTIVATE_THREAD_PREFX,
    CHANNEL_ID,
    INACTIVATE_THREAD_PREFIX,
    MAX_CHARS_PER_REPLY_MSG,
    MAX_THREAD_MESSAGES,
)
from sandbox_agent.factories import ChatModelFactory, EmbeddingModelFactory, VectorStoreFactory
from sandbox_agent.types import MessageLikeRepresentation
from sandbox_agent.utils import file_functions, file_operations
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
        """
        Initialize a ProxyObject.

        Args:
            guild (Optional[discord.abc.Snowflake]): The guild associated with the object.
        """
        super().__init__(id=0)
        self.guild: Optional[discord.abc.Snowflake] = guild


class SandboxAgent(DiscordClient):
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
        """Initialize the SandboxAgent."""
        super().__init__()
        self.chat_model = ChatModelFactory.create()
        self.embedding_model = EmbeddingModelFactory.create()
        self.vector_store = VectorStoreFactory.create(aiosettings.vector_store_type)
        self.agent = AgentExecutorFactory.create_plan_and_execute_agent()
        self.graph = WorkflowFactory.create()

    async def setup_hook(self) -> None:
        """
        Asynchronous setup hook for initializing the bot.

        This method is called to perform asynchronous setup tasks for the bot.
        It initializes the aiohttp session, sets up guild prefixes, retrieves
        bot application information, and loads extensions.

        It also sets the intents for members and message content to True.

        Raises
        ------
            Exception: If an extension fails to load, an exception is raised with
                        detailed error information.

        """

        # await super().setup_hook()

        self.session = aiohttp.ClientSession()
        self.prefixes: list[str] = [aiosettings.prefix]

        self.version = sandbox_agent.__version__
        self.guild_data: dict[Any, Any] = {}
        self.intents.members = True
        self.intents.message_content = True

        self.bot_app_info: discord.AppInfo = await self.application_info()
        self.owner_id: int = self.bot_app_info.owner.id  # pyright: ignore[reportAttributeAccessIssue]

        for ext in extensions():
            try:
                await self.load_extension(ext)
            except Exception as ex:
                print(f"Failed to load extension {ext} - exception: {ex}")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                LOGGER.error(f"Error Class: {str(ex.__class__)}")
                output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
                LOGGER.warning(output)
                LOGGER.error(f"exc_type: {exc_type}")
                LOGGER.error(f"exc_value: {exc_value}")
                traceback.print_tb(exc_traceback)
                raise

        await LOGGER.complete()

    @property
    def owner(self) -> discord.User:
        """
        Retrieve the owner of the bot.

        This property returns the owner of the bot as a discord.User object.
        The owner information is retrieved from the bot's application info.

        Returns
        -------
            discord.User: The owner of the bot.

        """
        return self.bot_app_info.owner  # pyright: ignore[reportAttributeAccessIssue]

    def _clear_gateway_data(self) -> None:
        """
        Clear gateway data older than one week.

        This method removes entries from the `identifies` and `resumes` dictionaries
        that are older than one week. It iterates through each shard's list of dates
        and deletes the entries that are older than the specified time frame.

        Returns
        -------
            None

        """
        one_week_ago = discord.utils.utcnow() - datetime.timedelta(days=7)
        for shard_id, dates in self.identifies.items():
            to_remove = [index for index, dt in enumerate(dates) if dt < one_week_ago]
            for index in reversed(to_remove):
                del dates[index]

        for shard_id, dates in self.resumes.items():
            to_remove = [index for index, dt in enumerate(dates) if dt < one_week_ago]
            for index in reversed(to_remove):
                del dates[index]

    async def before_identify_hook(self, shard_id: int, *, initial: bool) -> None:  # type: ignore
        """
        Perform actions before identifying the shard.

        This method is called before the bot identifies the shard with the Discord gateway.
        It clears old gateway data and appends the current timestamp to the identifies list
        for the given shard ID.

        Args:
        ----
            shard_id (int): The ID of the shard that is about to identify.
            initial (bool): Whether this is the initial identification of the shard.

        Returns:
        -------
            None

        """
        self._clear_gateway_data()
        self.identifies[shard_id].append(discord.utils.utcnow())
        await super().before_identify_hook(shard_id, initial=initial)

    async def on_command_error(self, ctx: Context, error: commands.CommandError) -> None:
        """
        Handle errors raised during command invocation.

        This method is called when an error is raised during the invocation of a command.
        It handles different types of command errors and sends appropriate messages to the user.

        Args:
        ----
            ctx (Context): The context in which the command was invoked.
            error (commands.CommandError): The error that was raised during command invocation.

        Returns:
        -------
            None

        """
        if isinstance(error, commands.NoPrivateMessage):
            await ctx.author.send("This command cannot be used in private messages.")
        elif isinstance(error, commands.DisabledCommand):
            await ctx.author.send("Sorry. This command is disabled and cannot be used.")
        elif isinstance(error, commands.CommandInvokeError):
            original = error.original  # pyright: ignore[reportAttributeAccessIssue]
            if not isinstance(original, discord.HTTPException):
                LOGGER.exception("In %s:", ctx.command.qualified_name, exc_info=original)
        elif isinstance(error, commands.ArgumentParsingError):
            await ctx.send(str(error))

        await LOGGER.complete()

    async def query_member_named(
        self, guild: discord.Guild, argument: str, *, cache: bool = False
    ) -> Optional[discord.Member]:
        """
        Query a member by their name, name + discriminator, or nickname.

        This asynchronous function searches for a member in the specified guild
        by their name, name + discriminator (e.g., username#1234), or nickname.
        It can optionally cache the results of the query.

        Args:
        ----
            guild (discord.Guild): The guild to query the member in.
            argument (str): The name, nickname, or name + discriminator combo to check.
            cache (bool): Whether to cache the results of the query. Defaults to False.

        Returns:
        -------
            Optional[discord.Member]: The member matching the query or None if not found.

        """
        if len(argument) > 5 and argument[-5] == "#":
            username, _, discriminator = argument.rpartition("#")
            members = await guild.query_members(username, limit=100, cache=cache)
            return discord.utils.get(members, name=username, discriminator=discriminator)
        else:
            members = await guild.query_members(argument, limit=100, cache=cache)

            return discord.utils.find(lambda m: m.name == argument or m.nick == argument, members)  # pylint: disable=consider-using-in # pyright: ignore[reportAttributeAccessIssue]

    async def get_or_fetch_member(self, guild: discord.Guild, member_id: int) -> Optional[discord.Member]:
        """
        Retrieve a member from the cache or fetch from the API if not found.

        This asynchronous function attempts to retrieve a member from the cache
        in the specified guild using the provided member ID. If the member is not
        found in the cache, it fetches the member from the Discord API. The function
        handles rate limiting and returns the member if found, or None if not found.

        Args:
        ----
            guild (discord.Guild): The guild to look in.
            member_id (int): The member ID to search for.

        Returns:
        -------
            Optional[discord.Member]: The member if found, or None if not found.

        """
        member = guild.get_member(member_id)
        if member is not None:
            return member

        shard: discord.ShardInfo = self.get_shard(guild.shard_id)  # type: ignore  # will never be None
        if shard.is_ws_ratelimited():
            try:
                member = await guild.fetch_member(member_id)
            except discord.HTTPException:
                return None
            else:
                return member

        members = await guild.query_members(limit=1, user_ids=[member_id], cache=True)
        return members[0] if members else None

    async def resolve_member_ids(
        self, guild: discord.Guild, member_ids: Iterable[int]
    ) -> AsyncIterator[discord.Member]:
        """
        Bulk resolve member IDs to member instances, if possible.

        This asynchronous function attempts to resolve a list of member IDs to their corresponding
        member instances within a specified guild. Members that cannot be resolved are discarded
        from the list. The function yields the resolved members lazily using an asynchronous iterator.

        Note:
        ----
            The order of the resolved members is not guaranteed to be the same as the input order.

        Args:
        ----
            guild (discord.Guild): The guild to resolve members from.
            member_ids (Iterable[int]): An iterable of member IDs to resolve.

        Yields:
        ------
            discord.Member: The resolved members.

        """
        needs_resolution = []
        for member_id in member_ids:
            member = guild.get_member(member_id)
            if member is not None:
                yield member
            else:
                needs_resolution.append(member_id)

        total_need_resolution = len(needs_resolution)
        if total_need_resolution == 1:
            shard: discord.ShardInfo = self.get_shard(guild.shard_id)  # type: ignore  # will never be None
            if shard.is_ws_ratelimited():
                try:
                    member = await guild.fetch_member(needs_resolution[0])
                except discord.HTTPException:
                    pass
                else:
                    yield member
            else:
                members = await guild.query_members(limit=1, user_ids=needs_resolution, cache=True)
                if members:
                    yield members[0]
        elif total_need_resolution <= 100:
            # Only a single resolution call needed here
            resolved = await guild.query_members(limit=100, user_ids=needs_resolution, cache=True)
            for member in resolved:
                yield member
        else:
            # We need to chunk these in bits of 100...
            for index in range(0, total_need_resolution, 100):
                to_resolve = needs_resolution[index : index + 100]
                members = await guild.query_members(limit=100, user_ids=to_resolve, cache=True)
                for member in members:
                    yield member

    async def on_ready(self) -> None:
        """
        Handle the event when the bot is ready.

        This method is called when the bot has successfully logged in and has completed
        its initial setup. It logs the bot's user information, sets the bot's presence,
        and prints the invite link. Additionally, it preloads guild data and logs the
        logger tree structure.

        Returns
        -------
            None

        """
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")
        self.invite = INVITE_LINK.format(self.user.id)
        self.guild_data = await preload_guild_data()
        print(
            f"""Logged in as {self.user}..
            Serving {len(self.users)} users in {len(self.guilds)} guilds
            Invite: {INVITE_LINK.format(self.user.id)}
        """
        )
        await self.change_presence(status=discord.Status.online, activity=discord.Game(aiosettings.bot_name))

        if not hasattr(self, "uptime"):
            self.uptime = discord.utils.utcnow()

        LOGGER.info(f"Ready: {self.user} (ID: {self.user.id})")

        LOGGER.info("LOGGING TREE:")
        await LOGGER.complete()
        await get_logger_tree_printout()

    async def on_shard_resumed(self, shard_id: int) -> None:
        """
        Handle the event when a shard resumes.

        This method is called when a shard successfully resumes its connection
        to the Discord gateway. It logs the shard ID and the timestamp of the
        resume event.

        Args:
        ----
            shard_id (int): The ID of the shard that resumed.

        Returns:
        -------
            None

        """
        await super().on_shard_resumed(shard_id)  # pyright: ignore[reportUnknownMemberType] # type: ignore

        LOGGER.info("Shard ID %s has resumed...", shard_id)
        self.resumes[shard_id].append(discord.utils.utcnow())
        await LOGGER.complete()

    async def check_for_attachments(self, message: discord.Message) -> str:
        """
        Check a Discord message for attachments and process image URLs.

        This asynchronous function examines a Discord message for attachments,
        processes Tenor GIF URLs, downloads and processes image URLs, and modifies
        the message content based on the extracted information.

        Args:
        ----
            message (discord.Message): The Discord message to check for attachments and process.

        Returns:
        -------
            str: The updated message content with extracted information.

        """
        message_content: str = message.content  # type: ignore
        url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")

        if "https://tenor.com/view/" in message_content:
            # Process Tenor GIF URL
            start_index = message_content.index("https://tenor.com/view/")
            end_index = message_content.find(" ", start_index)
            tenor_url = message_content[start_index:] if end_index == -1 else message_content[start_index:end_index]
            words = tenor_url.split("/")[-1].split("-")[:-1]
            sentence = " ".join(words)
            message_content = (
                f"{message_content.replace(tenor_url, '')} [{message.author.display_name} posts an animated {sentence}]"
            )
            return message_content.strip()
        elif url_pattern.search(message_content):
            # Process image URL
            url = url_pattern.search(message_content).group()
            await self.process_image(url)
        elif message.attachments:  # type: ignore
            # Process attached image
            image_url = message.attachments[0].url  # type: ignore
            await self.process_image(image_url)

        await LOGGER.complete()
        return message_content

    def get_attachments(
        self, message: discord.Message
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[str]]:
        """
        Retrieve attachment data from a Discord message.

        This function processes the attachments in a Discord message and converts each attachment
        to a dictionary format. It returns a tuple containing lists of dictionaries and file paths
        for further processing.

        Args:
        ----
            message (discord.Message): The Discord message containing attachments.

        Returns:
        -------
            Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[str]]:
            A tuple containing:
                - A list of dictionaries with attachment data.
                - A list of local attachment file paths.
                - A list of dictionaries with local attachment data.
                - A list of media file paths.

        """
        attachment_data_list_dicts = []
        local_attachment_file_list = []
        local_attachment_data_list_dicts = []
        media_filepaths = []

        for attm in message.attachments:  # pyright: ignore[reportAttributeAccessIssue]
            data = attachment_to_dict(attm)
            attachment_data_list_dicts.append(data)

        return attachment_data_list_dicts, local_attachment_file_list, local_attachment_data_list_dicts, media_filepaths

    def prepare_agent_input(
        self, message: Union[discord.Message, discord.Thread], user_real_name: str, surface_info: dict
    ) -> dict[str, Any]:
        """
        Prepare the agent input from the incoming Discord message.

        This function constructs the input dictionary to be sent to the agent based on the
        provided Discord message, user's real name, and surface information. It includes
        the message content, user_name, and any attachments if present.

        Args:
        ----
            message (discord.Message): The Discord message containing the user input.
            user_real_name (str): The real name of the user who sent the message.
            surface_info (Dict): The surface information related to the message.

        Returns:
        -------
            Dict[str, Any]: The input dictionary to be sent to the agent.

        """
        # ctx: Context = await self.get_context(message)  # type: ignore
        if isinstance(message, discord.Thread):
            agent_input = {"user_name": user_real_name, "message": message.starter_message.content}  # pyright: ignore[reportAttributeAccessIssue]
            attachments: list[discord.Attachment] = message.starter_message.attachments  # pyright: ignore[reportAttributeAccessIssue]
        elif isinstance(message, discord.Message):
            agent_input = {"user_name": user_real_name, "message": message.content}  # pyright: ignore[reportAttributeAccessIssue]
            attachments: list[discord.Attachment] = message.attachments  # pyright: ignore[reportAttributeAccessIssue]

        if len(attachments) > 0:  # pyright: ignore[reportAttributeAccessIssue]
            for attachment in attachments:  # pyright: ignore[reportAttributeAccessIssue]
                print(f"attachment -> {attachment}")  # pyright: ignore[reportAttributeAccessIssue]
                agent_input["file_name"] = attachment.filename  # pyright: ignore[reportAttributeAccessIssue]
                if attachment.content_type.startswith("image/"):  # pyright: ignore[reportAttributeAccessIssue]
                    agent_input["image_url"] = attachment.url  # pyright: ignore[reportAttributeAccessIssue]

        agent_input["surface_info"] = surface_info

        return agent_input

    def get_thread_id(self, message: Union[discord.Message, discord.Thread]) -> Optional[int]:
        """
        Get the thread ID from a Discord message or thread object.

        Args:
            message (Union[discord.Message, discord.Thread]): The Discord message or thread object.

        Returns:
            Optional[int]: The thread ID, or None if the message is not part of a thread.
        """
        if isinstance(message, discord.Thread):
            LOGGER.info(f"message is a thread, id: {message.id}")
            return message.id
        elif isinstance(message, discord.Message):
            LOGGER.info(f"message is a message, thread id: {message.thread.id if message.thread else None}")  # type: ignore
            return message.thread.id if message.thread else None  # type: ignore
        return None

    def get_session_id(self, message: Union[discord.Message, discord.Thread]) -> str:
        """
        Generate a session ID for the given message.

        This function generates a session ID based on the message context.
        The session ID is used as a key for the history session and as an identifier for logs.

        Args:
        ----
            message (discord.Message): The message or event dictionary.

        Returns:
        -------
            str: The generated session ID.

        Notes:
        -----
            - If the message is a direct message (DM), the session ID is based on the user ID.
            - If the message is from a guild (server) channel, the session ID is based on the channel ID.

        """
        # ctx: Context = await self.get_context(message)  # type: ignore
        if isinstance(message, discord.Thread):
            is_dm: bool = str(message.starter_message.channel.type) == "private"  # pyright: ignore[reportAttributeAccessIssue]
            user_id: int = message.starter_message.author.id  # pyright: ignore[reportAttributeAccessIssue]
            channel_id = message.starter_message.channel.name  # pyright: ignore[reportAttributeAccessIssue]
        elif isinstance(message, discord.Message):
            is_dm: bool = str(message.channel.type) == "private"  # pyright: ignore[reportAttributeAccessIssue]
            user_id: int = message.author.id  # pyright: ignore[reportAttributeAccessIssue]
            channel_id = message.channel.id  # pyright: ignore[reportAttributeAccessIssue]

        return f"discord_{user_id}" if is_dm else f"discord_{channel_id}"  # pyright: ignore[reportAttributeAccessIssue]

    async def handle_dm_from_user(self, message: discord.Message) -> bool:
        """
        Handle a direct message (DM) from a user.

        This asynchronous function processes a direct message (DM) received from a user.
        It prepares the surface information, constructs the agent input, and processes
        the user task using the AI agent. The function sends a temporary message to the
        user indicating that the request is being processed and then sends the agent's
        response in multiple messages if necessary.

        Args:
        ----
            message (discord.Message): The Discord message object representing the direct message.

        Returns:
        -------
            bool: True if the message was successfully processed, False otherwise.

        Example:
        -------
            >>> await bot.handle_dm_from_user(message)

        """
        ctx: Context = await self.get_context(message)
        user_id = ctx.message.author.id  # pyright: ignore[reportAttributeAccessIssue]
        # import bpdb; bpdb.set_trace()
        # Process DMs
        LOGGER.info("Processing direct message")

        # For direct messages, prepare the surface info
        surface_info = SurfaceInfo(surface=SurfaceType.DISCORD, type="direct_message", source="direct_message")

        # Convert surface_info to dictionary using the utility function
        surface_info_dict = surface_info.__dict__

        user_name = message.author.name  # pyright: ignore[reportAttributeAccessIssue]
        agent_input = self.prepare_agent_input(message, user_name, surface_info_dict)
        session_id = self.get_session_id(message)
        thread_id = self.get_thread_id(message)
        LOGGER.info(f"session_id: {session_id} thread_id: {thread_id} Agent input: {json.dumps(agent_input)}")

        temp_message = (
            "Processing your request, please wait... (this could take up to 1min, depending on the response size)"
        )
        # thread_ts is None for direct messages, so we use the ts of the original message
        orig_msg = await ctx.send(
            # embed=discord.Embed(description=temp_message),
            content=temp_message,
            delete_after=30.0,
        )
        # response = client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=temp_message)
        # response_ts = response["ts"]
        # For direct messages, remember chat based on the user:

        #          process_user_task: Must write to at least one of ['input', 'plan', 'past_steps', 'response'] | {}Traceback (most recent call last):

        #   File "/Users/malcolm/dev/bossjones/sandbox_agent/.venv/bin/pytest", line 8, in <module>
        #     sys.exit(console_main())
        # #
        agent_response_text = await self.process_user_task(session_id, str(agent_input), thread_id)

        # Update the slack response with the agent response
        # client.chat_update(channel=channel_id, ts=response_ts, text=agent_response_text)

        # Sometimes the response can be over 2000 characters, so we need to split it
        # into multiple messages, and send them one at a time
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", ".", "?", "!"],
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len,
        )

        responses = text_splitter.split_text(agent_response_text)

        for rsp in responses:
            await message.channel.send(rsp)  # pyright: ignore[reportAttributeAccessIssue]

        # await orig_msg.edit(content=agent_response_text)
        LOGGER.info(f"session_id: {session_id} Agent response: {json.dumps(agent_response_text)}")

        evaluator = Evaluator()

        # Evaluate the response against the question
        eval_result = evaluator.evaluate_prediction(agent_input, agent_response_text)
        # Log the evaluation result
        LOGGER.info(f"session_id: {session_id} Evaluation Result: {eval_result}")
        await LOGGER.complete()
        return True

    async def handle_message_from_channel(self, message: Union[discord.Message, discord.Thread]) -> bool:
        LOGGER.debug(f"message -> {message}")
        rich.inspect(message, all=True)
        # ctx: Context = await self.get_context(message)  # type: ignore
        if isinstance(message, discord.Thread):
            user_id = message.starter_message.author.id
            user_name = message.starter_message.author.name
            channel_id = message.starter_message.channel.name  # pyright: ignore[reportAttributeAccessIssue]
            channel = message.starter_message.channel  # pyright: ignore[reportAttributeAccessIssue]
        elif isinstance(message, discord.Message):
            user_id = message.author.id  # pyright: ignore[reportAttributeAccessIssue]
            user_name = message.author.name  # pyright: ignore[reportAttributeAccessIssue]
            channel_id = message.channel.name  # pyright: ignore[reportAttributeAccessIssue]
            channel = message.channel  # pyright: ignore[reportAttributeAccessIssue]

        # import bpdb; bpdb.set_trace()
        # Process DMs
        LOGGER.info("Processing channel message")

        # For direct messages, prepare the surface info
        surface_info = SurfaceInfo(surface=SurfaceType.DISCORD, type="channel", source=f"{channel_id}")  # pyright: ignore[reportAttributeAccessIssue]

        # Convert surface_info to dictionary using the utility function
        surface_info_dict = surface_info.__dict__

        agent_input = self.prepare_agent_input(message, user_name, surface_info_dict)
        session_id = self.get_session_id(message)  # type: ignore
        thread_id = self.get_thread_id(message)
        LOGGER.info(f"session_id: {session_id} Agent input: {json.dumps(agent_input)}")

        agent_response_text = await self.process_user_task(session_id, str(agent_input), thread_id)

        # Sometimes the response can be over 2000 characters, so we need to split it
        # into multiple messages, and send them one at a time
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", ".", "?", "!"],
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len,
        )

        responses = text_splitter.split_text(agent_response_text)

        # send responses to thread
        for rsp in responses:
            await message.send(rsp)  # pyright: ignore[reportAttributeAccessIssue]

        # await orig_msg.edit(content=agent_response_text)
        LOGGER.info(f"session_id: {session_id} Agent response: {json.dumps(agent_response_text)}")

        # Evaluate the response against the question
        eval_result = Evaluator().evaluate_prediction(agent_input, agent_response_text)
        # Log the evaluation result
        LOGGER.info(f"session_id: {session_id} Evaluation Result: {eval_result}")
        await LOGGER.complete()
        return True

    async def get_context(self, origin: Union[discord.Interaction, discord.Message], /, *, cls=Context) -> Context:
        """
        Retrieve the context for a Discord interaction or message.

        This asynchronous method retrieves the context for a Discord interaction or message
        and returns a Context object. It calls the superclass method to get the context
        based on the provided origin and class type.

        Args:
        ----
            origin (Union[discord.Interaction, discord.Message]): The Discord interaction or message to get the context from.
            cls (Context): The class type for the context object.

        Returns:
        -------
            Context: The context object retrieved for the provided origin.

        """
        return await super().get_context(origin, cls=cls)

    async def process_commands(self, message: discord.Message) -> None:
        """
        Process commands based on the received Discord message.

        This asynchronous function processes commands based on the provided Discord message.
        It retrieves the context for the message, logs information, and then invokes the command handling.
        It includes commented-out sections for potential future functionality like spam control and blacklisting.

        Args:
        ----
            message (discord.Message): The Discord message to process commands from.

        Returns:
        -------
            None

        """
        ctx = await self.get_context(message)

        LOGGER.info(f"ctx = {ctx}")

        # its a dm
        if str(message.channel.type) == "private":  # pyright: ignore[reportAttributeAccessIssue]
            await self.handle_dm_from_user(message)

        # text channel and GPT channel id
        elif str(message.channel.type) == "text" and message.channel.id == 1240294186201124929:  # pyright: ignore[reportAttributeAccessIssue]
            max_tokens = 512
            # let's do all validation and creation of threads here instead.
            try:
                # ignore messages not in a thread
                channel: Union[discord.VoiceChannel, discord.TextChannel, discord.Thread, discord.DMChannel] = (
                    message.channel  # pyright: ignore[reportAttributeAccessIssue]
                )  # pyright: ignore[reportAttributeAccessIssue]

                # if no thread defined, create it
                if not isinstance(channel, discord.Thread):
                    LOGGER.error("if no thread defined, create it")
                    embed = discord.Embed(
                        description=f"<@{ctx.message.author.id}> wants to chat! 🤖💬",  # pyright: ignore[reportAttributeAccessIssue]
                        color=discord.Color.green(),
                    )
                    embed.add_field(name="model", value=aiosettings.chat_model)
                    embed.add_field(name="temperature", value=aiosettings.llm_temperature, inline=True)
                    embed.add_field(name="max_tokens", value=max_tokens, inline=True)
                    embed.add_field(name=ctx.message.author.name, value=message)  # pyright: ignore[reportAttributeAccessIssue]

                    ctx_message: discord.Message = ctx.message  # pyright: ignore[reportAttributeAccessIssue]

                    LOGGER.debug(f"ctx_message = {ctx_message}")
                    rich.inspect(ctx_message, all=True)

                    # create the thread
                    thread: discord.Thread = await ctx_message.create_thread(
                        name=f"{ACTIVATE_THREAD_PREFX} {ctx.message.author.name[:20]} - {message.content[:30]}",  # pyright: ignore[reportAttributeAccessIssue]
                        # Specifies the slowmode rate limit for user in this channel, in seconds. The maximum value possible is ``21600``. By default no slowmode rate limit if this is ``None``.
                        slowmode_delay=1,
                        # The reason for creating a new thread. Shows up on the audit log.
                        reason="gpt-bot",
                        # The duration in minutes before a thread is automatically hidden from the channel list.
                        # If not provided, the channel's default auto archive duration is used.
                        # Must be one of ``60``, ``1440``, ``4320``, or ``10080``, if provided.
                        auto_archive_duration=60,
                    )
                    THREAD_DATA[thread.id] = GoobThreadConfig(
                        model=aiosettings.chat_model, max_tokens=max_tokens, temperature=aiosettings.llm_temperature
                    )
                # if it is a thread, just set channel equal to thread
                elif isinstance(channel, discord.Thread):  # type: ignore
                    LOGGER.error("channel is a thread")
                    thread: discord.Thread = (
                        channel  # mypy: disable-error-code="no-redef" # type: ignore  # type: ignore
                    )

                # ignore threads that are archived locked or title is not what we want
                if (
                    thread.archived  # pyright: ignore[reportAttributeAccessIssue]
                    or thread.locked  # pyright: ignore[reportAttributeAccessIssue]
                    or not thread.name.startswith(ACTIVATE_THREAD_PREFX)  # pyright: ignore[reportAttributeAccessIssue]
                ):
                    # ignore this thread
                    return

                if thread.message_count > MAX_THREAD_MESSAGES:  # pyright: ignore[reportAttributeAccessIssue]
                    # too many messages, no longer going to reply
                    await close_thread(thread=thread)
                    return
                else:
                    # add info to thread
                    await self.handle_message_from_channel(thread)

            except Exception as ex:
                print(f"{ex}")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(f"Error Class: {ex.__class__}")
                output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
                print(output)
                print(f"exc_type: {exc_type}")
                print(f"exc_value: {exc_value}")
                traceback.print_tb(exc_traceback)
                if aiosettings.dev_mode:
                    bpdb.pm()

        # Invokes the command given under the invocation context and
        # handles all the internal event dispatch mechanisms.
        await self.invoke(ctx)
        await LOGGER.complete()

    async def handle_user_task(self, message: discord.Message) -> JSONResponse:
        """
        Handle a user task received through a Discord message.

        This asynchronous function processes a user task received as a Discord message.
        It determines the surface type of the message (DM or channel), creates a SurfaceInfo
        instance, prepares the agent input, and processes the user task using the AI agent.
        It returns a JSON response with the agent's text response or raises an exception if
        processing fails.

        Args:
        ----
            message (discord.Message): The Discord message containing the user task.

        Returns:
        -------
            JSONResponse: A JSON response containing the agent's text response with a status code of 200 if successful.

        Raises:
        ------
            Exception: If an error occurs during the processing of the user task.

        """
        # Optional surface information
        surface = "discord"
        if isinstance(message.channel, discord.DMChannel):  # pyright: ignore[reportAttributeAccessIssue]
            LOGGER.debug("Message is from a DM channel")
            surface_type = "dm"
            source = "dm"
        elif isinstance(message.channel, discord.TextChannel):  # pyright: ignore[reportAttributeAccessIssue]
            LOGGER.debug("Message is from a text channel")
            surface_type = "channel"
            source = message.channel.name  # pyright: ignore[reportAttributeAccessIssue]

        # Convert surface to SurfaceType enum, default to UNKNOWN if not found
        surface_enum = SurfaceType[surface.upper()] if surface else SurfaceType.UNKNOWN

        # Create an instance of SurfaceInfo
        surface_info = SurfaceInfo(surface=surface_enum, type=surface_type, source=source)

        agent_input = {"user_name": message.author, "message": message.content, "surface_info": surface_info.__dict__}  # pyright: ignore[reportAttributeAccessIssue]

        # Modify the call to process_user_task to pass agent_input
        # if not is_streaming:
        session_id = self.get_session_id(message)  # type: ignore
        thread_id = self.get_thread_id(message)
        LOGGER.info(f"session_id: {session_id} thread_id: {thread_id} Agent input: {json.dumps(agent_input)}")
        try:
            # agent_response_text = await self.process_user_task(REQUEST_ID_CONTEXTVAR.get(), str(agent_input))
            agent_response_text = await self.process_user_task(session_id, agent_input, thread_id)
            return JSONResponse(content={"response": agent_response_text}, status_code=200)
        except Exception as ex:
            LOGGER.exception(f"Failed to process user task: {ex}")
            print(f"Failed to load extension {ex} - exception: {ex}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            LOGGER.error(f"Error Class: {str(ex.__class__)}")
            output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
            LOGGER.warning(output)
            LOGGER.error(f"exc_type: {exc_type}")
            LOGGER.error(f"exc_value: {exc_value}")
            traceback.print_tb(exc_traceback)
            raise
        finally:
            await LOGGER.complete()

    async def on_message(self, message: discord.Message) -> None:
        """
        Handle incoming messages and process commands.

        This method is called whenever a message is received. It logs the message details,
        processes any attachments, and then processes the message content as a command if applicable.

        Args:
        ----
            message (discord.Message): The message object received from Discord.

        Returns:
        -------
            None

        """
        LOGGER.info(f"message = {message}")
        LOGGER.info("ITS THIS ONE BOSS")

        LOGGER.info(f"You are in function: {CURRENTFUNCNAME()}")
        LOGGER.info(f"This function's caller was: {CURRENTFUNCNAME(1)}")

        # TODO: This is where all the AI logic is going to go
        LOGGER.info(
            f"Thread message to process - function: {CURRENTFUNCNAME()}, function caller: {CURRENTFUNCNAME(1)}, author: {message.author}, content: '{message.content[:50]}'"  # type: ignore
        )  # pyright: ignore[reportAttributeAccessIssue]
        if message.author.bot:
            LOGGER.info(f"Skipping message from bot itself, message.author.bot = {message.author.bot}")
            return

        # skip messages that start w/ cerebro's prefix
        if message.content.startswith("%"):  # pyright: ignore[reportAttributeAccessIssue]
            LOGGER.info("Skipping message that starts with %")
            return

        # Handle attachments first
        await self.process_attachments(message)
        if message.content.strip() != "":  # pyright: ignore[reportAttributeAccessIssue]
            # NOTE: dptest doesn't like this, disable so we can keep tests # async with message.channel.typing():  # pyright: ignore[reportAttributeAccessIssue]
            # Send everything to AI bot
            await self.process_commands(message)
        await LOGGER.complete()

    async def close(self) -> None:
        """
        Close the bot and its associated resources.

        This asynchronous method performs the necessary cleanup operations
        before shutting down the bot. It closes the aiohttp session and
        calls the superclass's close method to ensure proper shutdown.

        Returns
        -------
            None

        """
        await super().close()
        await self.session.close()

    async def start(self) -> None:  # type: ignore
        """
        Start the bot and connect to Discord.

        This asynchronous method initiates the bot's connection to Discord using the token
        specified in the settings. It ensures that the bot attempts to reconnect if the
        connection is lost.

        The method overrides the default `start` method from the `commands.Bot` class to
        include the bot's specific token and reconnection behavior.

        Returns
        -------
            None

        """
        await super().start(aiosettings.discord_token.get_secret_value(), reconnect=True)

    async def my_background_task(self) -> None:
        """
        Run a background task that sends a counter message to a specific channel every 60 seconds.

        This asynchronous method waits until the bot is ready, then continuously increments a counter
        and sends its value to a predefined Discord channel every 60 seconds. The channel ID is retrieved
        from the bot's settings.

        The method ensures that the task runs indefinitely until the bot is closed.

        Args:
        ----
            None

        Returns:
        -------
            None

        """
        await self.wait_until_ready()
        # """
        # Wait until the bot is ready before starting the monitoring loop.
        # """
        counter = 0
        # """
        # Initialize a counter to keep track of the number of monitoring iterations.
        # """
        # TEMPCHANGE: # channel = self.get_channel(DISCORD_GENERAL_CHANNEL)  # channel ID goes here
        channel = self.get_channel(aiosettings.discord_general_channel)  # channel ID goes here
        while not self.is_closed():
            # """
            # Continuously monitor the worker tasks until the bot is closed.
            # """
            counter += 1
            # """
            # Increment the counter for each monitoring iteration.
            # """
            await channel.send(counter)  # pyright: ignore[reportAttributeAccessIssue]  # type: ignore
            await asyncio.sleep(60)  # task runs every 60 seconds

    async def on_worker_monitor(self) -> None:
        """
        Monitor and log the status of worker tasks.

        This asynchronous method waits until the bot is ready, then continuously
        increments a counter and logs the status of worker tasks every 10 seconds.
        It prints the list of tasks and the number of tasks currently being processed.

        The method ensures that the monitoring runs indefinitely until the bot is closed.

        Returns
        -------
            None

        """
        await self.wait_until_ready()
        counter = 0
        # channel = self.get_channel(DISCORD_GENERAL_CHANNEL)  # channel ID goes here
        while not self.is_closed():
            counter += 1
            # await channel.send(counter)
            print(f" self.tasks = {self.tasks}")
            """
            Print the list of current tasks being processed.
            """
            print(f" len(self.tasks) = {len(self.tasks)}")
            """
            Print the number of tasks currently being processed.
            """
            await asyncio.sleep(10)
            """
            Sleep for 10 seconds before the next monitoring iteration.
            """

    # SOURCE: https://github.com/aronweiler/assistant/blob/a8abd34c6973c21bc248f4782f1428a810daf899/src/discord/rag_bot.py#L90
    async def process_attachments(self, message: discord.Message) -> None:
        """
        Process attachments in a Discord message.

        This asynchronous function processes attachments in a Discord message by downloading each attached file,
        storing it in a temporary directory, and then loading and processing the files. It sends a message to indicate
        the start of processing and handles any errors that occur during the download process.

        Args:
        ----
            message (discord.Message): The Discord message containing attachments to be processed.

        Returns:
        -------
            None

        """
        if len(message.attachments) <= 0:  # type: ignore
            return

        root_temp_dir = file_operations.create_temp_directory()
        uploaded_file_paths = []
        for attachment in message.attachments:  # type: ignore
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

    async def process_image(self, url: str) -> None:
        """
        Process an image from a given URL.

        Args:
            url (str): The URL of the image to process.
        """
        response = await file_operations.download_image(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")  # type: ignore
        # Add any additional image processing logic here

    async def write_attachments_to_disk(self, message: discord.Message) -> None:
        """
        Save attachments from a Discord message to disk.

        This asynchronous function processes the attachments in a Discord message,
        saves them to a temporary directory, and logs the file paths. It also handles
        any errors that occur during the download process.

        Args:
        ----
            message (discord.Message): The Discord message containing attachments to be saved.

        Returns:
        -------
            None

        """
        ctx = await self.get_context(message)
        attachment_data_list_dicts, local_attachment_file_list, local_attachment_data_list_dicts, media_filepaths = (
            self.get_attachments(message)
        )

        tmpdirname = await file_operations.aio_create_temp_directory()

        LOGGER.info(f"tmpdirname = {tmpdirname}")

        with Timer(text="\nTotal elapsed time: {:.1f}"):
            for an_attachment_dict in attachment_data_list_dicts:
                local_attachment_path = await file_operations.handle_save_attachment_locally(
                    an_attachment_dict, tmpdirname
                )
                local_attachment_file_list.append(local_attachment_path)

            for some_file in local_attachment_file_list:
                local_data_dict = file_operations.file_to_local_data_dict(some_file, tmpdirname)
                local_attachment_data_list_dicts.append(local_data_dict)
                path_to_image = file_functions.fix_path(local_data_dict["filename"])
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

    @commands.command()
    async def chat(self, ctx: commands.Context, *, message: str) -> None:  # type: ignore
        """
        Implement chat functionality.

        Args:
            ctx (commands.Context): The command context.
            message (str): The message to process.
        """
        response = await self.chat_model.agenerate([message])
        await ctx.send(response.generations[0][0].text)

    # Add more commands as needed

    @traceable
    async def process_user_task(
        self, session_id: str, user_task: str, thread_id: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Summary:
        Process a user task by invoking an agent executor and returning the output.

        Explanation:
        This function processes a user task by setting up an agent executor with the provided session ID, user task, and thread ID. It then invokes the agent executor with the user task input and returns the output generated by the agent. If an error occurs during processing, it logs the exception and returns an error message.

        Args:
        ----
        - self: The instance of the class.
        - session_id (str): The session ID for the user task.
        - user_task (str): The user task to be processed.
        - thread_id (Optional[int]): The ID of the thread associated with the user task. Defaults to None.

        Returns:
        -------
        - str: The output generated by the agent or an error message if processing fails.

        """
        LOGGER.debug(f"session_id = {session_id}")
        LOGGER.debug(f"user_task = {user_task}")
        LOGGER.debug(f"thread_id = {thread_id}")

        try:
            config = {
                "metadata": {
                    "session_id": session_id,
                    "thread_id": thread_id,
                },
                "callbacks": [StdOutCallbackHandler(), AsyncLoggingCallbackHandler()],
            }
            # inputs: Dict[str, str] = {"input": user_task}
            inputs: MessageLikeRepresentation = format_user_message(user_task)
            result = await self.graph.ainvoke(inputs, config=config, debug=True)
            LOGGER.error(f"type(result) = {type(result)}")
            # bpdb.set_trace()
            if aiosettings.agent_type == "basic" or aiosettings.agent_type == "advanced":
                # example results for basic:
                #                 >>> rich.print(messages)
                # [
                #     HumanMessage(
                #         content='{\'user_name\': \'ghfghf\', \'message\': "<@fghjghf> what\'s the answer to life the universe and everything", \'surface_info\': {\'surface\': \'discord\', \'type\': \'direct_message\', \'source\': \'direct_message\'}}',
                #         additional_kwargs={},
                #         response_metadata={},
                #         id='699975cd-21a3-40f2-a1d2-5991239755a0'
                #     ),
                #     AIMessage(
                #         content='The answer to life, the universe, and everything is 42.\n\nThis is a reference to Douglas Adams\' popular science fiction series "The Hitchhiker\'s Guide to the Galaxy." In the story, a supercomputer named Deep Thought calculates for 7.5 million years to
                # determine that 42 is the answer to the ultimate question of life, the universe, and everything.\n\nOf course, this is meant to be humorous, as 42 doesn\'t actually explain anything meaningful about life or the universe. The joke is that while we may search for profound
                # cosmic truths, sometimes the answers we find are absurd or unsatisfying.\n\nIn reality, questions about the meaning of life and the nature of the universe are complex philosophical and scientific topics that humanity continues to explore and debate.',
                #         additional_kwargs={},
                #         response_metadata={'id': 'msg_01Kgr2CW8Y8dqFeGJy9h9Fne', 'model': 'claude-3-5-sonnet-20240620', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 78, 'output_tokens': 171}},
                #         id='run-dc720191-97f7-4f3f-b2eb-7c4869244426-0',
                #         usage_metadata={'input_tokens': 78, 'output_tokens': 171, 'total_tokens': 249, 'input_token_details': {}}
                #     )
                # ]

                return result.get("messages", "No response generated by the agent.")[1].content
            elif aiosettings.agent_type == "plan_and_execute":
                return result.get("output", "No response generated by the agent.")
            else:
                raise ValueError(f"Unsupported agent type: {aiosettings.agent_type}")
        except Exception as e:
            LOGGER.exception(f"Error in process_user_task: {e}")
            return "An error occurred while processing the task."

    @traceable
    async def process_user_task_streaming(
        self,
        session_id: str,
        user_task: str,
        thread_id: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Process the user task by streaming the agent's response.

        This method invokes the agent's workflow graph to process the user's task.
        It streams the agent's response, including tool invocations and the final output,
        as an asynchronous iterator of strings.

        Args:
            session_id (str): The ID of the current session.
            user_task (str): The task provided by the user.
            thread_id (Optional[int], optional): The ID of the current thread. Defaults to None.

        Yields:
            str: The next chunk of the agent's response, including tool invocations and the final output.

        Raises:
            Exception: If an error occurs during the processing of the user task.
        """
        try:
            LOGGER.debug(f"session_id = {session_id}")
            LOGGER.debug(f"user_task = {user_task}")
            LOGGER.debug(f"thread_id = {thread_id}")
            # SOURCE: https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/#flights
            # We the can access the RunnableConfig for a given run to check the passenger_id of the user accessing this application. The LLM never has to provide these explicitly, they are provided for a given invocation of the graph so that each user cannot access other passengers' booking information.
            # agent_executor = self.setup_agent_executor(session_id, user_task)
            config = {
                # # @Web @https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html
                # # tags: List of strings to tag this call and any sub-calls (e.g., a Chain calling an LLM)
                # "tags": ["agent"],
                # # metadata: Dictionary of key-value pairs to store metadata for this call and any sub-calls
                # # Metadata for this call and any sub-calls (eg. a Chain calling an LLM). Keys should be strings, values should be JSON-serializable.
                "metadata": {
                    "session_id": session_id,
                    "thread_id": thread_id,
                },
                # # callbacks: List of callbacks to be called for this call and any sub-calls
                # # Callbacks for this call and any sub-calls (eg. a Chain calling an LLM). Tags are passed to all callbacks, metadata is passed to handle*Start callbacks.
                "callbacks": [StdOutCallbackHandler(), AsyncLoggingCallbackHandler()],
                # "callbacks": [
                #     {
                #         "name": "langsmith",
                #         "session_id": session_id,
                #         "thread_id": thread_id,
                #     }
                # ],
                # # run_name: Name for the tracer run for this call, defaults to the name of the class
                # "run_name": "process_user_task",
                # # max_concurrency: Maximum number of parallel calls to make, defaults to ThreadPoolExecutor's default if not provided
                # "max_concurrency": 1,
                # # recursion_limit: Maximum number of times a call can recurse, defaults to 25 if not provided
                # "recursion_limit": 1,
                # # run_id: Unique identifier for the tracer run for this call, a new UUID will be generated if not provided
                # "run_id": session_id,
                # configurable: Runtime values for attributes previously made configurable on this Runnable or sub-Runnables
                # "configurable": {
                #     # The passenger_id is used in our flight tools to fetch the user's flight information
                #     "passenger_id": "3442 587242",
                #     # Checkpoints are accessed by thread_id
                #     "thread_id": thread_id,
                # }
            }
            # inputs = {"input": user_task}
            inputs: MessageLikeRepresentation = format_user_message(user_task)
            # agent_executor = self.setup_agent_executor(session_id, user_task)
            # Loop through the agent executor stream
            async for chunk in self.graph.astream_log(inputs, config=config):
                # We want to pull any tool invocation and the tokens for the final response
                for op in chunk.ops:
                    # Extract the tool invocation from the agent executor stream
                    if op["op"] == "add" and "/logs/OpenAIToolsAgentOutputParser/final_output" in op["path"]:
                        try:
                            tool_invocations = op["value"]["output"]
                            for tool_invocation in tool_invocations:
                                # Directly access the 'log' attribute of the tool_invocation object
                                if hasattr(tool_invocation, "log"):
                                    LOGGER.info(f"Tool invocation: {tool_invocation.log}")
                                    yield tool_invocation.log

                        except AttributeError as e:
                            LOGGER.exception(f"An error occurred: {e}")
                            continue

                    # Logic to pull agent response from the streaming output
                    # Target paths ending with 'streamed_output_str/-', and check if 'value' is non-empty
                    if op["op"] == "add" and op["path"].endswith("/streamed_output_str/-") and op["value"]:
                        value = op["value"]  # Directly access the value
                        LOGGER.info(f"Chunk: {value}")
                        yield value

        except Exception as e:
            LOGGER.exception(f"Error in process_user_task_streaming: {e}")
            yield "An error occurred while processing the task."
        finally:
            await LOGGER.complete()
