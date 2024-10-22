"""Base classes and utilities for the Discord client."""

# pylint: disable=too-many-function-args
# mypy: disable-error-code="arg-type, var-annotated, list-item, no-redef, truthy-bool, return-value"
# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import asyncio
import base64
import io
import os
import pathlib
import sys
import typing

from collections import defaultdict
from collections.abc import AsyncIterator, Generator, Iterable
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, NoReturn, Optional

import aiohttp
import discord
import rich

from codetiming import Timer
from discord import Message as DiscordMessage
from logging_tree import printout
from loguru import logger as LOGGER

from sandbox_agent import shell
from sandbox_agent.base import GoobMessage, GoobThreadConfig
from sandbox_agent.bot_logger import generate_tree, get_lm_from_tree
from sandbox_agent.constants import INACTIVATE_THREAD_PREFIX, MAX_CHARS_PER_REPLY_MSG
from sandbox_agent.factories import guild_factory
from sandbox_agent.utils import async_


SEPARATOR_TOKEN = "<|endoftext|>"

HERE = os.path.dirname(__file__)

INVITE_LINK = "https://discordapp.com/api/oauth2/authorize?client_id={}&scope=bot&permissions=0"


HOME_PATH = os.environ.get("HOME")

COMMAND_RUNNER = {"dl_thumb": shell.run_coroutine_subprocess}
THREAD_DATA = defaultdict()


# Enums for surface values
class SurfaceType(str, Enum):
    DISCORD = "discord"
    UNKNOWN = "unknown"


# Dataclass for surface information using Pydantic's dataclass
@dataclass(config={"use_enum_values": True})
class SurfaceInfo:
    surface: SurfaceType
    type: str
    source: str


def unlink_orig_file(a_filepath: str) -> str:
    """
    Delete the specified file and return its path.

    This function deletes the file at the given file path and returns the path of the deleted file.
    It uses the `os.unlink` method to remove the file and logs the deletion using `rich.print`.

    Args:
    ----
        a_filepath (str): The path to the file to be deleted.

    Returns:
    -------
        str: The path of the deleted file.

    """
    rich.print(f"deleting ... {a_filepath}")
    os.unlink(f"{a_filepath}")
    return a_filepath


# https://github.com/discord-math/bot/blob/babb41b71a68b4b099684b3e1ed583f84083f971/plugins/log.py#L63
def path_for(attm: discord.Attachment, basedir: str = "./") -> pathlib.Path:
    """
    Generate a pathlib.Path object for an attachment with a specified base directory.

    This function constructs a pathlib.Path object for a given attachment 'attm' using the specified base directory 'basedir'.
    It logs the generated path for debugging purposes and returns the pathlib.Path object.

    Args:
    ----
        attm (discord.Attachment): The attachment for which the path is generated.
        basedir (str): The base directory path where the attachment file will be located. Default is the current directory.

    Returns:
    -------
        pathlib.Path: A pathlib.Path object representing the path for the attachment file.

    Example:
    -------
        >>> attm = discord.Attachment(filename="example.png")
        >>> path = path_for(attm, basedir="/attachments")
        >>> print(path)
        /attachments/example.png

    """
    p = pathlib.Path(basedir, str(attm.filename))  # pyright: ignore[reportAttributeAccessIssue]
    LOGGER.debug(f"path_for: p -> {p}")
    return p


# https://github.com/discord-math/bot/blob/babb41b71a68b4b099684b3e1ed583f84083f971/plugins/log.py#L63
async def save_attachment(attm: discord.Attachment, basedir: str = "./") -> None:
    """
    Save a Discord attachment to a specified directory.

    This asynchronous function saves a Discord attachment to the specified base directory.
    It constructs the path for the attachment, creates the necessary directories, and saves the attachment
    to the generated path. If an HTTPException occurs during saving, it retries the save operation.

    Args:
    ----
        attm (discord.Attachment): The attachment to be saved.
        basedir (str): The base directory path where the attachment file will be located. Default is the current directory.

    Returns:
    -------
        None

    """
    path = path_for(attm, basedir=basedir)
    LOGGER.debug(f"save_attachment: path -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ret_code = await attm.save(path, use_cached=True)
        await asyncio.sleep(5)
    except discord.HTTPException:
        await attm.save(path)

    await LOGGER.complete()


def attachment_to_dict(attm: discord.Attachment) -> dict[str, Any]:
    """
    Convert a discord.Attachment object to a dictionary.

    This function takes a discord.Attachment object and converts it into a dictionary
    containing relevant information about the attachment.

    The dictionary includes:
    - filename: The name of the file.
    - id: The unique identifier of the attachment.
    - proxy_url: The proxy URL of the attachment.
    - size: The size of the attachment in bytes.
    - url: The URL of the attachment.
    - spoiler: A boolean indicating if the attachment is a spoiler.
    - height: The height of the attachment (if applicable).
    - width: The width of the attachment (if applicable).
    - content_type: The MIME type of the attachment (if applicable).
    - attachment_obj: The original attachment object.

    Args:
    ----
        attm (discord.Attachment): The attachment object to be converted.

    Returns:
    -------
        Dict[str, Any]: A dictionary containing information about the attachment.

    """
    result = {
        "filename": attm.filename,  # pyright: ignore[reportAttributeAccessIssue]
        "id": attm.id,
        "proxy_url": attm.proxy_url,  # pyright: ignore[reportAttributeAccessIssue]
        "size": attm.size,  # pyright: ignore[reportAttributeAccessIssue]
        "url": attm.url,  # pyright: ignore[reportAttributeAccessIssue]
        "spoiler": attm.is_spoiler(),
    }
    if attm.height:  # pyright: ignore[reportAttributeAccessIssue]
        result["height"] = attm.height  # pyright: ignore[reportAttributeAccessIssue]
    if attm.width:  # pyright: ignore[reportAttributeAccessIssue]
        result["width"] = attm.width  # pyright: ignore[reportAttributeAccessIssue]
    if attm.content_type:  # pyright: ignore[reportAttributeAccessIssue]
        result["content_type"] = attm.content_type  # pyright: ignore[reportAttributeAccessIssue]

    result["attachment_obj"] = attm

    return result


def file_to_local_data_dict(fname: str, dir_root: str) -> dict[str, Any]:
    """
    Convert a file to a dictionary with metadata.

    This function takes a file path and a root directory, and converts the file
    into a dictionary containing metadata such as filename, size, extension, and
    a pathlib.Path object representing the file.

    Args:
    ----
        fname (str): The name of the file to be converted.
        dir_root (str): The root directory where the file is located.

    Returns:
    -------
        Dict[str, Any]: A dictionary containing metadata about the file.

    """
    file_api = pathlib.Path(fname)
    return {
        "filename": f"{dir_root}/{file_api.stem}{file_api.suffix}",
        "size": file_api.stat().st_size,
        "ext": f"{file_api.suffix}",
        "api": file_api,
    }


async def handle_save_attachment_locally(attm_data_dict: dict[str, Any], dir_root: str) -> str:
    """
    Save a Discord attachment locally.

    This asynchronous function saves a Discord attachment to a specified directory.
    It constructs the file path for the attachment, saves the attachment to the generated path,
    and returns the path of the saved file.

    Args:
    ----
        attm_data_dict (Dict[str, Any]): A dictionary containing information about the attachment.
        dir_root (str): The root directory where the attachment file will be saved.

    Returns:
    -------
        str: The path of the saved attachment file.

    """
    fname = f"{dir_root}/orig_{attm_data_dict['id']}_{attm_data_dict['filename']}"
    rich.print(f"Saving to ... {fname}")
    await attm_data_dict["attachment_obj"].save(fname, use_cached=True)
    await asyncio.sleep(1)
    return fname


# SOURCE: https://github.com/CrosswaveOmega/NikkiBot/blob/75c7ecd307f50390cfc798d39098fdb78535650c/cogs/AiCog.py#L237
async def download_image(url: str) -> BytesIO:  # type: ignore
    """
    Download an image from a given URL asynchronously.

    This asynchronous function uses aiohttp to make a GET request to the provided URL
    and downloads the image data. If the response status is 200 (OK), it reads the
    response data and returns it as a BytesIO object.

    Args:
    ----
        url (str): The URL of the image to download.

    Returns:
    -------
        BytesIO: A BytesIO object containing the downloaded image data.

    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                return io.BytesIO(data)


# SOURCE: https://github.com/CrosswaveOmega/NikkiBot/blob/7092ae6da21c86c7686549edd5c45335255b73ec/cogs/GlobalCog.py#L23
async def file_to_data_uri(file: discord.File) -> str:
    """
    Convert a discord.File object to a data URI.

    This asynchronous function reads the bytes from a discord.File object,
    base64 encodes the bytes, and constructs a data URI.

    Args:
    ----
        file (discord.File): The discord.File object to be converted.

    Returns:
    -------
        str: A data URI representing the file content.

    """
    with BytesIO(file.fp.read()) as f:  # pyright: ignore[reportAttributeAccessIssue]
        file_bytes = f.read()
    base64_encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:image;base64,{base64_encoded}"


# SOURCE: https://github.com/CrosswaveOmega/NikkiBot/blob/7092ae6da21c86c7686549edd5c45335255b73ec/cogs/GlobalCog.py#L23
async def data_uri_to_file(data_uri: str, filename: str) -> discord.File:
    """
    Convert a data URI to a discord.File object.

    This asynchronous function takes a data URI and a filename, decodes the base64 data,
    and creates a discord.File object with the decoded data.

    Args:
    ----
        data_uri (str): The data URI to be converted.
        filename (str): The name of the file to be created.

    Returns:
    -------
        discord.File: A discord.File object containing the decoded data.

    """
    # Split the data URI into its components
    metadata, base64_data = data_uri.split(",")
    # Get the content type from the metadata
    content_type = metadata.split(";")[0].split(":")[1]
    # Decode the base64 data
    file_bytes = base64.b64decode(base64_data)
    return discord.File(BytesIO(file_bytes), filename=filename, spoiler=False)


@async_.to_async
def get_logger_tree_printout() -> None:
    """
    Print the logger tree structure.

    This function prints the logger tree structure using the `printout` function
    from the `logging_tree` module. It is decorated with `@async_.to_async` to
    run asynchronously.
    """
    printout()


# SOURCE: https://realpython.com/how-to-make-a-discord-bot-python/#responding-to-messages
def dump_logger_tree() -> None:
    """
    Dump the logger tree structure.

    This function generates the logger tree structure using the `generate_tree` function
    and logs the tree structure using the `LOGGER.debug` method.
    """
    rootm = generate_tree()
    LOGGER.debug(rootm)


def dump_logger(logger_name: str) -> Any:
    """
    Dump the logger tree structure for a specific logger.

    This function generates the logger tree structure using the `generate_tree` function
    and retrieves the logger metadata for the specified logger name using the `get_lm_from_tree` function.
    It logs the retrieval process using the `LOGGER.debug` method.

    Args:
    ----
        logger_name (str): The name of the logger to retrieve the tree structure for.

    Returns:
    -------
        Any: The logger metadata for the specified logger name.

    """
    LOGGER.debug(f"getting logger {logger_name}")
    rootm = generate_tree()
    return get_lm_from_tree(rootm, logger_name)


def filter_empty_string(a_list: list[str]) -> list[str]:
    """
    Filter out empty strings from a list of strings.

    This function takes a list of strings and returns a new list with all empty strings removed.

    Args:
    ----
        a_list (List[str]): The list of strings to be filtered.

    Returns:
    -------
        List[str]: A new list containing only non-empty strings from the input list.

    """
    filter_object = filter(lambda x: x != "", a_list)
    return list(filter_object)


# SOURCE: https://docs.python.org/3/library/asyncio-queue.html
async def worker(name: str, queue: asyncio.Queue) -> NoReturn:
    """
    Process tasks from the queue.

    This asynchronous function continuously processes tasks from the provided queue.
    Each task is executed using the COMMAND_RUNNER dictionary, and the function
    notifies the queue when a task is completed.

    Args:
    ----
        name (str): The name of the worker.
        queue (asyncio.Queue): The queue from which tasks are retrieved.

    Returns:
    -------
        NoReturn: This function runs indefinitely and does not return.

    """
    LOGGER.info(f"starting working ... {name}")

    while True:
        # Get a "work item" out of the queue.
        co_cmd_task = await queue.get()
        print(f"co_cmd_task = {co_cmd_task}")

        # Sleep for the "co_cmd_task" seconds.

        await COMMAND_RUNNER[co_cmd_task.name](cmd=co_cmd_task.cmd, uri=co_cmd_task.uri)

        # Notify the queue that the "work item" has been processed.
        queue.task_done()

        print(f"{name} ran {co_cmd_task.name} with arguments {co_cmd_task}")


# SOURCE: https://realpython.com/python-async-features/#building-a-synchronous-web-server
async def co_task(name: str, queue: asyncio.Queue) -> AsyncIterator[None]:
    """
    Process tasks from the queue with timing.

    This asynchronous function processes tasks from the provided queue. It uses a timer to measure the elapsed time for each task. The function yields control back to the event loop after processing each task.

    Args:
    ----
        name (str): The name of the task.
        queue (asyncio.Queue): The queue from which tasks are retrieved.

    Yields:
    ------
        None: This function yields control back to the event loop after processing each task.

    """
    LOGGER.info(f"starting working ... {name}")

    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
    while not queue.empty():
        co_cmd_task = await queue.get()
        print(f"Task {name} running")
        timer.start()
        await COMMAND_RUNNER[co_cmd_task.name](cmd=co_cmd_task.cmd, uri=co_cmd_task.uri)
        timer.stop()
        yield
        await LOGGER.complete()


# SOURCE: https://github.com/makupi/cookiecutter-discord.py-postgres/blob/master/%7B%7Bcookiecutter.bot_slug%7D%7D/bot/__init__.py#L28
async def preload_guild_data() -> dict[int, dict[str, str]]:
    """
    Preload guild data.

    This function initializes and returns a dictionary containing guild data.
    Each guild is represented by its ID and contains a dictionary with the guild's prefix.

    Returns
    -------
        Dict[int, Dict[str, str]]: A dictionary where the keys are guild IDs and the values are dictionaries
        containing guild-specific data, such as the prefix.

    """
    LOGGER.info("preload_guild_data ... ")
    guilds = [guild_factory.Guild()]
    await LOGGER.complete()
    return {guild.id: {"prefix": guild.prefix} for guild in guilds}


def extensions() -> Iterable[str]:
    """
    Yield extension module paths.

    This function searches for Python files in the 'cogs' directory relative to the current file's directory.
    It constructs the module path for each file and yields it.

    Yields
    ------
        str: The module path for each Python file in the 'cogs' directory.

    """
    module_dir = pathlib.Path(HERE)
    files = pathlib.Path(module_dir.stem, "cogs").rglob("*.py")
    for file in files:
        LOGGER.debug(f"exension = {file.as_posix()[:-3].replace('/', '.')}")
        yield file.as_posix()[:-3].replace("/", ".")


async def details_from_file(path_to_media_from_cli: str, cwd: typing.Union[str, None] = None) -> tuple[str, str, str]:
    """
    Generate input and output file paths and retrieve the timestamp of the input file.

    This function takes a file path and an optional current working directory (cwd),
    and returns the input file path, output file path, and the timestamp of the input file.
    The timestamp is retrieved using platform-specific commands.

    Args:
    ----
        path_to_media_from_cli (str): The path to the media file provided via the command line.
        cwd (typing.Union[str, None], optional): The current working directory. Defaults to None.

    Returns:
    -------
        Tuple[str, str, str]: A tuple containing the input file path, output file path, and the timestamp of the input file.

    """
    p = pathlib.Path(path_to_media_from_cli)
    full_path_input_file = f"{p.stem}{p.suffix}"
    full_path_output_file = f"{p.stem}_smaller.mp4"
    rich.print(full_path_input_file)
    rich.print(full_path_output_file)
    if sys.platform == "darwin":
        get_timestamp = await shell._aio_run_process_and_communicate(
            ["gstat", "-c", "%y", f"{p.stem}{p.suffix}"], cwd=cwd
        )
    elif sys.platform == "linux":
        get_timestamp = await shell._aio_run_process_and_communicate(
            ["stat", "-c", "%y", f"{p.stem}{p.suffix}"], cwd=cwd
        )

    return full_path_input_file, full_path_output_file, get_timestamp


def discord_message_to_message(message: DiscordMessage) -> Optional[GoobMessage]:
    if (
        message.type == discord.MessageType.thread_starter_message
        and message.reference.cached_message
        and len(message.reference.cached_message.embeds) > 0
        and len(message.reference.cached_message.embeds[0].fields) > 0  # type: ignore
    ):
        field = message.reference.cached_message.embeds[0].fields[0]  # type: ignore
        if field.value:
            return GoobMessage(user=field.name, text=field.value)
    elif message.content:
        return GoobMessage(user=message.author.name, text=message.content)
    return None


def split_into_shorter_messages(message: str) -> list[str]:
    return [message[i : i + MAX_CHARS_PER_REPLY_MSG] for i in range(0, len(message), MAX_CHARS_PER_REPLY_MSG)]


def is_last_message_stale(interaction_message: DiscordMessage, last_message: DiscordMessage, bot_id: str) -> bool:
    return (
        last_message
        and last_message.id != interaction_message.id
        and last_message.author
        and last_message.author.id != bot_id
    )


async def close_thread(thread: discord.Thread):
    await thread.edit(name=INACTIVATE_THREAD_PREFIX)
    await thread.send(
        embed=discord.Embed(
            description="**Thread closed** - Context limit reached, closing...",
            color=discord.Color.blue(),
        )
    )
    await thread.edit(archived=True, locked=True)


# SOURCE: https://github.com/darren-rose/DiscordDocChatBot/blob/63a2f25d2cb8aaace6c1a0af97d48f664588e94e/main.py#L28
# TODO: maybe enable this
async def send_long_message(channel: Any, message: discord.Message, max_length: int = 2000) -> None:
    """
    Send a long message by splitting it into chunks and sending each chunk.

    This asynchronous function takes a message and splits it into chunks of
    maximum length 'max_length'. It then sends each chunk as a separate message
    to the specified channel.

    Args:
    ----
        channel (Any): The channel to send the message chunks to.
        message (discord.Message): The message to be split into chunks and sent.
        max_length (int): The maximum length of each message chunk. Default is 2000.

    Returns:
    -------
        None

    """
    chunks = [message[i : i + max_length] for i in range(0, len(message), max_length)]  # type: ignore
    for chunk in chunks:
        await channel.send(chunk)
