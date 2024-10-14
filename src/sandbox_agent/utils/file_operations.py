# pylint: disable=no-member
# pylint: disable=possibly-used-before-assignment
# pyright: reportImportCycles=false
# pyright: reportAttributeAccessIssue=false
# mypy: disable-error-code="index"
# mypy: disable-error-code="no-redef"

"""Utility functions for file system operations."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import pathlib
import sys
import uuid

from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

import aiohttp

from discord import Attachment, File
from loguru import logger as LOGGER
from PIL import Image

from sandbox_agent import shell


async def download_image(url: str) -> BytesIO:
    """Download an image from a given URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                return io.BytesIO(data)


async def file_to_data_uri(file: File) -> str:
    """Convert a discord.File object to a data URI."""
    with BytesIO(file.fp.read()) as f:
        file_bytes = f.read()
    base64_encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:image;base64,{base64_encoded}"


async def data_uri_to_file(data_uri: str, filename: str) -> File:
    """Convert a data URI to a discord.File object."""
    metadata, base64_data = data_uri.split(",")
    content_type = metadata.split(";")[0].split(":")[1]
    file_bytes = base64.b64decode(base64_data)
    return File(BytesIO(file_bytes), filename=filename, spoiler=False)


def file_to_local_data_dict(fname: str, dir_root: str) -> dict[str, Any]:
    """Convert a file to a dictionary with metadata."""
    file_api = pathlib.Path(fname)
    return {
        "filename": f"{dir_root}/{file_api.stem}{file_api.suffix}",
        "size": file_api.stat().st_size,
        "ext": f"{file_api.suffix}",
        "api": file_api,
    }


async def handle_save_attachment_locally(attm_data_dict: dict[str, Any], dir_root: str) -> str:
    """Save a Discord attachment locally."""
    fname = f"{dir_root}/orig_{attm_data_dict['id']}_{attm_data_dict['filename']}"
    print(f"Saving to ... {fname}")
    await attm_data_dict["attachment_obj"].save(fname, use_cached=True)
    await asyncio.sleep(1)
    return fname


async def details_from_file(path_to_media_from_cli: str, cwd: Union[str, None] = None) -> tuple[str, str, str]:
    """Generate input and output file paths and retrieve the timestamp of the input file."""
    p = pathlib.Path(path_to_media_from_cli)
    full_path_input_file = f"{p.stem}{p.suffix}"
    full_path_output_file = f"{p.stem}_smaller.mp4"
    print(full_path_input_file)
    print(full_path_output_file)
    if sys.platform == "darwin":
        get_timestamp = await shell._aio_run_process_and_communicate(
            ["gstat", "-c", "%y", f"{p.stem}{p.suffix}"], cwd=cwd
        )
    elif sys.platform == "linux":
        get_timestamp = await shell._aio_run_process_and_communicate(
            ["stat", "-c", "%y", f"{p.stem}{p.suffix}"], cwd=cwd
        )

    return full_path_input_file, full_path_output_file, get_timestamp


def create_temp_directory() -> str:
    """Create a temporary directory and return its path."""
    tmpdirname = f"temp/{str(uuid.uuid4())}"
    os.makedirs(os.path.dirname(tmpdirname), exist_ok=True)
    print("created temporary directory", tmpdirname)
    return tmpdirname


def get_file_tree(directory: str) -> list[str]:
    """Get the directory tree of the given directory."""
    return [str(p) for p in pathlib.Path(directory).rglob("*")]
