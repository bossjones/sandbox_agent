# pylint: disable=no-member
# pylint: disable=possibly-used-before-assignment
# pyright: reportImportCycles=false
# pyright: reportAttributeAccessIssue=false
# mypy: disable-error-code="index"
# mypy: disable-error-code="no-redef"

from __future__ import annotations

import asyncio
import base64
import os
import shutil

from io import BytesIO
from pathlib import Path
from typing import Any, List

import aiohttp

from discord import File

import pytest

from pytest_mock import MockerFixture

from sandbox_agent.utils.file_operations import (
    create_temp_directory,
    data_uri_to_file,
    details_from_file,
    download_image,
    file_to_data_uri,
    file_to_local_data_dict,
    get_file_tree,
    handle_save_attachment_locally,
)


@pytest.fixture
def mock_pdf_file(tmp_path: Path) -> Path:
    """
    Fixture to create a mock PDF file for testing purposes.

    This fixture creates a temporary directory and copies a test PDF file into it.
    The path to the mock PDF file is then returned for use in tests.

    Args:
    ----
        tmp_path (Path): The temporary path provided by pytest.

    Returns:
    -------
        Path: A Path object of the path to the mock PDF file.

    """
    test_pdf_path: Path = tmp_path / "Attack_on_Titan.pdf"
    shutil.copy("src/sandbox_agent/data/chroma/documents/Attack_on_Titan.pdf", test_pdf_path)
    return test_pdf_path


@pytest.mark.asyncio
async def test_download_image(mocker: MockerFixture) -> None:
    """
    Test the download_image function.

    Args:
        mocker (MockerFixture): Pytest fixture for mocking.
    """
    mock_response = mocker.AsyncMock()
    mock_response.status = 200
    mock_response.read.return_value = b"test_data"

    mock_session = mocker.AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response

    mocker.patch("aiohttp.ClientSession", return_value=mock_session)

    result = await download_image("test_url")
    assert isinstance(result, BytesIO)
    assert result.getvalue() == b"test_data"

@pytest.mark.asyncio
async def test_download_image_failure(mocker: MockerFixture) -> None:
    """
    Test the download_image function when the request fails.

    Args:
        mocker (MockerFixture): Pytest fixture for mocking.
    """
    mock_response = mocker.AsyncMock()
    mock_response.status = 404

    mock_session = mocker.AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response

    mocker.patch("aiohttp.ClientSession", return_value=mock_session)

    with pytest.raises(ValueError, match="Failed to download image. Status code: 404"):
        await download_image("test_url")


@pytest.mark.asyncio
async def test_file_to_data_uri(mocker: MockerFixture) -> None:
    """
    Test the file_to_data_uri function.

    Args:
        mocker (MockerFixture): Pytest fixture for mocking.
    """
    test_data = b"test_data"
    test_file = File(BytesIO(test_data), filename="test.txt")

    result = await file_to_data_uri(test_file)
    assert result == f"data:image;base64,{base64.b64encode(test_data).decode('ascii')}"


@pytest.mark.asyncio
async def test_data_uri_to_file() -> None:
    """Test the data_uri_to_file function."""
    test_data = b"test_data"
    test_data_uri = f"data:image;base64,{base64.b64encode(test_data).decode('ascii')}"

    result = await data_uri_to_file(test_data_uri, "test.txt")
    assert isinstance(result, File)
    assert result.filename == "test.txt"
    assert result.fp.read() == test_data


def test_file_to_local_data_dict(tmp_path: Path) -> None:
    """
    Test the file_to_local_data_dict function.

    Args:
        tmp_path (Path): Pytest fixture for temporary directory.
    """
    test_file = tmp_path / "test.txt"
    test_file.write_text("test_data")

    result = file_to_local_data_dict(str(test_file), str(tmp_path))
    assert result == {
        "filename": f"{tmp_path}/test.txt",
        "size": test_file.stat().st_size,
        "ext": ".txt",
        "api": test_file,
    }


@pytest.mark.asyncio
async def test_handle_save_attachment_locally(mocker: MockerFixture, tmp_path: Path) -> None:
    """
    Test the handle_save_attachment_locally function.

    Args:
        mocker (MockerFixture): Pytest fixture for mocking.
        tmp_path (Path): Pytest fixture for temporary directory.
    """
    test_data = {"id": "test_id", "filename": "test.txt", "attachment_obj": mocker.AsyncMock()}
    test_data["attachment_obj"].save.return_value = None

    result = await handle_save_attachment_locally(test_data, str(tmp_path))
    assert result == f"{tmp_path}/orig_test_id_test.txt"
    test_data["attachment_obj"].save.assert_called_once_with(result, use_cached=True)


@pytest.mark.asyncio
async def test_details_from_file(mocker: MockerFixture, tmp_path: Path) -> None:
    """
    Test the details_from_file function.

    Args:
        mocker (MockerFixture): Pytest fixture for mocking.
        tmp_path (Path): Pytest fixture for temporary directory.
    """
    test_file = tmp_path / "test.txt"
    test_file.write_text("test_data")

    mocker.patch("sys.platform", "darwin")
    mock_shell = mocker.patch("sandbox_agent.shell._aio_run_process_and_communicate")
    mock_shell.return_value = "test_timestamp"

    result = await details_from_file(str(test_file))
    assert result == ("test.txt", "test_smaller.mp4", "test_timestamp")


def test_create_temp_directory(mocker: MockerFixture) -> None:
    """
    Test the create_temp_directory function.

    Args:
        mocker (MockerFixture): Pytest fixture for mocking.
    """
    mock_uuid = mocker.patch("uuid.uuid4")
    mock_uuid.return_value = "test_uuid"

    result = create_temp_directory()
    assert result == "temp/test_uuid"


def test_get_file_tree(tmp_path: Path) -> None:
    """
    Test the get_file_tree function.

    Args:
        tmp_path (Path): Pytest fixture for temporary directory.
    """
    test_file1 = tmp_path / "test1.txt"
    test_file1.write_text("test_data1")
    test_file2 = tmp_path / "test2.txt"
    test_file2.write_text("test_data2")

    result = get_file_tree(str(tmp_path))
    assert set(result) == {str(test_file1), str(test_file2)}
