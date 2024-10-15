from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from discord import Attachment, Message

import pytest

from sandbox_agent.bot import SandboxAgent
from sandbox_agent.utils.file_operations import create_temp_directory


@pytest.fixture
def sandbox_agent():
    return SandboxAgent()

@pytest.mark.asyncio
async def test_process_attachments(sandbox_agent):
    message = AsyncMock(spec=Message)
    attachment = AsyncMock(spec=Attachment)
    attachment.filename = "test_file.txt"
    attachment.url = "http://example.com/test_file.txt"
    message.attachments = [attachment]

    with patch("aiohttp.ClientSession.get") as mock_get, \
         patch("builtins.open", create=True) as mock_open, \
         patch("sandbox_agent.utils.file_operations.create_temp_directory", return_value="/tmp/test_dir"):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read.return_value = b"test content"
        mock_get.return_value.__aenter__.return_value = mock_response

        await sandbox_agent.process_attachments(message)

        mock_get.assert_called_once_with("http://example.com/test_file.txt")
        mock_open.assert_called_once_with("/tmp/test_dir/test_file.txt", "wb")
        mock_open.return_value.__enter__().write.assert_called_once_with(b"test content")

@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_check_for_attachments_tenor_gif(sandbox_agent):
    message = MagicMock(spec=Message)
    message.content = "Check out this GIF https://tenor.com/view/funny-cat-dancing-gif-12345"
    message.author.display_name = "TestUser"

    result = await sandbox_agent.check_for_attachments(message)
    expected = "Check out this GIF [TestUser posts an animated funny cat dancing]"
    assert result == expected

@pytest.mark.asyncio
async def test_check_for_attachments_image_url(sandbox_agent):
    message = MagicMock(spec=Message)
    message.content = "https://example.com/image.jpg"

    with patch.object(sandbox_agent, "process_image") as mock_process_image:
        result = await sandbox_agent.check_for_attachments(message)

        assert result == "https://example.com/image.jpg"
        mock_process_image.assert_called_once_with("https://example.com/image.jpg")

@pytest.mark.asyncio
async def test_check_for_attachments_attached_image(sandbox_agent):
    message = MagicMock(spec=Message)
    message.content = "Check out this image"
    attachment = MagicMock(spec=Attachment)
    attachment.url = "http://example.com/attached_image.jpg"
    message.attachments = [attachment]

    with patch.object(sandbox_agent, "process_image") as mock_process_image:
        result = await sandbox_agent.check_for_attachments(message)

        assert result == "Check out this image"
        mock_process_image.assert_called_once_with("http://example.com/attached_image.jpg")

@pytest.mark.skip(reason="generated by cursor but not yet ready")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_process_image(sandbox_agent):
    url = "http://example.com/image.jpg"

    with patch("sandbox_agent.utils.file_operations.download_image") as mock_download, \
         patch("PIL.Image.open") as mock_image_open:
        mock_response = MagicMock()
        mock_response.content = b"fake image data"
        mock_download.return_value = mock_response
        mock_image = MagicMock()
        mock_image_open.return_value.convert.return_value = mock_image

        await sandbox_agent.process_image(url)

        mock_download.assert_called_once_with(url)
        mock_image_open.assert_called_once()
        mock_image.convert.assert_called_once_with("RGB")

# Add more test cases for other methods in the SandboxAgent class
