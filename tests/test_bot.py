from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any

from discord import Attachment, File, Message
from PIL import Image

import pytest

from sandbox_agent.bot import SandboxAgent
from sandbox_agent.utils.file_operations import create_temp_directory, download_image


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def sandbox_agent() -> SandboxAgent:
    """Fixture to create an instance of SandboxAgent for testing."""
    return SandboxAgent()


@pytest.mark.asyncio
async def test_process_attachments(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the process_attachments method of SandboxAgent.

    This test mocks the necessary dependencies and verifies that the attachments are processed correctly.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    message = mocker.AsyncMock(spec=Message)
    attachment = mocker.AsyncMock(spec=Attachment)
    attachment.filename = "test_file.txt"
    attachment.url = "http://example.com/test_file.txt"
    message.attachments = [attachment]

    mock_get = mocker.patch("aiohttp.ClientSession.get")
    mock_open = mocker.patch("builtins.open", create=True)
    mocker.patch("sandbox_agent.utils.file_operations.create_temp_directory", return_value="/tmp/test_dir")
    mock_response = mocker.AsyncMock()
    mock_response.status = 200
    mock_response.read.return_value = b"test content"
    mock_get.return_value.__aenter__.return_value = mock_response

    await sandbox_agent.process_attachments(message)

    mock_get.assert_called_once_with("http://example.com/test_file.txt")
    mock_open.assert_called_once_with("/tmp/test_dir/test_file.txt", "wb")
    mock_open.return_value.__enter__().write.assert_called_once_with(b"test content")


@pytest.mark.asyncio
async def test_check_for_attachments_tenor_gif(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the check_for_attachments method with a Tenor GIF URL.

    This test verifies that the method correctly identifies and processes a Tenor GIF URL.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    message = mocker.MagicMock(spec=Message)
    message.content = "Check out this GIF https://tenor.com/view/funny-cat-dancing-gif-12345"
    message.author.display_name = "TestUser"

    result = await sandbox_agent.check_for_attachments(message)
    expected = "Check out this GIF  [TestUser posts an animated funny cat dancing gif]"
    assert result == expected


@pytest.mark.asyncio
async def test_check_for_attachments_image_url(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the check_for_attachments method with an image URL.

    This test verifies that the method correctly identifies and processes an image URL.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    message = mocker.MagicMock(spec=Message)
    message.content = "https://example.com/image.jpg"

    mock_process_image = mocker.patch.object(sandbox_agent, "process_image")
    result = await sandbox_agent.check_for_attachments(message)

    assert result == "https://example.com/image.jpg"
    mock_process_image.assert_called_once_with("https://example.com/image.jpg")


@pytest.mark.asyncio
async def test_check_for_attachments_attached_image(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the check_for_attachments method with an attached image.

    This test verifies that the method correctly identifies and processes an attached image.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    message = mocker.MagicMock(spec=Message)
    message.content = "Check out this image"
    attachment = mocker.MagicMock(spec=Attachment)
    attachment.url = "http://example.com/attached_image.jpg"
    message.attachments = [attachment]

    mock_process_image = mocker.patch.object(sandbox_agent, "process_image")
    result = await sandbox_agent.check_for_attachments(message)

    assert result == "Check out this image"
    mock_process_image.assert_called_once_with("http://example.com/attached_image.jpg")


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_process_image(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the process_image method of SandboxAgent.

    This test mocks the necessary dependencies and verifies that the image is processed correctly.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    url = "http://example.com/image.jpg"

    mock_download = mocker.patch("sandbox_agent.utils.file_operations.download_image")
    mock_image_open = mocker.patch("PIL.Image.open")
    mock_response = mocker.MagicMock()
    mock_response.content = b"fake image data"
    mock_download.return_value = mock_response
    mock_image = mocker.MagicMock()
    mock_image_open.return_value.convert.return_value = mock_image

    await sandbox_agent.process_image(url)

    mock_download.assert_called_once_with(url)
    mock_image_open.assert_called_once()
    mock_image.convert.assert_called_once_with("RGB")


@pytest.mark.asyncio
async def test_on_message_bot_message(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the on_message method with a message from a bot.

    This test verifies that the method does not process messages from bots.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_message = mocker.AsyncMock(spec=Message)
    mock_message.author.bot = True

    await sandbox_agent.on_message(mock_message)

    mock_message.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_percent_message(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the on_message method with a message starting with a percent sign.

    This test verifies that the method does not process messages starting with a percent sign.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_message = mocker.AsyncMock(spec=Message)
    mock_message.author.bot = False
    mock_message.content = "%some command"

    await sandbox_agent.on_message(mock_message)

    mock_message.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_valid_message(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the on_message method with a valid message.

    This test verifies that the method processes valid messages correctly.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_message = mocker.AsyncMock(spec=Message)
    mock_message.author.bot = False
    mock_message.content = "Hello, bot!"

    mock_process_attachments = mocker.patch.object(sandbox_agent, "process_attachments")
    mock_process_commands = mocker.patch.object(sandbox_agent, "process_commands")
    await sandbox_agent.on_message(mock_message)

    mock_process_attachments.assert_called_once_with(mock_message)
    mock_process_commands.assert_called_once_with(mock_message)


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_chat_command(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the chat command of SandboxAgent.

    This test mocks the necessary dependencies and verifies that the chat command generates a response.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    ctx = mocker.AsyncMock()
    message = "Hello, bot!"

    mock_response = mocker.MagicMock()
    mock_response.generations = [[mocker.MagicMock(text="Bot response")]]

    mock_agenerate = mocker.patch.object(sandbox_agent.chat_model, "agenerate", return_value=mock_response)
    await sandbox_agent.chat(ctx, message=message)

    mock_agenerate.assert_called_once_with([message])
    ctx.send.assert_called_once_with("Bot response")


@pytest.mark.asyncio
async def test_check_for_attachments_no_attachments(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the check_for_attachments method with no attachments.

    This test verifies that the method returns the original message content when there are no attachments.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    message = mocker.MagicMock(spec=Message)
    message.content = "Hello, bot!"
    message.attachments = []

    result = await sandbox_agent.check_for_attachments(message)

    assert result == "Hello, bot!"


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_process_image_invalid_url(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the process_image method with an invalid image URL.

    This test verifies that the method handles invalid image URLs gracefully.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    url = "http://example.com/invalid_image.jpg"

    mock_download = mocker.patch(
        "sandbox_agent.utils.file_operations.download_image", side_effect=Exception("Download failed")
    )

    await sandbox_agent.process_image(url)

    mock_download.assert_called_once_with(url)


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_on_message_exception(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the on_message method when an exception occurs.

    This test verifies that the method handles exceptions gracefully and sends an error message.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_message = mocker.AsyncMock(spec=Message)
    mock_message.author.bot = False
    mock_message.content = "Hello, bot!"

    mock_process_attachments = mocker.patch.object(
        sandbox_agent, "process_attachments", side_effect=Exception("Processing failed")
    )

    await sandbox_agent.on_message(mock_message)

    mock_process_attachments.assert_called_once_with(mock_message)
    mock_message.channel.send.assert_called_once_with("An error occurred while processing the message.")


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_chat_command_empty_message(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the chat command with an empty message.

    This test verifies that the method sends a response indicating that the message is empty.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    ctx = mocker.AsyncMock()
    message = ""

    await sandbox_agent.chat(ctx, message=message)

    ctx.send.assert_called_once_with("Please provide a message to chat with the bot.")


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_get_attachments(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the get_attachments method of SandboxAgent.

    This test mocks the necessary dependencies and verifies that the attachment data is retrieved correctly.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    message = mocker.MagicMock(spec=Message)
    attachment = mocker.MagicMock(spec=Attachment)
    attachment.filename = "test_file.txt"
    attachment.url = "http://example.com/test_file.txt"
    message.attachments = [attachment]

    mock_attachment_to_dict = mocker.patch(
        "sandbox_agent.utils.attachment_to_dict",
        return_value={"filename": "test_file.txt", "url": "http://example.com/test_file.txt"},
    )

    attachment_data_list_dicts, local_attachment_file_list, local_attachment_data_list_dicts, media_filepaths = (
        sandbox_agent.get_attachments(message)
    )

    assert attachment_data_list_dicts == [{"filename": "test_file.txt", "url": "http://example.com/test_file.txt"}]
    assert local_attachment_file_list == []
    assert local_attachment_data_list_dicts == []
    assert media_filepaths == []
    mock_attachment_to_dict.assert_called_once_with(attachment)


@pytest.mark.asyncio
async def test_write_attachments_to_disk(sandbox_agent: SandboxAgent, mocker: MockerFixture):
    """
    Test the write_attachments_to_disk method of SandboxAgent.

    This test mocks the necessary dependencies and verifies that the attachments are saved to disk correctly.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    message = mocker.MagicMock(spec=Message)
    attachment = mocker.MagicMock(spec=Attachment)
    attachment.filename = "test_file.txt"
    attachment.url = "http://example.com/test_file.txt"
    message.attachments = [attachment]

    mock_get_context = mocker.patch.object(sandbox_agent, "get_context", return_value=mocker.AsyncMock())
    mock_get_attachments = mocker.patch.object(
        sandbox_agent,
        "get_attachments",
        return_value=([{"filename": "test_file.txt", "url": "http://example.com/test_file.txt"}], [], [], []),
    )

    mock_create_temp_directory = mocker.patch(
        "sandbox_agent.utils.file_operations.create_temp_directory", return_value="/tmp/test_dir"
    )
    mock_handle_save_attachment_locally = mocker.patch(
        "sandbox_agent.utils.file_operations.handle_save_attachment_locally", return_value="/tmp/test_dir/test_file.txt"
    )
    mock_file_to_local_data_dict = mocker.patch(
        "sandbox_agent.utils.file_operations.file_to_local_data_dict",
        return_value={"filename": "/tmp/test_dir/test_file.txt"},
    )
    mock_fix_path = mocker.patch(
        "sandbox_agent.utils.file_functions.fix_path", return_value="/tmp/test_dir/test_file.txt"
    )
    mock_details_from_file = mocker.patch(
        "sandbox_agent.utils.file_operations.details_from_file", return_value=("input_file", "output_file", "timestamp")
    )
    mock_get_file_tree = mocker.patch(
        "sandbox_agent.utils.file_operations.get_file_tree", return_value=["/tmp/test_dir/test_file.txt"]
    )

    await sandbox_agent.write_attachments_to_disk(message)

    mock_get_context.assert_called_once_with(message)
    mock_get_attachments.assert_called_once_with(message)
    mock_create_temp_directory.assert_called_once()
    mock_handle_save_attachment_locally.assert_called_once_with(
        {"filename": "test_file.txt", "url": "http://example.com/test_file.txt"}, "/tmp/test_dir"
    )
    mock_file_to_local_data_dict.assert_called_once_with("/tmp/test_dir/test_file.txt", "/tmp/test_dir")
    mock_fix_path.assert_called_once_with("/tmp/test_dir/test_file.txt")
    mock_details_from_file.assert_called_once_with("/tmp/test_dir/test_file.txt", cwd="/tmp/test_dir")
    mock_get_file_tree.assert_called_once_with("/tmp/test_dir")
