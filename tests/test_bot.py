from __future__ import annotations

import pathlib

from collections.abc import AsyncGenerator
from io import BytesIO
from typing import TYPE_CHECKING, Any

import discord
import discord.ext.test as dpytest
import pytest_asyncio

from discord import Attachment, File, Message
from discord.client import _LoopSentinel
from discord.ext.commands import Cog, command
from PIL import Image

import pytest

from sandbox_agent.bot import SandboxAgent
from sandbox_agent.utils.file_operations import create_temp_directory, download_image
from sandbox_agent.vendored.dpytest.tests.conftest import bot


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


# class Misc(Cog):
#     @command()
#     async def ping(self, ctx):
#         await ctx.send("Pong !")

#     @command()
#     async def echo(self, ctx, text: str):
#         await ctx.send(text)


@pytest_asyncio.fixture
# async def bot(mocker: MockerFixture, monkeypatch: MonkeyPatch):
async def mockbot(request: pytest.FixtureRequest) -> AsyncGenerator[SandboxAgent, None]:
    test_bot: SandboxAgent = SandboxAgent()

    # set up the loop
    if isinstance(test_bot.loop, _LoopSentinel):  # type: ignore
        await test_bot._async_setup_hook()  # type: ignore

    marks = request.function.pytestmark
    mark = None
    for mark in marks:
        if mark.name == "cogs":
            break

    if mark is not None:
        for extension in mark.args:
            await test_bot.load_extension("tests.internal." + extension)

    # await test_bot._async_setup_hook()  # type: ignore # setup the loop
    # await test_bot.add_cog(Misc())

    dpytest.configure(test_bot)

    yield test_bot

    # Teardown
    await dpytest.empty_queue()  # empty the global message queue as test teardown


@pytest_asyncio.fixture(autouse=True)
async def cleanup():
    yield
    await dpytest.empty_queue()


@pytest.mark.asyncio
async def test_process_attachments(mockbot: SandboxAgent, mocker: MockerFixture):
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

    await mockbot.process_attachments(message)

    mock_get.assert_called_once_with("http://example.com/test_file.txt")
    mock_open.assert_called_once_with("/tmp/test_dir/test_file.txt", "wb")
    mock_open.return_value.__enter__().write.assert_called_once_with(b"test content")


@pytest.mark.asyncio
async def test_check_for_attachments_tenor_gif(mockbot: SandboxAgent, mocker: MockerFixture):
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

    result = await mockbot.check_for_attachments(message)
    expected = "Check out this GIF  [TestUser posts an animated funny cat dancing gif]"
    assert result == expected


@pytest.mark.asyncio
async def test_check_for_attachments_image_url(mockbot: SandboxAgent, mocker: MockerFixture):
    """
    Test the check_for_attachments method with an image URL.

    This test verifies that the method correctly identifies and processes an image URL.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    message = mocker.MagicMock(spec=Message)
    message.content = "https://example.com/image.jpg"

    mock_process_image = mocker.patch.object(mockbot, "process_image")
    result = await mockbot.check_for_attachments(message)

    assert result == "https://example.com/image.jpg"
    mock_process_image.assert_called_once_with("https://example.com/image.jpg")


@pytest.mark.asyncio
async def test_verify_file_jpg(mockbot: SandboxAgent):
    guild = mockbot.guilds[0]
    channel = guild.text_channels[0]

    path_ = pathlib.Path(__file__).resolve().parent / "fixtures/screenshot_image_larger00000.JPEG"
    file_ = discord.File(path_)
    await channel.send(file=file_)
    assert dpytest.verify().message().attachment(path_)


# @pytest.mark.asyncio
# async def test_check_for_attachments_attached_image(mockbot: SandboxAgent, mocker: MockerFixture):
#     """
#     Test the check_for_attachments method with an attached image.

#     This test verifies that the method correctly identifies and processes an attached image.

#     Args:
#         sandbox_agent (SandboxAgent): The SandboxAgent instance.
#         mocker (MockerFixture): The pytest-mock fixture for mocking.
#     """
#     message = mocker.MagicMock(spec=Message)
#     message.content = "Check out this image"
#     attachment = mocker.MagicMock(spec=Attachment)
#     attachment.url = "http://example.com/attached_image.jpg"
#     message.attachments = [attachment]

#     mock_process_image = mocker.patch.object(mockbot, "process_image")
#     result = await mockbot.check_for_attachments(message)

#     assert result == "Check out this image"
#     mock_process_image.assert_called_once_with("http://example.com/attached_image.jpg")


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_process_image(mockbot: SandboxAgent, mocker: MockerFixture):
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

    await mockbot.process_image(url)

    mock_download.assert_called_once_with(url)
    mock_image_open.assert_called_once()
    mock_image.convert.assert_called_once_with("RGB")


@pytest.mark.asyncio
async def test_on_message_bot_message(mockbot: SandboxAgent, mocker: MockerFixture):
    """
    Test the on_message method with a message from a bot.

    This test verifies that the method does not process messages from bots.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    mock_message = mocker.AsyncMock(spec=Message)
    mock_message.author.bot = True

    await mockbot.on_message(mock_message)

    mock_message.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_percent_message(mockbot: SandboxAgent, mocker: MockerFixture):
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

    await mockbot.on_message(mock_message)

    mock_message.channel.send.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_valid_message(mockbot: SandboxAgent, mocker: MockerFixture):
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

    mock_process_attachments = mocker.patch.object(mockbot, "process_attachments")
    mock_process_commands = mocker.patch.object(mockbot, "process_commands")
    await mockbot.on_message(mock_message)

    mock_process_attachments.assert_called_once_with(mock_message)
    mock_process_commands.assert_called_once_with(mock_message)


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_chat_command(mockbot: SandboxAgent, mocker: MockerFixture):
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

    mock_agenerate = mocker.patch.object(mockbot.chat_model, "agenerate", return_value=mock_response)
    await mockbot.chat(ctx=ctx, message=message)

    mock_agenerate.assert_called_once_with([message])
    ctx.send.assert_called_once_with("Bot response")


@pytest.mark.asyncio
async def test_check_for_attachments_no_attachments(mockbot: SandboxAgent, mocker: MockerFixture):
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

    result = await mockbot.check_for_attachments(message)

    assert result == "Hello, bot!"


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_process_image_invalid_url(mockbot: SandboxAgent, mocker: MockerFixture):
    """
    Test the process_image method with an invalid image URL.

    This test verifies that the method handles invalid image URLs gracefully.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    url = "http://example.com/invalid_image.jpg"

    mock_download = mocker.patch(
        "mockbot.utils.file_operations.download_image", side_effect=Exception("Download failed")
    )

    await mockbot.process_image(url)

    mock_download.assert_called_once_with(url)


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_on_message_exception(mockbot: SandboxAgent, mocker: MockerFixture):
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
        mockbot, "process_attachments", side_effect=Exception("Processing failed")
    )

    await mockbot.on_message(mock_message)

    mock_process_attachments.assert_called_once_with(mock_message)
    mock_message.channel.send.assert_called_once_with("An error occurred while processing the message.")


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_chat_command_empty_message(mockbot: SandboxAgent, mocker: MockerFixture):
    """
    Test the chat command with an empty message.

    This test verifies that the method sends a response indicating that the message is empty.

    Args:
        sandbox_agent (SandboxAgent): The SandboxAgent instance.
        mocker (MockerFixture): The pytest-mock fixture for mocking.
    """
    ctx = mocker.AsyncMock()
    message = ""

    await mockbot.chat(ctx, message=message)

    ctx.send.assert_called_once_with("Please provide a message to chat with the bot.")


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_get_attachments(mockbot: SandboxAgent, mocker: MockerFixture):
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
        "mockbot.utils.attachment_to_dict",
        return_value={"filename": "test_file.txt", "url": "http://example.com/test_file.txt"},
    )

    attachment_data_list_dicts, local_attachment_file_list, local_attachment_data_list_dicts, media_filepaths = (
        mockbot.get_attachments(message)
    )

    assert attachment_data_list_dicts == [{"filename": "test_file.txt", "url": "http://example.com/test_file.txt"}]
    assert local_attachment_file_list == []
    assert local_attachment_data_list_dicts == []
    assert media_filepaths == []
    mock_attachment_to_dict.assert_called_once_with(attachment)


@pytest.mark.skip(reason="This test is not yet implemented")
@pytest.mark.flaky
@pytest.mark.asyncio
async def test_write_attachments_to_disk(mockbot: SandboxAgent, mocker: MockerFixture):
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

    mock_get_context = mocker.patch.object(mockbot, "get_context", return_value=mocker.AsyncMock())
    mock_get_attachments = mocker.patch.object(
        mockbot,
        "get_attachments",
        return_value=([{"filename": "test_file.txt", "url": "http://example.com/test_file.txt"}], [], [], []),
    )

    mock_create_temp_directory = mocker.patch(
        "mockbot.utils.file_operations.create_temp_directory", return_value="/tmp/test_dir"
    )
    mock_handle_save_attachment_locally = mocker.patch(
        "mockbot.utils.file_operations.handle_save_attachment_locally", return_value="/tmp/test_dir/test_file.txt"
    )
    mock_file_to_local_data_dict = mocker.patch(
        "mockbot.utils.file_operations.file_to_local_data_dict",
        return_value={"filename": "/tmp/test_dir/test_file.txt"},
    )
    mock_fix_path = mocker.patch("mockbot.utils.file_functions.fix_path", return_value="/tmp/test_dir/test_file.txt")
    mock_details_from_file = mocker.patch(
        "mockbot.utils.file_operations.details_from_file", return_value=("input_file", "output_file", "timestamp")
    )
    mock_get_file_tree = mocker.patch(
        "mockbot.utils.file_operations.get_file_tree", return_value=["/tmp/test_dir/test_file.txt"]
    )

    await mockbot.write_attachments_to_disk(message)

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


@pytest.mark.integration
class TestBotWithDPyTest:
    @pytest.mark.asyncio
    @pytest.mark.cogs("cogs.misc")
    async def test_ping(self, mockbot: SandboxAgent):
        await dpytest.message("?ping")
        assert dpytest.verify().message().content("Pong !")

    @pytest.mark.cogs("cogs.echo")
    @pytest.mark.asyncio
    async def test_echo(self, mockbot: SandboxAgent):
        await dpytest.message("?echo Hello world")
        assert dpytest.verify().message().contains().content("Hello")

    @pytest.mark.skip(reason="This test is not yet implemented")
    @pytest.mark.flaky
    @pytest.mark.asyncio
    async def test_on_ready(self, mockbot: SandboxAgent):
        """
        Test the on_ready event.

        This test verifies that the bot logs the 'Logged in as...' message when the on_ready event is triggered.
        """
        with pytest.raises(RuntimeError):
            await mockbot.on_ready()

    @pytest.mark.skip(reason="This test is not yet implemented")
    @pytest.mark.flaky
    @pytest.mark.asyncio
    @pytest.mark.discordonly
    async def test_process_commands_help_command(self, mockbot: SandboxAgent):
        """
        Test the process_commands method with the !help command.

        This test verifies that the bot responds with the help message when the !help command is used.
        """
        message = dpytest.backend.make_message("!help", author=dpytest.backend.make_user(bot=False))
        await dpytest.message(message)

        assert dpytest.verify().message().contains().content("Here are the available commands:")

    @pytest.mark.skip(reason="something wrong with make_message")
    @pytest.mark.flaky
    @pytest.mark.asyncio
    @pytest.mark.discordonly
    async def test_process_commands_unknown_command(self, mockbot: SandboxAgent):
        """
        Test the process_commands method with an unknown command.

        This test verifies that the bot responds with an unknown command message when an unrecognized command is used.
        """
        message = dpytest.backend.make_message(
            "!unknown_command", author=dpytest.backend.make_user("FakeApp", "0001", None)
        )
        await dpytest.message(message)

        assert (
            dpytest.verify()
            .message()
            .contains()
            .content("Unknown command. Please use !help to see available commands.")
        )

    @pytest.mark.skip(reason="something wrong with make_message")
    @pytest.mark.flaky
    @pytest.mark.asyncio
    @pytest.mark.discordonly
    async def test_process_commands_chat_command(self, mockbot: SandboxAgent, mocker: MockerFixture):
        """
        Test the process_commands method with the !chat command.

        This test verifies that the bot invokes the chat command when the !chat command is used.
        """
        # ctx = mocker.AsyncMock()
        # guild: discord.Guild = mockbot.guilds[0]
        # channel: discord.abc.Messageable = guild.channels[0]
        # message = dpytest.backend.make_message("!chat Hello, bot!", channel=channel, author=dpytest.backend.make_user("FakeApp", "0001", None))

        # mock_chat = mocker.patch.object(mockbot, "chat")
        # await dpytest.message(message)

        await dpytest.message("%ping")
        assert dpytest.verify().message().content("Pong !")

        # mock_chat.assert_called_once_with(ctx, message="Hello, bot!")

    # @pytest.mark.asyncio
    # @pytest.mark.discordonly
    # async def test_process_commands_image_command(self, mockbot: SandboxAgent, mocker: MockerFixture):
    #     """
    #     Test the process_commands method with the !image command.

    #     This test verifies that the bot invokes the image command when the !image command is used.
    #     """
    #     ctx = mocker.AsyncMock()
    #     guild: discord.Guild = mockbot.guilds[0]
    #     channel: discord.abc.Messageable = guild.channels[0]
    #     message = dpytest.backend.make_message(
    #         "!image cat", channel=channel, author=dpytest.backend.make_user("FakeApp", "0001", None)
    #     )

    #     mock_image = mocker.patch.object(mockbot, "image")
    #     await dpytest.message(message)

    #     mock_image.assert_called_once_with(ctx, query="cat")

    @pytest.mark.asyncio
    @pytest.mark.discordonly
    async def test_verify_file_text(self, mockbot: SandboxAgent):
        guild = mockbot.guilds[0]
        channel = guild.text_channels[0]

        path_ = pathlib.Path(__file__).resolve().parent / "fixtures/loremimpsum.txt"
        file_ = discord.File(path_)
        await channel.send(file=file_)
        assert dpytest.verify().message().attachment(path_)

    @pytest.mark.asyncio
    @pytest.mark.discordonly
    async def test_verify_file_jpg(self, mockbot: SandboxAgent):
        guild = mockbot.guilds[0]
        channel = guild.text_channels[0]

        path_ = pathlib.Path(__file__).resolve().parent / "fixtures/unit-tests.jpg"
        file_ = discord.File(path_)
        await channel.send(file=file_)
        assert dpytest.verify().message().attachment(path_)

    @pytest.mark.asyncio
    async def test_dm_send(self, mockbot: SandboxAgent):
        guild = mockbot.guilds[0]
        await guild.members[0].send("hi")

        assert dpytest.verify().message().content("hi")

    @pytest.mark.skip(reason="something wrong with make_message")
    @pytest.mark.flaky
    @pytest.mark.asyncio
    @pytest.mark.cogs("cogs.echo")
    async def test_dm_message(self, mockbot: SandboxAgent):
        guild = mockbot.guilds[0]
        member = guild.members[0]
        dm = await member.create_dm()
        await dpytest.message("!echo Ah-Ha!", dm)

        assert dpytest.verify().message().content("Ah-Ha!")

    # @pytest.mark.asyncio
    # @pytest.mark.discordonly
    # async def test_on_message_attachment(self, mockbot: SandboxAgent):
    #     """
    #     Test the on_message event with an attachment.

    #     This test verifies that the bot processes the attachment correctly when a message with an attachment is sent.
    #     """
    #     attachment_file = File(BytesIO(b"test content"), filename="test_file.txt")
    #     guild: discord.Guild = mockbot.guilds[0]
    #     channel: discord.abc.Messageable = guild.channels[0]
    #     message = dpytest.backend.make_message(
    #         attachments=[attachment_file], channel=channel, author=dpytest.backend.make_user("FakeApp", "0001", None)
    #     )
    #     await dpytest.message(message)

    #     assert dpytest.verify().message().contains().content("[User posts a file named test_file.txt]")

    # @pytest.mark.asyncio
    # @pytest.mark.discordonly
    # async def test_on_message_image_url(self, mockbot: SandboxAgent):
    #     """
    #     Test the on_message event with an image URL.

    #     This test verifies that the bot processes the image URL correctly when a message with an image URL is sent.
    #     """
    #     guild: discord.Guild = mockbot.guilds[0]
    #     channel: discord.abc.Messageable = guild.channels[0]
    #     message = dpytest.backend.make_message(
    #         content="Check out this image https://example.com/image.jpg",
    #         channel=channel,
    #         author=dpytest.backend.make_user("FakeApp", "0001", None),
    #     )
    #     await dpytest.message(message)

    #     assert dpytest.verify().message().contains().content("[User posts an image]")

    # @pytest.mark.asyncio
    # @pytest.mark.discordonly
    # async def test_on_message_tenor_gif(self, mockbot: SandboxAgent):
    #     """
    #     Test the on_message event with a Tenor GIF URL.

    #     This test verifies that the bot processes the Tenor GIF URL correctly when a message with a Tenor GIF URL is sent.
    #     """
    #     guild: discord.Guild = mockbot.guilds[0]
    #     channel: discord.abc.Messageable = guild.channels[0]
    #     message = dpytest.backend.make_message(
    #         content="Check out this GIF https://tenor.com/view/funny-cat-dancing-gif-12345",
    #         channel=channel,
    #         author=dpytest.backend.make_user("FakeApp", "0001", None),
    #     )
    #     await dpytest.message(message)

    #     assert dpytest.verify().message().contains().content("[User posts an animated funny cat dancing gif]")

    # @pytest.mark.asyncio
    # async def test_process_image_url(self, mockbot: SandboxAgent, mocker: MockerFixture):
    #     """
    #     Test the process_image_url method.

    #     This test verifies that the bot processes the image URL correctly and performs any necessary image processing.
    #     """
    #     url = "https://example.com/image.jpg"
    #     mock_download_image = mocker.patch("sandbox_agent.utils.file_operations.download_image")
    #     mock_response = mocker.MagicMock()
    #     mock_response.content = b"fake image data"
    #     mock_download_image.return_value = mock_response
    #     mock_image = mocker.MagicMock(spec=Image.Image)
    #     mocker.patch("PIL.Image.open", return_value=mock_image)

    #     await mockbot.process_image(url)

    #     mock_download_image.assert_called_once_with(url)
    #     mock_image.convert.assert_called_once_with("RGB")

    # @pytest.mark.asyncio
    # async def test_chat_command_with_message(self, mockbot: SandboxAgent, mocker: MockerFixture):
    #     """
    #     Test the chat command.

    #     This test verifies that the bot generates a response using the chat model when the !chat command is used.
    #     """
    #     ctx = mocker.AsyncMock()
    #     message = "Hello, bot!"
    #     mock_chat_model = mocker.AsyncMock()
    #     mock_chat_model.agenerate.return_value = mocker.AsyncMock(generations=[[mocker.AsyncMock(text="Hello, user!")]])
    #     mockbot.chat_model = mock_chat_model

    #     await mockbot.chat(ctx, message=message)

    #     mock_chat_model.agenerate.assert_called_once_with([message])
    #     ctx.send.assert_called_once_with("Hello, user!")

    # @pytest.mark.integration
    # @pytest.mark.vcronly
    # @pytest.mark.default_cassette("test_process_user_task.yaml")
    # @pytest.mark.vcr(
    #     allow_playback_repeats=True,
    #     match_on=["method", "scheme", "port", "path", "query", "headers"],
    #     ignore_localhost=False,
    # )
    # @pytest.mark.asyncio
    # async def test_process_user_task(self, mockbot: SandboxAgent, vcr: Any) -> None:
    #     """
    #     Test the process_user_task method of the SandboxAgent class.

    #     This test mocks the graph.ainvoke method to return a predefined output and verifies that
    #     the process_user_task method returns the expected output.

    #     Args:
    #     ----
    #         mockbot (SandboxAgent): The SandboxAgent instance to test.
    #         vcr (Any): The vcr fixture for recording and replaying HTTP requests.

    #     Returns:
    #     -------
    #         None

    #     """
    #     # Call the process_user_task method with test inputs
    #     session_id = "test_session"
    #     user_task = "what's the answer to life, the universe, and everything?"
    #     thread_id = 123

    #     result = await mockbot.process_user_task(session_id, user_task, thread_id)

    #     # Verify that the process_user_task method returns the expected output
    #     assert result == "42"
    #     assert vcr.play_count == 0

    # @pytest.mark.integration
    # @pytest.mark.vcronly
    # @pytest.mark.default_cassette("test_process_user_task_error.yaml")
    # @pytest.mark.vcr(
    #     allow_playback_repeats=True,
    #     match_on=["method", "scheme", "port", "path", "query", "headers"],
    #     ignore_localhost=False,
    # )
    # @pytest.mark.asyncio
    # async def test_process_user_task_error(self, mockbot: SandboxAgent, vcr: Any) -> None:
    #     """
    #     Test the process_user_task method of the SandboxAgent class when an error occurs.

    #     This test mocks the graph.ainvoke method to raise an exception and verifies that
    #     the process_user_task method returns an error message.

    #     Args:
    #     ----
    #         mockbot (SandboxAgent): The SandboxAgent instance to test.
    #         vcr (Any): The vcr fixture for recording and replaying HTTP requests.

    #     Returns:
    #     -------
    #         None

    #     """
    #     # Mock the graph.ainvoke method to raise an exception
    #     mockbot.graph.ainvoke.side_effect = Exception("Test error")

    #     # Call the process_user_task method with test inputs
    #     session_id = "test_session"
    #     user_task = "what's the answer to life, the universe, and everything?"
    #     thread_id = 123

    #     result = await mockbot.process_user_task(session_id, user_task, thread_id)

    #     # Verify that the process_user_task method returns the expected error message
    #     assert result == "An error occurred while processing the task."
    #     assert vcr.play_count == 0
