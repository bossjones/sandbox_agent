"""Test the sandbox_agent CLI."""

from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING

from typer.testing import CliRunner

import pytest

from sandbox_agent import cli
from sandbox_agent.cli import (
    APP,
    ChromaChoices,
    about,
    deps,
    entry,
    go,
    handle_sigterm,
    main,
    run_bot,
    run_bot_with_redis,
    run_pyright,
)


if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

runner = CliRunner()

# @pytest.fixture()
# def mod(monkeypatch: MonkeyPatch):
#     with monkeypatch.context() as m:
#         monkeypatch.syspath_prepend(list(cli.__path__)[0])
#         from sandbox_agent import cli

#         return cli

# @pytest.fixture()
# def app(mod):
#     return mod.app


class TestApp:
    """Test the sandbox_agent CLI."""

    def test_version(self) -> None:
        """Test the version command."""
        result = runner.invoke(APP, ["version"])
        assert result.exit_code == 0
        assert "sandbox_agent version:" in result.stdout

    def test_help(self) -> None:
        """Test the help command."""
        result = runner.invoke(APP, ["--help"])
        assert result.exit_code == 0
        assert "About command" in result.stdout

    def test_about(self) -> None:
        """Test the about command."""
        result = runner.invoke(APP, ["about"])
        assert result.exit_code == 0
        assert "This is GoobBot CLI" in result.stdout

    def test_deps(self) -> None:
        """Test the deps command."""
        result = runner.invoke(APP, ["deps"])
        assert result.exit_code == 0
        assert "sandbox_agent version:" in result.stdout
        assert "langchain_version:" in result.stdout
        assert "langchain_community_version:" in result.stdout
        assert "langchain_core_version:" in result.stdout
        assert "langchain_openai_version:" in result.stdout
        assert "langchain_text_splitters_version:" in result.stdout
        assert "langchain_chroma_version:" in result.stdout
        assert "chromadb_version:" in result.stdout
        assert "langsmith_version:" in result.stdout
        assert "pydantic_version:" in result.stdout
        assert "pydantic_settings_version:" in result.stdout
        assert "ruff_version:" in result.stdout

    @pytest.mark.parametrize("choice", list(ChromaChoices))
    def test_chroma_choices(self, choice: ChromaChoices) -> None:
        """Test the ChromaChoices enum."""
        assert choice in ChromaChoices
        assert choice.value == choice.name

    @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
    @pytest.mark.flaky
    def test_load_commands(self) -> None:
        """Test the load_commands function."""
        from sandbox_agent.cli import load_commands

        load_commands()
        command_names = [cmd.name for cmd in APP.registered_commands]
        assert "chroma" in command_names

    def test_version_callback(self) -> None:
        """Test the version_callback function."""
        from typer import Exit

        from sandbox_agent.cli import version_callback

        with pytest.raises(Exit):
            version_callback(True)

    def test_show(self) -> None:
        """Test the show command."""
        result = runner.invoke(APP, ["show"])
        assert result.exit_code == 0
        assert "Show sandbox_agent" in result.stdout

    @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
    @pytest.mark.flaky
    def test_main(self, mocker: MockerFixture) -> None:
        """Test the main function."""
        # Mock load_commands
        mock_load_commands = mocker.patch("sandbox_agent.cli.load_commands")

        with pytest.raises(SystemExit):
            main()

        # Assert that load_commands was called
        mock_load_commands.assert_called_once()

    @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
    @pytest.mark.flaky
    def test_entry(self, mocker: MockerFixture) -> None:
        """Test the entry function."""
        # Mock load_commands
        mock_load_commands = mocker.patch("sandbox_agent.cli.load_commands")

        with pytest.raises(SystemExit):
            entry()

        # Assert that load_commands was called
        mock_load_commands.assert_called_once()

    # @pytest.mark.asyncio
    # async def test_run_bot(self) -> None:
    #     """Test the run_bot function."""
    #     await run_bot()
    #     # Add assertions based on the expected behavior of run_bot

    def test_run_pyright(self, mocker: MockerFixture) -> None:
        """Test the run_pyright function."""
        # Mock repo_typing.run_pyright
        mock_run_pyright = mocker.patch("sandbox_agent.utils.repo_typing.run_pyright")

        result = runner.invoke(APP, ["run-pyright"])
        assert result.exit_code == 0
        assert "Generating type stubs for SandboxAgentAI" in result.stdout

        # Assert that repo_typing.run_pyright was called
        mock_run_pyright.assert_called_once()

    def test_go(self, mocker: MockerFixture) -> None:
        """
        Test the go function.

        Args:
            mocker (MockerFixture): The pytest-mock fixture for mocking.
        """
        # Mock the run_bot function
        mock_run_bot = mocker.patch("sandbox_agent.cli.run_bot")

        result = runner.invoke(APP, ["go"])
        assert result.exit_code == 0
        assert "Starting up SandboxAgentAI Bot" in result.stdout

        # Assert that run_bot was called
        mock_run_bot.assert_called_once()

    # def test_handle_sigterm(self) -> None:
    #     """Test the handle_sigterm function."""
    #     with pytest.raises(SystemExit):
    #         handle_sigterm(signal.SIGTERM, None)

    # def test_run_bot_with_redis_success(self, mocker: MockerFixture) -> None:
    #     """Test the run_bot_with_redis function when it runs successfully."""
    #     # Mock SandboxAgent
    #     mock_sandbox_agent = mocker.patch("sandbox_agent.cli.SandboxAgent")
    #     mock_bot_instance = mocker.AsyncMock()
    #     mock_sandbox_agent.return_value = mock_bot_instance

    #     # Run the function
    #     asyncio.run(run_bot_with_redis())

    #     # Assert that SandboxAgent was called and the bot was started
    #     mock_sandbox_agent.assert_called_once()
    #     mock_bot_instance.start.assert_awaited_once()

    # def test_run_bot_with_redis_exception(self, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
    #     """Test the run_bot_with_redis function when an exception occurs."""
    #     # Mock SandboxAgent to raise an exception
    #     mock_sandbox_agent = mocker.patch("sandbox_agent.cli.SandboxAgent")
    #     mock_sandbox_agent.side_effect = Exception("Test exception")

    #     # Run the function
    #     asyncio.run(run_bot_with_redis())

    #     # Assert that the exception was logged
    #     assert "Test exception" in caplog.text

    @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
    @pytest.mark.flaky
    def test_load_commands_success(self) -> None:
        """Test the load_commands function when it loads commands successfully."""
        # Run the function
        from sandbox_agent.cli import load_commands

        load_commands()

        from sandbox_agent.cli import APP

        # Assert that the command was added to the app
        assert "dummy" in [cmd.name for cmd in APP.registered_commands]

    # def test_load_commands_no_app(self, mocker: MockerFixture) -> None:
    #     """Test the load_commands function when a subcommand module doesn't have an app."""
    #     # Mock os.listdir and import_module
    #     mock_listdir = mocker.patch("os.listdir")
    #     mock_listdir.return_value = ["test_cmd.py"]
    #     mock_import_module = mocker.patch("importlib.import_module")
    #     mock_module = mocker.MagicMock()
    #     mock_module.app = None
    #     mock_import_module.return_value = mock_module

    #     # Run the function
    #     load_commands()

    #     # Assert that no commands were added to the app
    #     assert len(APP.registered_commands) == 0

    # @pytest.mark.asyncio
    # async def test_run_bot_exception(mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
    #     """Test the run_bot function when an exception occurs."""
    #     # Mock SandboxAgent to raise an exception
    #     mock_sandbox_agent = mocker.patch("sandbox_agent.cli.SandboxAgent")
    #     mock_sandbox_agent.side_effect = Exception("Test exception")

    #     # Run the function
    #     await run_bot()

    #     # Assert that the exception was logged and handled
    #     assert "Test exception" in caplog.text
    #     assert "Error Class: <class 'Exception'>" in caplog.text
    #     assert "[UNEXPECTED] Exception: Test exception" in caplog.text


@pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
@pytest.mark.flaky
class TestPgVectorAppCommands:
    """Test the pgvector commands."""

    @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
    @pytest.mark.flaky
    def test_export_collection(self, mocker: MockerFixture) -> None:
        """Test the export_collection command."""
        mock_export_collection_data = mocker.spy(cli, "export_collection_data")
        folder_path = "test_folder"
        collection = "test_collection"

        result = runner.invoke(APP, ["export-collection", folder_path, collection])
        assert result.exit_code == 0
        assert f"Running export_collection with folder_path={folder_path}, collection={collection}" in result.stdout

        mock_export_collection_data.assert_called_once_with(folder_path=folder_path, collection=collection)

    @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
    @pytest.mark.flaky
    def test_import_collection(self, mocker: MockerFixture) -> None:
        """Test the import_collection command."""
        mock_import_collection_data = mocker.spy(cli, "import_collection_data")
        folder_path = "test_folder"
        collection = "test_collection"

        result = runner.invoke(APP, ["import-collection", folder_path, collection])
        assert result.exit_code == 0
        assert f"Running import_collection with folder_path={folder_path}, collection={collection}" in result.stdout

        mock_import_collection_data.assert_called_once_with(folder_path=folder_path, collection=collection)

    @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
    @pytest.mark.flaky
    def test_import_file(self, mocker: MockerFixture) -> None:
        """Test the import_file command."""
        mock_index_file_folder = mocker.spy(cli, "index_file_folder")
        file_path = "test_file"
        collection = "test_collection"
        options = '{"option1": "value1"}'

        result = runner.invoke(APP, ["import-file", file_path, collection, "--options", options])
        assert result.exit_code == 0
        assert (
            f"Running import_file with file_path={file_path}, collection={collection}, options={options}"
            in result.stdout
        )

        mock_index_file_folder.assert_called_once_with(file_path=file_path, collection=collection, option1="value1")

    @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
    @pytest.mark.flaky
    def test_db_current(self, mocker: MockerFixture) -> None:
        """Test the db_current command."""
        mock_current = mocker.spy(cli, "current")
        result = runner.invoke(APP, ["db-current", "--verbose"])
        assert result.exit_code == 0
        assert "Running db_current with verbose=True" in result.stdout

        mock_current.assert_called_once_with(True)

    @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
    @pytest.mark.flaky
    def test_db_upgrade(self, mocker: MockerFixture) -> None:
        """Test the db_upgrade command."""
        mock_upgrade = mocker.spy(cli, "upgrade")
        revision = "12345"

        result = runner.invoke(APP, ["db-upgrade", "--revision", revision])
        assert result.exit_code == 0
        assert f"Running db_upgrade with revision={revision}" in result.stdout

        mock_upgrade.assert_called_once_with(revision)

    @pytest.mark.skip(reason="This is a work in progress and it is currently expected to fail")
    @pytest.mark.flaky
    def test_db_downgrade(self, mocker: MockerFixture) -> None:
        """Test the db_downgrade command."""
        mock_downgrade = mocker.spy(cli, "downgrade")
        revision = "12345"

        result = runner.invoke(APP, ["db-downgrade", "--revision", revision])
        assert result.exit_code == 0
        assert f"Running db_downgrade with revision={revision}" in result.stdout

        mock_downgrade.assert_called_once_with(revision)
