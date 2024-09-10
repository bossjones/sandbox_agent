"""Test the sandbox_agent CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

import pytest

from sandbox_agent.cli import APP, ChromaChoices, about, deps, entry, go, handle_sigterm, main, run_bot, run_pyright


if TYPE_CHECKING:
    from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock

    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture
runner = CliRunner()


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

    @pytest.mark.asyncio
    async def test_run_bot(self) -> None:
        """Test the run_bot function."""
        await run_bot()
        # Add assertions based on the expected behavior of run_bot

    def test_run_pyright(self, mocker: MockerFixture) -> None:
        """Test the run_pyright function."""
        # Mock repo_typing.run_pyright
        mock_run_pyright = mocker.patch("sandbox_agent.utils.repo_typing.run_pyright")

        result = runner.invoke(APP, ["run-pyright"])
        assert result.exit_code == 0
        assert "Generating type stubs for GoobAI" in result.stdout

        # Assert that repo_typing.run_pyright was called
        mock_run_pyright.assert_called_once()

    def test_go(self) -> None:
        """Test the go function."""
        result = runner.invoke(APP, ["go"])
        assert result.exit_code == 0
        assert "Starting up GoobAI Bot" in result.stdout

    # def test_handle_sigterm(self) -> None:
    #     """Test the handle_sigterm function."""
    #     with pytest.raises(SystemExit):
    #         handle_sigterm(signal.SIGTERM, None)
