"""Test the sandbox_agent CLI."""

from __future__ import annotations

from typer.testing import CliRunner

import pytest

from sandbox_agent.cli import APP, ChromaChoices


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
