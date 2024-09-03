"""Test the sandbox_agent CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from sandbox_agent.cli import APP


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
        """Test the help command."""
        result = runner.invoke(APP, ["about"])
        assert result.exit_code == 0
        assert "This is GoobBot CLI" in result.stdout
