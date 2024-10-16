"""Generate dspy.Signatures"""

from __future__ import annotations

import asyncio

import typer

from sandbox_agent.asynctyper import AsyncTyper


app = AsyncTyper(help="dummy command")


@app.command("dummy")
def cli_dummy_cmd(prompt: str):
    """Generate a new dspy.Module. Example: dspygen sig new 'text -> summary'"""
    return f"dummy cmd: {prompt}"


@app.command()
async def aio_cli_dummy_cmd() -> str:
    """Returns information about the bot."""
    await asyncio.sleep(1)
    return "slept for 1 second"


if __name__ == "__main__":
    app()
