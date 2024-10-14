"""Entry point for the sandbox agent."""

from __future__ import annotations

import asyncio

from sandbox_agent.bot import SandboxAgent
from sandbox_agent.config import aiosettings


async def main():
    bot = SandboxAgent()
    await bot.start(aiosettings.discord_token.get_secret_value())


if __name__ == "__main__":
    asyncio.run(main())
