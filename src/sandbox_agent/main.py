"""Entry point for the sandbox agent."""

from __future__ import annotations

import asyncio

from loguru import logger as LOGGER

from sandbox_agent.aio_settings import aiosettings
from sandbox_agent.bot import SandboxAgent


async def main():
    async with SandboxAgent() as bot:
        # if aiosettings.enable_redis:
        #     bot.pool = pool
        await bot.start()

    await LOGGER.complete()


if __name__ == "__main__":
    asyncio.run(main())
