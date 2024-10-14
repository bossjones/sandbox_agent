"""Discord client for the sandbox agent."""

from __future__ import annotations

import discord

from discord.ext import commands

from sandbox_agent.aio_settings import aiosettings


class DiscordClient(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=aiosettings.prefix, intents=intents)

    async def on_ready(self):
        print(f"Logged in as {self.user}")
