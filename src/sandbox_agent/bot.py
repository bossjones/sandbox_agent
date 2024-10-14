"""Main bot implementation for the sandbox agent."""

from __future__ import annotations

from discord.ext import commands

from sandbox_agent.clients.discord import DiscordClient
from sandbox_agent.config import aiosettings
from sandbox_agent.factories import ChatModelFactory, EmbeddingModelFactory, VectorStoreFactory


class SandboxAgent(DiscordClient):
    def __init__(self):
        super().__init__()
        self.chat_model = ChatModelFactory.create()
        self.embedding_model = EmbeddingModelFactory.create()
        self.vector_store = VectorStoreFactory.create(aiosettings.vector_store_type)
        # Initialize other components as needed

    async def on_message(self, message):
        if message.author == self.user:
            return

        await self.process_commands(message)

    @commands.command()
    async def chat(self, ctx, *, message):
        # Implement chat functionality here
        response = await self.chat_model.agenerate([message])
        await ctx.send(response.generations[0][0].text)

    # Add more commands as needed
