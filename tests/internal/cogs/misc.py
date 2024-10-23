from __future__ import annotations

from discord.ext import commands
from discord.ext.commands import Cog, command


class Misc(Cog):
    # Silence the default on_error handler
    async def cog_command_error(self, ctx, error):
        pass

    @command()
    async def ping(self, ctx):
        await ctx.send("Pong !")


async def setup(bot):
    await bot.add_cog(Misc())
