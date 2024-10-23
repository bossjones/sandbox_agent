# pylint: disable=too-many-function-args
# mypy: disable-error-code="arg-type, var-annotated, list-item, no-redef, truthy-bool, return-value"
# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import re

from io import BytesIO
from typing import TYPE_CHECKING

import discord
import requests
import torch

from discord.ext import commands
from loguru import logger as LOGGER
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor  # pyright: ignore[reportAttributeAccessIssue]

from sandbox_agent.factories import cmd_factory, guild_factory


if TYPE_CHECKING:
    from sandbox_agent.bot import SandboxAgent


class ImageCaptionCog(commands.Cog, name="image_caption"):
    """Cog for generating image captions using the BLIP model."""

    def __init__(self, bot: SandboxAgent):
        """
        Initialize the ImageCaptionCog.

        Args:
            bot (SandboxAgent): The bot instance.
        """
        self.bot = bot
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=torch.float32
        ).to("cpu")

        LOGGER.info(f" self.model = {self.model}")
        LOGGER.info(f" type(self.model) = {type(self.model)}")
        LOGGER.info(f" self.processor = {self.processor}")
        LOGGER.info(f" type(self.processor) = {type(self.processor)}")

        LOGGER.complete()

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Event triggered when the cog is ready."""
        print(f"{type(self).__name__} Cog ready.")

    @commands.Cog.listener()
    async def on_guild_join(self, guild: guild_factory.Guild) -> None:
        """
        Event triggered when the bot joins a new guild.

        Args:
            guild (guild_factory.Guild): The joined guild.
        """
        _ = await guild_factory.Guild(id=guild.id)

    @commands.command(name="image_comment")
    async def image_comment(self, message: discord.Message, message_content: str) -> str:
        """
        Generate a caption for an image in a message.

        Args:
            message (discord.Message): The message containing the image.
            message_content (str): The content of the message.

        Returns:
            str: The updated message content with the generated caption.
        """
        # Check if the message content is a URL
        url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        if "https://tenor.com/view/" in message_content:
            # Extract the Tenor GIF URL from the message content
            start_index = message_content.index("https://tenor.com/view/")
            end_index = message_content.find(" ", start_index)
            if end_index == -1:
                tenor_url = message_content[start_index:]
            else:
                tenor_url = message_content[start_index:end_index]
            # Split the URL on forward slashes
            parts = tenor_url.split("/")
            # Extract the relevant words from the URL
            words = parts[-1].split("-")[:-1]
            # Join the words into a sentence
            sentence = " ".join(words)
            message_content = f"{message_content} [{message.author.display_name} posts an animated {sentence} ]"
            return message_content.replace(tenor_url, "")
        elif url_pattern.match(message_content):
            # Download the image from the URL and convert it to a PIL image
            response = requests.get(message_content)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Download the image from the message and convert it to a PIL image
            image_url = message.attachments[0].url  # pyright: ignore[reportAttributeAccessIssue]
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")

        # Generate the image caption
        caption = self.caption_image(image)
        message_content = f"{message_content} [{message.author.display_name} posts a picture of {caption}]"
        return message_content

    def caption_image(self, raw_image: Image.Image) -> str:
        """
        Generate a caption for an image using the BLIP model.

        Args:
            raw_image (Image.Image): The input image.

        Returns:
            str: The generated caption.
        """
        inputs = self.processor(raw_image.convert("RGB"), return_tensors="pt").to("cpu", torch.float32)
        out = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(out[0], skip_special_tokens=True)


async def setup(bot: SandboxAgent) -> None:
    """
    Set up the ImageCaptionCog.

    Args:
        bot (SandboxAgent): The bot instance.
    """
    await bot.add_cog(ImageCaptionCog(bot))
