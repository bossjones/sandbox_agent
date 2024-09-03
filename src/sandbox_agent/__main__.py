#!/usr/bin/env python
"""sandbox_agent.main."""

from __future__ import annotations

import logging


rootlogger = logging.getLogger()
handler_logger = logging.getLogger("handler")

name_logger = logging.getLogger(__name__)
logging.getLogger("asyncio").setLevel(logging.DEBUG)  # noqa # type: ignore

from sandbox_agent.cli import main


main()

# TEMPCHANGE: DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
# TEMPCHANGE: DISCORD_ADMIN = os.environ.get("DISCORD_ADMIN_USER_ID")
# TEMPCHANGE: DISCORD_GUILD = os.environ.get("DISCORD_SERVER_ID")
# TEMPCHANGE: DISCORD_GENERAL_CHANNEL = 908894727779258390
