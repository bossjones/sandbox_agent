# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
# pyright: reportImportCycles=false

"""sandbox_agent: A Python package for gooby things."""

from __future__ import annotations

import logging

from sandbox_agent.__version__ import __version__


logging.getLogger("asyncio").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("faker").setLevel(logging.DEBUG)
logging.getLogger("sentry_sdk").setLevel(logging.WARNING)

# # -------------------------------------------------------------------------------
# # pyright: reportMissingTypeStubs=false
# # pylint: disable=no-member
# # pylint: disable=no-value-for-parameter
# # pyright: reportAttributeAccessIssue=false
# # pyright: reportImportCycles=false

# """sandbox_agent: A Python package for gooby things."""

# from __future__ import annotations

# from loguru import logger

# import sandbox_agent

# from sandbox_agent.__version__ import __version__


# logger.disable("sandbox_agent")

# # import logging
# # logging.getLogger("asyncio").setLevel(logging.DEBUG)
# # logging.getLogger("httpx").setLevel(logging.DEBUG)
# # logging.getLogger("faker").setLevel(logging.DEBUG)
# # logging.getLogger("sentry_sdk").setLevel(logging.WARNING)
