# tests/test_logger.py
from __future__ import annotations

import logging

from collections.abc import Generator

from _pytest.logging import LogCaptureFixture
from loguru import logger

import pytest

from src.sandbox_agent.bot_logger import (
    InterceptHandler,
    InterceptHandlerImproved,
    filter_out_serialization_errors,
    global_log_config,
    reset_logging,
)


# # SOURCE: https://loguru.readthedocs.io/en/stable/resources/migration.html#replacing-caplog-fixture-from-pytest-library
# # @pytest.fixture(autouse=True)
# def caplog(caplog: LogCaptureFixture):
#     handler_id = logger.add(
#         caplog.handler,
#         format="{message}",
#         level="DEBUG",
#         filter=lambda record: record["level"].no >= caplog.handler.level,
#         enqueue=True,  # Set to 'True' if your test is spawning child processes.
#     )
#     yield caplog
#     logger.remove(handler_id)


# # SOURCE: https://loguru.readthedocs.io/en/stable/resources/migration.html#replacing-caplog-fixture-from-pytest-library
# @pytest.fixture
# def reportlog(pytestconfig):
#     logging_plugin = pytestconfig.pluginmanager.getplugin("logging-plugin")
#     handler_id = logger.add(logging_plugin.report_handler, format="{message}")
#     yield
#     logger.remove(handler_id)


# @pytest.fixture
# def setup_logging() -> Generator[None, None, None]:
#     """
#     Set up the logging environment for tests.

#     This fixture configures the standard logging module to use the InterceptHandlerImproved,
#     ensuring that log messages are redirected to loguru.

#     Yields:
#         None
#     """
#     # Remove all existing handlers
#     logger.remove()
#     logging.basicConfig(handlers=[InterceptHandlerImproved()], level=logging.DEBUG, force=True)
#     yield
#     # Clean up after test
#     logging.getLogger().handlers = []


def test_global_log_config(capsys, caplog):
    """
    Test the global_log_config function to ensure it sets up the logger correctly.

    Args:
        capsys: Pytest fixture to capture stdout and stderr.

    Asserts:
        The logger is set up with the correct log level.
    """
    global_log_config(log_level=logging.INFO, json=False)
    logger.info("Testing global log config.")
    captured = capsys.readouterr()
    assert "Logger set up with log level: 20" in captured.out


# def test_reset_logging(setup_logging, tmp_path):
#     """
#     Test the reset_logging function to ensure it configures logging correctly.

#     Args:
#         setup_logging: The fixture to set up the logging environment.
#         tmp_path: Pytest fixture to provide a temporary directory.

#     Asserts:
#         Log files are created in the specified directory.
#     """
#     log_dir = tmp_path / "logs"
#     log_dir.mkdir()
#     reset_logging(str(log_dir), console_log_level="DEBUG", backup_count=1)
#     logger.debug("Testing reset logging.")
#     log_files = list(log_dir.glob("*.log"))
#     assert len(log_files) > 0


def test_filter_out_serialization_errors():
    """
    Test the filter_out_serialization_errors function to ensure it filters messages correctly.

    Asserts:
        Messages matching the filter patterns are excluded.
    """
    record = {"message": "Orjson serialization failed: some error"}
    assert not filter_out_serialization_errors(record)

    record = {"message": "This is a normal log message"}
    assert filter_out_serialization_errors(record)
