from __future__ import annotations

from discord.ext import commands


class UserBlacklisted(commands.CheckFailure):
    """Thrown when a user is attempting something, but is blacklisted."""

    def __init__(self, message="User is blacklisted!"):
        self.message = message
        super().__init__(self.message)


class UserNotOwner(commands.CheckFailure):
    """Thrown when a user is attempting something, but is not an owner of the bot."""

    def __init__(self, message="User is not an owner of the bot!"):
        self.message = message
        super().__init__(self.message)


"""Custom exceptions for the sandbox agent."""


class SandboxAgentException(Exception):
    """Base exception for sandbox agent."""

    pass


class ModelNotFoundError(SandboxAgentException):
    """Raised when a requested model is not found."""

    pass


class ConfigurationError(SandboxAgentException):
    """Raised when there's an error in the configuration."""

    pass
