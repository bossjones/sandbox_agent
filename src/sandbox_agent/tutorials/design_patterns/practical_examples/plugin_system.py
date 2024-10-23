# pragma: exclude file
"""
Creating a Plugin System using the Strategy Pattern

The Strategy Pattern is a behavioral design pattern that allows you to define a family of algorithms,
put each of them into a separate class, and make their objects interchangeable. This pattern is a good
choice when you need to provide multiple ways to process the same type of data, which makes it suitable
for designing a plugin system.

In this module, we use the Strategy Pattern to create a plugin system that can dynamically add new
functionality to a core application.
"""

# Concept of Plugin System with Strategy Pattern
"""
The main idea is to encapsulate each plugin (or strategy) into its own class with a common interface. Then, we use a context object to switch between strategies. This allows for changing the behavior of the program at runtime without modifying the main codebase.
"""

# Code Example in Python

# Here is a simplified example of a text processing application with a plugin system:
from __future__ import annotations

from abc import ABC, abstractmethod


# Define the common interface for all strategies
class Plugin(ABC):
    """
    Abstract base class for plugin implementations.
    """

    @abstractmethod
    def process(self, text: str) -> str:
        """
        Abstract method to process the given text.

        Args:
            text (str): The text to process.

        Returns:
            str: The processed text.
        """
        pass


# Implement specific strategies
class LowerCasePlugin(Plugin):
    """
    Concrete implementation of a plugin that converts text to lowercase.
    """

    def process(self, text: str) -> str:
        """
        Process the given text by converting it to lowercase.

        Args:
            text (str): The text to process.

        Returns:
            str: The lowercase version of the text.
        """
        return text.lower()


class UpperCasePlugin(Plugin):
    """
    Concrete implementation of a plugin that converts text to uppercase.
    """

    def process(self, text: str) -> str:
        """
        Process the given text by converting it to uppercase.

        Args:
            text (str): The text to process.

        Returns:
            str: The uppercase version of the text.
        """
        return text.upper()


# Context
class TextProcessor:
    """
    The context class that maintains a reference to a plugin and delegates text processing to it.
    """

    def __init__(self, plugin: Plugin) -> None:
        """
        Initialize the text processor with a plugin.

        Args:
            plugin (Plugin): The plugin to use for text processing.
        """
        self.plugin = plugin

    def set_plugin(self, plugin: Plugin) -> None:
        """
        Set the plugin to use for text processing.

        Args:
            plugin (Plugin): The plugin to set.
        """
        self.plugin = plugin

    def process(self, text: str) -> str:
        """
        Process the given text using the current plugin.

        Args:
            text (str): The text to process.

        Returns:
            str: The processed text.
        """
        return self.plugin.process(text)


if __name__ == "__main__":
    # Create plugins
    lower_case_plugin = LowerCasePlugin()
    upper_case_plugin = UpperCasePlugin()

    # Create text processor and use a plugin
    text_processor = TextProcessor(lower_case_plugin)
    print(text_processor.process("Hello, World!"))  # Output: 'hello, world!'

    # Change plugin at runtime
    text_processor.set_plugin(upper_case_plugin)
    print(text_processor.process("Hello, World!"))  # Output: 'HELLO, WORLD!'
"""
In this code:
- Plugin is an abstract base class that defines a common interface for all plugins.
- LowerCasePlugin and UpperCasePlugin are concrete strategies that implement the Plugin interface.
- TextProcessor is the context that maintains a reference to a Plugin object and delegates the text processing to it.

With this setup, we can easily add new plugins by creating new classes that implement the Plugin interface and integrate them into our application by setting them as the current plugin in TextProcessor. The Strategy Pattern enables us to add new behaviors to our application without changing its existing code.
"""
