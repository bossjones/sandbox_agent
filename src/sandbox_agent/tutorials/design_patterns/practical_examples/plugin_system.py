"""
Creating a Plugin System using the Strategy Pattern

The Strategy Pattern is a behavioral design pattern that allows you to define a family of algorithms, put each of them into a separate class, and make their objects interchangeable. This pattern is a good choice when you need to provide multiple ways to process the same type of data, which makes it suitable for designing a plugin system.

In this section, we will use the Strategy Pattern to create a plugin system that can dynamically add new functionality to a core application.
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
    @abstractmethod
    def process(self, text):
        pass


# Implement specific strategies
class LowerCasePlugin(Plugin):
    def process(self, text):
        return text.lower()


class UpperCasePlugin(Plugin):
    def process(self, text):
        return text.upper()


# Context
class TextProcessor:
    def __init__(self, plugin):
        self.plugin = plugin

    def set_plugin(self, plugin):
        self.plugin = plugin

    def process(self, text):
        return self.plugin.process(text)


if __name__ == "__main__":
    # Create strategies
    lower_case_plugin = LowerCasePlugin()
    upper_case_plugin = UpperCasePlugin()

    # Create context and use a strategy
    text_processor = TextProcessor(lower_case_plugin)
    print(text_processor.process("Hello, World!"))  # Output: 'hello, world!'

    # Change strategy at runtime
    text_processor.set_plugin(upper_case_plugin)
    print(text_processor.process("Hello, World!"))  # Output: 'HELLO, WORLD!'

"""
In this code:
- Plugin is an abstract base class that defines a common interface for all plugins.
- LowerCasePlugin and UpperCasePlugin are concrete strategies that implement the Plugin interface.
- TextProcessor is the context that maintains a reference to a Plugin object and delegates the text processing to it.

With this setup, we can easily add new plugins by creating new classes that implement the Plugin interface and integrate them into our application by setting them as the current plugin in TextProcessor. The Strategy Pattern enables us to add new behaviors to our application without changing its existing code.
"""
