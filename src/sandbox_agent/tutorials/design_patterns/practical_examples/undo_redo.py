"""
Implementing Undo/Redo Functionality using the Memento Pattern

The Memento Pattern is a behavioral design pattern that allows you to save and restore the previous state of an object without revealing the details of its implementation. This pattern is particularly useful for implementing undo/redo functionality in applications.

In this module, we use the Memento Pattern to create a simple undo/redo system for a text editor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Memento(ABC):
    """
    Abstract base class for memento implementations.
    """

    @abstractmethod
    def get_state(self) -> str:
        """
        Abstract method to get the state stored in the memento.

        Returns:
            str: The state stored in the memento.
        """
        pass


class TextMemento(Memento):
    """
    Concrete implementation of a memento for storing text state.
    """

    def __init__(self, text: str) -> None:
        """
        Initialize the text memento with the given text.

        Args:
            text (str): The text to store in the memento.
        """
        self._text = text

    def get_state(self) -> str:
        """
        Get the text state stored in the memento.

        Returns:
            str: The text state stored in the memento.
        """
        return self._text


class TextEditor:
    """
    Originator class representing a text editor that creates and restores mementos.
    """

    def __init__(self) -> None:
        """
        Initialize the text editor with empty text.
        """
        self._text = ""

    def set_text(self, text: str) -> None:
        """
        Set the text in the editor.

        Args:
            text (str): The text to set in the editor.
        """
        self._text = text

    def get_text(self) -> str:
        """
        Get the current text in the editor.

        Returns:
            str: The current text in the editor.
        """
        return self._text

    def create_memento(self) -> TextMemento:
        """
        Create a memento with the current text state.

        Returns:
            TextMemento: The created memento with the current text state.
        """
        return TextMemento(self._text)

    def restore_memento(self, memento: TextMemento) -> None:
        """
        Restore the text state from the given memento.

        Args:
            memento (TextMemento): The memento to restore the text state from.
        """
        self._text = memento.get_state()


class History:
    """
    Caretaker class responsible for managing the history of mementos.
    """

    def __init__(self) -> None:
        """
        Initialize the history with empty undo and redo stacks.
        """
        self._undo_stack: list[TextMemento] = []
        self._redo_stack: list[TextMemento] = []

    def push(self, memento: TextMemento) -> None:
        """
        Push a memento onto the undo stack.

        Args:
            memento (TextMemento): The memento to push onto the undo stack.
        """
        self._undo_stack.append(memento)
        self._redo_stack.clear()

    def undo(self) -> TextMemento | None:
        """
        Pop a memento from the undo stack and push it onto the redo stack.

        Returns:
            TextMemento | None: The popped memento from the undo stack, or None if the stack is empty.
        """
        if self._undo_stack:
            memento = self._undo_stack.pop()
            self._redo_stack.append(memento)
            return memento
        return None

    def redo(self) -> TextMemento | None:
        """
        Pop a memento from the redo stack and push it onto the undo stack.

        Returns:
            TextMemento | None: The popped memento from the redo stack, or None if the stack is empty.
        """
        if self._redo_stack:
            memento = self._redo_stack.pop()
            self._undo_stack.append(memento)
            return memento
        return None


if __name__ == "__main__":
    editor = TextEditor()
    history = History()

    editor.set_text("Hello")
    history.push(editor.create_memento())

    editor.set_text("Hello, world!")
    history.push(editor.create_memento())

    editor.set_text("Hello, world! How are you?")
    print(editor.get_text())  # Output: Hello, world! How are you?

    editor.restore_memento(history.undo())
    print(editor.get_text())  # Output: Hello, world!

    editor.restore_memento(history.undo())
    print(editor.get_text())  # Output: Hello

    editor.restore_memento(history.redo())
    print(editor.get_text())  # Output: Hello, world!
