# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Memento Pattern

Overview of the Memento Pattern

The Memento pattern is a behavioral design pattern that lets you save and restore the previous state of an object
without revealing the details of its implementation. It provides the ability to restore an object to its previous state (undo via rollback).
"""

from __future__ import annotations

from typing import Any


class Memento:
    """
    The Memento class represents a snapshot of the state of the Editor.
    It stores the state and provides a method to retrieve the saved state.
    """

    def __init__(self, state: str) -> None:
        """
        Initialize the Memento with the given state.

        Args:
            state (str): The state to be saved.
        """
        self._state = state

    def get_saved_state(self) -> str:
        """
        Get the saved state.

        Returns:
            str: The saved state.
        """
        return self._state


class Editor:
    """
    The Editor class represents the originator that creates and restores mementos.
    It has methods to manipulate the text, save the current state, and restore to a previous state.
    """

    def __init__(self) -> None:
        """
        Initialize the Editor with an empty text.
        """
        self._text = ""

    def type(self, words: str) -> None:
        """
        Append the given words to the text.

        Args:
            words (str): The words to be appended.
        """
        self._text += " " + words

    def content(self) -> str:
        """
        Get the current content of the text.

        Returns:
            str: The current content of the text.
        """
        return self._text

    def save(self) -> Memento:
        """
        Save the current state of the Editor.

        Returns:
            Memento: The memento object representing the current state.
        """
        return Memento(self._text)

    def restore(self, memento: Memento) -> None:
        """
        Restore the Editor to a previous state using the given memento.

        Args:
            memento (Memento): The memento object representing the state to restore.
        """
        self._text = memento.get_saved_state()


# Client code
editor = Editor()

# Type some stuff
editor.type("This is the first sentence.")
editor.type("This is second.")

# Save the state to restore to: This is the first sentence. This is second.
saved = editor.save()

# Type some more
editor.type("And this is third.")

# Content before restoring
print(editor.content())  # Outputs: This is the first sentence. This is second. And this is third.

# Restoring to last saved state
editor.restore(saved)

print(editor.content())  # Outputs: This is the first sentence. This is second.


"""
In this example, Editor is our Originator, Memento is our Memento, and our caretaker is the client that executes the save and restore operations.

The Memento pattern can provide the ability to restore an object to its previous state, which is useful in case of error or other events.
It can provide a sophisticated undo capability while keeping the details of the originator's implementation hidden from the client.
However, saving and restoring a large memento might consume lots of resources.
"""
