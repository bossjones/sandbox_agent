"""
Memento Pattern

Overview of the Memento Pattern

The Memento pattern is a behavioral design pattern that lets you save and restore the previous state of an object
without revealing the details of its implementation. It provides the ability to restore an object to its previous state (undo via rollback).
"""

from __future__ import annotations


class Memento:
    def __init__(self, state):
        self._state = state

    def get_saved_state(self):
        return self._state


class Editor:
    def __init__(self):
        self._text = ""

    def type(self, words):
        self._text += " " + words

    def content(self):
        return self._text

    def save(self):
        return Memento(self._text)

    def restore(self, memento):
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
