"""
Building a Text Editor using the Memento and Command Patterns

The Memento Pattern is a design pattern that provides the ability to restore an object to its previous state (undo via rollback). This is especially useful in cases of changes that are supposed to be temporary and are only needed for a short period.

The Command Pattern is a design pattern that encapsulates a request as an object, thereby allowing users to parameterize clients with queues, requests, and operations. The pattern promotes the idea of encapsulation and decoupling and provides a means to separate the responsibility of issuing a command from knowing how to perform it.

In this section, we will combine these two patterns to create a simple text editor with undo functionality.
"""

# Concept of the Text Editor with Memento and Command Patterns
"""
The main idea is to have a Command abstract base class, with each concrete command encapsulating an action performed on the text, such as WriteCommand or DeleteCommand. Each command holds a memento—a snapshot of the editor's content before the operation is executed—allowing us to undo the operation if needed.
"""

# Code Example in Python

# Here is a simplified text editor example that supports writing and undoing:
from __future__ import annotations


class Memento:
    def __init__(self, text):
        self.text = text


class TextEditor:
    def __init__(self):
        self.commands = []
        self.undo_stack = []
        self.text = ""

    def write(self, text):
        memento = Memento(self.text)
        cmd = WriteCommand(self, text, memento)
        cmd.execute()
        self.commands.append(cmd)

    def undo(self):
        if self.commands:
            cmd = self.commands.pop()
            cmd.undo()


class Command:
    def __init__(self, editor, memento):
        self.editor = editor
        self.memento = memento

    def undo(self):
        self.editor.text = self.memento.text


class WriteCommand(Command):
    def __init__(self, editor, text, memento):
        super().__init__(editor, memento)
        self.text = text

    def execute(self):
        self.editor.text += self.text


if __name__ == "__main__":
    editor = TextEditor()
    editor.write("Hello, ")
    editor.write("world!")
    print(editor.text)  # "Hello, world!"
    editor.undo()
    print(editor.text)  # "Hello, "
    editor.undo()
    print(editor.text)  # ""

"""
In this code:
- Memento holds the state of the TextEditor.
- TextEditor is the "originator" that uses mementos to save and restore its state.
- Command is an abstract base class that represents a command.
- WriteCommand is a concrete command that appends text to the editor and can undo this operation.

With this setup, the text editor supports undoable writing. The Command pattern allows us to encapsulate each operation on the text, and the Memento pattern allows us to save and restore the text's state, providing the undo functionality.

This is a simple example, but it demonstrates the essence of combining the Command and Memento patterns. In a real-world text editor, we could support more operations (like deletion or formatting), each encapsulated in its own command class.
"""
