"""
Building a Text Editor using the Memento and Command Patterns

The Memento Pattern is a design pattern that provides the ability to restore an object to its previous state (undo via rollback). This is especially useful in cases of changes that are supposed to be temporary and are only needed for a short period.

The Command Pattern is a design pattern that encapsulates a request as an object, thereby allowing users to parameterize clients with queues, requests, and operations. The pattern promotes the idea of encapsulation and decoupling and provides a means to separate the responsibility of issuing a command from knowing how to perform it.

In this module, we combine these two patterns to create a simple text editor with undo functionality.
"""

from __future__ import annotations


class Memento:
    """
    Memento class to store the state of the TextEditor.
    """

    def __init__(self, text: str) -> None:
        """
        Initialize the memento with the given text.

        Args:
            text (str): The text to store in the memento.
        """
        self.text = text


class TextEditor:
    """
    TextEditor class representing the originator that uses mementos to save and restore its state.
    """

    def __init__(self) -> None:
        """
        Initialize the text editor with empty commands, undo stack, and text.
        """
        self.commands: list[Command] = []
        self.undo_stack: list[Memento] = []
        self.text = ""

    def write(self, text: str) -> None:
        """
        Write text to the editor and create a WriteCommand.

        Args:
            text (str): The text to write to the editor.
        """
        memento = Memento(self.text)
        cmd = WriteCommand(self, text, memento)
        cmd.execute()
        self.commands.append(cmd)

    def undo(self) -> None:
        """
        Undo the last command if there are any commands in the history.
        """
        if self.commands:
            cmd = self.commands.pop()
            cmd.undo()


class Command:
    """
    Abstract base class for command implementations.
    """

    def __init__(self, editor: TextEditor, memento: Memento) -> None:
        """
        Initialize the command with the text editor and a memento.

        Args:
            editor (TextEditor): The text editor to perform the command on.
            memento (Memento): The memento to store the editor's state.
        """
        self.editor = editor
        self.memento = memento

    def undo(self) -> None:
        """
        Undo the command by restoring the editor's state from the memento.
        """
        self.editor.text = self.memento.text


class WriteCommand(Command):
    """
    Concrete command class for writing text to the editor.
    """

    def __init__(self, editor: TextEditor, text: str, memento: Memento) -> None:
        """
        Initialize the write command with the text editor, text to write, and a memento.

        Args:
            editor (TextEditor): The text editor to write the text to.
            text (str): The text to write to the editor.
            memento (Memento): The memento to store the editor's state before writing.
        """
        super().__init__(editor, memento)
        self.text = text

    def execute(self) -> None:
        """
        Execute the write command by appending the text to the editor.
        """
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
