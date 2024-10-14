"""
State Pattern

Overview of the State Pattern

The State pattern is a behavioral design pattern that lets an object alter its behavior when its internal state
changes. The object will appear to change its class. It is a clean way for an object to partially change its behavior
at runtime without resorting to large monolithic conditional statements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DocumentState(ABC):
    @abstractmethod
    def handle_publish(self) -> DocumentState:
        """
        Handle the publish action.

        Returns:
            DocumentState: The new state after handling the publish action.
        """
        pass

    @abstractmethod
    def handle_reject(self) -> DocumentState:
        """
        Handle the reject action.

        Returns:
            DocumentState: The new state after handling the reject action.
        """
        pass


class Draft(DocumentState):
    def handle_publish(self) -> DocumentState:
        """
        Handle the publish action in the Draft state.

        Returns:
            DocumentState: The Moderation state.
        """
        return Moderation()

    def handle_reject(self) -> DocumentState:
        """
        Handle the reject action in the Draft state.

        Returns:
            DocumentState: The Draft state (no change).
        """
        return self


class Moderation(DocumentState):
    def handle_publish(self) -> DocumentState:
        """
        Handle the publish action in the Moderation state.

        Returns:
            DocumentState: The Published state.
        """
        return Published()

    def handle_reject(self) -> DocumentState:
        """
        Handle the reject action in the Moderation state.

        Returns:
            DocumentState: The Draft state.
        """
        return Draft()


class Published(DocumentState):
    def handle_publish(self) -> DocumentState:
        """
        Handle the publish action in the Published state.

        Returns:
            DocumentState: The Published state (no change).
        """
        return self

    def handle_reject(self) -> DocumentState:
        """
        Handle the reject action in the Published state.

        Returns:
            DocumentState: The Draft state.
        """
        return Draft()


class Document:
    def __init__(self):
        """
        Initialize the Document with the Draft state.
        """
        self.state = Draft()

    def publish(self) -> None:
        """
        Publish the document by transitioning to the next state based on the current state.
        """
        self.state = self.state.handle_publish()

    def reject(self) -> None:
        """
        Reject the document by transitioning to the next state based on the current state.
        """
        self.state = self.state.handle_reject()


# Client code
document = Document()

print(type(document.state).__name__)  # Outputs: Draft
document.publish()
print(type(document.state).__name__)  # Outputs: Moderation
document.reject()
print(type(document.state).__name__)  # Outputs: Draft
document.publish()
print(type(document.state).__name__)  # Outputs: Moderation
document.publish()
print(type(document.state).__name__)  # Outputs: Published


"""
In this example, DocumentState is an abstract base class that defines an interface for all states.
Draft, Moderation, and Published are concrete states that implement the DocumentState interface.
The Document class uses a DocumentState to define its current state.

The State pattern can be beneficial when a class changes behavior based on its state. It makes an object's
behavior more versatile and easier to understand. However, it can lead to a large number of classes as states
are encapsulated in separate classes.
"""
