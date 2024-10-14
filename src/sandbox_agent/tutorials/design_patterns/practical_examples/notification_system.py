"""
Implementing a Notification System using the Observer Pattern

The Observer Pattern is a design pattern that establishes a one-to-many relationship between objects so that when one object changes its state, all its dependents are notified and updated automatically. This pattern is often used to implement event handling systems.

In this module, we use the Observer Pattern to create a notification system where multiple observer objects are notified whenever a subject object changes its state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Observer(ABC):
    """
    Abstract base class for observer implementations.
    """

    @abstractmethod
    def update(self, message: str) -> None:
        """
        Abstract method to update the observer with a message.

        Args:
            message (str): The message to update the observer with.
        """
        pass


class User(Observer):
    """
    Concrete implementation of an observer representing a user.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a user with a name.

        Args:
            name (str): The name of the user.
        """
        self.name = name

    def update(self, message: str) -> None:
        """
        Update the user with a message.

        Args:
            message (str): The message to update the user with.
        """
        print(f"{self.name} received message: {message}")


class NotificationSystem:
    """
    The subject that maintains a list of observers and notifies them when its state changes.
    """

    def __init__(self) -> None:
        """
        Initialize the notification system with an empty list of observers.
        """
        self.observers: list[Observer] = []

    def register(self, observer: Observer) -> None:
        """
        Register an observer with the notification system.

        Args:
            observer (Observer): The observer to register.
        """
        self.observers.append(observer)

    def unregister(self, observer: Observer) -> None:
        """
        Unregister an observer from the notification system.

        Args:
            observer (Observer): The observer to unregister.
        """
        self.observers.remove(observer)

    def notify(self, message: str) -> None:
        """
        Notify all registered observers with a message.

        Args:
            message (str): The message to notify the observers with.
        """
        for observer in self.observers:
            observer.update(message)


if __name__ == "__main__":
    # Create users
    alice = User("Alice")
    bob = User("Bob")

    # Create notification system and register users
    notification_system = NotificationSystem()
    notification_system.register(alice)
    notification_system.register(bob)

    # Send notification
    notification_system.notify("This is a test message.")

"""
In this code:
- Observer is an interface that defines the update method, which is called when the subject's state changes.
- User is a concrete observer that implements the Observer interface.
- NotificationSystem is the subject that maintains a list of observers and notifies them when its state changes.

With this setup, whenever a notification is sent through the notification system, all registered users (observers) will receive the notification message. The Observer pattern allows us to add, remove, and notify as many users as we like, independently of the users or the notification system.
"""
