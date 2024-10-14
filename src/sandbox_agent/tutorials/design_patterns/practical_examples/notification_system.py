"""
Implementing a Notification System using the Observer Pattern

The Observer Pattern is a design pattern that establishes a one-to-many relationship between objects so that when one object changes its state, all its dependents are notified and updated automatically. This pattern is often used to implement event handling systems.

In this section, we will use the Observer Pattern to create a notification system where multiple observer objects are notified whenever a subject object changes its state.
"""

# Concept of Notification System with Observer Pattern
"""
The main idea is to have a Subject class and an Observer interface. Observers register themselves with the subject to receive notifications. When the state of the subject changes, it notifies all registered observers.
"""

# Code Example in Python

# Here is a simplified notification system example:
from __future__ import annotations


class Observer:
    def update(self, message: str) -> None:
        pass


class User(Observer):
    def __init__(self, name):
        self.name = name

    def update(self, message: str) -> None:
        print(f"{self.name} received message: {message}")


class NotificationSystem:
    def __init__(self):
        self.observers = []

    def register(self, observer: Observer):
        self.observers.append(observer)

    def unregister(self, observer: Observer):
        self.observers.remove(observer)

    def notify(self, message: str):
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
