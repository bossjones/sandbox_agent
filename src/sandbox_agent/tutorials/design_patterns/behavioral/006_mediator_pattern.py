# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Mediator Pattern
Overview of the Mediator Pattern

The Mediator pattern is a behavioral design pattern that reduces coupling between components of a program by making them communicate indirectly,
through a special mediator object. The Mediator pattern provides central authority over a group of objects by encapsulating how these objects interact.
This model is useful for scenarios where there is a need to manage complex conditions in which every object is aware of any state change in any other object in the group.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Mediator(ABC):
    """
    The Mediator interface declares a method used by components to notify the
    mediator about various events. The Mediator may react to these events and
    pass the execution to other components.
    """

    @abstractmethod
    def notify(self, sender: object, event: str) -> None:
        """
        Notify the mediator about an event.

        Args:
            sender (object): The sender component.
            event (str): The event that occurred.
        """
        pass


class ConcreteMediator(Mediator):
    """
    Concrete Mediators implement cooperative behavior by coordinating several
    components.
    """

    def notify(self, sender: object, event: str) -> None:
        """
        Handle notifications from components.

        Args:
            sender (object): The sender component.
            event (str): The event that occurred.
        """
        if event == "event_A":
            print("Mediator reacts on event_A and triggers following operations:")
            # logic related to event_A
        elif event == "event_B":
            print("Mediator reacts on event_B and triggers following operations:")
            # logic related to event_B


class Component1:
    """
    The Base Component provides the basic functionality of storing a mediator's
    instance inside component objects.
    """

    def __init__(self, mediator: Mediator) -> None:
        """
        Initialize the Component1 with a mediator.

        Args:
            mediator (Mediator): The mediator instance.
        """
        self._mediator = mediator

    def do_something(self) -> None:
        """
        Perform an action and notify the mediator.
        """
        print("Component1 does something.")
        self._mediator.notify(self, "event_A")


class Component2:
    """
    The Base Component provides the basic functionality of storing a mediator's
    instance inside component objects.
    """

    def __init__(self, mediator: Mediator) -> None:
        """
        Initialize the Component2 with a mediator.

        Args:
            mediator (Mediator): The mediator instance.
        """
        self._mediator = mediator

    def do_something(self) -> None:
        """
        Perform an action and notify the mediator.
        """
        print("Component2 does something.")
        self._mediator.notify(self, "event_B")


# Client code
mediator = ConcreteMediator()
component1 = Component1(mediator)
component2 = Component2(mediator)

component1.do_something()  # Component1 does something.
# Mediator reacts on event_A and triggers following operations:
component2.do_something()  # Component2 does something.
# Mediator reacts on event_B and triggers following operations:

"""
In this example, Mediator is an abstract base class with a notify method.
ConcreteMediator is a concrete mediator that coordinates communication between Component1 and Component2.

The Mediator pattern reduces the dependencies between communicating objects, thereby reducing coupling and simplifying the reuse of objects.
It simplifies maintenance of the code by centralizing control logic. However, a downside of the Mediator pattern is that it can evolve into a god object,
coupled to all other classes, making the system more complex.
"""
