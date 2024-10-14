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
    @abstractmethod
    def notify(self, sender: object, event: str) -> None:
        pass


class ConcreteMediator(Mediator):
    def notify(self, sender: object, event: str) -> None:
        if event == "event_A":
            print("Mediator reacts on event_A and triggers following operations:")
            # logic related to event_A
        elif event == "event_B":
            print("Mediator reacts on event_B and triggers following operations:")
            # logic related to event_B


class Component1:
    def __init__(self, mediator: Mediator):
        self._mediator = mediator

    def do_something(self):
        print("Component1 does something.")
        self._mediator.notify(self, "event_A")


class Component2:
    def __init__(self, mediator: Mediator):
        self._mediator = mediator

    def do_something(self):
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
