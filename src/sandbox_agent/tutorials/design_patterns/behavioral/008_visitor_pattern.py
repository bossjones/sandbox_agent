# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Visitor Pattern

Overview of the Visitor Pattern

The Visitor pattern is a behavioral design pattern that allows you to add new behaviors to existing class hierarchies
without altering any existing code. You can introduce a new operation to a class without changing the classes on which it operates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class CarPart(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass


class Wheel(CarPart):
    def accept(self, visitor):
        visitor.visit_wheel(self)


class Engine(CarPart):
    def accept(self, visitor):
        visitor.visit_engine(self)


class Car:
    def __init__(self):
        self._parts = [Wheel(), Engine()]

    def accept(self, visitor):
        for part in self._parts:
            part.accept(visitor)


class Visitor(ABC):
    @abstractmethod
    def visit_wheel(self, wheel):
        pass

    @abstractmethod
    def visit_engine(self, engine):
        pass


class CarDisplayVisitor(Visitor):
    def visit_wheel(self, wheel):
        print("Displaying Wheel.")

    def visit_engine(self, engine):
        print("Displaying Engine.")


class CarRepairVisitor(Visitor):
    def visit_wheel(self, wheel):
        print("Repairing Wheel.")

    def visit_engine(self, engine):
        print("Repairing Engine.")


# Client code
car = Car()

display_visitor = CarDisplayVisitor()
car.accept(display_visitor)  # Outputs: Displaying Wheel. Displaying Engine.

repair_visitor = CarRepairVisitor()
car.accept(repair_visitor)  # Outputs: Repairing Wheel. Repairing Engine.


"""
In this example, Car, Wheel, and Engine are our Elements, CarPart is our Element interface,
Visitor is our Visitor interface, and CarDisplayVisitor and CarRepairVisitor are our Concrete Visitors.

The Visitor pattern can allow you to add operations to objects without changing their classes.
It can gather related operations into a single class, rather than having to change or extend many classes.
However, a downside of the Visitor pattern is that it can be hard to add new types of elements, as this would require changing all visitor classes.
"""
