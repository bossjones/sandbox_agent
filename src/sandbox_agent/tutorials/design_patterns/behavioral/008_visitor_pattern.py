# pragma: exclude file
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
from typing import List


class CarPart(ABC):
    """
    Abstract base class representing a car part.
    """

    @abstractmethod
    def accept(self, visitor: Visitor) -> None:
        """
        Accept a visitor and allow it to visit the car part.

        Args:
            visitor (Visitor): The visitor to accept.
        """
        pass


class Wheel(CarPart):
    """
    Concrete class representing a wheel.
    """

    def accept(self, visitor: Visitor) -> None:
        """
        Accept a visitor and allow it to visit the wheel.

        Args:
            visitor (Visitor): The visitor to accept.
        """
        visitor.visit_wheel(self)


class Engine(CarPart):
    """
    Concrete class representing an engine.
    """

    def accept(self, visitor: Visitor) -> None:
        """
        Accept a visitor and allow it to visit the engine.

        Args:
            visitor (Visitor): The visitor to accept.
        """
        visitor.visit_engine(self)


class Car:
    """
    Class representing a car composed of car parts.
    """

    def __init__(self) -> None:
        """
        Initialize the car with a list of car parts.
        """
        self._parts: list[CarPart] = [Wheel(), Engine()]

    def accept(self, visitor: Visitor) -> None:
        """
        Accept a visitor and allow it to visit all car parts.

        Args:
            visitor (Visitor): The visitor to accept.
        """
        for part in self._parts:
            part.accept(visitor)


class Visitor(ABC):
    """
    Abstract base class representing a visitor.
    """

    @abstractmethod
    def visit_wheel(self, wheel: Wheel) -> None:
        """
        Visit a wheel.

        Args:
            wheel (Wheel): The wheel to visit.
        """
        pass

    @abstractmethod
    def visit_engine(self, engine: Engine) -> None:
        """
        Visit an engine.

        Args:
            engine (Engine): The engine to visit.
        """
        pass


class CarDisplayVisitor(Visitor):
    """
    Concrete visitor class for displaying car parts.
    """

    def visit_wheel(self, wheel: Wheel) -> None:
        """
        Visit a wheel and display it.

        Args:
            wheel (Wheel): The wheel to visit.
        """
        print("Displaying Wheel.")

    def visit_engine(self, engine: Engine) -> None:
        """
        Visit an engine and display it.

        Args:
            engine (Engine): The engine to visit.
        """
        print("Displaying Engine.")


class CarRepairVisitor(Visitor):
    """
    Concrete visitor class for repairing car parts.
    """

    def visit_wheel(self, wheel: Wheel) -> None:
        """
        Visit a wheel and repair it.

        Args:
            wheel (Wheel): The wheel to visit.
        """
        print("Repairing Wheel.")

    def visit_engine(self, engine: Engine) -> None:
        """
        Visit an engine and repair it.

        Args:
            engine (Engine): The engine to visit.
        """
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
