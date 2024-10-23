# pragma: exclude file
# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Bridge Pattern

Overview of the Bridge Pattern

The Bridge pattern is a design pattern that separates an object's interface from its implementation. This means you
can change the implementation of an abstraction without changing the abstraction's interface. This pattern is most
useful when both the class and what it does vary often.

Implementing the Bridge Pattern in Python

Let's consider an example where we have a Shape class and multiple different Color implementations. The Shape needs
to use Color, but we don't want to hard code which Color it uses. So, we use the Bridge pattern to decouple the
Shape from the Color.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Color(ABC):
    @abstractmethod
    def fill_color(self) -> str:
        """
        Fill the color.

        Returns:
            str: The color filled.
        """
        pass


class RedColor(Color):
    def fill_color(self) -> str:
        """
        Fill the color red.

        Returns:
            str: The color red.
        """
        return "Red"


class BlueColor(Color):
    def fill_color(self) -> str:
        """
        Fill the color blue.

        Returns:
            str: The color blue.
        """
        return "Blue"


class Shape(ABC):
    def __init__(self, color: Color):
        """
        Initialize the Shape with the given Color.

        Args:
            color (Color): The color to fill the shape.
        """
        self.color = color

    @abstractmethod
    def color_it(self) -> str:
        """
        Color the shape.

        Returns:
            str: The colored shape.
        """
        pass


class Square(Shape):
    def __init__(self, color: Color):
        """
        Initialize the Square with the given Color.

        Args:
            color (Color): The color to fill the square.
        """
        super().__init__(color)

    def color_it(self) -> str:
        """
        Color the square.

        Returns:
            str: The colored square.
        """
        return f"Square filled with {super().color.fill_color()} color."


class Circle(Shape):
    def __init__(self, color: Color):
        """
        Initialize the Circle with the given Color.

        Args:
            color (Color): The color to fill the circle.
        """
        super().__init__(color)

    def color_it(self) -> str:
        """
        Color the circle.

        Returns:
            str: The colored circle.
        """
        return f"Circle filled with {super().color.fill_color()} color."


red = RedColor()
square = Square(red)
print(square.color_it())  # Outputs: Square filled with Red color

blue = BlueColor()
circle = Circle(blue)
print(circle.color_it())  # Outputs: Circle filled with Blue color.


# In this example, Shape (the "abstraction" role) uses Color (the "implementor" role), but Shape doesn't depend on
# specific subclasses of Color. This decouples the Shape from the Color, so you can add new Shape or Color subclasses
# without changing the existing classes.

# The Bridge pattern can help to reduce the proliferation of classes when the class and its functionality both can
# have different combinations. It improves decoupling between an abstraction and its implementation, reducing
# compile-time dependencies. The downside is that it can increase the complexity of the code by introducing more
# classes.

# ---

# Let me know if you need further clarification!
