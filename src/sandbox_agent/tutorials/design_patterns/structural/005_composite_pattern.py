# pragma: exclude file
"""
Composite Pattern

Overview of the Composite Pattern

The Composite pattern is a design pattern that allows treating individual objects and compositions of objects
uniformly. It composes objects into tree structures to represent part-whole hierarchies. It can be used to create a
tree-like structure of simple and composite objects which makes the client interface simpler.

Implementing the Composite Pattern in Python

Let's take an example where we have a structure for graphic design. A graphic could be either an ellipse or a
composition of several graphics. Every graphic can be printed.
"""

from __future__ import annotations

from typing import List


class Graphic:
    def print(self) -> None:
        """
        Print the graphic.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError()


class Ellipse(Graphic):
    def print(self) -> None:
        """
        Print the ellipse.
        """
        print("Ellipse")


class CompositeGraphic(Graphic):
    def __init__(self):
        """
        Initialize the CompositeGraphic with an empty list of graphics.
        """
        self.graphics: list[Graphic] = []

    def print(self) -> None:
        """
        Print all the graphics in the composition.
        """
        for graphic in self.graphics:
            graphic.print()

    def add(self, graphic: Graphic) -> None:
        """
        Add a graphic to the composition.

        Args:
            graphic (Graphic): The graphic to be added.
        """
        self.graphics.append(graphic)

    def remove(self, graphic: Graphic) -> None:
        """
        Remove a graphic from the composition.

        Args:
            graphic (Graphic): The graphic to be removed.
        """
        self.graphics.remove(graphic)


# Client code
ellipse1 = Ellipse()
ellipse2 = Ellipse()
ellipse3 = Ellipse()

graphic = CompositeGraphic()
graphic1 = CompositeGraphic()

graphic1.add(ellipse1)
graphic1.add(ellipse2)
graphic.add(graphic1)
graphic.add(ellipse3)

graphic.print()  # Outputs: Ellipse, Ellipse, Ellipse


# In this example, **Ellipse** and **CompositeGraphic** share a common interface, **Graphic**. **CompositeGraphic**
# uses that interface to manage child objects (**graphics**), which could be either simple **Ellipse** instances or
# other **CompositeGraphic** instances.

# The Composite pattern allows clients to treat individual objects and compositions uniformly. It can simplify your
# code, especially when dealing with complex tree-like structures. However, it can make your design overly general.
# It's worth using this pattern only when your models form a hierarchy or when you have a good reason to treat
# composed and individual objects the same way.

# ---

# Let me know if you need further clarification!
