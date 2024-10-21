# pragma: exclude file
# creational design patterns

"""
Builder pattern

Overview of the Builder Pattern The Builder pattern is a design pattern designed to provide a flexible solution to various object creation problems in object-oriented programming. The intent of the Builder design pattern is to separate the construction of a complex object from its representation. By doing so, the same construction process can create different representations. This pattern is particularly useful when the object creation involves multiple steps, and these steps could change in different representations. Implementing the Builder Pattern in Python Consider a simplified example where we are creating a complex Pizza object:

Envall, Henrik. Python Design Patterns: A Very Practical Guide (pp. 15-16). Kindle Edition.
"""

from __future__ import annotations


class Pizza:
    """Represents a pizza with various toppings."""

    def __init__(self, builder: PizzaBuilder) -> None:
        """
        Initialize a Pizza instance with a PizzaBuilder.

        Args:
            builder (PizzaBuilder): The builder used to construct the pizza.
        """
        self.garlic = builder.garlic
        self.extra_cheese = builder.extra_cheese

    def __str__(self) -> str:
        """
        Return a string representation of the pizza.

        Returns:
            str: A string describing the pizza's toppings.
        """
        garlic = "yes" if self.garlic else "no"
        cheese = "yes" if self.extra_cheese else "no"
        return f"Garlic: {garlic}, Extra cheese: {cheese}"


class PizzaBuilder:
    """Builds a pizza with various toppings."""

    def __init__(self) -> None:
        """Initialize a PizzaBuilder instance with default topping values."""
        self.extra_cheese = False
        self.garlic = False

    def add_garlic(self) -> PizzaBuilder:
        """
        Add garlic to the pizza.

        Returns:
            PizzaBuilder: The updated PizzaBuilder instance.
        """
        self.garlic = True
        return self

    def add_extra_cheese(self) -> PizzaBuilder:
        """
        Add extra cheese to the pizza.

        Returns:
            PizzaBuilder: The updated PizzaBuilder instance.
        """
        self.extra_cheese = True
        return self

    def build(self) -> Pizza:
        """
        Build and return the constructed Pizza instance.

        Returns:
            Pizza: The constructed Pizza instance.
        """
        return Pizza(self)


pizza_builder = PizzaBuilder()
pizza = pizza_builder.add_garlic().add_extra_cheese().build()
print(pizza)
