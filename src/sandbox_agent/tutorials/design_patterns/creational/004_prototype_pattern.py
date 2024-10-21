# pragma: exclude file
# creational design patterns

"""
Prototype Pattern

Overview of the Prototype Pattern The Prototype pattern involves creating an instance of a class and then copying that instance to make new instances, rather than creating a new instance from scratch. The Prototype pattern can be a more efficient way to create objects in some cases, especially when the object creation is a heavy operation. Implementing the Prototype Pattern in Python Python makes it easy to implement the Prototype pattern in a variety of ways. One of the simplest ways is to use the copy module's copy() or deepcopy() functions, depending on whether you want a shallow or deep copy of the object. Let's look at an example where we have a Car class:

Envall, Henrik. Python Design Patterns: A Very Practical Guide (p. 17). Kindle Edition.
"""

from __future__ import annotations

import copy


class Car:
    """
    Represents a car with a model and color.
    """

    def __init__(self, model: str, color: str) -> None:
        """
        Initialize a Car instance with a model and color.

        Args:
            model (str): The model of the car.
            color (str): The color of the car.
        """
        self.model = model
        self.color = color

    def __str__(self) -> str:
        """
        Return a string representation of the car.

        Returns:
            str: A string describing the car's color and model.
        """
        return f"{self.color} {self.model}"

    def clone(self) -> Car:
        """
        Create and return a shallow copy of the car.

        Returns:
            Car: A new Car instance with the same attributes as the original.
        """
        return copy.copy(self)


# In this Python code, we implement a clone() method in the Car class that returns a copy of the current instance. Here's how you can use it:


car1 = Car("SAAB 99 Turbo", "Red")
print(car1)
car2 = car1.clone()
print(car2)
car2.color = "Blue"
print(car1)
print(car2)

# In this example, car1 is cloned to create car2. Changing car2 does not affect car1 because car2 is a separate instance with its own data. This is because copy() creates a shallow copy of the object. The Prototype pattern can be a useful alternative to direct construction of objects, especially when creating an object involves a costly operation, such as reading from a database or making a network call. By creating a prototype and cloning it, you can potentially save system resources and improve performance. It also supports the principles of class encapsulation.

# Envall, Henrik. Python Design Patterns: A Very Practical Guide (p. 18). Kindle Edition.
