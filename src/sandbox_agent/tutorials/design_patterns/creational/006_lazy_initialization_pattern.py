# creational design patterns

"""
Continuing our exploration of Creational Design Patterns, let's delve into the Lazy Initialization Pattern. This pattern is a tactic used to defer the instantiation of an object until the moment it is first needed. This strategy is particularly beneficial when dealing with resource-intensive objects.

Overview of the Lazy Initialization Pattern
In scenarios where an object is expensive to create, in terms of memory or time, and the object isn't immediately required or potentially might not be needed at all, the Lazy Initialization Pattern plays a critical role. It avoids unnecessary resource consumption by delaying the object's creation until it's indispensable.

Implementing the Lazy Initialization Pattern in Python
A Pythonic approach to the Lazy Initialization Pattern often involves the use of properties. Consider the following example:
"""

from __future__ import annotations


class ExpensiveObject:
    """
    Represents an expensive object that is resource-intensive to create.
    """

    def __init__(self) -> None:
        """
        Initialize an ExpensiveObject instance.
        """
        print("Creating an instance of ExpensiveObject is resource-intensive")


class Resource:
    """
    Manages access to an ExpensiveObject instance using lazy initialization.
    """

    def __init__(self) -> None:
        """
        Initialize a Resource instance with a None reference to an ExpensiveObject.
        """
        self._expensive_object: ExpensiveObject | None = None

    @property
    def expensive_object(self) -> ExpensiveObject:
        """
        Get the ExpensiveObject instance, creating it if it doesn't exist.

        Returns:
            ExpensiveObject: The ExpensiveObject instance.
        """
        if self._expensive_object is None:
            print("Initializing ExpensiveObject...")
            self._expensive_object = ExpensiveObject()
        return self._expensive_object


# Here, we've made the initialization of ExpensiveObject lazy. It's not instantiated when an instance of Resource is created. The ExpensiveObject is only created when we first try to access the expensive_object attribute.
res = Resource()  # ExpensiveObject isn't instantiated here
obj = res.expensive_object  # ExpensiveObject is created here
