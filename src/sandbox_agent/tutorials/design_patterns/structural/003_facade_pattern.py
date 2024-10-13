"""
Facade Pattern

Overview of the Facade Pattern

The Facade pattern is a design pattern that provides a simplified interface to a larger body of code, like a library
or a subsystem. It's called a facade because it provides a simpler face or interface to the underlying more complex
system.

The Facade design pattern is often used when a system is very complex or difficult to understand because the system
has a large number of interdependent classes or its source code is unavailable. It hides the complexities of the
larger system and provides a simpler interface to the client.

Implementing the Facade Pattern in Python

Let's consider an example where we have three complex subsystems: SubsystemA, SubsystemB, and SubsystemC. We can use
the Facade pattern to provide a simpler interface to these subsystems.
"""

from __future__ import annotations


class SubsystemA:
    def operation_a(self) -> str:
        """
        Perform operation A.

        Returns:
            str: The result of operation A.
        """
        return "Subsystem A, Method A\n"


class SubsystemB:
    def operation_b(self) -> str:
        """
        Perform operation B.

        Returns:
            str: The result of operation B.
        """
        return "Subsystem B, Method B\n"


class SubsystemC:
    def operation_c(self) -> str:
        """
        Perform operation C.

        Returns:
            str: The result of operation C.
        """
        return "Subsystem C, Method C\n"


class Facade:
    def __init__(self):
        """
        Initialize the Facade with instances of the subsystems.
        """
        self._subsystem_a = SubsystemA()
        self._subsystem_b = SubsystemB()
        self._subsystem_c = SubsystemC()

    def operation(self) -> str:
        """
        Perform a facade operation that uses the subsystems.

        Returns:
            str: The combined results of the subsystem operations.
        """
        results = ""
        results += self._subsystem_a.operation_a()
        results += self._subsystem_b.operation_b()
        results += self._subsystem_c.operation_c()
        return results


# Client Code
facade = Facade()
print(facade.operation())


# In this example, Facade provides a simple `operation` method that uses SubsystemA, SubsystemB, and SubsystemC. The
# client only needs to interact with the Facade, not the subsystem classes.

# The Facade pattern simplifies the client's interaction with a complex system. It doesn't prevent the client from
# accessing the system directly if it needs to, but it makes the system easier to use for most use cases. The Facade
# pattern can also be a point of decoupling, helping to minimize dependencies in your code. However, if used
# excessively or inappropriately, it can lead to problems of hiding necessary complexity.

# ---

# Let me know if you need further assistance!
