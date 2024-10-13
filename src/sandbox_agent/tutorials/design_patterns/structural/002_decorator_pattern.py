"""
Decorator Pattern

Overview of the Decorator Pattern

The Decorator pattern is a design pattern that allows behavior to be added to an individual object, either statically
or dynamically, without affecting the behavior of other objects from the same class. This type of design pattern comes
under structural patterns as this pattern acts as a wrapper to the existing class.

Implementing the Decorator Pattern in Python

Python's decorators allow you to extend and modify the behavior of a callable (functions, methods, or classes) without
permanently modifying the callable itself.

Any sufficiently generic functionality you can tack onto an existing class or function's behavior makes a great use
case for decoration. This includes the following:
- Logging
- Enforcing access control and authentication
- Instrumentation and timing functions
- Rate-limiting
- Caching, and more.
"""

from __future__ import annotations

from typing import Callable


def uppercase_decorator(function: Callable[[], str]) -> Callable[[], str]:
    """
    Decorator that converts the result of the decorated function to uppercase.

    Args:
        function (Callable[[], str]): The function to be decorated.

    Returns:
        Callable[[], str]: The decorated function.
    """

    def wrapper() -> str:
        """
        Wrapper function that calls the decorated function and converts its result to uppercase.

        Returns:
            str: The uppercase version of the decorated function's result.
        """
        func = function()
        make_uppercase = func.upper()
        return make_uppercase

    return wrapper


@uppercase_decorator
def say_hello() -> str:
    """
    Function that returns a greeting message.

    Returns:
        str: The greeting message.
    """
    return "hello"


print(say_hello())  # Outputs: HELLO


# In this example, `@uppercase_decorator` is a decorator that changes the behavior of the `say_hello` function.
# Instead of just returning 'hello', the function now returns 'HELLO'. The decorator accomplishes this by defining a
# wrapper function that calls the original function, modifies its result, and returns the modified result.

# The Decorator pattern is a flexible way to add responsibilities to objects without subclassing. It can often be a
# simple alternative to creating subclasses solely to extend functionality. However, it introduces complexity and can
# result in many small objects in your system, which can make debugging more challenging.

# ---

# Let me know if you need further assistance with this.
