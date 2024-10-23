# pragma: exclude file
# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Interpreter Pattern
Overview of the Interpreter Pattern

The Interpreter pattern is a behavioral design pattern that deals with evaluating sentences in a language.
It provides a way to include language elements in a program to match and manipulate the given problem's syntax.
This pattern involves implementing an expression interface which tells how to interpret a particular context.
This pattern is used in compilers, parsers, or macro expansions.
"""

from __future__ import annotations

from typing import Dict


class Interpreter:
    """
    The Interpreter class interprets and executes the operations specified in the expressions.
    """

    def __init__(self) -> None:
        """
        Initialize the Interpreter with an empty dictionary of variables.
        """
        self.variables: dict[str, int] = {}

    def evaluate(self, expression: str) -> None:
        """
        Evaluate the given expression.

        Args:
            expression (str): The expression to be evaluated.
        """
        tokens = expression.split()
        operator = tokens[0]

        if operator == "add":
            self.variables[tokens[1]] = int(tokens[2]) + int(tokens[3])
        elif operator == "subtract":
            self.variables[tokens[1]] = int(tokens[2]) - int(tokens[3])
        else:
            raise ValueError(f"Invalid operation {operator}")


# Client code
interpreter = Interpreter()

interpreter.evaluate("add result 10 20")
print(interpreter.variables)  # Outputs: {'result': 30}

interpreter.evaluate("subtract result 30 10")
print(interpreter.variables)  # Outputs: {'result': 20}

"""
In this example, Interpreter is our interpreter class that interprets and executes the operations specified in the expressions.

The Interpreter pattern provides a way to include language elements in a program to match and manipulate the given problem's syntax.
This can be useful when you want to define a simple language for problems where the syntax is simple and efficiency isn't a critical concern.
However, the Interpreter pattern can become complicated when the grammar of the language becomes more complex.
For complex languages, tools like parser generators are a better option.
"""
