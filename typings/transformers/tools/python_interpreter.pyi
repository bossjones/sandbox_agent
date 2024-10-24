"""
This type stub file was generated by pyright.
"""

import ast
from typing import Any, Callable, Dict

class InterpretorError(ValueError):
    """
    An error raised when the interpretor cannot evaluate a Python expression, due to syntax error or unsupported
    operations.
    """
    ...


def evaluate(code: str, tools: Dict[str, Callable], state=..., chat_mode=...): # -> Any | dict[Any, Any] | str | list[Any] | None:
    """
    Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (`str`):
            The code to evaluate.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpretorError`.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` should contain the initial inputs but will be
            updated by this function to contain all variables as they are evaluated.
        chat_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the function is called from `Agent.chat`.
    """
    ...

def evaluate_ast(expression: ast.AST, state: Dict[str, Any], tools: Dict[str, Callable]): # -> Any | dict[Any, Any] | str | list[Any] | None:
    """
    Evaluate an absract syntax tree using the content of the variables stored in a state and only evaluating a given
    set of functions.

    This function will recurse trough the nodes of the tree provided.

    Args:
        expression (`ast.AST`):
            The code to evaluate, as an abastract syntax tree.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` is updated if need be when the evaluation
            encounters assignements.
        tools (`Dict[str, Callable]`):
            The functions that may be called during the evaluation. Any call to another function will fail with an
            `InterpretorError`.
    """
    ...

def evaluate_assign(assign, state, tools):
    ...

def evaluate_call(call, state, tools):
    ...

def evaluate_subscript(subscript, state, tools):
    ...

def evaluate_name(name, state, tools):
    ...

def evaluate_condition(condition, state, tools): # -> Any | bool:
    ...

def evaluate_if(if_statement, state, tools): # -> None:
    ...

def evaluate_for(for_loop, state, tools): # -> None:
    ...
