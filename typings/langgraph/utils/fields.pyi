"""
This type stub file was generated by pyright.
"""

from typing import Any, Type

_DEFAULT_KEYS: frozenset[str] = ...
def get_field_default(name: str, type_: Any, schema: Type[Any]) -> Any:
    """Determine the default value for a field in a state schema.

    This is based on:
        If TypedDict:
            - Required/NotRequired
            - total=False -> everything optional
        - Type annotation (Optional/Union[None])
    """
    ...