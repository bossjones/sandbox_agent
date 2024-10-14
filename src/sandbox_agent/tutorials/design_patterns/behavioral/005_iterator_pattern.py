# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]
# pyright: ignore[reportCallInDefaultInitializer]
# pyright: ignore[reportAttributeAccessIssue]
# pyright: ignore[reportCallInDefaultInitializer]

"""
Iterator Pattern
Overview of the Iterator Pattern

The Iterator pattern is a behavioral design pattern that lets you traverse elements of a collection without exposing its underlying representation
(list, stack, tree, etc.). It provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class MyIterator:
    def __init__(self, items: list[Any] = []) -> None:  # pyright: ignore[reportAttributeAccessIssue,reportCallInDefaultInitializer]
        """
        Initialize the MyIterator with an optional list of items.

        Args:
            items (list[Any], optional): The list of items to iterate over. Defaults to [].
        """
        self._items = items
        self._index = 0

    def __iter__(self) -> Iterator[Any]:
        """
        Return the iterator object itself.

        Returns:
            Iterator[Any]: The iterator object.
        """
        return self

    def __next__(self) -> Any:
        """
        Return the next item from the iterator.

        Returns:
            Any: The next item in the iterator.

        Raises:
            StopIteration: If there are no more items to return.
        """
        if self._index < len(self._items):
            result = self._items[self._index]
            self._index += 1
            return result
        raise StopIteration


# Client code
my_iterator = MyIterator(["Apple", "Banana", "Cherry"])
for item in my_iterator:
    print(item)  # Outputs: Apple, Banana, Cherry

"""
In this example, MyIterator class is an iterator over a collection of items. It includes __iter__ method to return the iterator itself
and __next__ method to return the next value from the iterator.

The Iterator pattern provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
It lets you traverse elements of different collections in a similar way with a single iterator interface. However, an iterator can be
somewhat overkill if your collection is quite simple and easy to traverse with public APIs.
"""
