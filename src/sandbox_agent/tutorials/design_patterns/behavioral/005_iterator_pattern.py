# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]

"""
Iterator Pattern
Overview of the Iterator Pattern

The Iterator pattern is a behavioral design pattern that lets you traverse elements of a collection without exposing its underlying representation
(list, stack, tree, etc.). It provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
"""

from __future__ import annotations


class MyIterator:
    def __init__(self, items=[]):
        self._items = items
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
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
