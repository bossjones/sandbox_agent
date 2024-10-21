# pragma: exclude file
"""
Private Class Data Pattern

Overview of the Private Class Data Pattern

The Private Class Data design pattern seeks to reduce exposure of attributes by limiting their visibility. It reduces
the number of class attributes by encapsulating them in a single **Data** object. This provides better protection to
the class data, preventing unintended changes that might lead to an invalid or inconsistent state of the object.

Implementing the Private Class Data Pattern in Python

In Python, although we do not have direct access modifiers like `private` or `protected`, the convention is to denote
private attributes with a leading underscore. However, the Private Class Data pattern can still be implemented using
a separate class to encapsulate data attributes. Here is an example:
"""

from __future__ import annotations


class Data:
    def __init__(self, data: int):
        """
        Initialize the Data object with the given data.

        Args:
            data (int): The data to be encapsulated.
        """
        self._data = data

    @property
    def data(self) -> int:
        """
        Get the encapsulated data.

        Returns:
            int: The encapsulated data.
        """
        return self._data


class MyClass:
    def __init__(self, data: int):
        """
        Initialize MyClass with the given data.

        Args:
            data (int): The data to be encapsulated.
        """
        self._data = Data(data)

    def display_data(self) -> None:
        """
        Display the encapsulated data.
        """
        print(self._data.data)


# Client code
obj = MyClass(5)
obj.display_data()  # Outputs: 5


# In the above example, the **Data** class encapsulates the `_data` attribute. **MyClass** stores an instance of
# **Data** instead of storing `data` directly. A property decorator `@property` is used in the **Data** class to
# allow read access to `_data` but doesn't provide a setter method, which prevents modification of `_data`.

# This pattern provides a way to protect object state and prevent unauthorized changes. However, it may introduce
# some extra complexity and is not usually used in Python as there are other idiomatic ways to control access to
# attributes, such as name mangling and property decorators.

# ---

# Let me know if you need further clarification!
